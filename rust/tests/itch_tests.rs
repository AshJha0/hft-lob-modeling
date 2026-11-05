//! ITCH decoder on hand-built frames, and the FastBook operations it drives.

use lobmod::book::{LimitOrderBook, Side};
use lobmod::fast_book::{FastBook, FastConfig, FastError};
use lobmod::itch_decode::{apply, decode, message_length, DecodeError, Frames, Message};

fn put(b: &mut Vec<u8>, v: u64, width: usize) {
    b.extend_from_slice(&v.to_be_bytes()[8 - width..]);
}

fn header(kind: u8) -> Vec<u8> {
    let mut b = vec![kind];
    put(&mut b, 0x0102, 2);
    put(&mut b, 0x0304, 2);
    put(&mut b, 0x0A0B_0C0D_0E0F, 6);
    b
}

fn add(reference: u64, side: u8, shares: u64, price: u64) -> Vec<u8> {
    let mut b = header(b'A');
    put(&mut b, reference, 8);
    b.push(side);
    put(&mut b, shares, 4);
    b.extend_from_slice(b"AAPL    ");
    put(&mut b, price, 4);
    b
}

fn exec(reference: u64, shares: u64) -> Vec<u8> {
    let mut b = header(b'E');
    put(&mut b, reference, 8);
    put(&mut b, shares, 4);
    put(&mut b, 77, 8);
    b
}

fn cancel(reference: u64, shares: u64) -> Vec<u8> {
    let mut b = header(b'X');
    put(&mut b, reference, 8);
    put(&mut b, shares, 4);
    b
}

fn del(reference: u64) -> Vec<u8> {
    let mut b = header(b'D');
    put(&mut b, reference, 8);
    b
}

fn replace(reference: u64, next: u64, shares: u64, price: u64) -> Vec<u8> {
    let mut b = header(b'U');
    put(&mut b, reference, 8);
    put(&mut b, next, 8);
    put(&mut b, shares, 4);
    put(&mut b, price, 4);
    b
}

fn must(b: &[u8]) -> Message<'_> {
    decode(b).unwrap().unwrap()
}

#[test]
fn length_table() {
    let want: [(u8, usize); 18] = [
        (b'A', 36),
        (b'F', 40),
        (b'E', 31),
        (b'C', 36),
        (b'X', 23),
        (b'D', 19),
        (b'U', 35),
        (b'P', 44),
        (b'Q', 40),
        (b'R', 39),
        (b'S', 12),
        (b'H', 25),
        (b'Y', 20),
        (b'L', 26),
        (b'I', 50),
        (b'J', 35),
        (b'K', 28),
        (b'V', 35),
    ];
    assert_eq!((0..=255u8).filter(|&t| message_length(t) != 0).count(), 18);
    for (t, n) in want {
        assert_eq!(message_length(t), n);
        let mut b = vec![b'B'; n + 1];
        b[0] = t;
        for short in 1..n {
            assert_eq!(decode(&b[..short]), Err(DecodeError::Truncated));
        }
        assert_eq!(decode(&b), Err(DecodeError::BadLength));
    }
}

#[test]
fn fields() {
    let a = add(0x1122_3344_5566_7788, b'S', 300, 1_234_500);
    let m = must(&a);
    assert_eq!((m.kind, m.locate, m.tracking), (b'A', 0x0102, 0x0304));
    assert_eq!(m.timestamp_ns, 0x0A0B_0C0D_0E0F);
    assert_eq!((m.reference, m.side), (0x1122_3344_5566_7788, b'S'));
    assert_eq!((m.shares, m.price), (300, 1_234_500));

    let mut f = add(9, b'B', 1, 2);
    f[0] = b'F';
    f.extend_from_slice(b"MPID");
    let m = must(&f);
    assert_eq!((m.kind, m.reference, m.side), (b'F', 9, b'B'));
    assert_eq!(m.stock, Some(b"AAPL    "));
    assert_eq!(m.attribution, Some(b"MPID"));
    assert!(std::ptr::eq(m.stock.unwrap().as_ptr(), f[24..].as_ptr()));

    let e = exec(5, 40);
    let m = must(&e);
    assert_eq!((m.kind, m.reference, m.shares), (b'E', 5, 40));
    assert_eq!((m.match_number, m.price), (77, 0));

    let mut c = exec(6, 41);
    c[0] = b'C';
    c.push(b'Y');
    put(&mut c, 999, 4);
    let m = must(&c);
    assert_eq!((m.kind, m.reference, m.shares), (b'C', 6, 41));
    assert_eq!((m.match_number, m.printable, m.price), (77, b'Y', 999));

    let x = cancel(7, 3);
    let m = must(&x);
    assert_eq!((m.kind, m.reference, m.shares), (b'X', 7, 3));
    let d = del(8);
    assert_eq!(must(&d).reference, 8);
    let u = replace(10, 11, 12, 13);
    let m = must(&u);
    assert_eq!((m.reference, m.new_reference), (10, 11));
    assert_eq!((m.shares, m.price), (12, 13));

    let mut p = add(0, b'B', 50, 60);
    p[0] = b'P';
    put(&mut p, 88, 8);
    let m = must(&p);
    assert_eq!((m.kind, m.side, m.shares), (b'P', b'B', 50));
    assert_eq!((m.price, m.match_number), (60, 88));
}

#[test]
fn refusals() {
    assert_eq!(decode(&[]), Err(DecodeError::Truncated));
    assert_eq!(decode(b"Zxxx"), Err(DecodeError::UnknownType));
    assert_eq!(decode(&add(1, b'x', 1, 1)), Err(DecodeError::BadSide));
    let mut s = header(b'S');
    s.push(b'O');
    assert_eq!(decode(&s), Ok(None));
}

#[test]
fn frames() {
    let mut buf = Vec::new();
    for m in [add(1, b'B', 1, 1), del(1)] {
        put(&mut buf, m.len() as u64, 2);
        buf.extend_from_slice(&m);
    }
    put(&mut buf, 50, 2); // a frame cut short
    buf.push(b'A');
    let mut it = Frames::new(&buf);
    assert_eq!(it.next().map(|f| (f.len(), f[0])), Some((36, b'A')));
    assert_eq!(it.next().map(|f| (f.len(), f[0])), Some((19, b'D')));
    assert_eq!(it.next(), None);
    assert_eq!(it.rest().len(), 3);
}

#[test]
fn arbitrary_bytes_never_panic() {
    let mut s = 1u64;
    let mut byte = move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (s >> 56) as u8
    };
    let kinds = b"AFECXDUPQRSHYLIJKVZ";
    let mut book = small_book(1, 256, 32);
    for _ in 0..200_000 {
        let k = kinds[usize::from(byte()) % kinds.len()];
        let n = message_length(k) + usize::from(byte() % 4 == 0) * usize::from(byte() % 3);
        let mut b: Vec<u8> = (0..n.max(1)).map(|_| byte()).collect();
        b[0] = k;
        if let Ok(Some(mut m)) = decode(&b) {
            m.reference &= 0x1F;
            m.new_reference &= 0x1F;
            m.price &= 0x1FF;
            m.shares &= 0xFF;
            let _ = apply(&mut book, &m);
        }
    }
}

fn small_book(min_price_ticks: i64, n_levels: i64, max_orders: u32) -> FastBook {
    FastBook::new(FastConfig {
        min_price_ticks,
        n_levels,
        max_orders,
    })
    .unwrap()
}

#[test]
fn mirror() {
    let mut b = small_book(1000, 1000, 16);
    let mut feed = |b: &mut FastBook, x: Vec<u8>| apply(b, &must(&x));
    feed(&mut b, add(1, b'B', 100, 1500)).unwrap();
    feed(&mut b, add(2, b'B', 50, 1500)).unwrap();
    feed(&mut b, add(3, b'S', 70, 1510)).unwrap();
    assert_eq!(b.best_bid(), Some(1500));
    assert_eq!(b.best_bid_size(), 150);
    assert_eq!(b.best_ask(), Some(1510));
    feed(&mut b, exec(1, 30)).unwrap();
    feed(&mut b, cancel(1, 20)).unwrap();
    assert_eq!(b.best_bid_size(), 100);
    assert_eq!(b.n_orders(Side::Bid), 2);
    feed(&mut b, exec(2, 50)).unwrap(); // full execution removes it
    assert!(!b.has_order(2));
    feed(&mut b, replace(1, 4, 25, 1505)).unwrap(); // side comes from the order
    assert!(!b.has_order(1) && b.has_order(4));
    assert_eq!((b.best_bid(), b.best_bid_size()), (Some(1505), 25));
    feed(&mut b, del(4)).unwrap();
    assert_eq!(b.best_bid(), None);
    assert_eq!(b.total_size(Side::Bid), 0);
    assert_eq!(feed(&mut b, del(4)), Err(FastError::UnknownId));
    assert_eq!(feed(&mut b, exec(3, 71)), Err(FastError::BadSize));
    assert_eq!(b.best_ask_size(), 70);
    let mut p = add(0, b'B', 50, 1510);
    p[0] = b'P';
    put(&mut p, 88, 8);
    feed(&mut b, p).unwrap(); // a hidden trade leaves the book alone
    assert_eq!(b.best_ask_size(), 70);
}

#[test]
fn rest_does_not_match() {
    let mut b = small_book(1, 100, 8);
    b.rest(Side::Ask, 50, 10, 1).unwrap();
    b.rest(Side::Bid, 60, 10, 2).unwrap(); // crossed, as before an open
    assert_eq!((b.best_bid(), b.best_ask()), (Some(60), Some(50)));
    assert_eq!(b.rest(Side::Bid, 60, 10, 2), Err(FastError::DuplicateId));
    assert_eq!(b.rest(Side::Bid, 600, 10, 3), Err(FastError::OutOfBand));
    assert_eq!(b.rest(Side::Bid, 60, 0, 3), Err(FastError::BadSize));
}

#[test]
fn replace_refusals_change_nothing() {
    let mut b = small_book(1, 100, 8);
    b.rest(Side::Bid, 50, 10, 1).unwrap();
    b.rest(Side::Bid, 40, 10, 2).unwrap();
    assert_eq!(b.replace(9, 3, 50, 10), Err(FastError::UnknownId));
    assert_eq!(b.replace(1, 2, 50, 10), Err(FastError::DuplicateId));
    assert_eq!(b.replace(1, 3, 500, 10), Err(FastError::OutOfBand));
    assert_eq!(b.replace(1, 3, 50, 0), Err(FastError::BadSize));
    assert!(b.has_order(1));
    assert_eq!(b.total_size(Side::Bid), 20);
    b.replace(1, 1, 45, 7).unwrap(); // the same id is legal
    assert_eq!((b.best_bid(), b.best_bid_size()), (Some(45), 7));
}

#[test]
fn reduce_matches_reference() {
    let mut r = LimitOrderBook::new(1.0).unwrap();
    let mut b = small_book(1, 100, 8);
    let mut no_sink = |_: i64, _: i64, _: i64| {};
    for id in 1..=3 {
        r.limit_ticks(Side::Bid, 50, 10, id).unwrap();
        b.limit(Side::Bid, 50, 10, id, &mut no_sink).unwrap();
    }
    assert_eq!(b.reduce(1, 4).unwrap(), r.reduce(1, 4).unwrap());
    assert_eq!(b.reduce(2, 10).unwrap(), r.reduce(2, 10).unwrap());
    assert!(!b.has_order(2) && !r.has_order(2));
    assert_eq!(b.reduce(1, 7), Err(FastError::BadSize));
    assert!(r.reduce(1, 7).is_err());
    let mut order = Vec::new();
    let mut rec = |id: i64, _: i64, _: i64| order.push(id);
    assert_eq!(b.market(Side::Ask, 100, &mut rec), Ok(84));
    assert_eq!(order, [1, 3]); // 1 kept its place
}
