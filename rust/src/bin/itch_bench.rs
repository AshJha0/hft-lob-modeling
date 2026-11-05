//! ITCH decode, and decode into FastBook, on synthetic frames.  The generator
//! is integer-only and matches `cpp/fast/itch_bench.cpp` call for call, so the
//! two print the same state line.  Method: `docs/LATENCY.md`.
//!
//! Usage: `cargo run --release --bin itch_bench -- [messages] [live] [repeats]`

use std::time::Instant;

use lobmod::book::Side;
use lobmod::fast_book::{FastBook, FastConfig};
use lobmod::itch_decode::{apply, decode, Frames};

const BAND_LO: i64 = 1_000_000 - 65_536;
const BAND_LEVELS: i64 = 1 << 17;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

struct Live {
    reference: u64,
    shares: u32,
    buy: bool,
}

fn num(out: &mut Vec<u8>, v: u64, width: usize) {
    out.extend_from_slice(&v.to_be_bytes()[8 - width..]);
}

fn begin(out: &mut Vec<u8>, kind: u8, len: u64, ts: u64) {
    num(out, len, 2);
    out.push(kind);
    num(out, 1, 2);
    num(out, 0, 2);
    num(out, ts, 6);
}

fn quote(rng: &mut Rng, mid: u32, buy: bool) -> u32 {
    let d = (rng.next() % 64) as u32;
    if buy {
        mid - 1 - d
    } else {
        mid + 1 + d
    }
}

fn make_stream(n: usize, live: usize) -> Vec<u8> {
    let mut rng = Rng(42);
    let mut out = Vec::with_capacity(n * 36);
    let mut book: Vec<Live> = Vec::new();
    let (mut next_ref, mut next_match, mut mid) = (0u64, 0u64, 1_000_000u32);
    for i in 0..n {
        if i.is_multiple_of(1024) {
            mid = if rng.next() & 1 != 0 {
                mid + 1
            } else {
                mid - 1
            };
        }
        let ts = i as u64 * 1000;
        let r = rng.next() % 100;
        let sz = book.len();
        if sz < live || (r < 45 && sz < 2 * live) {
            let buy = rng.next() & 1 != 0;
            let px = quote(&mut rng, mid, buy);
            let shares = (1 + rng.next() % 500) as u32;
            let attributed = r.is_multiple_of(8);
            if attributed {
                begin(&mut out, b'F', 40, ts);
            } else {
                begin(&mut out, b'A', 36, ts);
            }
            next_ref += 1;
            num(&mut out, next_ref, 8);
            out.push(if buy { b'B' } else { b'S' });
            num(&mut out, u64::from(shares), 4);
            out.extend_from_slice(b"SYNTH   ");
            num(&mut out, u64::from(px), 4);
            if attributed {
                out.extend_from_slice(b"MPID");
            }
            book.push(Live {
                reference: next_ref,
                shares,
                buy,
            });
            continue;
        }
        if r >= 98 {
            begin(&mut out, b'P', 44, ts);
            num(&mut out, 0, 8);
            out.push(b'B');
            num(&mut out, 100, 4);
            out.extend_from_slice(b"SYNTH   ");
            num(&mut out, u64::from(mid), 4);
            next_match += 1;
            num(&mut out, next_match, 8);
            continue;
        }
        let at = (rng.next() % sz as u64) as usize;
        if r >= 92 {
            let px = quote(&mut rng, mid, book[at].buy);
            let shares = (1 + rng.next() % 500) as u32;
            begin(&mut out, b'U', 35, ts);
            num(&mut out, book[at].reference, 8);
            next_ref += 1;
            num(&mut out, next_ref, 8);
            num(&mut out, u64::from(shares), 4);
            num(&mut out, u64::from(px), 4);
            book[at].reference = next_ref;
            book[at].shares = shares;
            continue;
        }
        if r >= 70 && book[at].shares > 1 {
            let by = (1 + rng.next() % u64::from(book[at].shares - 1)) as u32;
            if r < 80 {
                begin(&mut out, b'X', 23, ts);
                num(&mut out, book[at].reference, 8);
                num(&mut out, u64::from(by), 4);
            } else {
                let with_price = r >= 88;
                if with_price {
                    begin(&mut out, b'C', 36, ts);
                } else {
                    begin(&mut out, b'E', 31, ts);
                }
                num(&mut out, book[at].reference, 8);
                num(&mut out, u64::from(by), 4);
                next_match += 1;
                num(&mut out, next_match, 8);
                if with_price {
                    out.push(b'Y');
                    num(&mut out, u64::from(mid), 4);
                }
            }
            book[at].shares -= by;
            continue;
        }
        begin(&mut out, b'D', 19, ts);
        num(&mut out, book[at].reference, 8);
        book.swap_remove(at);
    }
    out
}

fn decode_pass(buf: &[u8]) -> u64 {
    let mut sum = 0u64;
    for frame in Frames::new(buf) {
        let m = decode(frame).expect("valid frame").expect("book message");
        sum = sum
            .wrapping_add(m.reference)
            .wrapping_add(u64::from(m.shares))
            .wrapping_add(u64::from(m.price));
    }
    sum
}

fn book_pass(buf: &[u8], book: &mut FastBook) {
    for frame in Frames::new(buf) {
        let m = decode(frame).expect("valid frame").expect("book message");
        apply(book, &m).expect("consistent stream");
    }
}

fn report(stage: &str, n: usize, secs: &mut [f64]) {
    secs.sort_by(f64::total_cmp);
    let scale = 1e9 / n as f64;
    println!(
        "| {stage} | {n} | {:.1} | {:.1} |",
        secs[secs.len() / 2] * scale,
        secs[0] * scale
    );
}

#[cfg(target_os = "linux")]
fn pin() -> bool {
    extern "C" {
        fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
    }
    let mask = [1u64 << 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    // SAFETY: the mask is a live 128-byte buffer, the size of a cpu_set_t.
    unsafe { sched_setaffinity(0, std::mem::size_of_val(&mask), mask.as_ptr()) == 0 }
}

#[cfg(not(target_os = "linux"))]
fn pin() -> bool {
    false
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg =
        |i: usize, default: usize| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(default);
    let (n, live, repeats) = (arg(1, 5_000_000), arg(2, 20_000), arg(3, 7));
    let pinned = pin();
    let buf = make_stream(n, live);
    println!(
        "Rust pinned={} messages={n} live={live} repeats={repeats} bytes={}",
        u8::from(pinned),
        buf.len()
    );
    println!("| stage | messages | median ns/msg | best ns/msg |\n|---|---|---|---|");
    let (mut dec, mut full) = (Vec::new(), Vec::new());
    let mut sum = 0u64;
    for _ in 0..repeats {
        let t0 = Instant::now();
        sum = sum.wrapping_add(decode_pass(&buf));
        dec.push(t0.elapsed().as_secs_f64());
    }
    let cfg = FastConfig {
        min_price_ticks: BAND_LO,
        n_levels: BAND_LEVELS,
        max_orders: 1 << 17,
    };
    let mut state = (0, 0, 0, 0);
    for _ in 0..repeats {
        let mut book = FastBook::new(cfg).expect("valid config");
        let t0 = Instant::now();
        book_pass(&buf, &mut book);
        full.push(t0.elapsed().as_secs_f64());
        state = (
            book.best_bid().unwrap_or(0),
            book.best_ask().unwrap_or(0),
            book.total_size(Side::Bid),
            book.total_size(Side::Ask),
        );
    }
    report("decode", n, &mut dec);
    report("decode + FastBook", n, &mut full);
    println!(
        "state bid={} ask={} bid_size={} ask_size={} checksum={}",
        state.0,
        state.1,
        state.2,
        state.3,
        sum / repeats as u64
    );
}
