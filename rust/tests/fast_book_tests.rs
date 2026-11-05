//! FastBook against the reference book: identical fills and quotes on seeded
//! random operation sequences, plus the error codes.

use lobmod::book::{Fill, LimitOrderBook, Side, MAX_TICKS};
use lobmod::fast_book::{FastBook, FastConfig, FastError};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % (hi - lo + 1) as u64) as i64
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn same(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a == b
}

fn same_state(r: &LimitOrderBook, f: &FastBook) {
    assert_eq!(r.best_bid(), f.best_bid());
    assert_eq!(r.best_ask(), f.best_ask());
    assert_eq!(r.best_bid_size(), f.best_bid_size());
    assert_eq!(r.best_ask_size(), f.best_ask_size());
    assert_eq!(r.total_size(Side::Bid), f.total_size(Side::Bid));
    assert_eq!(r.total_size(Side::Ask), f.total_size(Side::Ask));
    assert!(same(r.mid(), f.mid()));
    assert!(same(r.spread(), f.spread()));
    assert!(same(r.microprice(), f.microprice()));
}

fn same_deep_state(r: &LimitOrderBook, f: &FastBook) {
    for side in [Side::Bid, Side::Ask] {
        assert_eq!(r.n_orders(side) as i64, f.n_orders(side));
        for k in [1, 3, 10, 1000] {
            assert_eq!(r.depth(side, k).unwrap(), f.depth(side, k));
        }
    }
}

fn differential(seed: u64, lo: i64, hi: i64, steps: usize, id_stride: i64) {
    let mut rng = Rng(seed);
    let mut r = LimitOrderBook::with_options(1.0, false).unwrap();
    let mut f = FastBook::new(FastConfig {
        min_price_ticks: lo,
        n_levels: hi - lo + 1,
        max_orders: 4096,
    })
    .unwrap();
    let mut ids: Vec<i64> = Vec::new();
    let mut counter = 0i64;
    for step in 0..steps {
        let side = if rng.unit() < 0.5 {
            Side::Bid
        } else {
            Side::Ask
        };
        let op = rng.unit();
        let mut got: Vec<Fill> = Vec::new();
        let mut sink = |maker_id, price, size| {
            got.push(Fill {
                maker_id,
                price,
                size,
            })
        };
        if op < 0.55 {
            let (px, size) = (rng.range(lo, hi), rng.range(1, 20));
            counter += 1;
            let id = if counter % 2 == 1 { counter } else { -counter } * id_stride;
            let want = r.limit_ticks(side, px, size, id).unwrap();
            let rest = f.limit(side, px, size, id, &mut sink).unwrap();
            assert_eq!(rest, want.1);
            assert_eq!(got, want.0);
            if rest > 0 {
                ids.push(id);
            }
        } else if op < 0.80 && !ids.is_empty() {
            let at = (rng.next() % ids.len() as u64) as usize;
            let id = ids.swap_remove(at);
            if r.has_order(id) {
                assert_eq!(f.cancel(id), Ok(r.cancel(id).unwrap()));
            } else {
                assert_eq!(f.cancel(id), Err(FastError::UnknownId));
            }
        } else if op < 0.97 {
            let size = rng.range(1, 60);
            let want = r.market(side, size).unwrap();
            let unfilled = f.market(side, size, &mut sink).unwrap();
            assert_eq!(unfilled, want.1);
            assert_eq!(got, want.0);
        } else if !ids.is_empty() {
            let id = ids[(rng.next() % ids.len() as u64) as usize];
            if r.has_order(id) {
                assert!(r.limit_ticks(Side::Bid, lo, 1, id).is_err());
                assert_eq!(
                    f.limit(Side::Bid, lo, 1, id, &mut sink),
                    Err(FastError::DuplicateId)
                );
                assert!(got.is_empty());
            }
        }
        same_state(&r, &f);
        if step % 64 == 0 {
            same_deep_state(&r, &f);
        }
    }
    same_deep_state(&r, &f);
    for id in ids {
        assert_eq!(r.has_order(id), f.has_order(id));
    }
}

#[test]
fn dense_band_matches_reference() {
    for seed in 1..=40 {
        differential(seed, 90, 110, 3000, 1);
    }
}

#[test]
fn sparse_band_matches_reference() {
    for seed in 101..=140 {
        differential(seed, 1, 5000, 3000, 7);
    }
}

#[test]
fn wide_band_and_spread_ids_match_reference() {
    for seed in 201..=240 {
        differential(seed, 1000, 300_000, 2000, 1 << 40);
    }
}

#[test]
fn error_codes_leave_the_book_alone() {
    let mut f = FastBook::new(FastConfig {
        min_price_ticks: 100,
        n_levels: 11,
        max_orders: 2,
    })
    .unwrap();
    let mut none = |_: i64, _: i64, _: i64| panic!("no fill expected");
    let b = Side::Bid;
    assert_eq!(f.limit(b, 100, 0, 1, &mut none), Err(FastError::BadSize));
    assert_eq!(f.limit(b, 0, 1, 1, &mut none), Err(FastError::BadPrice));
    assert_eq!(f.limit(b, 99, 1, 1, &mut none), Err(FastError::OutOfBand));
    assert_eq!(f.limit(b, 111, 1, 1, &mut none), Err(FastError::OutOfBand));
    assert_eq!(f.market(b, -1, &mut none), Err(FastError::BadSize));
    assert_eq!(f.cancel(1), Err(FastError::UnknownId));
    assert_eq!(f.limit(b, 100, 5, 1, &mut none), Ok(5));
    assert_eq!(
        f.limit(b, 110, 5, 1, &mut none),
        Err(FastError::DuplicateId)
    );
    assert_eq!(
        f.limit(b, 110, MAX_TICKS, 2, &mut none),
        Err(FastError::SideOverflow)
    );
    assert_eq!(f.limit(b, 110, 5, 2, &mut none), Ok(5));
    assert_eq!(
        f.limit(Side::Ask, 100, 1, 3, &mut none),
        Err(FastError::PoolFull)
    );
    assert_eq!(f.total_size(b), 10);
    assert_eq!(f.cancel(1), Ok(5));
    assert_eq!(f.market(Side::Ask, 9, &mut |_, _, _| {}), Ok(4));
    assert_eq!(f.best_bid(), None);
    assert!(f.mid().is_nan());
    assert!(FastBook::new(FastConfig {
        min_price_ticks: 0,
        n_levels: 10,
        max_orders: 10
    })
    .is_none());
}
