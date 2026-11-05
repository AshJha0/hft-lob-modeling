//! Property tests: the reference book against a deliberately naive model, the
//! same idea as python/tests/test_book_properties.py.  The model keeps one flat
//! list of resting orders and finds each match by sorting it, so it shares no
//! data structures with the book.  Seeded, with no extra dependency.

use lobmod::book::{Fill, LimitOrderBook, Side};
use lobmod::rng::Lcg64;

#[derive(Clone, Copy)]
struct Resting {
    arrival: u64,
    side: Side,
    price: i64,
    id: i64,
    size: i64,
}

#[derive(Default)]
struct Naive {
    orders: Vec<Resting>,
    clock: u64,
}

impl Naive {
    fn execute(&mut self, side: Side, mut size: i64, limit: Option<i64>) -> (Vec<Fill>, i64) {
        let mut idx: Vec<usize> = (0..self.orders.len())
            .filter(|&i| self.orders[i].side != side)
            .collect();
        idx.sort_by_key(|&i| {
            let o = &self.orders[i];
            (
                if side == Side::Bid { o.price } else { -o.price },
                o.arrival,
            )
        });
        let mut fills = Vec::new();
        for i in idx {
            let o = &mut self.orders[i];
            let beyond = limit.is_some_and(|l| match side {
                Side::Bid => o.price > l,
                Side::Ask => o.price < l,
            });
            if size == 0 || beyond {
                break;
            }
            let take = size.min(o.size);
            o.size -= take;
            size -= take;
            fills.push(Fill {
                maker_id: o.id,
                price: o.price,
                size: take,
            });
        }
        self.orders.retain(|o| o.size > 0);
        (fills, size)
    }

    fn limit(&mut self, side: Side, price: i64, size: i64, id: i64) -> (Vec<Fill>, i64) {
        let (fills, rest) = self.execute(side, size, Some(price));
        if rest > 0 {
            self.clock += 1;
            self.orders.push(Resting {
                arrival: self.clock,
                side,
                price,
                id,
                size: rest,
            });
        }
        (fills, rest)
    }

    fn cancel(&mut self, id: i64) -> i64 {
        let at = self.orders.iter().position(|o| o.id == id).unwrap();
        self.orders.remove(at).size
    }

    fn best(&self, side: Side) -> Option<i64> {
        let prices = self.orders.iter().filter(|o| o.side == side);
        match side {
            Side::Bid => prices.map(|o| o.price).max(),
            Side::Ask => prices.map(|o| o.price).min(),
        }
    }

    fn total(&self, side: Side) -> i64 {
        let on_side = self.orders.iter().filter(|o| o.side == side);
        on_side.map(|o| o.size).sum()
    }
}

/// Seeded integer in `[lo, hi]`.
fn draw(rng: &mut Lcg64, lo: i64, hi: i64) -> i64 {
    lo + (rng.uniform() * (hi - lo + 1) as f64) as i64
}

#[test]
fn book_matches_naive_model() {
    for seed in 1..=40u64 {
        let mut rng = Lcg64::new(seed).unwrap();
        let mut book = LimitOrderBook::new(1.0).unwrap();
        let mut model = Naive::default();
        let mut next_id = 1;
        for _ in 0..400 {
            let op = rng.uniform();
            let side = if rng.uniform() < 0.5 {
                Side::Bid
            } else {
                Side::Ask
            };
            if op < 0.55 {
                let (price, size) = (draw(&mut rng, 90, 110), draw(&mut rng, 1, 19));
                let got = book.limit_ticks(side, price, size, next_id).unwrap();
                assert_eq!(got, model.limit(side, price, size, next_id));
                next_id += 1;
            } else if op < 0.8 && !model.orders.is_empty() {
                let at = draw(&mut rng, 0, model.orders.len() as i64 - 1) as usize;
                let id = model.orders[at].id;
                assert_eq!(book.cancel(id).unwrap(), model.cancel(id));
            } else {
                let size = draw(&mut rng, 1, 39);
                let got = book.market(side, size).unwrap();
                assert_eq!(got, model.execute(side, size, None));
            }
            assert_eq!(book.best_bid(), model.best(Side::Bid));
            assert_eq!(book.best_ask(), model.best(Side::Ask));
            assert_eq!(book.total_size(Side::Bid), model.total(Side::Bid));
            assert_eq!(book.total_size(Side::Ask), model.total(Side::Ask));
            if let (Some(b), Some(a)) = (book.best_bid(), book.best_ask()) {
                assert!(b < a, "crossed book {b} {a}");
            }
        }
    }
}
