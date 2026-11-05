//! Zero-intelligence order-flow generator (pinned, seeded).
//!
//! Flow loosely inspired by Smith et al. (2003), not their model: Poisson event-type rates around the best quotes
//! with geometric placement of passive limit orders.  Everything is pinned —
//! RNG ([`Lcg64`]), draw order, placement and size formulas — so that every
//! language regenerates the identical 5000-event stream.
//!
//! Pinned algorithm
//! ----------------
//! Seed phase (events 1..12): for `i = 0..5` submit a buy limit at `999 - i`
//! size 10 then a sell limit at `1001 + i` size 10 (ids = seq numbers).
//!
//! Stochastic phase (events 13..n).  With `nb, ns` the live order counts per
//! side, the event-type rates are (this exact order):
//!
//! ```text
//! limit buy 1.0 | limit sell 1.0 | cancel buy 0.04*nb | cancel sell 0.04*ns
//! market buy 0.25 | market sell 0.25
//! ```
//!
//! Draw `u1 = uniform()` and pick the type by cumulative probability in the
//! order above.  Then:
//!
//! * limit: `u2 = uniform()`; if `u2 < 0.12` and the opposite best exists the
//!   order is *aggressive* — priced AT the opposite best (a crossed limit
//!   that executes on arrival).  Otherwise passive: `k = geometric(0.35)` and
//!   price `opp_best -/+ (1 + k)` (or `1000 -/+ (1 + k)` when the opposite
//!   side is empty), clamped to >= 1 tick.  Size `1 + randint(5)`; id = seq.
//! * cancel: uniform pick from the side's live order ids sorted ascending:
//!   `ids[randint(len(ids))]`.  (Rate is 0 when the side is empty.)
//! * market: `u3 = uniform()`; size `40 + randint(40)` if `u3 < 0.02` (an
//!   occasional sweep that walks several levels) else `1 + randint(3)`.
//!
//! Every event is applied to an internal book immediately so that quotes and
//! live-order sets evolve exactly as a replay of the emitted stream would.

use crate::book::{LimitOrderBook, Side};
use crate::error::{LobError, Result};
use crate::replay::{apply_event, Event, EventType};
use crate::rng::Lcg64;

const RATE_LIMIT: f64 = 1.0;
const RATE_MARKET: f64 = 0.25;
const RATE_CANCEL_PER_ORDER: f64 = 0.04;
const P_AGGRESSIVE: f64 = 0.12;
const P_GEOMETRIC: f64 = 0.35;
const P_SWEEP: f64 = 0.02;
const REF_PRICE: i64 = 1000;

/// Generate a deterministic zero-intelligence event stream.
///
/// # Errors
/// [`LobError::InvalidInput`] if `n_events < 12` (the first 12 events seed
/// the book).
pub fn generate_events(n_events: usize, seed: u64) -> Result<Vec<Event>> {
    if n_events < 12 {
        return Err(LobError::InvalidInput(format!(
            "n_events must be an integer >= 12, got {n_events}"
        )));
    }
    let mut rng = Lcg64::new(seed)?;
    let mut book = LimitOrderBook::with_options(1.0, false)?;
    let mut events: Vec<Event> = Vec::with_capacity(n_events);

    fn emit(
        book: &mut LimitOrderBook,
        events: &mut Vec<Event>,
        etype: EventType,
        side: Side,
        price: i64,
        size: i64,
        order_id: i64,
    ) -> Result<()> {
        let event = Event {
            seq: events.len() as i64 + 1,
            etype,
            side,
            price,
            size,
            order_id,
        };
        apply_event(book, &event)?;
        events.push(event);
        Ok(())
    }

    // -- seed phase: 6 levels each side, size 10
    for i in 0..6 {
        let id = events.len() as i64 + 1;
        emit(
            &mut book,
            &mut events,
            EventType::Limit,
            Side::Bid,
            REF_PRICE - 1 - i,
            10,
            id,
        )?;
        let id = events.len() as i64 + 1;
        emit(
            &mut book,
            &mut events,
            EventType::Limit,
            Side::Ask,
            REF_PRICE + 1 + i,
            10,
            id,
        )?;
    }

    // -- stochastic phase
    while events.len() < n_events {
        let seq = events.len() as i64 + 1;
        let nb = book.n_orders(Side::Bid) as f64;
        let ns = book.n_orders(Side::Ask) as f64;
        let rates = [
            RATE_LIMIT,                 // limit buy
            RATE_LIMIT,                 // limit sell
            RATE_CANCEL_PER_ORDER * nb, // cancel buy
            RATE_CANCEL_PER_ORDER * ns, // cancel sell
            RATE_MARKET,                // market buy
            RATE_MARKET,                // market sell
        ];
        let total: f64 = rates.iter().sum();
        let x = rng.uniform() * total;
        let mut kind = 0usize;
        let mut acc = rates[0];
        while x >= acc && kind < 5 {
            kind += 1;
            acc += rates[kind];
        }

        match kind {
            0 | 1 => {
                // limit order
                let buy = kind == 0;
                let opp_best = if buy {
                    book.best_ask()
                } else {
                    book.best_bid()
                };
                let u2 = rng.uniform();
                let price = if u2 < P_AGGRESSIVE && opp_best.is_some() {
                    opp_best.expect("checked is_some")
                } else {
                    let k = rng.geometric(P_GEOMETRIC)?;
                    let anchor = opp_best.unwrap_or(REF_PRICE);
                    if buy {
                        anchor - (1 + k)
                    } else {
                        anchor + (1 + k)
                    }
                };
                let price = price.max(1);
                let size = 1 + rng.randint(5)? as i64;
                let side = if buy { Side::Bid } else { Side::Ask };
                emit(
                    &mut book,
                    &mut events,
                    EventType::Limit,
                    side,
                    price,
                    size,
                    seq,
                )?;
            }
            2 | 3 => {
                // cancel (rate is 0 when the side is empty, so ids is non-empty)
                let side = if kind == 2 { Side::Bid } else { Side::Ask };
                let ids = book.order_ids(side);
                let oid = ids[rng.randint(ids.len())?];
                emit(&mut book, &mut events, EventType::Cancel, side, 0, 0, oid)?;
            }
            _ => {
                // market order
                let side = if kind == 4 { Side::Bid } else { Side::Ask };
                let u3 = rng.uniform();
                let size = if u3 < P_SWEEP {
                    40 + rng.randint(40)? as i64
                } else {
                    1 + rng.randint(3)? as i64
                };
                emit(&mut book, &mut events, EventType::Market, side, 0, size, 0)?;
            }
        }
    }
    Ok(events)
}
