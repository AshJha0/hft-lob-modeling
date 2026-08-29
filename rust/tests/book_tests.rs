//! Book mechanics: invariants over the full bundled replay, FIFO / price
//! priority via targeted sequences, market orders walking and exhausting the
//! book, crossed limits, cancels, NaN guards, tick validation.

mod common;

use lobmod::{
    apply_event, generate_events, read_events_csv, replay, EventType, LimitOrderBook, LobError,
    Side,
};

fn book() -> LimitOrderBook {
    LimitOrderBook::new(1.0).expect("tick 1.0 is valid")
}

// ------------------------------------------------------- full-replay invariants

#[test]
fn replay_invariants_and_size_conservation() {
    let events = read_events_csv(&common::data_path("lob_events.csv")).expect("events");
    assert_eq!(events.len(), 5000);
    let mut book = book();
    let mut total_limit = 0i64; // size submitted via limit orders
    let mut limit_taker = 0i64; // marketable-limit executions on arrival
    let mut cancelled = 0i64;
    for e in &events {
        match e.etype {
            EventType::Limit => {
                let (_, rested) = book
                    .limit(e.side, e.price as f64, e.size, e.order_id)
                    .expect("valid limit");
                total_limit += e.size;
                limit_taker += e.size - rested;
            }
            EventType::Cancel => cancelled += book.cancel(e.order_id).expect("valid cancel"),
            EventType::Market => {
                book.market(e.side, e.size).expect("valid market");
            }
        }
        // price sanity after every event: the book is never crossed
        if let (Some(b), Some(a)) = (book.best_bid(), book.best_ask()) {
            assert!(b < a, "crossed book after event {}: bid {b} >= ask {a}", e.seq);
        }
    }
    // every fill consumes a resting limit (maker side), so:
    // limit size in = taker-executed on arrival + cancelled + maker-filled + still resting
    let maker_filled: i64 = book.trades.iter().map(|f| f.size).sum();
    let resting = book.total_size(Side::Bid) + book.total_size(Side::Ask);
    assert_eq!(total_limit, limit_taker + cancelled + maker_filled + resting);
    assert!(maker_filled > 0 && cancelled > 0);
}

#[test]
fn replay_matches_zi_regeneration() {
    // the bundled stream is the pinned ZI generator's output (seed 20240613)
    let bundled = read_events_csv(&common::data_path("lob_events.csv")).expect("events");
    let generated = generate_events(5000, 20240613).expect("generate");
    assert_eq!(bundled, generated);
}

#[test]
fn zi_generator_validation() {
    assert!(matches!(
        generate_events(11, 1),
        Err(LobError::InvalidInput(_))
    ));
    assert_eq!(generate_events(12, 1).expect("seed phase only").len(), 12);
}

// ------------------------------------------------------------ priority + fills

#[test]
fn fifo_time_priority_within_level() {
    let mut b = book();
    b.limit(Side::Bid, 100.0, 5, 1).unwrap();
    b.limit(Side::Bid, 100.0, 7, 2).unwrap();
    let (fills, unfilled) = b.market(Side::Ask, 6).unwrap();
    assert_eq!(unfilled, 0);
    // order 1 (earlier) fills first and fully, order 2 partially
    assert_eq!(fills.len(), 2);
    assert_eq!((fills[0].maker_id, fills[0].size), (1, 5));
    assert_eq!((fills[1].maker_id, fills[1].size), (2, 1));
    assert!(!b.has_order(1));
    assert_eq!(b.best_bid_size(), 6);
}

#[test]
fn price_priority_across_levels() {
    let mut b = book();
    b.limit(Side::Bid, 99.0, 5, 1).unwrap();
    b.limit(Side::Bid, 100.0, 5, 2).unwrap();
    let (fills, _) = b.market(Side::Ask, 3).unwrap();
    // the better (higher) bid fills first, at its own price
    assert_eq!((fills[0].maker_id, fills[0].price), (2, 100));
}

#[test]
fn market_order_walks_multiple_levels() {
    let mut b = book();
    b.limit(Side::Ask, 101.0, 5, 1).unwrap();
    b.limit(Side::Ask, 102.0, 5, 2).unwrap();
    b.limit(Side::Ask, 103.0, 5, 3).unwrap();
    let (fills, unfilled) = b.market(Side::Bid, 12).unwrap();
    assert_eq!(unfilled, 0);
    let prices: Vec<i64> = fills.iter().map(|f| f.price).collect();
    assert_eq!(prices, vec![101, 102, 103]);
    assert_eq!(fills[2].size, 2);
    assert_eq!(b.best_ask(), Some(103));
    assert_eq!(b.best_ask_size(), 3);
}

#[test]
fn market_bigger_than_book_reports_unfilled() {
    let mut b = book();
    b.limit(Side::Ask, 101.0, 4, 1).unwrap();
    b.limit(Side::Ask, 102.0, 6, 2).unwrap();
    let (fills, unfilled) = b.market(Side::Bid, 25).unwrap();
    assert_eq!(fills.iter().map(|f| f.size).sum::<i64>(), 10);
    assert_eq!(unfilled, 15); // reported, not an error
    assert_eq!(b.best_ask(), None);
    assert!(b.mid().is_nan());
}

#[test]
fn crossed_limit_executes_then_rests() {
    let mut b = book();
    b.limit(Side::Ask, 101.0, 5, 1).unwrap();
    b.limit(Side::Ask, 103.0, 5, 2).unwrap();
    // buy limit at 102 crosses 101 but must not trade through to 103
    let (fills, rested) = b.limit(Side::Bid, 102.0, 8, 3).unwrap();
    assert_eq!(fills.len(), 1);
    assert_eq!((fills[0].maker_id, fills[0].price, fills[0].size), (1, 101, 5));
    assert_eq!(rested, 3);
    assert_eq!(b.best_bid(), Some(102));
    assert_eq!(b.best_bid_size(), 3);
    assert_eq!(b.best_ask(), Some(103));
}

#[test]
fn crossed_limit_fully_filled_does_not_rest() {
    let mut b = book();
    b.limit(Side::Ask, 101.0, 10, 1).unwrap();
    let (fills, rested) = b.limit(Side::Bid, 101.0, 4, 2).unwrap();
    assert_eq!(rested, 0);
    assert_eq!(fills[0].price, 101); // executes at the resting order's price
    assert!(!b.has_order(2));
    assert_eq!(b.best_ask_size(), 6);
}

// --------------------------------------------------------------- cancels + ids

#[test]
fn cancel_semantics() {
    let mut b = book();
    b.limit(Side::Bid, 100.0, 5, 1).unwrap();
    assert_eq!(b.cancel(1).unwrap(), 5);
    assert_eq!(b.best_bid(), None);
    // unknown / already-cancelled ids are errors
    assert!(matches!(b.cancel(1), Err(LobError::UnknownOrder(1))));
    assert!(matches!(b.cancel(999), Err(LobError::UnknownOrder(999))));
    // a fully filled order can no longer be cancelled
    b.limit(Side::Ask, 101.0, 3, 2).unwrap();
    b.market(Side::Bid, 3).unwrap();
    assert!(b.cancel(2).is_err());
}

#[test]
fn duplicate_live_order_id_rejected() {
    let mut b = book();
    b.limit(Side::Bid, 100.0, 5, 7).unwrap();
    assert!(b.limit(Side::Bid, 99.0, 5, 7).is_err());
    // after cancel the id may be reused
    b.cancel(7).unwrap();
    assert!(b.limit(Side::Bid, 99.0, 5, 7).is_ok());
}

// --------------------------------------------------------- guards + validation

#[test]
fn empty_side_nan_guards() {
    let mut b = book();
    assert!(b.mid().is_nan());
    assert!(b.spread().is_nan());
    assert!(b.microprice().is_nan());
    assert_eq!(b.best_bid_size(), 0);
    assert_eq!(b.depth(Side::Bid, 3).unwrap(), 0);
    b.limit(Side::Bid, 100.0, 5, 1).unwrap();
    assert!(b.mid().is_nan(), "one-sided book still has no mid");
    b.limit(Side::Ask, 102.0, 15, 2).unwrap();
    assert_eq!(b.mid(), 101.0);
    assert_eq!(b.spread(), 2.0);
    // microprice leans toward the thin (bid) side
    assert_eq!(b.microprice(), (15.0 * 100.0 + 5.0 * 102.0) / 20.0);
}

#[test]
fn depth_sums_top_k_levels() {
    let mut b = book();
    b.limit(Side::Bid, 100.0, 1, 1).unwrap();
    b.limit(Side::Bid, 99.0, 2, 2).unwrap();
    b.limit(Side::Bid, 98.0, 4, 3).unwrap();
    b.limit(Side::Bid, 97.0, 8, 4).unwrap();
    assert_eq!(b.depth(Side::Bid, 3).unwrap(), 7);
    assert_eq!(b.depth(Side::Bid, 10).unwrap(), 15); // fewer levels: sum what exists
    assert!(b.depth(Side::Bid, 0).is_err());
}

#[test]
fn tick_validation() {
    assert!(LimitOrderBook::new(0.0).is_err());
    assert!(LimitOrderBook::new(-1.0).is_err());
    assert!(LimitOrderBook::new(f64::NAN).is_err());
    let mut b = LimitOrderBook::new(0.5).expect("valid tick");
    assert!(b.limit(Side::Bid, 100.25, 5, 1).is_err()); // off-grid
    assert!(b.limit(Side::Bid, -1.0, 5, 1).is_err());
    assert!(b.limit(Side::Bid, f64::INFINITY, 5, 1).is_err());
    assert!(b.limit(Side::Bid, 100.5, 5, 1).is_ok()); // on-grid: 201 ticks
    assert_eq!(b.best_bid(), Some(201));
    assert!(b.limit(Side::Bid, 100.0, 0, 2).is_err()); // non-positive size
    assert!(b.market(Side::Bid, -3).is_err());
}

#[test]
fn replay_records_after_each_event_and_rejects_empty() {
    assert!(replay(&[], 3, 1.0).is_err());
    let events = read_events_csv(&common::data_path("lob_events.csv")).expect("events");
    let st = replay(&events[..20], 3, 1.0).expect("replay prefix");
    assert_eq!(st.len(), 20);
    // after the 12-event seed phase: bids 994..999, asks 1001..1006, size 10 each
    assert_eq!(st.best_bid[11], 999.0);
    assert_eq!(st.best_ask[11], 1001.0);
    assert_eq!(st.mid[11], 1000.0);
    assert_eq!(st.bid_depth_k[11], 30.0);

    // replaying the stream through apply_event matches the recorded states
    let mut b = book();
    for e in &events[..20] {
        apply_event(&mut b, e).expect("apply");
    }
    assert_eq!(b.best_bid().map(|p| p as f64), Some(st.best_bid[19]));
    assert_eq!(b.best_ask().map(|p| p as f64), Some(st.best_ask[19]));
}

#[test]
fn read_events_csv_rejects_bad_input() {
    let dir = std::env::temp_dir();
    let bad_header = dir.join("lobmod_bad_header.csv");
    std::fs::write(&bad_header, "a,b,c\n1,L,B,999,10,1\n").unwrap();
    assert!(read_events_csv(bad_header.to_str().unwrap()).is_err());
    let bad_type = dir.join("lobmod_bad_type.csv");
    std::fs::write(
        &bad_type,
        "seq,type,side,price,size,order_id\n1,X,B,999,10,1\n",
    )
    .unwrap();
    assert!(read_events_csv(bad_type.to_str().unwrap()).is_err());
    assert!(read_events_csv("/nonexistent/path.csv").is_err());
}
