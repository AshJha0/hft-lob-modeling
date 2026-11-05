//! Book mechanics: invariants over the full bundled replay, FIFO / price
//! priority via targeted sequences, market orders walking and exhausting the
//! book, crossed limits, cancels, NaN guards, tick validation.

mod common;

use lobmod::{
    apply_event, generate_events, read_events_csv, replay, write_events_csv, EventType, Fill,
    Lcg64, LimitOrderBook, LobError, Side, MAX_SEED, MAX_TICKS,
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
            assert!(
                b < a,
                "crossed book after event {}: bid {b} >= ask {a}",
                e.seq
            );
        }
    }
    // every fill consumes a resting limit (maker side), so:
    // limit size in = taker-executed on arrival + cancelled + maker-filled + still resting
    let maker_filled: i64 = book.trades.iter().map(|f| f.size).sum();
    let resting = book.total_size(Side::Bid) + book.total_size(Side::Ask);
    assert_eq!(
        total_limit,
        limit_taker + cancelled + maker_filled + resting
    );
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
    assert_eq!(
        (fills[0].maker_id, fills[0].price, fills[0].size),
        (1, 101, 5)
    );
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
    assert!(replay(&[], 3).is_err());
    let events = read_events_csv(&common::data_path("lob_events.csv")).expect("events");
    let st = replay(&events[..20], 3).expect("replay prefix");
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

// ---------------------------------------------------------------- review additions

fn scratch(name: &str, text: &str) -> String {
    let path = std::env::temp_dir().join(format!("lobmod_rs_{name}"));
    std::fs::write(&path, text).expect("scratch file");
    path.to_string_lossy().into_owned()
}

#[test]
fn price_rounding_to_zero_ticks_rejected() {
    // PT-1
    let mut b = book();
    assert!(b.limit(Side::Bid, 1e-12, 5, 1).is_err());
    let mut b2 = LimitOrderBook::new(0.01).unwrap();
    assert!(b2.limit(Side::Ask, 4e-12, 1, 2).is_err());
    for bk in [&b, &b2] {
        assert!(bk.best_bid().is_none() && bk.best_ask().is_none());
        assert_eq!(bk.n_orders(Side::Bid) + bk.n_orders(Side::Ask), 0);
        assert!(bk.trades.is_empty());
    }
    assert!(b.limit_ticks(Side::Bid, 0, 5, 1).is_err());
    assert!(b.limit_ticks(Side::Bid, -5, 5, 1).is_err());
}

#[test]
fn large_tick_counts() {
    // PT-2: fine-tick products need 64-bit tick counts; overflow is an error
    let mut b = LimitOrderBook::new(1e-8).unwrap();
    let (f1, r1) = b.limit(Side::Ask, 60000.0, 1, 1).unwrap();
    assert!(f1.is_empty() && r1 == 1);
    assert_eq!(b.best_ask(), Some(6_000_000_000_000));
    let (f2, r2) = b.limit(Side::Bid, 60000.0, 1, 2).unwrap();
    assert_eq!(
        f2,
        vec![Fill {
            maker_id: 1,
            price: 6_000_000_000_000,
            size: 1
        }]
    );
    assert_eq!(r2, 0);
    assert!(b.best_ask().is_none() && b.best_bid().is_none());
    assert!(b.limit(Side::Bid, 1e300, 1, 3).is_err()); // no saturation, no sweep
    assert!(b.best_bid().is_none());
    assert!(b.limit(Side::Bid, 1e19, 1, 3).is_err());
    assert!(b.limit_ticks(Side::Bid, MAX_TICKS + 1, 1, 3).is_err());
    b.limit_ticks(Side::Bid, MAX_TICKS, 1, 3).unwrap();
    assert_eq!(b.best_bid(), Some(MAX_TICKS));
}

#[test]
fn tick_rounding_rule_is_half_up_within_tolerance() {
    let mut b = LimitOrderBook::new(0.5).unwrap();
    b.limit(Side::Bid, 99.5, 1, 1).unwrap();
    assert_eq!(b.best_bid(), Some(199));
    b.limit(Side::Bid, 100.0000000001, 1, 2).unwrap(); // residual 2e-10 <= 1e-9 * 200
    assert_eq!(b.best_bid(), Some(200));
    assert!(b.limit(Side::Bid, 100.004, 1, 3).is_err());
    assert!(b.limit(Side::Bid, 0.75, 1, 3).is_err()); // exactly 1.5 ticks: off-grid
}

#[test]
fn limit_ticks_matches_limit() {
    let mut a = LimitOrderBook::new(0.25).unwrap();
    let mut c = LimitOrderBook::new(0.25).unwrap();
    a.limit(Side::Ask, 25.5, 4, 1).unwrap();
    c.limit_ticks(Side::Ask, 102, 4, 1).unwrap();
    assert_eq!(a.best_ask(), Some(102));
    assert_eq!(c.best_ask(), Some(102));
    let (fa, ra) = a.limit(Side::Bid, 26.0, 6, 2).unwrap();
    let (fc, rc) = c.limit_ticks(Side::Bid, 104, 6, 2).unwrap();
    assert_eq!(fa, fc);
    assert_eq!((ra, rc), (2, 2));
}

#[test]
fn replay_prices_are_tick_counts() {
    // PT-4: Event.price is in ticks; replay never rescales it
    let events = read_events_csv(&common::data_path("lob_events.csv")).expect("events");
    let st = replay(&events[..12], 3).unwrap();
    assert_eq!(st.best_bid[11], 999.0);
    assert_eq!(st.best_ask[11], 1001.0);
    let mut manual = LimitOrderBook::new(0.5).unwrap();
    for e in &events[..12] {
        manual
            .limit(e.side, e.price as f64 * 0.5, e.size, e.order_id)
            .unwrap();
    }
    assert_eq!(manual.best_bid(), Some(999));
    assert!(replay(&events[..12], 0).is_err());
}

#[test]
fn depth_monotone_in_k_over_replay() {
    // PT-11
    let events = read_events_csv(&common::data_path("lob_events.csv")).expect("events");
    let mut book = book();
    for e in &events[..500] {
        apply_event(&mut book, e).unwrap();
        assert_eq!(book.depth(Side::Bid, 1).unwrap(), book.best_bid_size());
        assert_eq!(book.depth(Side::Ask, 1).unwrap(), book.best_ask_size());
        for side in [Side::Bid, Side::Ask] {
            for k in 1..=5 {
                assert!(book.depth(side, k).unwrap() <= book.depth(side, k + 1).unwrap());
            }
        }
    }
}

#[test]
fn market_on_empty_book_and_reuse_of_cancelled_id() {
    // PT-12
    let mut b = book();
    assert_eq!(b.market(Side::Bid, 5).unwrap(), (vec![], 5));
    assert_eq!(b.market(Side::Ask, 3).unwrap(), (vec![], 3));
    b.limit(Side::Bid, 100.0, 2, 7).unwrap();
    assert_eq!(b.cancel(7).unwrap(), 2);
    let (fills, rested) = b.limit(Side::Ask, 101.0, 4, 7).unwrap(); // id free again
    assert!(fills.is_empty() && rested == 4 && b.has_order(7));
}

#[test]
fn crossed_sell_limit_stops_at_its_price_and_price_improvement() {
    // PT-13
    let mut b = book();
    b.limit(Side::Bid, 100.0, 4, 1).unwrap();
    b.limit(Side::Bid, 99.0, 4, 2).unwrap();
    let (fills, rested) = b.limit(Side::Ask, 100.0, 10, 3).unwrap();
    assert_eq!(
        fills,
        vec![Fill {
            maker_id: 1,
            price: 100,
            size: 4
        }]
    );
    assert_eq!(rested, 6);
    assert_eq!((b.best_ask(), b.best_bid()), (Some(100), Some(99)));
    assert_eq!(b.best_bid_size(), 4); // the 99 bid was never touched
    let (fills, rested) = b.limit(Side::Ask, 95.0, 2, 4).unwrap(); // executes at the resting 99
    assert_eq!(
        fills,
        vec![Fill {
            maker_id: 2,
            price: 99,
            size: 2
        }]
    );
    assert_eq!(rested, 0);
}

#[test]
fn record_trades_flag_and_clear_history() {
    let mut b = LimitOrderBook::with_options(1.0, false).unwrap();
    b.limit(Side::Ask, 101.0, 5, 1).unwrap();
    let (fills, _) = b.market(Side::Bid, 2).unwrap();
    assert_eq!(fills.len(), 1);
    assert!(b.trades.is_empty() && !b.record_trades());
    assert_eq!(b.best_ask_size(), 3); // matching unaffected
    let mut c = book();
    c.limit(Side::Ask, 101.0, 5, 1).unwrap();
    c.market(Side::Bid, 2).unwrap();
    assert_eq!(c.trades.len(), 1);
    c.clear_history();
    assert!(c.trades.is_empty());
    assert_eq!(c.best_ask_size(), 3);
}

#[test]
fn seed_domain_is_pinned() {
    assert!(Lcg64::new(0).is_ok());
    assert!(Lcg64::new(MAX_SEED).is_ok());
    assert!(Lcg64::new(MAX_SEED + 1).is_err()); // never silently accepted
    assert!(generate_events(12, u64::MAX).is_err());
}

#[test]
fn events_csv_robustness() {
    // PT-15: the pinned CSV rules
    let body = "seq,type,side,price,size,order_id\n1,L,B,999,10,1\n2,L,S,1001,10,2\n";
    let ok = read_events_csv(&scratch("ok.csv", body)).unwrap();
    assert_eq!(ok.len(), 2);
    assert_eq!((ok[1].price, ok[1].side), (1001, Side::Ask));
    let crlf = body.replace('\n', "\r\n");
    assert_eq!(read_events_csv(&scratch("crlf.csv", &crlf)).unwrap(), ok);
    assert_eq!(
        read_events_csv(&scratch("trail.csv", &format!("{body}\n\n"))).unwrap(),
        ok
    );
    assert_eq!(
        read_events_csv(&scratch("noeol.csv", body.trim_end())).unwrap(),
        ok
    );
    let bad = [
        ("comma.csv", format!("{body}3,M,B,0,2,0,\n")),
        ("float.csv", format!("{body}3,L,B,999.5,1,3\n")),
        ("space.csv", format!("{body}3, L,B,999,1,3\n")),
        ("plus.csv", format!("{body}3,L,B,+999,1,3\n")),
        ("type.csv", format!("{body}3,X,B,999,1,3\n")),
        ("side.csv", format!("{body}3,L,Q,999,1,3\n")),
        (
            "blank.csv",
            "seq,type,side,price,size,order_id\n1,L,B,999,10,1\n\n2,L,S,1001,10,2\n".to_string(),
        ),
        (
            "header.csv",
            "seq,side,type,price,size,order_id\n1,B,L,999,10,1\n".to_string(),
        ),
        ("empty.csv", String::new()),
        ("big.csv", format!("{body}3,L,B,9223372036854775808,1,3\n")),
    ];
    for (name, text) in &bad {
        assert!(read_events_csv(&scratch(name, text)).is_err(), "{name}");
    }
    assert!(read_events_csv("/nonexistent/dir/x.csv").is_err());
}

#[test]
fn events_csv_round_trip_is_byte_identical() {
    let events = read_events_csv(&common::data_path("lob_events.csv")).expect("events");
    let path = scratch("stream.csv", "");
    write_events_csv(&path, &events).unwrap();
    assert_eq!(read_events_csv(&path).unwrap(), events);
    let written = std::fs::read(&path).unwrap();
    let bundled = std::fs::read(common::data_path("lob_events.csv")).unwrap();
    assert_eq!(written, bundled);
}

// ---- 64-bit edge rules pinned in API_SPEC 2.1/2.3 --------------------------

fn quoted(bid: i64, ask: i64) -> LimitOrderBook {
    let mut b = LimitOrderBook::new(1.0).unwrap();
    b.limit_ticks(Side::Bid, bid, 1, 1).unwrap();
    b.limit_ticks(Side::Ask, ask, 1, 2).unwrap();
    b
}

#[test]
fn mid_is_exact_sum_rounded_once() {
    assert_eq!(
        quoted(5_000_000_000_000_000_000, 5_000_000_000_000_000_001).mid(),
        5e18
    );
    assert_eq!(
        quoted((1 << 53) + 1, (1 << 53) + 2).mid(),
        ((1i64 << 53) + 2) as f64
    );
    assert_eq!(
        quoted(1 << 62, (1 << 62) + 1025).mid(),
        ((1i64 << 62) + 1024) as f64
    );
}

#[test]
fn side_capacity_rejected_before_mutation() {
    let mut b = LimitOrderBook::new(1.0).unwrap();
    b.limit_ticks(Side::Bid, 100, MAX_TICKS, 1).unwrap();
    assert!(b.limit_ticks(Side::Bid, 100, MAX_TICKS, 2).is_err());
    assert!(b.limit_ticks(Side::Bid, 99, 1, 2).is_err());
    assert_eq!(b.total_size(Side::Bid), MAX_TICKS);
    assert_eq!(b.n_orders(Side::Bid), 1);
    assert_eq!(b.best_bid_size(), MAX_TICKS);
    assert_eq!(b.depth(Side::Bid, 5).unwrap(), MAX_TICKS);
    b.market(Side::Ask, 1).unwrap(); // a fill frees capacity
    b.limit_ticks(Side::Bid, 99, 1, 2).unwrap();
    assert_eq!(b.total_size(Side::Bid), MAX_TICKS);
    b.cancel(1).unwrap(); // so does a cancel
    assert_eq!(b.total_size(Side::Bid), 1);
    b.limit_ticks(Side::Ask, 200, MAX_TICKS, 3).unwrap();
    assert!(b.limit_ticks(Side::Ask, 200, 1, 4).is_err());
    assert_eq!(b.total_size(Side::Ask), MAX_TICKS);
}

#[test]
fn grid_tolerance_is_capped_below_half_a_tick() {
    let mut b = LimitOrderBook::new(0.01).unwrap();
    assert!(b.limit(Side::Bid, 5000000.004, 1, 1).is_err());
    b.limit(Side::Bid, 5000000.01, 1, 1).unwrap();
    assert_eq!(b.best_bid(), Some(500000001));
}

#[test]
fn grid_validation_stops_at_2_pow_52() {
    let mut b = LimitOrderBook::new(1.0).unwrap();
    assert!(b.limit(Side::Bid, 4503599627370496.0, 1, 1).is_err());
    b.limit(Side::Bid, 4503599627370495.0, 1, 1).unwrap();
    assert_eq!(b.best_bid(), Some(4503599627370495));
}

#[test]
fn infinite_ratio_is_an_error() {
    let mut b = LimitOrderBook::new(1e-300).unwrap();
    assert!(b.limit(Side::Bid, 1e308, 1, 1).is_err());
}

#[test]
fn geometric_out_of_range_is_an_error() {
    assert!(Lcg64::new(1).unwrap().geometric(1e-300).is_err());
}

#[test]
fn rosenbrock_4d_iteration_count_is_pinned() {
    let rosen = |x: &[f64]| {
        let mut total = 0.0;
        for i in 0..x.len() - 1 {
            let (a, c) = (x[i + 1] - x[i] * x[i], 1.0 - x[i]);
            total += 100.0 * (a * a) + c * c;
        }
        total
    };
    let r = lobmod::nelder_mead_default(rosen, &[-1.2, 1.0, 0.7, 1.3]).unwrap();
    assert_eq!(r.n_iter, 308);
}
