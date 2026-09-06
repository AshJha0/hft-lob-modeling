//! Event stream schema, CSV input, and deterministic book replay.
//!
//! An event stream is an ordered list of [`Event`] rows.  Replaying a stream
//! through a fresh [`LimitOrderBook`] is fully deterministic, so recorded
//! golden book snapshots must be reproduced exactly by every language
//! implementation.
//!
//! After *each* event the replay records the top-of-book state; the arrays in
//! [`BookStates`] are the single input to the feature pipeline
//! ([`crate::features`]).
//!
//! [`Event::price`] is a *tick count*, never a currency price: [`replay`]
//! takes no tick size and drives a `tick_size = 1` book through
//! [`LimitOrderBook::limit_ticks`].  CSV parsing follows the pinned rules in
//! [`crate::csv`]; `type` must be one of `L`/`C`/`M` and `side` one of
//! `B`/`S` at parse time.

use std::fs;
use std::io::Write;

use crate::book::{LimitOrderBook, Side};
use crate::csv::{parse_int64_strict, read_csv_lines};
use crate::error::{LobError, Result};

/// Order-flow event type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventType {
    /// Limit order add ('L').
    Limit,
    /// Cancel by id ('C').
    Cancel,
    /// Market order ('M').
    Market,
}

impl EventType {
    /// The single-character code used by the event CSV schema.
    pub fn code(self) -> char {
        match self {
            EventType::Limit => 'L',
            EventType::Cancel => 'C',
            EventType::Market => 'M',
        }
    }

    /// Parse the CSV code; unknown types are an error per the contract.
    pub fn from_code(code: &str) -> Result<EventType> {
        match code {
            "L" => Ok(EventType::Limit),
            "C" => Ok(EventType::Cancel),
            "M" => Ok(EventType::Market),
            other => Err(LobError::InvalidInput(format!(
                "unknown event type {other:?}"
            ))),
        }
    }
}

/// One order-flow event.
///
/// For `L`: price (ticks), size, new order id.  For `C`: `price = 0`,
/// `size = 0`, id of the order to cancel (side echoes the resting side, not
/// used by replay).  For `M`: `price = 0`, `order_id = 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Event {
    /// 1-based sequence number.
    pub seq: i64,
    /// Event type.
    pub etype: EventType,
    /// Side ('L'/'M'; for 'C' it echoes the resting order's side).
    pub side: Side,
    /// Price in ticks ('L' only; 0 otherwise).
    pub price: i64,
    /// Order size ('L'/'M'; 0 for 'C').
    pub size: i64,
    /// Order id ('L': new id; 'C': id to cancel; 'M': 0).
    pub order_id: i64,
}

/// Top-of-book state after each replayed event (vectors of length `n_events`).
///
/// Sides that are empty at an event produce NaN in the price-derived fields
/// and 0 in the size fields for that row.
#[derive(Debug, Clone)]
pub struct BookStates {
    /// Best bid price in ticks (NaN if none).
    pub best_bid: Vec<f64>,
    /// Best ask price in ticks (NaN if none).
    pub best_ask: Vec<f64>,
    /// Total size at best bid (0 if none).
    pub bid_size: Vec<f64>,
    /// Total size at best ask (0 if none).
    pub ask_size: Vec<f64>,
    /// Summed size over top-k bid levels.
    pub bid_depth_k: Vec<f64>,
    /// Summed size over top-k ask levels.
    pub ask_depth_k: Vec<f64>,
    /// `(bid + ask) / 2` in ticks (NaN guarded).
    pub mid: Vec<f64>,
    /// Size-weighted mid (NaN guarded).
    pub microprice: Vec<f64>,
    /// `ask - bid` in ticks (NaN guarded).
    pub spread: Vec<f64>,
}

impl BookStates {
    /// Number of recorded events.
    pub fn len(&self) -> usize {
        self.mid.len()
    }

    /// True when no events were recorded.
    pub fn is_empty(&self) -> bool {
        self.mid.is_empty()
    }
}

const HEADER: &str = "seq,type,side,price,size,order_id";

/// Read an event stream CSV with the pinned header
/// `seq,type,side,price,size,order_id` (rules: [`crate::csv`]).
///
/// # Errors
/// [`LobError::Io`] on file problems, [`LobError::Parse`] /
/// [`LobError::InvalidInput`] on a malformed header or row.
pub fn read_events_csv(path: &str) -> Result<Vec<Event>> {
    let lines = read_csv_lines(path, HEADER)?;
    let mut events = Vec::with_capacity(lines.len());
    for (i, line) in lines.iter().enumerate() {
        let context = format!("event row {}", i + 1);
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != 6 {
            return Err(LobError::Parse(format!(
                "malformed {context}: expected 6 fields, got {}: {line:?}",
                fields.len()
            )));
        }
        events.push(Event {
            seq: parse_int64_strict(fields[0], &context)?,
            etype: EventType::from_code(fields[1])?,
            side: Side::from_code(fields[2])?,
            price: parse_int64_strict(fields[3], &context)?,
            size: parse_int64_strict(fields[4], &context)?,
            order_id: parse_int64_strict(fields[5], &context)?,
        });
    }
    Ok(events)
}

/// Write an event stream CSV with the pinned header (`\n` line endings).
///
/// # Errors
/// [`LobError::Io`] if the file cannot be written.
pub fn write_events_csv(path: &str, events: &[Event]) -> Result<()> {
    let mut out = String::with_capacity(32 * (events.len() + 1));
    out.push_str(HEADER);
    out.push('\n');
    for e in events {
        out.push_str(&format!(
            "{},{},{},{},{},{}\n",
            e.seq,
            e.etype.code(),
            e.side.code(),
            e.price,
            e.size,
            e.order_id
        ));
    }
    let mut file = fs::File::create(path).map_err(|e| LobError::Io(format!("{path}: {e}")))?;
    file.write_all(out.as_bytes())
        .map_err(|e| LobError::Io(format!("{path}: {e}")))
}

/// Apply one event to the book.
///
/// # Errors
/// Any book-level validation error (e.g. cancel of an unknown id).
pub fn apply_event(book: &mut LimitOrderBook, event: &Event) -> Result<()> {
    match event.etype {
        EventType::Limit => {
            book.limit_ticks(event.side, event.price, event.size, event.order_id)?;
        }
        EventType::Cancel => {
            book.cancel(event.order_id)?;
        }
        EventType::Market => {
            book.market(event.side, event.size)?;
        }
    }
    Ok(())
}

/// Replay `events` through a fresh `tick_size = 1` book (event prices are
/// tick counts), recording state after each event.
///
/// # Errors
/// [`LobError::InvalidInput`] if the stream is empty, `depth_k == 0`, or an
/// event is invalid.
pub fn replay(events: &[Event], depth_k: usize) -> Result<BookStates> {
    if depth_k == 0 {
        return Err(LobError::InvalidInput(
            "depth_k must be a positive integer, got 0".to_string(),
        ));
    }
    if events.is_empty() {
        return Err(LobError::InvalidInput("event stream is empty".to_string()));
    }
    let n = events.len();
    let mut st = BookStates {
        best_bid: Vec::with_capacity(n),
        best_ask: Vec::with_capacity(n),
        bid_size: Vec::with_capacity(n),
        ask_size: Vec::with_capacity(n),
        bid_depth_k: Vec::with_capacity(n),
        ask_depth_k: Vec::with_capacity(n),
        mid: Vec::with_capacity(n),
        microprice: Vec::with_capacity(n),
        spread: Vec::with_capacity(n),
    };
    let mut book = LimitOrderBook::with_options(1.0, false)?;
    for event in events {
        apply_event(&mut book, event)?;
        st.best_bid
            .push(book.best_bid().map_or(f64::NAN, |p| p as f64));
        st.best_ask
            .push(book.best_ask().map_or(f64::NAN, |p| p as f64));
        st.bid_size.push(book.best_bid_size() as f64);
        st.ask_size.push(book.best_ask_size() as f64);
        st.bid_depth_k.push(book.depth(Side::Bid, depth_k)? as f64);
        st.ask_depth_k.push(book.depth(Side::Ask, depth_k)? as f64);
        st.mid.push(book.mid());
        st.microprice.push(book.microprice());
        st.spread.push(book.spread());
    }
    Ok(st)
}
