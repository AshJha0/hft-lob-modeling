#ifndef LOBMOD_REPLAY_HPP
#define LOBMOD_REPLAY_HPP

/// \file replay.hpp
/// \brief Event stream schema, CSV i/o, and deterministic book replay.
///
/// An event stream is an ordered list of Event rows.  Replaying a stream
/// through a fresh LimitOrderBook is fully deterministic, so recorded golden
/// book snapshots must be reproduced exactly by every language
/// implementation.  After *each* event the replay records the top-of-book
/// state; the BookStates arrays are the single input to the feature
/// pipeline (features.hpp).
///
/// Event::price is a *tick count*, never a currency price: replay() takes
/// no tick size and drives a tick_size = 1 book through limit_ticks().
/// CSV parsing follows the pinned rules in csv.hpp; type must be one of
/// L/C/M and side one of B/S at parse time.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "lobmod/book.hpp"

namespace lobmod {

/// One order-flow event (CSV schema `seq,type,side,price,size,order_id`).
struct Event {
    std::int64_t seq;      ///< 1-based sequence number
    char etype;            ///< 'L' (limit add), 'C' (cancel), 'M' (market)
    char side;             ///< 'B' or 'S' (for 'C': echoes the resting side)
    std::int64_t price;    ///< price in ticks ('L' only; 0 otherwise)
    std::int64_t size;     ///< order size ('L'/'M'; 0 for 'C')
    std::int64_t order_id; ///< 'L': new id; 'C': id to cancel; 'M': 0
};

/// One event of schema v2: header
/// `seq,ts_ns,type,side,price,size,order_id,new_order_id`.
struct EventV2 {
    std::int64_t seq;          ///< 1-based sequence number
    std::int64_t ts_ns;        ///< non-negative nanoseconds (0 when lifted from v1)
    char etype;                ///< 'L','C','M', or 'R' reduce, 'U' replace, 'E' execute
    char side;                 ///< 'B' or 'S' (echoes the resting side for C/R/U/E)
    std::int64_t price;        ///< price in ticks ('L' and 'U'; 0 otherwise)
    std::int64_t size;         ///< L/M size; R size removed; U new size; E executed; C 0
    std::int64_t order_id;     ///< 'L': new id; C/R/U/E: resting id; 'M': 0
    std::int64_t new_order_id; ///< 'U': replacement id; 0 otherwise
};

/// Top-of-book state after each replayed event (arrays of length n_events).
/// Empty sides produce NaN in the price-derived arrays and 0 in the sizes.
struct BookStates {
    std::vector<double> best_bid;    ///< best bid price in ticks (NaN if none)
    std::vector<double> best_ask;    ///< best ask price in ticks (NaN if none)
    std::vector<double> bid_size;    ///< total size at best bid (0 if none)
    std::vector<double> ask_size;    ///< total size at best ask (0 if none)
    std::vector<double> bid_depth_k; ///< summed size over top-k bid levels
    std::vector<double> ask_depth_k; ///< summed size over top-k ask levels
    std::vector<double> mid;         ///< (bid+ask)/2 in ticks (NaN guarded)
    std::vector<double> microprice;  ///< size-weighted mid (NaN guarded)
    std::vector<double> spread;      ///< ask-bid in ticks (NaN guarded)

    std::size_t size() const { return mid.size(); }
};

/// Read an event stream CSV with the pinned header
/// `seq,type,side,price,size,order_id`.
/// \throws std::invalid_argument on a malformed header or row, or an
///         unreadable file.
std::vector<Event> read_events_csv(const std::string& path);

/// Write an event stream CSV with the pinned header ('\n' line endings).
/// \throws std::invalid_argument if the file cannot be written.
void write_events_csv(const std::string& path, const std::vector<Event>& events);

/// Apply one event to the book.
/// \throws std::invalid_argument on an unknown event type or any book-level
///         validation error (e.g. cancel of an unknown id).
void apply_event(LimitOrderBook& book, const Event& event);

/// Read a v2 event CSV, or a v1 file lifted with ts_ns = 0; the header selects
/// the schema.
/// \throws std::invalid_argument on a bad header or row, or an unreadable file.
std::vector<EventV2> read_events_v2_csv(const std::string& path);

/// Write schema v2 events ('\n' line endings).
/// \throws std::invalid_argument if the file cannot be written.
void write_events_v2_csv(const std::string& path, const std::vector<EventV2>& events);

/// Apply one schema v2 event to the book.
/// \throws std::invalid_argument as apply_event().
void apply_event_v2(LimitOrderBook& book, const EventV2& event);

/// As replay(), for schema v2 events.
BookStates replay_v2(const std::vector<EventV2>& events, int depth_k = 3);

/// Replay events through a fresh tick_size = 1 book (event prices are tick
/// counts), recording the top-of-book state after each event.
/// \throws std::invalid_argument if the stream is empty, depth_k <= 0, or
///         an event is bad.
BookStates replay(const std::vector<Event>& events, int depth_k = 3);

} // namespace lobmod

#endif // LOBMOD_REPLAY_HPP
