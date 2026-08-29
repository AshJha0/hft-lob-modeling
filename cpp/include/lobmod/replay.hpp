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

/// Apply one event to the book.
/// \throws std::invalid_argument on an unknown event type or any book-level
///         validation error (e.g. cancel of an unknown id).
void apply_event(LimitOrderBook& book, const Event& event);

/// Replay events through a fresh book (given tick size), recording the
/// top-of-book state after each event.
/// \throws std::invalid_argument if the stream is empty or an event is bad.
BookStates replay(const std::vector<Event>& events, int depth_k = 3,
                  double tick_size = 1.0);

} // namespace lobmod

#endif // LOBMOD_REPLAY_HPP
