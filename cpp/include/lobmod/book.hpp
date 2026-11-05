#ifndef LOBMOD_BOOK_HPP
#define LOBMOD_BOOK_HPP

/// \file book.hpp
/// \brief Limit order book with price-time (FIFO) priority on a discrete
///        tick grid.
///
/// Prices live on a grid of integer multiples of the tick size; internally a
/// price is stored as an integer number of ticks.  Each price level holds a
/// FIFO queue of resting orders (id, remaining size): earlier arrivals at a
/// level fill first (time priority), better prices fill first (price
/// priority).  A crossed limit order first executes as a taker against every
/// crossing level (at the resting orders' prices), and only the unfilled
/// remainder rests.  A market order larger than the opposite depth fills
/// what exists and *reports* the remainder — never an error.
///
/// Quantities needing both sides (mid, spread, microprice) return NaN when a
/// side is empty (guarded, never a crash).
///
/// Tick conversion (pinned, identical in every port): ratio = price / tick,
/// must be below 2^52, rounded half up; accepted iff
/// |ratio - ticks| <= min(1e-9 * max(1, ratio), 0.01) and ticks >= 1.  A bad
/// price is an error — never a negative price and never an unbounded market
/// order.  limit_ticks() takes the integer tick count directly, up to
/// kMaxTicks (9e18; the replay path).
///
/// Size capacity (pinned): a limit add is rejected before any mutation if the
/// side's running total plus the submitted size exceeds kMaxTicks, so every
/// size sum fits an int64.
///
/// Memory: trades() and log() grow for the life of the book; construct with
/// record_trades = false or call clear_history() for long sessions.

#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lobmod {

/// Largest accepted tick count / size (fits a signed 64-bit integer).
constexpr std::int64_t kMaxTicks = 9000000000000000000LL;

/// One execution against a resting order.
struct Fill {
    std::int64_t maker_id; ///< id of the resting (passive) order that was hit
    std::int64_t price;    ///< execution price in ticks (the resting order's price)
    std::int64_t size;     ///< executed quantity (> 0)
};

/// One entry of the book's event log.
struct BookLogEntry {
    char etype;            ///< 'L', 'C', 'M', 'R' or 'E' (replace logs 'C' then 'L')
    char side;             ///< 'B' or 'S'
    std::int64_t price;    ///< price in ticks (0 for 'M'; cancelled level for 'C')
    std::int64_t size;     ///< submitted size ('L'/'M'); removed size for 'C'
    std::int64_t order_id; ///< order id ('L'/'C'); 0 for 'M'
    std::int64_t executed; ///< quantity executed on arrival (taker side)
    std::int64_t unfilled; ///< unfilled market-order remainder ('M' only)
};

/// Price-time priority limit order book (see file docs).
class LimitOrderBook {
public:
    static constexpr char BUY = 'B';
    static constexpr char SELL = 'S';

    /// \param tick_size positive finite tick; every incoming price must be an
    ///        integer multiple of it (see the pinned tick conversion above).
    /// \param record_trades keep every Fill in trades() and every event in
    ///        log(); false disables both (matching is unaffected).
    /// \throws std::invalid_argument on a non-positive or non-finite tick.
    explicit LimitOrderBook(double tick_size = 1.0, bool record_trades = true);

    /// Highest bid price in ticks, or nullopt if the bid side is empty.
    std::optional<std::int64_t> best_bid() const;
    /// Lowest ask price in ticks, or nullopt if the ask side is empty.
    std::optional<std::int64_t> best_ask() const;
    /// Total resting size at the best bid level (0 if the side is empty).
    std::int64_t best_bid_size() const;
    /// Total resting size at the best ask level (0 if the side is empty).
    std::int64_t best_ask_size() const;

    /// Mid price in ticks: the exact integer bid + ask rounded once to the
    /// nearest double, then halved; NaN if either side is empty.
    double mid() const;
    /// Bid-ask spread in ticks; NaN if either side is empty.
    double spread() const;
    /// Size-weighted mid (q_a * P_b + q_b * P_a) / (q_a + q_b); NaN if either
    /// side is empty.  Leans toward the thin side — the more likely direction
    /// of the next mid move.
    double microprice() const;

    /// Total resting size over the top k price levels on a side (fewer
    /// levels: sums what exists).
    /// \throws std::invalid_argument on a bad side or k <= 0.
    std::int64_t depth(char side, int k) const;

    /// Sorted ids of live resting orders on a side (the pinned ordering used
    /// by the zero-intelligence generator's cancel selection).
    std::vector<std::int64_t> order_ids(char side) const;
    /// True if the order id is currently resting in the book.
    bool has_order(std::int64_t order_id) const;
    /// Total resting size on a side across all levels (O(1)).
    std::int64_t total_size(char side) const;
    /// Number of live resting orders on a side.
    std::int64_t n_orders(char side) const;

    /// Submit a limit order.  A crossed limit (buy at/above best ask, sell
    /// at/below best bid) first executes against every crossing level, at the
    /// resting orders' prices, never trading through its own limit; the
    /// unfilled remainder rests at the limit price.
    /// \returns (fills, rested size) — rested size is 0 if fully executed.
    /// \throws std::invalid_argument on bad side/price/size, an off-grid
    ///         price, a price at or above 2^52 ticks, a duplicate live order
    ///         id, or a size taking the side's total above kMaxTicks.
    std::pair<std::vector<Fill>, std::int64_t>
    limit(char side, double price, std::int64_t size, std::int64_t order_id);

    /// Submit a limit order priced directly in ticks (no float conversion);
    /// identical semantics to limit().  price_ticks must lie in
    /// [1, kMaxTicks].
    /// \throws std::invalid_argument as limit().
    std::pair<std::vector<Fill>, std::int64_t>
    limit_ticks(char side, std::int64_t price_ticks, std::int64_t size,
                std::int64_t order_id);

    /// Cancel a resting order by id; returns the size removed.
    /// \throws std::invalid_argument if the id is not live (never rested,
    ///         already cancelled, or fully filled).
    std::int64_t cancel(std::int64_t order_id);

    /// Take by_size off a resting order, keeping its queue position; the whole
    /// remaining size removes it.  \return the size left resting.
    /// \throws std::invalid_argument on an unknown id, a bad size, or by_size
    ///         above the remaining size (book unchanged).
    std::int64_t reduce(std::int64_t order_id, std::int64_t by_size);

    /// Fill a named resting order at its own price, partly or fully, keeping
    /// its queue position on a partial fill.  \return the Fill, also recorded
    /// in trades().
    /// \throws std::invalid_argument as reduce().
    Fill execute(std::int64_t order_id, std::int64_t size);

    /// Swap a resting order for new_id on the same side at the tail of the new
    /// level, losing priority.  Equals cancel then limit_ticks, but every check
    /// runs first; a crossing price executes.  new_id may equal order_id.
    /// \throws std::invalid_argument on an unknown order_id, new_id live on
    ///         another order, a bad price or size, or a size taking the side
    ///         above kMaxTicks once the old order has gone (book unchanged).
    std::pair<std::vector<Fill>, std::int64_t>
    replace(std::int64_t order_id, std::int64_t new_id, double price, std::int64_t size);

    /// replace() priced directly in ticks.
    std::pair<std::vector<Fill>, std::int64_t>
    replace_ticks(std::int64_t order_id, std::int64_t new_id, std::int64_t price_ticks,
                  std::int64_t size);

    /// Submit a market order that walks the opposite side.  A market order
    /// bigger than the opposite depth fills what exists; the remainder is
    /// *reported* in the second element — not an error.
    /// \returns (fills, unfilled).
    std::pair<std::vector<Fill>, std::int64_t> market(char side, std::int64_t size);

    /// All executions so far, in execution order.
    const std::vector<Fill>& trades() const { return trades_; }
    /// Event log (one entry per limit/cancel/market call).
    const std::vector<BookLogEntry>& log() const { return log_; }
    /// The validated tick size.
    double tick_size() const { return tick_size_; }
    /// Whether fills/events are being recorded.
    bool record_trades() const { return record_trades_; }
    /// Drop the accumulated trades() and log() (book unchanged).
    void clear_history();

private:
    struct Order {
        std::int64_t id;
        std::int64_t size;
    };
    using Level = std::deque<Order>;
    using Side = std::map<std::int64_t, Level>; // price ticks -> FIFO queue

    std::int64_t to_ticks(double price) const;
    static std::int64_t check_ticks(std::int64_t ticks);
    static void check_side(char side);
    static void check_size(std::int64_t size);
    std::pair<std::vector<Fill>, std::int64_t>
    match(char side, std::int64_t size, const std::int64_t* limit_ticks);

    /// Take qty off a live order in place; returns what is left of it.
    struct Shrunk {
        char side;
        std::int64_t ticks;
        std::int64_t left;
    };
    Shrunk shrink(std::int64_t order_id, std::int64_t qty, const char* what);

    double tick_size_;
    bool record_trades_;
    Side bids_;
    Side asks_;
    /// running total resting size per side (never above kMaxTicks)
    std::int64_t bid_total_ = 0;
    std::int64_t ask_total_ = 0;
    /// order id -> (side, price ticks) for live resting orders
    std::unordered_map<std::int64_t, std::pair<char, std::int64_t>> orders_;
    std::vector<Fill> trades_;
    std::vector<BookLogEntry> log_;
};

} // namespace lobmod

#endif // LOBMOD_BOOK_HPP
