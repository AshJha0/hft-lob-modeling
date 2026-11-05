#ifndef LOBMOD_FAST_BOOK_HPP
#define LOBMOD_FAST_BOOK_HPP

/// \file fast_book.hpp
/// \brief Preallocated limit order book with the matching semantics of
///        lobmod::LimitOrderBook, for a bounded price band.
///
/// Same rules as the reference book for limit, cancel and market: price-time
/// priority, crossed limits execute first at the resting prices, the remainder
/// rests, market orders report what could not fill.  Prices are tick counts
/// only; the double-price conversion stays in the reference book.
///
/// Layout:
///  * a flat ladder of levels per side, one entry per tick of the band;
///  * a hierarchical bitset per side, so finding the next live level costs at
///    most four word operations however wide the gap is, plus a cached best;
///  * one node pool with an intrusive doubly linked FIFO per level, so a
///    cancel unlinks in O(1);
///  * cached level, side and order-count totals;
///  * an open-addressing id map (linear probing, backward-shift deletion, no
///    tombstones, always at most half full).
///
/// All memory is taken in the constructor, which is the only place that can
/// throw.  Operations return a Status and never allocate.
///
/// Where it differs from the reference book, deliberately:
///  * a limit priced outside the band returns Status::OutOfBand and changes
///    nothing, even if it would have crossed.  Market orders are unaffected,
///    because everything resting is inside the band;
///  * when the pool is full every limit returns Status::PoolFull and changes
///    nothing, including one that would have executed in full.  The check runs
///    before matching so that a rejected order has no partial effect;
///  * errors are return codes rather than exceptions;
///  * there is no trade or event log: fills go to the caller's sink.
///
/// The sink is any callable taking (maker_id, price_ticks, size).  It runs in
/// the middle of matching, so it must not call back into the book, and it must
/// not throw: the operations are noexcept.
///
/// The id hash is not keyed.  A feed chosen to collide would degrade the map
/// to long probe runs, so ids from an untrusted source need remapping first.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "lobmod/book.hpp"

namespace lobmod {
namespace fast {

enum class Side : std::uint8_t { Buy = 0, Sell = 1 };

enum class Status : std::uint8_t {
    Ok = 0,
    BadSize,      ///< size <= 0 or above kMaxTicks
    BadPrice,     ///< price below 1 tick or above kMaxTicks
    OutOfBand,    ///< valid price outside the configured band
    DuplicateId,  ///< the id is already resting
    SideOverflow, ///< the side's total would pass kMaxTicks
    PoolFull,     ///< no free node; nothing was done
    UnknownId,    ///< cancel of an id that is not resting
};

/// status, plus: rested size (limit), unfilled size (market) or removed size
/// (cancel).  value is 0 unless status is Ok.
struct OpResult {
    Status status;
    std::int64_t value;
};

struct Config {
    std::int64_t min_price_ticks = 1;   ///< lowest price in the band (>= 1)
    std::int64_t n_levels = 1 << 16;    ///< ticks in the band, at most 2^24
    std::uint32_t max_orders = 1 << 20; ///< live orders, at most 2^30
};

namespace detail {

inline int msb64(std::uint64_t w) {
#if defined(__GNUC__) || defined(__clang__)
    return 63 - __builtin_clzll(w);
#else
    int n = 0;
    while (w >>= 1) ++n;
    return n;
#endif
}

inline int lsb64(std::uint64_t w) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(w);
#else
    return msb64(w & (~w + 1));
#endif
}

/// Up to four levels of 64-way summary words over n positions.
class LevelBits {
public:
    explicit LevelBits(std::size_t n) {
        std::size_t count = n;
        std::size_t total = 0;
        do {
            count = (count + 63) / 64;
            off_[depth_++] = total;
            total += count;
        } while (count > 1);
        words_.assign(total, 0);
    }

    void set(std::size_t i) {
        for (int h = 0; h < depth_; ++h) {
            std::uint64_t& w = words_[off_[h] + (i >> 6)];
            const bool was_empty = w == 0;
            w |= std::uint64_t{1} << (i & 63);
            if (!was_empty) return;
            i >>= 6;
        }
    }

    void clear(std::size_t i) {
        for (int h = 0; h < depth_; ++h) {
            std::uint64_t& w = words_[off_[h] + (i >> 6)];
            w &= ~(std::uint64_t{1} << (i & 63));
            if (w != 0) return;
            i >>= 6;
        }
    }

    /// Highest set position strictly below i, or -1.
    std::int64_t prev(std::size_t i) const {
        for (int h = 0; h < depth_; ++h) {
            const std::size_t idx = i >> 6;
            const std::uint64_t mask = (std::uint64_t{1} << (i & 63)) - 1;
            const std::uint64_t w = words_[off_[h] + idx] & mask;
            if (w != 0) {
                std::size_t pos = idx * 64 + static_cast<std::size_t>(msb64(w));
                for (int g = h - 1; g >= 0; --g) {
                    pos = pos * 64 + static_cast<std::size_t>(msb64(words_[off_[g] + pos]));
                }
                return static_cast<std::int64_t>(pos);
            }
            i = idx;
        }
        return -1;
    }

    /// Lowest set position strictly above i, or -1.
    std::int64_t next(std::size_t i) const {
        for (int h = 0; h < depth_; ++h) {
            const std::size_t idx = i >> 6;
            const int bit = static_cast<int>(i & 63);
            const std::uint64_t mask = bit == 63 ? 0 : ~std::uint64_t{0} << (bit + 1);
            const std::uint64_t w = words_[off_[h] + idx] & mask;
            if (w != 0) {
                std::size_t pos = idx * 64 + static_cast<std::size_t>(lsb64(w));
                for (int g = h - 1; g >= 0; --g) {
                    pos = pos * 64 + static_cast<std::size_t>(lsb64(words_[off_[g] + pos]));
                }
                return static_cast<std::int64_t>(pos);
            }
            i = idx;
        }
        return -1;
    }

private:
    std::vector<std::uint64_t> words_;
    std::size_t off_[4] = {0, 0, 0, 0};
    int depth_ = 0;
};

/// id -> pool handle.  Handles are stored plus one; zero marks an empty slot,
/// so every int64 value is usable as an id.
class IdMap {
public:
    static constexpr std::size_t kNone = ~std::size_t{0};

    explicit IdMap(std::uint32_t max_orders) {
        int bits = 4;
        while ((std::uint64_t{1} << bits) < 2 * std::uint64_t{max_orders}) ++bits;
        slots_.assign(std::size_t{1} << bits, Slot{0, 0});
        mask_ = slots_.size() - 1;
        shift_ = 64 - bits;
    }

    std::size_t find(std::int64_t id) const {
        std::size_t i = home(id);
        while (slots_[i].handle1 != 0) {
            if (slots_[i].id == id) return i;
            i = (i + 1) & mask_;
        }
        return kNone;
    }

    /// The slot holding `id`, or the empty slot that ends its probe run.  One
    /// walk then serves both the duplicate check and the insert.
    std::size_t probe(std::int64_t id) const {
        std::size_t i = home(id);
        while (slots_[i].handle1 != 0 && slots_[i].id != id) i = (i + 1) & mask_;
        return i;
    }

    bool occupied(std::size_t slot) const { return slots_[slot].handle1 != 0; }

    std::uint32_t handle_at(std::size_t slot) const { return slots_[slot].handle1 - 1; }

    /// `slot` must come from probe() with no map change since.
    void insert_at(std::size_t slot, std::int64_t id, std::uint32_t handle) {
        slots_[slot] = Slot{id, handle + 1};
    }

    /// Backward-shift deletion: pull later entries of the probe run into the
    /// hole unless that would move one before its home slot.
    void erase_at(std::size_t hole) {
        std::size_t j = hole;
        for (;;) {
            j = (j + 1) & mask_;
            if (slots_[j].handle1 == 0) break;
            const std::size_t k = home(slots_[j].id);
            if (((j - k) & mask_) >= ((j - hole) & mask_)) {
                slots_[hole] = slots_[j];
                hole = j;
            }
        }
        slots_[hole].handle1 = 0;
    }

private:
    struct Slot {
        std::int64_t id;
        std::uint32_t handle1;
    };

    std::size_t home(std::int64_t id) const {
        std::uint64_t x = static_cast<std::uint64_t>(id);
        x ^= x >> 32;
        return static_cast<std::size_t>((x * 0x9E3779B97F4A7C15ULL) >> shift_);
    }

    std::vector<Slot> slots_;
    std::size_t mask_ = 0;
    int shift_ = 0;
};

} // namespace detail

class FastBook {
public:
    /// \throws std::invalid_argument on a bad band or pool size, and
    ///         std::bad_alloc if the memory is not available.
    explicit FastBook(const Config& cfg)
        : min_price_(check(cfg).min_price_ticks),
          n_levels_(cfg.n_levels),
          sides_{SideState(cfg.n_levels), SideState(cfg.n_levels)},
          nodes_(cfg.max_orders),
          ids_(cfg.max_orders) {
        for (std::uint32_t i = 0; i < cfg.max_orders; ++i) {
            nodes_[i].next = i + 1 < cfg.max_orders ? i + 1 : kNil;
        }
        free_ = 0;
    }

    FastBook(const FastBook&) = delete;
    FastBook& operator=(const FastBook&) = delete;

    /// Submit a limit order; value is the rested size.
    template <class Sink>
    OpResult limit(Side side, std::int64_t price, std::int64_t size, std::int64_t id,
                   Sink&& sink) noexcept {
        if (size <= 0 || size > kMaxTicks) return {Status::BadSize, 0};
        if (price < 1 || price > kMaxTicks) return {Status::BadPrice, 0};
        if (price < min_price_ || price - min_price_ >= n_levels_) {
            return {Status::OutOfBand, 0};
        }
        const std::size_t slot = ids_.probe(id);
        if (ids_.occupied(slot)) return {Status::DuplicateId, 0};
        SideState& own = state(side);
        if (size > kMaxTicks - own.total) return {Status::SideOverflow, 0};
        if (free_ == kNil) return {Status::PoolFull, 0};

        const auto level = static_cast<std::uint32_t>(price - min_price_);
        if (!crosses(side, price)) { // no matching, so the map and slot stay as probed
            rest_order(own, side, level, size, id, slot);
            return {Status::Ok, size};
        }
        const std::int64_t left = match(side, size, price, true, sink);
        if (left > 0) rest_order(own, side, level, left, id, ids_.probe(id));
        return {Status::Ok, left};
    }

    /// Add an order without matching, for mirroring a feed: the venue has
    /// already decided what rests, and a feed book can be locked or crossed
    /// before the open.  Same refusals as limit(); value is the rested size.
    OpResult rest(Side side, std::int64_t price, std::int64_t size, std::int64_t id) noexcept {
        if (size <= 0 || size > kMaxTicks) return {Status::BadSize, 0};
        if (price < 1 || price > kMaxTicks) return {Status::BadPrice, 0};
        if (price < min_price_ || price - min_price_ >= n_levels_) {
            return {Status::OutOfBand, 0};
        }
        const std::size_t slot = ids_.probe(id);
        if (ids_.occupied(slot)) return {Status::DuplicateId, 0};
        SideState& own = state(side);
        if (size > kMaxTicks - own.total) return {Status::SideOverflow, 0};
        if (free_ == kNil) return {Status::PoolFull, 0};
        rest_order(own, side, static_cast<std::uint32_t>(price - min_price_), size, id, slot);
        return {Status::Ok, size};
    }

    /// Move a resting order to a new id, price and size on the same side; it
    /// loses its queue place.  `new_id` may equal `id`.  Any refusal changes
    /// nothing.  Value is the rested size.
    OpResult replace(std::int64_t id, std::int64_t new_id, std::int64_t price,
                     std::int64_t size) noexcept {
        const std::size_t slot = ids_.find(id);
        if (slot == detail::IdMap::kNone) return {Status::UnknownId, 0};
        if (size <= 0 || size > kMaxTicks) return {Status::BadSize, 0};
        if (price < 1 || price > kMaxTicks) return {Status::BadPrice, 0};
        if (price < min_price_ || price - min_price_ >= n_levels_) {
            return {Status::OutOfBand, 0};
        }
        if (new_id != id && ids_.find(new_id) != detail::IdMap::kNone) {
            return {Status::DuplicateId, 0};
        }
        const Node& n = nodes_[ids_.handle_at(slot)];
        const Side side = n.side;
        if (size > kMaxTicks - (state(side).total - n.size)) return {Status::SideOverflow, 0};
        cancel(id);
        return rest(side, price, size, new_id);
    }

    /// Take `by` off a resting order, keeping its queue place; reducing to zero
    /// removes it.  Value is the size left.  BadSize if `by` is not in
    /// [1, resting size], and the order is then unchanged.
    OpResult reduce(std::int64_t id, std::int64_t by) noexcept {
        const std::size_t slot = ids_.find(id);
        if (slot == detail::IdMap::kNone) return {Status::UnknownId, 0};
        Node& n = nodes_[ids_.handle_at(slot)];
        if (by <= 0 || by > n.size) return {Status::BadSize, 0};
        if (by == n.size) {
            cancel(id);
            return {Status::Ok, 0};
        }
        SideState& s = state(n.side);
        n.size -= by;
        s.levels[n.level].total -= by;
        s.total -= by;
        return {Status::Ok, n.size};
    }

    /// Submit a market order; value is the unfilled size.
    template <class Sink>
    OpResult market(Side side, std::int64_t size, Sink&& sink) noexcept {
        if (size <= 0 || size > kMaxTicks) return {Status::BadSize, 0};
        return {Status::Ok, match(side, size, 0, false, sink)};
    }

    /// Cancel a resting order; value is the size removed.
    OpResult cancel(std::int64_t id) noexcept {
        const std::size_t slot = ids_.find(id);
        if (slot == detail::IdMap::kNone) return {Status::UnknownId, 0};
        const std::uint32_t h = ids_.handle_at(slot);
        ids_.erase_at(slot);
        Node& n = nodes_[h];
        SideState& s = state(n.side);
        Level& lv = s.levels[n.level];
        const std::int64_t removed = n.size;
        if (n.prev != kNil) {
            nodes_[n.prev].next = n.next;
        } else {
            lv.head = n.next;
        }
        if (n.next != kNil) {
            nodes_[n.next].prev = n.prev;
        } else {
            lv.tail = n.prev;
        }
        lv.total -= removed;
        s.total -= removed;
        --s.orders;
        const std::uint32_t level = n.level;
        const Side side = n.side;
        release(h);
        if (lv.head == kNil) level_emptied(s, side, level);
        return {Status::Ok, removed};
    }

    bool has_bid() const { return sides_[0].best >= 0; }
    bool has_ask() const { return sides_[1].best >= 0; }
    /// Best prices in ticks; 0 when the side is empty (0 is never a price).
    std::int64_t best_bid() const { return best_price(sides_[0]); }
    std::int64_t best_ask() const { return best_price(sides_[1]); }
    std::int64_t best_bid_size() const { return best_size(sides_[0]); }
    std::int64_t best_ask_size() const { return best_size(sides_[1]); }

    /// Same pinned formulas as the reference book; NaN if a side is empty.
    double mid() const {
        if (!has_bid() || !has_ask()) return kNaN;
        return 0.5 * static_cast<double>(static_cast<std::uint64_t>(best_bid()) +
                                         static_cast<std::uint64_t>(best_ask()));
    }
    double spread() const {
        if (!has_bid() || !has_ask()) return kNaN;
        return static_cast<double>(best_ask() - best_bid());
    }
    double microprice() const {
        if (!has_bid() || !has_ask()) return kNaN;
        const double qb = static_cast<double>(best_bid_size());
        const double qa = static_cast<double>(best_ask_size());
        return (qa * static_cast<double>(best_bid()) +
                qb * static_cast<double>(best_ask())) / (qa + qb);
    }

    /// Total size over the top k live levels; 0 for k <= 0.
    std::int64_t depth(Side side, int k) const {
        const SideState& s = state(side);
        std::int64_t total = 0;
        std::int64_t at = s.best;
        for (int taken = 0; at >= 0 && taken < k; ++taken) {
            total += s.levels[static_cast<std::size_t>(at)].total;
            at = side == Side::Buy ? s.bits.prev(static_cast<std::size_t>(at))
                                   : s.bits.next(static_cast<std::size_t>(at));
        }
        return total;
    }

    std::int64_t total_size(Side side) const { return state(side).total; }
    std::int64_t n_orders(Side side) const { return state(side).orders; }
    bool has_order(std::int64_t id) const { return ids_.find(id) != detail::IdMap::kNone; }

private:
    static constexpr std::uint32_t kNil = 0xFFFFFFFFu;
    static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

    struct Node {
        std::int64_t id = 0;
        std::int64_t size = 0;
        std::uint32_t prev = kNil;
        std::uint32_t next = kNil;
        std::uint32_t level = 0;
        Side side = Side::Buy;
    };
    struct Level {
        std::int64_t total = 0;
        std::uint32_t head = kNil;
        std::uint32_t tail = kNil;
    };
    struct SideState {
        explicit SideState(std::int64_t n)
            : levels(static_cast<std::size_t>(n)), bits(static_cast<std::size_t>(n)) {}
        std::vector<Level> levels;
        detail::LevelBits bits;
        std::int64_t best = -1; ///< ladder index of the best level, or -1
        std::int64_t total = 0;
        std::int64_t orders = 0;
    };

    static const Config& check(const Config& c) {
        if (c.min_price_ticks < 1 || c.n_levels < 1 || c.n_levels > (1 << 24) ||
            c.min_price_ticks > kMaxTicks ||
            c.n_levels - 1 > kMaxTicks - c.min_price_ticks) {
            throw std::invalid_argument("FastBook: bad price band");
        }
        if (c.max_orders < 1 || c.max_orders > (1u << 30)) {
            throw std::invalid_argument("FastBook: max_orders must be in [1, 2^30]");
        }
        return c;
    }

    SideState& state(Side s) { return sides_[static_cast<int>(s)]; }
    const SideState& state(Side s) const { return sides_[static_cast<int>(s)]; }

    std::int64_t best_price(const SideState& s) const {
        return s.best < 0 ? 0 : min_price_ + s.best;
    }
    static std::int64_t best_size(const SideState& s) {
        return s.best < 0 ? 0 : s.levels[static_cast<std::size_t>(s.best)].total;
    }

    void release(std::uint32_t h) {
        nodes_[h].next = free_;
        free_ = h;
    }

    void level_emptied(SideState& s, Side side, std::uint32_t level) {
        s.bits.clear(level);
        if (s.best == static_cast<std::int64_t>(level)) {
            s.best = side == Side::Buy ? s.bits.prev(level) : s.bits.next(level);
        }
    }

    bool crosses(Side side, std::int64_t price) const {
        const SideState& opp = state(side == Side::Buy ? Side::Sell : Side::Buy);
        if (opp.best < 0) return false;
        const std::int64_t touch = min_price_ + opp.best;
        return side == Side::Buy ? touch <= price : touch >= price;
    }

    void rest_order(SideState& s, Side side, std::uint32_t level, std::int64_t size,
                    std::int64_t id, std::size_t slot) {
        const std::uint32_t h = free_;
        Node& n = nodes_[h];
        free_ = n.next;
        Level& lv = s.levels[level];
        n.id = id;
        n.size = size;
        n.level = level;
        n.side = side;
        n.prev = lv.tail;
        n.next = kNil;
        if (lv.tail != kNil) {
            nodes_[lv.tail].next = h;
        } else {
            lv.head = h;
            s.bits.set(level);
            const std::int64_t at = level;
            if (s.best < 0 || (side == Side::Buy ? at > s.best : at < s.best)) s.best = at;
        }
        lv.tail = h;
        lv.total += size;
        s.total += size;
        ++s.orders;
        ids_.insert_at(slot, id, h);
    }

    template <class Sink>
    std::int64_t match(Side side, std::int64_t size, std::int64_t limit, bool has_limit,
                       Sink& sink) {
        const Side opp_side = side == Side::Buy ? Side::Sell : Side::Buy;
        SideState& opp = state(opp_side);
        std::int64_t remaining = size;
        while (remaining > 0 && opp.best >= 0) {
            const std::int64_t price = min_price_ + opp.best;
            if (has_limit && (side == Side::Buy ? price > limit : price < limit)) break;
            const std::uint32_t level = static_cast<std::uint32_t>(opp.best);
            Level& lv = opp.levels[level];
            while (remaining > 0 && lv.head != kNil) {
                const std::uint32_t h = lv.head;
                Node& n = nodes_[h];
                const std::int64_t take = remaining < n.size ? remaining : n.size;
                n.size -= take;
                remaining -= take;
                lv.total -= take;
                sink(n.id, price, take);
                if (n.size == 0) {
                    lv.head = n.next;
                    if (lv.head != kNil) {
                        nodes_[lv.head].prev = kNil;
                    } else {
                        lv.tail = kNil;
                    }
                    ids_.erase_at(ids_.find(n.id));
                    --opp.orders;
                    release(h);
                }
            }
            if (lv.head == kNil) level_emptied(opp, opp_side, level);
        }
        opp.total -= size - remaining;
        return remaining;
    }

    std::int64_t min_price_;
    std::int64_t n_levels_;
    SideState sides_[2];
    std::vector<Node> nodes_;
    detail::IdMap ids_;
    std::uint32_t free_ = kNil;
};

} // namespace fast
} // namespace lobmod

#endif // LOBMOD_FAST_BOOK_HPP
