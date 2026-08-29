#include "lobmod/book.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace lobmod {

LimitOrderBook::LimitOrderBook(double tick_size) : tick_size_(tick_size) {
    if (!(std::isfinite(tick_size) && tick_size > 0.0)) {
        throw std::invalid_argument("tick_size must be positive and finite, got " +
                                    std::to_string(tick_size));
    }
}

std::int64_t LimitOrderBook::to_ticks(double price) const {
    if (!(std::isfinite(price) && price > 0.0)) {
        throw std::invalid_argument("price must be positive and finite, got " +
                                    std::to_string(price));
    }
    const double ratio = price / tick_size_;
    const double ticks = std::round(ratio);
    if (std::abs(ratio - ticks) > 1e-9 * std::max(1.0, std::abs(ratio))) {
        throw std::invalid_argument("price " + std::to_string(price) +
                                    " is not a multiple of tick_size " +
                                    std::to_string(tick_size_));
    }
    return static_cast<std::int64_t>(ticks);
}

void LimitOrderBook::check_side(char side) {
    if (side != BUY && side != SELL) {
        throw std::invalid_argument(std::string("side must be 'B' or 'S', got '") +
                                    side + "'");
    }
}

void LimitOrderBook::check_size(std::int64_t size) {
    if (size <= 0) {
        throw std::invalid_argument("size must be a positive integer, got " +
                                    std::to_string(size));
    }
}

std::optional<std::int64_t> LimitOrderBook::best_bid() const {
    if (bids_.empty()) return std::nullopt;
    return bids_.rbegin()->first;
}

std::optional<std::int64_t> LimitOrderBook::best_ask() const {
    if (asks_.empty()) return std::nullopt;
    return asks_.begin()->first;
}

std::int64_t LimitOrderBook::best_bid_size() const {
    if (bids_.empty()) return 0;
    std::int64_t s = 0;
    for (const Order& o : bids_.rbegin()->second) s += o.size;
    return s;
}

std::int64_t LimitOrderBook::best_ask_size() const {
    if (asks_.empty()) return 0;
    std::int64_t s = 0;
    for (const Order& o : asks_.begin()->second) s += o.size;
    return s;
}

double LimitOrderBook::mid() const {
    const auto b = best_bid();
    const auto a = best_ask();
    if (!b || !a) return std::nan("");
    return 0.5 * static_cast<double>(*b + *a);
}

double LimitOrderBook::spread() const {
    const auto b = best_bid();
    const auto a = best_ask();
    if (!b || !a) return std::nan("");
    return static_cast<double>(*a - *b);
}

double LimitOrderBook::microprice() const {
    const auto b = best_bid();
    const auto a = best_ask();
    if (!b || !a) return std::nan("");
    const double qb = static_cast<double>(best_bid_size());
    const double qa = static_cast<double>(best_ask_size());
    return (qa * static_cast<double>(*b) + qb * static_cast<double>(*a)) / (qa + qb);
}

std::int64_t LimitOrderBook::depth(char side, int k) const {
    check_side(side);
    if (k <= 0) {
        throw std::invalid_argument("k must be a positive integer, got " +
                                    std::to_string(k));
    }
    std::int64_t total = 0;
    int taken = 0;
    if (side == BUY) {
        for (auto it = bids_.rbegin(); it != bids_.rend() && taken < k; ++it, ++taken) {
            for (const Order& o : it->second) total += o.size;
        }
    } else {
        for (auto it = asks_.begin(); it != asks_.end() && taken < k; ++it, ++taken) {
            for (const Order& o : it->second) total += o.size;
        }
    }
    return total;
}

std::vector<std::int64_t> LimitOrderBook::order_ids(char side) const {
    check_side(side);
    std::vector<std::int64_t> ids;
    for (const auto& [id, loc] : orders_) {
        if (loc.first == side) ids.push_back(id);
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

bool LimitOrderBook::has_order(std::int64_t order_id) const {
    return orders_.count(order_id) > 0;
}

std::int64_t LimitOrderBook::total_size(char side) const {
    check_side(side);
    const Side& levels = side == BUY ? bids_ : asks_;
    std::int64_t total = 0;
    for (const auto& [price, queue] : levels) {
        (void)price;
        for (const Order& o : queue) total += o.size;
    }
    return total;
}

std::int64_t LimitOrderBook::n_orders(char side) const {
    check_side(side);
    std::int64_t n = 0;
    for (const auto& [id, loc] : orders_) {
        (void)id;
        if (loc.first == side) ++n;
    }
    return n;
}

std::pair<std::vector<Fill>, std::int64_t>
LimitOrderBook::match(char side, std::int64_t size, const std::int64_t* limit_ticks) {
    std::vector<Fill> fills;
    std::int64_t remaining = size;
    Side& opp = side == BUY ? asks_ : bids_;
    while (remaining > 0 && !opp.empty()) {
        const std::int64_t best =
            side == BUY ? opp.begin()->first : opp.rbegin()->first;
        if (limit_ticks != nullptr) {
            if (side == BUY && best > *limit_ticks) break;
            if (side == SELL && best < *limit_ticks) break;
        }
        Level& queue = opp[best];
        while (remaining > 0 && !queue.empty()) {
            Order& order = queue.front();
            const std::int64_t take = std::min(remaining, order.size);
            order.size -= take;
            remaining -= take;
            fills.push_back(Fill{order.id, best, take});
            if (order.size == 0) {
                orders_.erase(order.id);
                queue.pop_front();
            }
        }
        if (queue.empty()) opp.erase(best);
    }
    trades_.insert(trades_.end(), fills.begin(), fills.end());
    return {std::move(fills), remaining};
}

std::pair<std::vector<Fill>, std::int64_t>
LimitOrderBook::limit(char side, double price, std::int64_t size,
                      std::int64_t order_id) {
    check_side(side);
    check_size(size);
    const std::int64_t ticks = to_ticks(price);
    if (orders_.count(order_id) > 0) {
        throw std::invalid_argument("order_id " + std::to_string(order_id) +
                                    " is already live in the book");
    }
    auto [fills, remaining] = match(side, size, &ticks);
    if (remaining > 0) {
        Side& levels = side == BUY ? bids_ : asks_;
        levels[ticks].push_back(Order{order_id, remaining});
        orders_[order_id] = {side, ticks};
    }
    log_.push_back(BookLogEntry{'L', side, ticks, size, order_id,
                                size - remaining, 0});
    return {std::move(fills), remaining};
}

std::int64_t LimitOrderBook::cancel(std::int64_t order_id) {
    const auto it = orders_.find(order_id);
    if (it == orders_.end()) {
        throw std::invalid_argument("cancel of unknown order_id " +
                                    std::to_string(order_id));
    }
    const auto [side, ticks] = it->second;
    orders_.erase(it);
    Side& levels = side == BUY ? bids_ : asks_;
    Level& queue = levels[ticks];
    std::int64_t removed = 0;
    for (auto qit = queue.begin(); qit != queue.end(); ++qit) {
        if (qit->id == order_id) {
            removed = qit->size;
            queue.erase(qit);
            break;
        }
    }
    if (queue.empty()) levels.erase(ticks);
    log_.push_back(BookLogEntry{'C', side, ticks, removed, order_id, 0, 0});
    return removed;
}

std::pair<std::vector<Fill>, std::int64_t>
LimitOrderBook::market(char side, std::int64_t size) {
    check_side(side);
    check_size(size);
    auto [fills, unfilled] = match(side, size, nullptr);
    log_.push_back(BookLogEntry{'M', side, 0, size, 0, size - unfilled, unfilled});
    return {std::move(fills), unfilled};
}

} // namespace lobmod
