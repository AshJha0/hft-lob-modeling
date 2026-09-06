#include "lobmod/zi.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "lobmod/rng.hpp"

namespace lobmod {

namespace {
constexpr double kRateLimit = 1.0;
constexpr double kRateMarket = 0.25;
constexpr double kRateCancelPerOrder = 0.04;
constexpr double kPAggressive = 0.12;
constexpr double kPGeometric = 0.35;
constexpr double kPSweep = 0.02;
constexpr std::int64_t kRefPrice = 1000;
} // namespace

std::vector<Event> generate_events(std::int64_t n_events, std::int64_t seed) {
    if (n_events < 12) {
        throw std::invalid_argument("n_events must be an integer >= 12, got " +
                                    std::to_string(n_events));
    }
    Lcg64 rng(seed);
    LimitOrderBook book(1.0, /*record_trades=*/false);
    std::vector<Event> events;
    events.reserve(static_cast<std::size_t>(n_events));

    const auto emit = [&](char etype, char side, std::int64_t price,
                          std::int64_t size, std::int64_t order_id) {
        const Event event{static_cast<std::int64_t>(events.size()) + 1, etype,
                          side, price, size, order_id};
        apply_event(book, event);
        events.push_back(event);
    };

    // -- seed phase: 6 levels each side, size 10
    for (std::int64_t i = 0; i < 6; ++i) {
        emit('L', 'B', kRefPrice - 1 - i, 10,
             static_cast<std::int64_t>(events.size()) + 1);
        emit('L', 'S', kRefPrice + 1 + i, 10,
             static_cast<std::int64_t>(events.size()) + 1);
    }

    // -- stochastic phase
    while (static_cast<std::int64_t>(events.size()) < n_events) {
        const std::int64_t seq = static_cast<std::int64_t>(events.size()) + 1;
        const double nb = static_cast<double>(book.n_orders('B'));
        const double ns = static_cast<double>(book.n_orders('S'));
        const double rates[6] = {
            kRateLimit,                  // limit buy
            kRateLimit,                  // limit sell
            kRateCancelPerOrder * nb,    // cancel buy
            kRateCancelPerOrder * ns,    // cancel sell
            kRateMarket,                 // market buy
            kRateMarket,                 // market sell
        };
        double total = 0.0;
        for (double r : rates) total += r;
        const double x = rng.uniform() * total;
        int kind = 0;
        double acc = rates[0];
        while (x >= acc && kind < 5) {
            ++kind;
            acc += rates[kind];
        }

        if (kind == 0 || kind == 1) { // limit order
            const bool buy = kind == 0;
            const auto opp_best = buy ? book.best_ask() : book.best_bid();
            const double u2 = rng.uniform();
            std::int64_t price;
            if (u2 < kPAggressive && opp_best.has_value()) {
                price = *opp_best; // aggressive: priced at the opposite best
            } else {
                const std::int64_t k = rng.geometric(kPGeometric);
                const std::int64_t anchor = opp_best.value_or(kRefPrice);
                price = buy ? anchor - (1 + k) : anchor + (1 + k);
            }
            price = std::max<std::int64_t>(price, 1);
            const std::int64_t size = 1 + rng.randint(5);
            emit('L', buy ? 'B' : 'S', price, size, seq);
        } else if (kind == 2 || kind == 3) { // cancel
            const char side = kind == 2 ? 'B' : 'S';
            const auto ids = book.order_ids(side);
            const std::int64_t oid =
                ids[static_cast<std::size_t>(rng.randint(
                    static_cast<std::int64_t>(ids.size())))];
            emit('C', side, 0, 0, oid);
        } else { // market order
            const char side = kind == 4 ? 'B' : 'S';
            const double u3 = rng.uniform();
            const std::int64_t size =
                u3 < kPSweep ? 40 + rng.randint(40) : 1 + rng.randint(3);
            emit('M', side, 0, size, 0);
        }
    }
    return events;
}

} // namespace lobmod
