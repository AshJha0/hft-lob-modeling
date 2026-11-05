// Reference book against FastBook on one deep-book script.  Method and caveats
// are in docs/LATENCY.md.  Usage: fast_book_bench [ops] [live_orders] [repeats]
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "fast_book.hpp"

#if defined(__linux__)
#include <sched.h>
#endif
#if defined(__x86_64__)
#include <x86intrin.h>
#endif

namespace {

using lobmod::LimitOrderBook;
using lobmod::fast::Config;
using lobmod::fast::FastBook;
using lobmod::fast::Side;
using Clock = std::chrono::steady_clock;

#if defined(__x86_64__)
inline std::uint64_t t_begin() {
    _mm_lfence();
    const std::uint64_t t = __rdtsc();
    _mm_lfence();
    return t;
}
inline std::uint64_t t_end() {
    unsigned aux;
    const std::uint64_t t = __rdtscp(&aux);
    _mm_lfence();
    return t;
}
#else
inline std::uint64_t t_begin() {
    return static_cast<std::uint64_t>(Clock::now().time_since_epoch().count());
}
inline std::uint64_t t_end() { return t_begin(); }
#endif

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

double ticks_per_ns() {
    const auto c0 = Clock::now();
    const std::uint64_t t0 = t_begin();
    while (seconds_since(c0) < 0.2) {
    }
    const std::uint64_t t1 = t_end();
    return static_cast<double>(t1 - t0) / (seconds_since(c0) * 1e9);
}

std::uint64_t timer_overhead() {
    std::vector<std::uint64_t> d(1000000);
    for (auto& x : d) {
        const std::uint64_t a = t_begin();
        x = t_end() - a;
    }
    std::nth_element(d.begin(), d.begin() + static_cast<std::ptrdiff_t>(d.size() / 2), d.end());
    return d[d.size() / 2];
}

struct Rng {
    std::uint64_t s;
    std::uint64_t next() {
        std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    std::int64_t range(std::int64_t lo, std::int64_t hi) {
        return lo + static_cast<std::int64_t>(next() % static_cast<std::uint64_t>(hi - lo + 1));
    }
    double unit() { return static_cast<double>(next() >> 11) * 0x1p-53; }
};

enum OpType { kAdd = 0, kCancel = 1, kCross = 2, kMarket = 3, kOpTypes = 4 };
const char* const kOpNames[kOpTypes] = {"add", "cancel", "marketable limit", "market"};

struct Op {
    std::int64_t price, size, id;
    std::uint8_t type;
    Side side;
};

constexpr std::int64_t kBandLo = 1;
constexpr std::int64_t kBandLevels = 1 << 17;
constexpr std::int64_t kMid = kBandLevels / 2;
/// Set in main from the live-order target: an oversized pool and id map would
/// measure cache misses that a sized deployment does not have.
std::uint32_t g_pool = 1 << 17;

auto no_sink = [](std::int64_t, std::int64_t, std::int64_t) {};

/// Distance from the touch: mostly near it, thinning out to 500 ticks.
std::int64_t distance(Rng& rng) {
    std::int64_t d = 0;
    while (d < 500 && rng.unit() < 0.97) ++d;
    return d;
}

/// Builds the script by running it through a shadow FastBook, so every cancel
/// hits a live order.  The first `prefill` entries are adds run untimed.
std::vector<Op> make_script(std::size_t n_ops, std::size_t live, std::size_t* prefill) {
    Rng rng{42};
    FastBook shadow(Config{kBandLo, kBandLevels, g_pool});
    std::vector<Op> ops;
    ops.reserve(n_ops + 2 * live);
    std::vector<std::int64_t> ids;
    std::int64_t next_id = 0;
    std::int64_t mid = kMid;
    auto add = [&](Side side) {
        const std::int64_t d = 1 + distance(rng);
        const std::int64_t px = side == Side::Buy ? mid - d : mid + d;
        const std::int64_t bb = shadow.best_bid();
        const std::int64_t ba = shadow.best_ask();
        // stay passive: a drifting mid must not turn this into a taker
        if ((side == Side::Buy && ba != 0 && px >= ba) ||
            (side == Side::Sell && bb != 0 && px <= bb)) {
            return;
        }
        const Op op{px, rng.range(1, 100), ++next_id, kAdd, side};
        shadow.limit(side, op.price, op.size, op.id, no_sink);
        ops.push_back(op);
        ids.push_back(op.id);
    };
    while (ids.size() < live) add(rng.unit() < 0.5 ? Side::Buy : Side::Sell);
    *prefill = ops.size();
    const std::size_t end = *prefill + n_ops;
    while (ops.size() < end) {
        if (ops.size() % 1000 == 0) mid += rng.unit() < 0.5 ? -1 : 1;
        const Side side = rng.unit() < 0.5 ? Side::Buy : Side::Sell;
        const std::size_t resting =
            static_cast<std::size_t>(shadow.n_orders(Side::Buy) + shadow.n_orders(Side::Sell));
        const double u = rng.unit();
        const double p_add = resting > live ? 0.40 : 0.55;
        if (u < p_add) {
            add(side);
        } else if (u < 0.95) {
            while (!ids.empty()) {
                const std::size_t at = static_cast<std::size_t>(rng.next() % ids.size());
                const std::int64_t id = ids[at];
                ids[at] = ids.back();
                ids.pop_back();
                if (shadow.has_order(id)) {
                    shadow.cancel(id);
                    ops.push_back(Op{0, 0, id, kCancel, side});
                    break;
                }
            }
        } else if (u < 0.98) {
            const std::int64_t touch = side == Side::Buy ? shadow.best_ask() : shadow.best_bid();
            if (touch == 0) continue;
            const std::int64_t reach = rng.range(0, 2);
            const Op op{side == Side::Buy ? touch + reach : touch - reach, rng.range(1, 300),
                        ++next_id, kCross, side};
            const auto r = shadow.limit(side, op.price, op.size, op.id, no_sink);
            ops.push_back(op);
            if (r.value > 0) ids.push_back(op.id);
        } else {
            const Op op{0, rng.range(1, 300), 0, kMarket, side};
            shadow.market(side, op.size, no_sink);
            ops.push_back(op);
        }
    }
    return ops;
}

inline std::uint64_t mix(std::int64_t id, std::int64_t px, std::int64_t sz) {
    return static_cast<std::uint64_t>(id ^ (px * 31 + sz));
}

struct RefAdapter {
    static constexpr const char* kName = "reference";
    LimitOrderBook book{1.0, false};
    std::uint64_t checksum = 0;
    void absorb(const std::vector<lobmod::Fill>& fills) {
        for (const auto& f : fills) checksum += mix(f.maker_id, f.price, f.size);
    }
    void run(const Op& op) {
        const char side = op.side == Side::Buy ? 'B' : 'S';
        switch (op.type) {
        case kCancel:
            book.cancel(op.id);
            break;
        case kMarket:
            absorb(book.market(side, op.size).first);
            break;
        default:
            absorb(book.limit_ticks(side, op.price, op.size, op.id).first);
        }
    }
};

struct FastAdapter {
    static constexpr const char* kName = "FastBook";
    FastBook book{Config{kBandLo, kBandLevels, g_pool}};
    std::uint64_t checksum = 0;
    void run(const Op& op) {
        auto sink = [this](std::int64_t id, std::int64_t px, std::int64_t sz) {
            checksum += mix(id, px, sz);
        };
        switch (op.type) {
        case kCancel:
            book.cancel(op.id);
            break;
        case kMarket:
            book.market(op.side, op.size, sink);
            break;
        default:
            book.limit(op.side, op.price, op.size, op.id, sink);
        }
    }
};

struct Stats {
    double p50, p99, p999, max;
};

Stats summarise(std::vector<std::uint64_t>& d, std::uint64_t overhead, double tpn) {
    std::sort(d.begin(), d.end());
    auto at = [&](double q) {
        const std::size_t i = static_cast<std::size_t>(q * static_cast<double>(d.size() - 1));
        const std::uint64_t v = d[i] > overhead ? d[i] - overhead : 0;
        return static_cast<double>(v) / tpn;
    };
    return Stats{at(0.5), at(0.99), at(0.999), at(1.0)};
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

template <class Adapter>
std::uint64_t measure(const std::vector<Op>& ops, std::size_t prefill, int repeats,
                      std::uint64_t overhead, double tpn) {
    std::size_t counts[kOpTypes] = {0, 0, 0, 0};
    for (std::size_t i = prefill; i < ops.size(); ++i) ++counts[ops[i].type];
    std::vector<Stats> runs[kOpTypes];
    std::vector<double> throughput;
    std::uint64_t checksum = 0;
    for (int r = 0; r < repeats; ++r) {
        std::vector<std::uint64_t> lat[kOpTypes];
        std::size_t fill[kOpTypes] = {0, 0, 0, 0};
        for (int t = 0; t < kOpTypes; ++t) lat[t].assign(counts[t], 0); // touched up front
        {
            auto a = std::make_unique<Adapter>();
            for (std::size_t i = 0; i < prefill; ++i) a->run(ops[i]);
            for (std::size_t i = prefill; i < ops.size(); ++i) {
                const std::uint64_t t0 = t_begin();
                a->run(ops[i]);
                const std::uint64_t t1 = t_end();
                lat[ops[i].type][fill[ops[i].type]++] = t1 - t0;
            }
            checksum = a->checksum;
        }
        for (int t = 0; t < kOpTypes; ++t) runs[t].push_back(summarise(lat[t], overhead, tpn));
        {
            auto a = std::make_unique<Adapter>();
            for (std::size_t i = 0; i < prefill; ++i) a->run(ops[i]);
            const auto c0 = Clock::now();
            for (std::size_t i = prefill; i < ops.size(); ++i) a->run(ops[i]);
            const double s = seconds_since(c0);
            throughput.push_back(static_cast<double>(ops.size() - prefill) / s);
            if (a->checksum != checksum) {
                std::printf("checksum drift in %s\n", Adapter::kName);
                std::exit(1);
            }
        }
    }
    for (int t = 0; t < kOpTypes; ++t) {
        std::vector<double> p50, p99, p999;
        double mx = 0;
        for (const Stats& s : runs[t]) {
            p50.push_back(s.p50);
            p99.push_back(s.p99);
            p999.push_back(s.p999);
            mx = std::max(mx, s.max);
        }
        std::printf("| %s | %s | %zu | %.0f | %.0f | %.0f | %.0f |\n", Adapter::kName,
                    kOpNames[t], counts[t], median(p50), median(p99), median(p999), mx);
    }
    std::printf("%s untimed throughput: %.2f M ops/s (median of %d)\n", Adapter::kName,
                median(throughput) / 1e6, repeats);
    return checksum;
}

template <class Adapter>
double reverse_cancel(std::int64_t n) {
    auto a = std::make_unique<Adapter>();
    for (std::int64_t i = 1; i <= n; ++i) a->run(Op{kMid, 10, i, kAdd, Side::Buy});
    const auto c0 = Clock::now();
    for (std::int64_t i = n; i >= 1; --i) a->run(Op{0, 0, i, kCancel, Side::Buy});
    return seconds_since(c0) * 1e9 / static_cast<double>(n);
}

} // namespace

int main(int argc, char** argv) {
    const std::size_t n_ops = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 2000000;
    const std::size_t live = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 20000;
    const int repeats = argc > 3 ? std::atoi(argv[3]) : 5;
    while (g_pool < 4 * live) g_pool *= 2;
    bool pinned = false;
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(2, &set);
    pinned = sched_setaffinity(0, sizeof set, &set) == 0;
#endif
    const double tpn = ticks_per_ns();
    const std::uint64_t overhead = timer_overhead();
    std::printf("C++ pinned=%d ticks/ns=%.3f timer_overhead=%.1f ns ops=%zu live=%zu repeats=%d pool=%u\n",
                pinned ? 1 : 0, tpn, static_cast<double>(overhead) / tpn, n_ops, live, repeats,
                g_pool);
    std::size_t prefill = 0;
    const std::vector<Op> ops = make_script(n_ops, live, &prefill);
    std::printf("| book | op | count | p50 ns | p99 ns | p99.9 ns | max ns |\n"
                "|---|---|---|---|---|---|---|\n");
    const std::uint64_t a = measure<RefAdapter>(ops, prefill, repeats, overhead, tpn);
    const std::uint64_t b = measure<FastAdapter>(ops, prefill, repeats, overhead, tpn);
    std::printf("fills checksum %s\n", a == b ? "identical" : "DIFFERENT");
    std::printf("| orders in one level | reference ns/cancel | FastBook ns/cancel |\n|---|---|---|\n");
    for (std::int64_t n : {1000, 10000, 100000}) {
        const double r = reverse_cancel<RefAdapter>(n);
        const double f = reverse_cancel<FastAdapter>(n);
        std::printf("| %lld | %.0f | %.0f |\n", static_cast<long long>(n), r, f);
    }
    return a == b ? 0 : 1;
}
