// FastBook against the reference book, plus the zero-allocation check.  Plain
// main() so that the standalone cpp/fast build needs no GoogleTest.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <vector>

#include "fast_book.hpp"

namespace {

std::uint64_t g_news = 0;

} // namespace

// Counting replacements: every global allocation in this binary passes here.
void* operator new(std::size_t n) {
    ++g_news;
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

namespace {

using lobmod::Fill;
using lobmod::LimitOrderBook;
using lobmod::fast::Config;
using lobmod::fast::FastBook;
using lobmod::fast::Side;
using lobmod::fast::Status;

int g_failures = 0;

#define CHECK(cond)                                                    \
    do {                                                               \
        if (!(cond)) {                                                 \
            std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); \
            if (++g_failures > 20) std::exit(1);                       \
        }                                                              \
    } while (0)

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
    bool chance(double p) { return unit() < p; }
};

char code(Side s) { return s == Side::Buy ? 'B' : 'S'; }

bool same_double(double a, double b) { return (std::isnan(a) && std::isnan(b)) || a == b; }

void same_state(const LimitOrderBook& ref, const FastBook& fb) {
    CHECK(ref.best_bid().value_or(0) == fb.best_bid());
    CHECK(ref.best_ask().value_or(0) == fb.best_ask());
    CHECK(ref.best_bid_size() == fb.best_bid_size());
    CHECK(ref.best_ask_size() == fb.best_ask_size());
    CHECK(ref.total_size('B') == fb.total_size(Side::Buy));
    CHECK(ref.total_size('S') == fb.total_size(Side::Sell));
    CHECK(same_double(ref.mid(), fb.mid()));
    CHECK(same_double(ref.spread(), fb.spread()));
    CHECK(same_double(ref.microprice(), fb.microprice()));
}

void same_deep_state(const LimitOrderBook& ref, const FastBook& fb) {
    CHECK(ref.n_orders('B') == fb.n_orders(Side::Buy));
    CHECK(ref.n_orders('S') == fb.n_orders(Side::Sell));
    for (int k : {1, 3, 10, 1000}) {
        CHECK(ref.depth('B', k) == fb.depth(Side::Buy, k));
        CHECK(ref.depth('S', k) == fb.depth(Side::Sell, k));
    }
}

bool same_fills(const std::vector<Fill>& a, const std::vector<Fill>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].maker_id != b[i].maker_id || a[i].price != b[i].price ||
            a[i].size != b[i].size) {
            return false;
        }
    }
    return true;
}

/// id_stride spreads ids over the 64-bit range so that the map's probing and
/// negative ids get exercised, not only small counters.
void differential(std::uint64_t seed, std::int64_t lo, std::int64_t hi, int steps,
                  std::int64_t id_stride) {
    Rng rng{seed};
    LimitOrderBook ref(1.0, false);
    FastBook fb(Config{lo, hi - lo + 1, 4096});
    std::vector<std::int64_t> ids;
    std::vector<Fill> got;
    auto sink = [&got](std::int64_t id, std::int64_t px, std::int64_t sz) {
        got.push_back(Fill{id, px, sz});
    };
    got.reserve(8192);
    std::int64_t counter = 0;
    for (int step = 0; step < steps; ++step) {
        const Side side = rng.chance(0.5) ? Side::Buy : Side::Sell;
        const double op = rng.unit();
        got.clear();
        if (op < 0.55) {
            const std::int64_t px = rng.range(lo, hi);
            const std::int64_t size = rng.range(1, 20);
            ++counter;
            const std::int64_t id = (counter % 2 ? counter : -counter) * id_stride;
            const auto want = ref.limit_ticks(code(side), px, size, id);
            const auto res = fb.limit(side, px, size, id, sink);
            CHECK(res.status == Status::Ok);
            CHECK(res.value == want.second);
            CHECK(same_fills(want.first, got));
            if (want.second > 0) ids.push_back(id);
        } else if (op < 0.80 && !ids.empty()) {
            const std::size_t at = static_cast<std::size_t>(rng.next() % ids.size());
            const std::int64_t id = ids[at];
            ids[at] = ids.back();
            ids.pop_back();
            if (ref.has_order(id)) {
                const std::int64_t want = ref.cancel(id);
                const auto res = fb.cancel(id);
                CHECK(res.status == Status::Ok);
                CHECK(res.value == want);
            } else {
                CHECK(fb.cancel(id).status == Status::UnknownId);
            }
        } else if (op < 0.97) {
            const std::int64_t size = rng.range(1, 60);
            const auto want = ref.market(code(side), size);
            const auto res = fb.market(side, size, sink);
            CHECK(res.status == Status::Ok);
            CHECK(res.value == want.second);
            CHECK(same_fills(want.first, got));
        } else if (!ids.empty()) {
            // a live id must be refused by both and change nothing
            const std::int64_t id = ids[static_cast<std::size_t>(rng.next() % ids.size())];
            if (ref.has_order(id)) {
                bool threw = false;
                try {
                    ref.limit_ticks('B', lo, 1, id);
                } catch (const std::invalid_argument&) {
                    threw = true;
                }
                CHECK(threw);
                CHECK(fb.limit(Side::Buy, lo, 1, id, sink).status == Status::DuplicateId);
                CHECK(got.empty());
            }
        }
        same_state(ref, fb);
        if (step % 64 == 0) same_deep_state(ref, fb);
    }
    same_deep_state(ref, fb);
    for (std::int64_t id : ids) CHECK(ref.has_order(id) == fb.has_order(id));
}

void test_differential() {
    for (std::uint64_t seed = 1; seed <= 40; ++seed) {
        differential(seed, 90, 110, 3000, 1);                    // dense, deep queues
        differential(seed + 100, 1, 5000, 3000, 7);              // sparse, long gaps
        differential(seed + 200, 1000, 300000, 2000, 1LL << 40); // several bitset levels
    }
}

void test_codes() {
    FastBook fb(Config{100, 11, 2});
    auto none = [](std::int64_t, std::int64_t, std::int64_t) { CHECK(false); };
    CHECK(fb.limit(Side::Buy, 100, 0, 1, none).status == Status::BadSize);
    CHECK(fb.limit(Side::Buy, 0, 1, 1, none).status == Status::BadPrice);
    CHECK(fb.limit(Side::Buy, 99, 1, 1, none).status == Status::OutOfBand);
    CHECK(fb.limit(Side::Buy, 111, 1, 1, none).status == Status::OutOfBand);
    CHECK(fb.market(Side::Buy, -1, none).status == Status::BadSize);
    CHECK(fb.cancel(1).status == Status::UnknownId);
    CHECK(fb.limit(Side::Buy, 100, 5, 1, none).value == 5);
    CHECK(fb.limit(Side::Buy, 110, 5, 1, none).status == Status::DuplicateId);
    CHECK(fb.limit(Side::Buy, 110, lobmod::kMaxTicks, 2, none).status == Status::SideOverflow);
    CHECK(fb.limit(Side::Buy, 110, 5, 2, none).status == Status::Ok);
    // pool full: refused untouched, even though this order would fully execute
    CHECK(fb.limit(Side::Sell, 100, 1, 3, none).status == Status::PoolFull);
    CHECK(fb.total_size(Side::Buy) == 10);
    CHECK(fb.cancel(1).value == 5);
    CHECK(fb.market(Side::Sell, 9, [](std::int64_t, std::int64_t, std::int64_t) {}).value == 4);
    CHECK(!fb.has_bid());
    CHECK(std::isnan(fb.mid()));
    bool threw = false;
    try {
        FastBook bad(Config{0, 10, 10});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

void test_no_allocation() {
    FastBook fb(Config{1, 1 << 16, 1 << 16});
    Rng rng{7};
    std::vector<std::int64_t> live;
    live.reserve(1 << 20);
    std::int64_t filled = 0;
    auto sink = [&filled](std::int64_t, std::int64_t, std::int64_t sz) { filled += sz; };
    const std::uint64_t before = g_news;
    std::int64_t id = 0;
    for (int i = 0; i < 1000000; ++i) {
        const std::uint64_t r = rng.next() % 100;
        const Side side = rng.chance(0.5) ? Side::Buy : Side::Sell;
        if (r < 50) {
            const auto res = fb.limit(side, rng.range(30000, 34000), rng.range(1, 50), ++id, sink);
            if (res.value > 0 && live.size() < live.capacity()) live.push_back(id);
        } else if (r < 90 && !live.empty()) {
            const std::size_t at = static_cast<std::size_t>(rng.next() % live.size());
            fb.cancel(live[at]);
            live[at] = live.back();
            live.pop_back();
        } else {
            fb.market(side, rng.range(1, 100), sink);
        }
    }
    const std::uint64_t after = g_news;
    CHECK(after == before);
    CHECK(filled > 0);
    std::printf("allocations during 1000000 operations: %llu (filled %lld)\n",
                static_cast<unsigned long long>(after - before),
                static_cast<long long>(filled));
}

} // namespace

int main() {
    test_codes();
    test_differential();
    test_no_allocation();
    if (g_failures != 0) {
        std::printf("%d failures\n", g_failures);
        return 1;
    }
    std::printf("fast book tests OK\n");
    return 0;
}
