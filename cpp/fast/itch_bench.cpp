// ITCH decode, and decode into FastBook, on synthetic frames.  The generator
// is integer-only and matches rust/src/bin/itch_bench.rs call for call, so the
// two print the same state line.  Method: docs/LATENCY.md.
// Usage: itch_bench [messages] [live_orders] [repeats]
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "itch_decode.hpp"

#if defined(__linux__)
#include <sched.h>
#endif

namespace {

using lobmod::fast::Config;
using lobmod::fast::FastBook;
using lobmod::fast::Side;
using lobmod::fast::Status;
using namespace lobmod::itch;
using Clock = std::chrono::steady_clock;

constexpr std::int64_t kBandLo = 1000000 - 65536;
constexpr std::int64_t kBandLevels = 1 << 17;

struct Rng {
    std::uint64_t s;
    std::uint64_t next() {
        std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
};

struct Live {
    std::uint64_t ref;
    std::uint32_t shares;
    bool buy;
};

class Writer {
public:
    explicit Writer(std::vector<std::uint8_t>& out) : out_(out) {}
    void begin(char type, std::size_t len, std::uint64_t ts) {
        num(len, 2);
        out_.push_back(static_cast<std::uint8_t>(type));
        num(1, 2);
        num(0, 2);
        num(ts, 6);
    }
    void num(std::uint64_t v, int width) {
        for (int i = width - 1; i >= 0; --i) out_.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
    }
    void bytes(const char* s, int n) { out_.insert(out_.end(), s, s + n); }

private:
    std::vector<std::uint8_t>& out_;
};

std::vector<std::uint8_t> make_stream(std::size_t n, std::size_t live) {
    Rng rng{42};
    std::vector<std::uint8_t> out;
    out.reserve(n * 36);
    Writer w(out);
    std::vector<Live> book;
    std::uint64_t next_ref = 0, next_match = 0;
    std::uint32_t mid = 1000000;
    auto quote = [&](bool buy) {
        const auto d = static_cast<std::uint32_t>(rng.next() % 64);
        return buy ? mid - 1 - d : mid + 1 + d;
    };
    for (std::size_t i = 0; i < n; ++i) {
        if (i % 1024 == 0) mid = (rng.next() & 1) != 0 ? mid + 1 : mid - 1;
        const std::uint64_t ts = i * 1000;
        const std::uint64_t r = rng.next() % 100;
        const std::size_t sz = book.size();
        if (sz < live || (r < 45 && sz < 2 * live)) {
            const bool buy = (rng.next() & 1) != 0;
            const std::uint32_t px = quote(buy);
            const auto shares = static_cast<std::uint32_t>(1 + rng.next() % 500);
            const bool attributed = r % 8 == 0;
            w.begin(attributed ? 'F' : 'A', attributed ? 40 : 36, ts);
            w.num(++next_ref, 8);
            w.bytes(buy ? "B" : "S", 1);
            w.num(shares, 4);
            w.bytes("SYNTH   ", 8);
            w.num(px, 4);
            if (attributed) w.bytes("MPID", 4);
            book.push_back(Live{next_ref, shares, buy});
            continue;
        }
        if (r >= 98) {
            w.begin('P', 44, ts);
            w.num(0, 8);
            w.bytes("B", 1);
            w.num(100, 4);
            w.bytes("SYNTH   ", 8);
            w.num(mid, 4);
            w.num(++next_match, 8);
            continue;
        }
        const auto at = static_cast<std::size_t>(rng.next() % sz);
        Live& o = book[at];
        if (r >= 92) {
            const std::uint32_t px = quote(o.buy);
            const auto shares = static_cast<std::uint32_t>(1 + rng.next() % 500);
            w.begin('U', 35, ts);
            w.num(o.ref, 8);
            w.num(++next_ref, 8);
            w.num(shares, 4);
            w.num(px, 4);
            o.ref = next_ref;
            o.shares = shares;
            continue;
        }
        if (r >= 70 && o.shares > 1) {
            const auto by = static_cast<std::uint32_t>(1 + rng.next() % (o.shares - 1));
            if (r < 80) {
                w.begin('X', 23, ts);
                w.num(o.ref, 8);
                w.num(by, 4);
            } else {
                const bool with_price = r >= 88;
                w.begin(with_price ? 'C' : 'E', with_price ? 36 : 31, ts);
                w.num(o.ref, 8);
                w.num(by, 4);
                w.num(++next_match, 8);
                if (with_price) {
                    w.bytes("Y", 1);
                    w.num(mid, 4);
                }
            }
            o.shares -= by;
            continue;
        }
        w.begin('D', 19, ts);
        w.num(o.ref, 8);
        book[at] = book.back();
        book.pop_back();
    }
    return out;
}

std::uint64_t decode_pass(const std::vector<std::uint8_t>& buf) {
    std::uint64_t sum = 0;
    std::size_t at = 0, len = 0;
    const std::uint8_t* msg = nullptr;
    Message m;
    while (next_frame(buf.data(), buf.size(), &at, &msg, &len)) {
        if (decode(msg, len, m) != Decode::Ok) std::abort();
        sum += m.ref + m.shares + m.price;
    }
    return sum;
}

/// Returns the book so the caller can print its state.
void book_pass(const std::vector<std::uint8_t>& buf, FastBook& book) {
    std::size_t at = 0, len = 0;
    const std::uint8_t* msg = nullptr;
    Message m;
    while (next_frame(buf.data(), buf.size(), &at, &msg, &len)) {
        if (decode(msg, len, m) != Decode::Ok || apply(book, m) != Status::Ok) std::abort();
    }
}

void report(const char* stage, std::size_t n, std::vector<double>& secs) {
    std::sort(secs.begin(), secs.end());
    const double scale = 1e9 / static_cast<double>(n);
    std::printf("| %s | %zu | %.1f | %.1f |\n", stage, n, secs[secs.size() / 2] * scale,
                secs[0] * scale);
}

double since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

} // namespace

int main(int argc, char** argv) {
    const std::size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 5000000;
    const std::size_t live = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 20000;
    const int repeats = argc > 3 ? std::atoi(argv[3]) : 7;
    bool pinned = false;
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(2, &set);
    pinned = sched_setaffinity(0, sizeof set, &set) == 0;
#endif
    const std::vector<std::uint8_t> buf = make_stream(n, live);
    std::printf("C++ pinned=%d messages=%zu live=%zu repeats=%d bytes=%zu\n", pinned ? 1 : 0, n,
                live, repeats, buf.size());
    std::printf("| stage | messages | median ns/msg | best ns/msg |\n|---|---|---|---|\n");
    std::vector<double> dec, full;
    std::uint64_t sum = 0;
    for (int r = 0; r < repeats; ++r) {
        const auto t0 = Clock::now();
        sum += decode_pass(buf);
        dec.push_back(since(t0));
    }
    const Config cfg{kBandLo, kBandLevels, 1u << 17};
    std::int64_t bid = 0, ask = 0, bids = 0, asks = 0;
    for (int r = 0; r < repeats; ++r) {
        FastBook book(cfg);
        const auto t0 = Clock::now();
        book_pass(buf, book);
        full.push_back(since(t0));
        bid = book.best_bid();
        ask = book.best_ask();
        bids = book.total_size(Side::Buy);
        asks = book.total_size(Side::Sell);
    }
    report("decode", n, dec);
    report("decode + FastBook", n, full);
    std::printf("state bid=%lld ask=%lld bid_size=%lld ask_size=%lld checksum=%llu\n",
                static_cast<long long>(bid), static_cast<long long>(ask),
                static_cast<long long>(bids), static_cast<long long>(asks),
                static_cast<unsigned long long>(sum / static_cast<std::uint64_t>(repeats)));
    return 0;
}
