// ITCH decoder on hand-built frames, and the FastBook operations it drives.
// Plain main(), like the other FastBook tests, so no GoogleTest is needed.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "itch_decode.hpp"

using lobmod::LimitOrderBook;
using lobmod::fast::Config;
using lobmod::fast::FastBook;
using lobmod::fast::Side;
using lobmod::fast::Status;
using namespace lobmod::itch;

namespace {

int failures = 0;

#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::printf("FAIL line %d: %s\n", __LINE__, #cond);         \
            ++failures;                                                 \
        }                                                               \
    } while (0)

using Bytes = std::vector<std::uint8_t>;

void put(Bytes& b, std::uint64_t v, int width) {
    for (int i = width - 1; i >= 0; --i) b.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
}

Bytes header(char type) {
    Bytes b{static_cast<std::uint8_t>(type)};
    put(b, 0x0102, 2);           // locate
    put(b, 0x0304, 2);           // tracking
    put(b, 0x0A0B0C0D0E0FULL, 6); // timestamp
    return b;
}

void text(Bytes& b, const char* s, std::size_t n) { b.insert(b.end(), s, s + n); }

Bytes add(std::uint64_t ref, char side, std::uint32_t shares, std::uint32_t price) {
    Bytes b = header('A');
    put(b, ref, 8);
    b.push_back(static_cast<std::uint8_t>(side));
    put(b, shares, 4);
    text(b, "AAPL    ", 8);
    put(b, price, 4);
    return b;
}

Bytes exec(std::uint64_t ref, std::uint32_t shares) {
    Bytes b = header('E');
    put(b, ref, 8);
    put(b, shares, 4);
    put(b, 77, 8);
    return b;
}

Bytes cancel(std::uint64_t ref, std::uint32_t shares) {
    Bytes b = header('X');
    put(b, ref, 8);
    put(b, shares, 4);
    return b;
}

Bytes del(std::uint64_t ref) {
    Bytes b = header('D');
    put(b, ref, 8);
    return b;
}

Bytes replace(std::uint64_t ref, std::uint64_t next, std::uint32_t shares, std::uint32_t price) {
    Bytes b = header('U');
    put(b, ref, 8);
    put(b, next, 8);
    put(b, shares, 4);
    put(b, price, 4);
    return b;
}

Message must(const Bytes& b) {
    Message m;
    CHECK(decode(b.data(), b.size(), m) == Decode::Ok);
    return m;
}

void test_table() {
    const struct {
        char t;
        std::size_t n;
    } want[] = {{'A', 36}, {'F', 40}, {'E', 31}, {'C', 36}, {'X', 23}, {'D', 19},
                {'U', 35}, {'P', 44}, {'Q', 40}, {'R', 39}, {'S', 12}, {'H', 25},
                {'Y', 20}, {'L', 26}, {'I', 50}, {'J', 35}, {'K', 28}, {'V', 35}};
    int known = 0;
    for (int t = 0; t < 256; ++t) known += message_length(static_cast<std::uint8_t>(t)) != 0;
    CHECK(known == 18);
    for (const auto& w : want) {
        CHECK(message_length(static_cast<std::uint8_t>(w.t)) == w.n);
        Bytes b(w.n + 1, 'B');
        b[0] = static_cast<std::uint8_t>(w.t);
        Message m;
        for (std::size_t n = 1; n < w.n; ++n) CHECK(decode(b.data(), n, m) == Decode::Truncated);
        CHECK(decode(b.data(), w.n + 1, m) == Decode::BadLength);
    }
}

void test_fields() {
    Message m = must(add(0x1122334455667788ULL, 'S', 300, 1234500));
    CHECK(m.type == 'A' && m.locate == 0x0102 && m.tracking == 0x0304);
    CHECK(m.timestamp_ns == 0x0A0B0C0D0E0FULL);
    CHECK(m.ref == 0x1122334455667788ULL && m.side == 'S');
    CHECK(m.shares == 300 && m.price == 1234500);

    Bytes f = add(9, 'B', 1, 2);
    f[0] = 'F';
    text(f, "MPID", 4);
    m = must(f);
    CHECK(m.type == 'F' && m.ref == 9 && m.side == 'B');
    CHECK(m.stock == f.data() + 24 && std::memcmp(m.stock, "AAPL    ", 8) == 0);
    CHECK(m.attribution == f.data() + 36 && std::memcmp(m.attribution, "MPID", 4) == 0);

    m = must(exec(5, 40));
    CHECK(m.type == 'E' && m.ref == 5 && m.shares == 40 && m.match == 77 && m.price == 0);

    Bytes c = exec(6, 41);
    c[0] = 'C';
    c.push_back('Y');
    put(c, 999, 4);
    m = must(c);
    CHECK(m.type == 'C' && m.ref == 6 && m.shares == 41 && m.match == 77);
    CHECK(m.printable == 'Y' && m.price == 999);

    m = must(cancel(7, 3));
    CHECK(m.type == 'X' && m.ref == 7 && m.shares == 3);
    m = must(del(8));
    CHECK(m.type == 'D' && m.ref == 8);
    m = must(replace(10, 11, 12, 13));
    CHECK(m.type == 'U' && m.ref == 10 && m.new_ref == 11 && m.shares == 12 && m.price == 13);

    Bytes p = add(0, 'B', 50, 60);
    p[0] = 'P';
    put(p, 88, 8);
    m = must(p);
    CHECK(m.type == 'P' && m.side == 'B' && m.shares == 50 && m.price == 60 && m.match == 88);
}

void test_refusals() {
    Message m;
    CHECK(decode(nullptr, 0, m) == Decode::Truncated);
    const std::uint8_t unknown[4] = {'Z', 0, 0, 0};
    CHECK(decode(unknown, 4, m) == Decode::UnknownType);
    const Bytes bad = add(1, 'x', 1, 1);
    CHECK(decode(bad.data(), bad.size(), m) == Decode::BadSide);
    Bytes s = header('S');
    s.push_back('O');
    CHECK(decode(s.data(), s.size(), m) == Decode::Ignored);
    CHECK(m.type == 'S' && m.locate == 0x0102);
}

void test_frames() {
    Bytes buf;
    const Bytes a = add(1, 'B', 1, 1);
    const Bytes d = del(1);
    put(buf, a.size(), 2);
    buf.insert(buf.end(), a.begin(), a.end());
    put(buf, d.size(), 2);
    buf.insert(buf.end(), d.begin(), d.end());
    put(buf, 50, 2); // a frame cut short
    buf.push_back('A');
    std::size_t at = 0, len = 0;
    const std::uint8_t* msg = nullptr;
    CHECK(next_frame(buf.data(), buf.size(), &at, &msg, &len) && len == 36 && msg[0] == 'A');
    CHECK(next_frame(buf.data(), buf.size(), &at, &msg, &len) && len == 19 && msg[0] == 'D');
    const std::size_t stuck = at;
    CHECK(!next_frame(buf.data(), buf.size(), &at, &msg, &len));
    CHECK(at == stuck && at + 3 == buf.size());
}

void test_mirror() {
    FastBook b(Config{1000, 1000, 16});
    const auto feed = [&b](const Bytes& x) { return apply(b, must(x)); };
    CHECK(feed(add(1, 'B', 100, 1500)) == Status::Ok);
    CHECK(feed(add(2, 'B', 50, 1500)) == Status::Ok);
    CHECK(feed(add(3, 'S', 70, 1510)) == Status::Ok);
    CHECK(b.best_bid() == 1500 && b.best_bid_size() == 150 && b.best_ask() == 1510);
    CHECK(feed(exec(1, 30)) == Status::Ok);
    CHECK(feed(cancel(1, 20)) == Status::Ok);
    CHECK(b.best_bid_size() == 100 && b.n_orders(Side::Buy) == 2);
    CHECK(feed(exec(2, 50)) == Status::Ok); // full execution removes it
    CHECK(!b.has_order(2) && b.best_bid_size() == 50);
    CHECK(feed(replace(1, 4, 25, 1505)) == Status::Ok); // side comes from the order
    CHECK(!b.has_order(1) && b.has_order(4));
    CHECK(b.best_bid() == 1505 && b.best_bid_size() == 25);
    CHECK(feed(del(4)) == Status::Ok);
    CHECK(b.best_bid() == 0 && b.total_size(Side::Buy) == 0);
    CHECK(feed(del(4)) == Status::UnknownId);
    CHECK(feed(exec(3, 71)) == Status::BadSize);
    CHECK(b.best_ask_size() == 70);
    // A hidden trade leaves the book alone.
    Bytes p = add(0, 'B', 50, 1510);
    p[0] = 'P';
    put(p, 88, 8);
    CHECK(feed(p) == Status::Ok && b.best_ask_size() == 70);
}

void test_rest_does_not_match() {
    FastBook b(Config{1, 100, 8});
    auto sink = [](std::int64_t, std::int64_t, std::int64_t) { CHECK(false); };
    CHECK(b.rest(Side::Sell, 50, 10, 1).status == Status::Ok);
    CHECK(b.rest(Side::Buy, 60, 10, 2).status == Status::Ok); // crossed, as before an open
    CHECK(b.best_bid() == 60 && b.best_ask() == 50);
    CHECK(b.rest(Side::Buy, 60, 10, 2).status == Status::DuplicateId);
    CHECK(b.rest(Side::Buy, 600, 10, 3).status == Status::OutOfBand);
    CHECK(b.rest(Side::Buy, 60, 0, 3).status == Status::BadSize);
    CHECK(b.cancel(1).status == Status::Ok && b.cancel(2).status == Status::Ok);
    (void)sink;
}

void test_replace_refusals_change_nothing() {
    FastBook b(Config{1, 100, 8});
    CHECK(b.rest(Side::Buy, 50, 10, 1).status == Status::Ok);
    CHECK(b.rest(Side::Buy, 40, 10, 2).status == Status::Ok);
    CHECK(b.replace(9, 3, 50, 10).status == Status::UnknownId);
    CHECK(b.replace(1, 2, 50, 10).status == Status::DuplicateId);
    CHECK(b.replace(1, 3, 500, 10).status == Status::OutOfBand);
    CHECK(b.replace(1, 3, 50, 0).status == Status::BadSize);
    CHECK(b.has_order(1) && b.best_bid() == 50 && b.total_size(Side::Buy) == 20);
    CHECK(b.replace(1, 1, 45, 7).status == Status::Ok); // same id is legal
    CHECK(b.best_bid() == 45 && b.best_bid_size() == 7);
}

/// reduce against the reference book, including the queue place it keeps.
void test_reduce_matches_reference() {
    LimitOrderBook ref(1.0, false);
    FastBook b(Config{1, 100, 8});
    auto sink = [](std::int64_t, std::int64_t, std::int64_t) {};
    for (std::int64_t id = 1; id <= 3; ++id) {
        ref.limit_ticks('B', 50, 10, id);
        CHECK(b.limit(Side::Buy, 50, 10, id, sink).status == Status::Ok);
    }
    CHECK(b.reduce(1, 4).value == ref.reduce(1, 4));
    CHECK(b.reduce(2, 10).value == ref.reduce(2, 10));
    CHECK(!b.has_order(2) && !ref.has_order(2));
    CHECK(b.reduce(1, 7).status == Status::BadSize);
    bool threw = false;
    try {
        ref.reduce(1, 7);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    std::vector<std::int64_t> order;
    auto rec = [&order](std::int64_t id, std::int64_t, std::int64_t) { order.push_back(id); };
    CHECK(b.market(Side::Sell, 100, rec).value == 84);
    CHECK(order.size() == 2 && order[0] == 1 && order[1] == 3); // 1 kept its place
}

} // namespace

int main() {
    test_table();
    test_fields();
    test_refusals();
    test_frames();
    test_mirror();
    test_rest_does_not_match();
    test_replace_refusals_change_nothing();
    test_reduce_matches_reference();
    if (failures == 0) std::puts("itch tests passed");
    return failures == 0 ? 0 : 1;
}
