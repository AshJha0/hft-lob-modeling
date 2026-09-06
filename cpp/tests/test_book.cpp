/// Book mechanics: price-time priority, crossing limits, market orders,
/// cancels, NaN guards, tick validation, replay invariants, and the pinned
/// zero-intelligence generator.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <gtest/gtest.h>

#include "lobmod/book.hpp"
#include "lobmod/replay.hpp"
#include "lobmod/rng.hpp"
#include "lobmod/zi.hpp"
#include "test_data.hpp"

namespace {

using lobmod::LimitOrderBook;

/// Write `text` to a scratch file and return its path.
std::string scratch(const std::string& name, const std::string& text) {
    const std::string path = testing::TempDir() + "lobmod_" + name;
    std::ofstream out(path, std::ios::binary);
    out << text;
    return path;
}

std::string slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
}

TEST(Rng, PinnedLcgBasics) {
    lobmod::Lcg64 a(42), b(42);
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(a.next_u64(), b.next_u64()); // deterministic per seed
    }
    lobmod::Lcg64 c(7);
    for (int i = 0; i < 1000; ++i) {
        const double u = c.uniform();
        ASSERT_GE(u, 0.0);
        ASSERT_LT(u, 1.0);
    }
    // first state from seed 0 is the increment itself
    lobmod::Lcg64 z(0);
    EXPECT_EQ(z.next_u64(), 1442695040888963407ULL);
    EXPECT_THROW(lobmod::Lcg64(-1), std::invalid_argument);
    EXPECT_THROW(c.exponential(0.0), std::invalid_argument);
    EXPECT_THROW(c.geometric(0.0), std::invalid_argument);
    EXPECT_THROW(c.geometric(1.0), std::invalid_argument);
    EXPECT_THROW(c.randint(0), std::invalid_argument);
}

TEST(Book, FifoTimePriorityWithinLevel) {
    LimitOrderBook book;
    book.limit('S', 101, 5, 1); // first in queue at 101
    book.limit('S', 101, 5, 2); // second
    book.limit('S', 101, 5, 3); // third
    const auto [fills, unfilled] = book.market('B', 7);
    ASSERT_EQ(fills.size(), 2u);
    EXPECT_EQ(fills[0].maker_id, 1); // FIFO: earliest order fills first
    EXPECT_EQ(fills[0].size, 5);
    EXPECT_EQ(fills[1].maker_id, 2);
    EXPECT_EQ(fills[1].size, 2);
    EXPECT_EQ(unfilled, 0);
    EXPECT_FALSE(book.has_order(1));
    EXPECT_TRUE(book.has_order(2)); // partially filled, still resting
    EXPECT_EQ(book.best_ask_size(), 3 + 5);
}

TEST(Book, PricePriorityAcrossLevels) {
    LimitOrderBook book;
    book.limit('B', 99, 4, 1);
    book.limit('B', 100, 4, 2); // better bid must fill first
    book.limit('B', 98, 4, 3);
    const auto [fills, unfilled] = book.market('S', 6);
    ASSERT_EQ(fills.size(), 2u);
    EXPECT_EQ(fills[0].maker_id, 2);
    EXPECT_EQ(fills[0].price, 100);
    EXPECT_EQ(fills[1].maker_id, 1);
    EXPECT_EQ(fills[1].price, 99);
    EXPECT_EQ(unfilled, 0);
    EXPECT_EQ(*book.best_bid(), 99);
}

TEST(Book, MarketOrderWalksMultipleLevels) {
    LimitOrderBook book;
    book.limit('S', 101, 3, 1);
    book.limit('S', 102, 3, 2);
    book.limit('S', 103, 3, 3);
    const auto [fills, unfilled] = book.market('B', 8);
    ASSERT_EQ(fills.size(), 3u);
    EXPECT_EQ(fills[0].price, 101);
    EXPECT_EQ(fills[1].price, 102);
    EXPECT_EQ(fills[2].price, 103);
    EXPECT_EQ(fills[2].size, 2);
    EXPECT_EQ(unfilled, 0);
    EXPECT_EQ(book.total_size('S'), 1);
}

TEST(Book, MarketOrderBiggerThanBookReportsRemainder) {
    LimitOrderBook book;
    book.limit('S', 101, 3, 1);
    book.limit('S', 102, 2, 2);
    const auto [fills, unfilled] = book.market('B', 10);
    ASSERT_EQ(fills.size(), 2u);
    EXPECT_EQ(unfilled, 5); // reported, not an error
    EXPECT_FALSE(book.best_ask().has_value());
    EXPECT_TRUE(std::isnan(book.mid()));
}

TEST(Book, CrossedLimitExecutesThenRests) {
    LimitOrderBook book;
    book.limit('S', 101, 3, 1);
    book.limit('S', 102, 3, 2);
    // crossing buy limit at 101: takes level 101, then rests the remainder —
    // it must NOT trade through its own limit price into 102.
    const auto [fills, rested] = book.limit('B', 101, 5, 10);
    ASSERT_EQ(fills.size(), 1u);
    EXPECT_EQ(fills[0].price, 101); // executed at the resting order's price
    EXPECT_EQ(fills[0].size, 3);
    EXPECT_EQ(rested, 2);
    EXPECT_EQ(*book.best_bid(), 101);
    EXPECT_EQ(*book.best_ask(), 102);
    EXPECT_EQ(book.best_bid_size(), 2);
}

TEST(Book, CrossedLimitAtBetterPriceStopsAtItsLimit) {
    LimitOrderBook book;
    book.limit('B', 100, 4, 1);
    book.limit('B', 99, 4, 2);
    const auto [fills, rested] = book.limit('S', 100, 10, 3);
    ASSERT_EQ(fills.size(), 1u);
    EXPECT_EQ(fills[0].price, 100);
    EXPECT_EQ(fills[0].size, 4);
    EXPECT_EQ(rested, 6); // rests at 100, does not hit the 99 bid
    EXPECT_EQ(*book.best_ask(), 100);
    EXPECT_EQ(*book.best_bid(), 99);
}

TEST(Book, CancelErrorsAndRemoval) {
    LimitOrderBook book;
    book.limit('B', 100, 4, 1);
    EXPECT_THROW(book.cancel(99), std::invalid_argument); // never existed
    EXPECT_EQ(book.cancel(1), 4);
    EXPECT_THROW(book.cancel(1), std::invalid_argument); // already cancelled
    book.limit('S', 101, 2, 5);
    book.market('B', 2);
    EXPECT_THROW(book.cancel(5), std::invalid_argument); // fully filled
}

TEST(Book, EmptySideNanGuards) {
    LimitOrderBook book;
    EXPECT_TRUE(std::isnan(book.mid()));
    EXPECT_TRUE(std::isnan(book.spread()));
    EXPECT_TRUE(std::isnan(book.microprice()));
    EXPECT_EQ(book.best_bid_size(), 0);
    book.limit('B', 100, 4, 1);
    EXPECT_TRUE(std::isnan(book.mid())); // still no ask
    book.limit('S', 102, 4, 2);
    EXPECT_DOUBLE_EQ(book.mid(), 101.0);
    EXPECT_DOUBLE_EQ(book.spread(), 2.0);
    EXPECT_DOUBLE_EQ(book.microprice(), 101.0); // equal sizes: plain mid
    // market order into an EMPTY side: everything reported unfilled
    LimitOrderBook empty_book;
    const auto [fills, unfilled] = empty_book.market('B', 5);
    EXPECT_TRUE(fills.empty());
    EXPECT_EQ(unfilled, 5);
}

TEST(Book, MicropriceLeansTowardThinSide) {
    LimitOrderBook book;
    book.limit('B', 100, 1, 1); // thin bid
    book.limit('S', 102, 9, 2); // thick ask
    // microprice = (qa*Pb + qb*Pa)/(qa+qb) = (9*100 + 1*102)/10 = 100.2
    EXPECT_DOUBLE_EQ(book.microprice(), 100.2);
}

TEST(Book, ValidationErrors) {
    EXPECT_THROW(LimitOrderBook(0.0), std::invalid_argument);
    EXPECT_THROW(LimitOrderBook(-1.0), std::invalid_argument);
    LimitOrderBook book(0.5);
    EXPECT_THROW(book.limit('B', 100.3, 1, 1), std::invalid_argument); // off-grid
    EXPECT_THROW(book.limit('B', -1.0, 1, 1), std::invalid_argument);
    EXPECT_THROW(book.limit('X', 100.0, 1, 1), std::invalid_argument);
    EXPECT_THROW(book.limit('B', 100.0, 0, 1), std::invalid_argument);
    EXPECT_THROW(book.limit('B', 100.0, -3, 1), std::invalid_argument);
    book.limit('B', 100.0, 1, 1);
    EXPECT_THROW(book.limit('B', 99.5, 1, 1), std::invalid_argument); // dup id
    EXPECT_THROW(book.depth('B', 0), std::invalid_argument);
    EXPECT_THROW(book.depth('Q', 1), std::invalid_argument);
    EXPECT_THROW(book.market('B', 0), std::invalid_argument);
    EXPECT_NO_THROW(book.limit('B', 99.5, 1, 2)); // on the 0.5 grid
}

TEST(Book, DepthTopKLevels) {
    LimitOrderBook book;
    book.limit('B', 100, 1, 1);
    book.limit('B', 99, 2, 2);
    book.limit('B', 98, 4, 3);
    book.limit('B', 97, 8, 4);
    EXPECT_EQ(book.depth('B', 1), 1);
    EXPECT_EQ(book.depth('B', 3), 7);
    EXPECT_EQ(book.depth('B', 10), 15); // fewer levels: sums what exists
    EXPECT_EQ(book.depth('S', 3), 0);
}

TEST(Replay, InvariantsOverFullStream) {
    const auto& events = testdata::events();
    lobmod::LimitOrderBook book;
    std::map<long long, char> submit_side; // limit order id -> side
    for (const auto& e : events) {
        lobmod::apply_event(book, e);
        if (e.etype == 'L') submit_side[e.order_id] = e.side;
        const auto b = book.best_bid();
        const auto a = book.best_ask();
        if (b && a) {
            ASSERT_LT(*b, *a) << "crossed book after event " << e.seq;
        }
    }
    // size conservation per side: limit size added = still resting +
    // cancelled + filled as maker + executed as taker on arrival.
    for (char side : {'B', 'S'}) {
        long long added = 0, taker_exec = 0, cancelled = 0;
        for (const auto& entry : book.log()) {
            if (entry.etype == 'L' && entry.side == side) {
                added += entry.size;
                taker_exec += entry.executed;
            } else if (entry.etype == 'C' && entry.side == side) {
                cancelled += entry.size;
            }
        }
        long long maker_fills = 0;
        for (const auto& f : book.trades()) {
            const auto it = submit_side.find(f.maker_id);
            ASSERT_TRUE(it != submit_side.end());
            if (it->second == side) maker_fills += f.size;
        }
        const long long resting = book.total_size(side);
        EXPECT_EQ(added, resting + cancelled + maker_fills + taker_exec)
            << "size conservation violated on side " << side;
    }
}

TEST(Replay, ErrorsAndSchema) {
    EXPECT_THROW(lobmod::replay({}), std::invalid_argument);
    lobmod::LimitOrderBook book;
    const lobmod::Event bad{1, 'Z', 'B', 100, 1, 1};
    EXPECT_THROW(lobmod::apply_event(book, bad), std::invalid_argument);
    EXPECT_THROW(lobmod::read_events_csv("/nonexistent/file.csv"),
                 std::invalid_argument);
    const auto& events = testdata::events();
    ASSERT_EQ(events.size(), 5000u);
    EXPECT_EQ(events.front().seq, 1);
    EXPECT_EQ(events.back().seq, 5000);
}

TEST(Zi, RegeneratesBundledStreamExactly) {
    // The pinned generator + pinned LCG must reproduce the committed CSV
    // event for event (seed and count from data/dgp_params.json).
    const auto generated = lobmod::generate_events(5000, 20240613);
    const auto& bundled = testdata::events();
    ASSERT_EQ(generated.size(), bundled.size());
    for (std::size_t i = 0; i < bundled.size(); ++i) {
        ASSERT_EQ(generated[i].seq, bundled[i].seq) << "event " << i + 1;
        ASSERT_EQ(generated[i].etype, bundled[i].etype) << "event " << i + 1;
        ASSERT_EQ(generated[i].side, bundled[i].side) << "event " << i + 1;
        ASSERT_EQ(generated[i].price, bundled[i].price) << "event " << i + 1;
        ASSERT_EQ(generated[i].size, bundled[i].size) << "event " << i + 1;
        ASSERT_EQ(generated[i].order_id, bundled[i].order_id) << "event " << i + 1;
    }
}

TEST(Zi, Validation) {
    EXPECT_THROW(lobmod::generate_events(11, 1), std::invalid_argument);
    EXPECT_THROW(lobmod::generate_events(5000, -1), std::invalid_argument);
    const auto ev = lobmod::generate_events(12, 123);
    EXPECT_EQ(ev.size(), 12u); // seed phase only
    EXPECT_EQ(ev[0].price, 999);
    EXPECT_EQ(ev[1].price, 1001);
}

// ------------------------------------------------------------ review additions

TEST(Book, PriceRoundingToZeroTicksRejected) {
    // PT-1
    LimitOrderBook b(1.0);
    EXPECT_THROW(b.limit('B', 1e-12, 5, 1), std::invalid_argument);
    LimitOrderBook b2(0.01);
    EXPECT_THROW(b2.limit('S', 4e-12, 1, 2), std::invalid_argument);
    for (const LimitOrderBook* book : {&b, &b2}) {
        EXPECT_FALSE(book->best_bid().has_value());
        EXPECT_FALSE(book->best_ask().has_value());
        EXPECT_EQ(book->n_orders('B'), 0);
        EXPECT_EQ(book->n_orders('S'), 0);
        EXPECT_TRUE(book->log().empty());
    }
    EXPECT_THROW(b.limit_ticks('B', 0, 5, 1), std::invalid_argument);
    EXPECT_THROW(b.limit_ticks('B', -5, 5, 1), std::invalid_argument);
}

TEST(Book, LargeTickCounts) {
    // PT-2: fine-tick products need 64-bit tick counts; overflow is an error
    LimitOrderBook b(1e-8);
    const auto [f1, r1] = b.limit('S', 60000.0, 1, 1);
    EXPECT_TRUE(f1.empty());
    EXPECT_EQ(r1, 1);
    ASSERT_TRUE(b.best_ask().has_value());
    EXPECT_EQ(*b.best_ask(), 6000000000000LL);
    const auto [f2, r2] = b.limit('B', 60000.0, 1, 2);
    ASSERT_EQ(f2.size(), 1u);
    EXPECT_EQ(f2[0].maker_id, 1);
    EXPECT_EQ(f2[0].price, 6000000000000LL);
    EXPECT_EQ(f2[0].size, 1);
    EXPECT_EQ(r2, 0);
    EXPECT_FALSE(b.best_ask().has_value());
    EXPECT_FALSE(b.best_bid().has_value());
    EXPECT_THROW(b.limit('B', 1e300, 1, 3), std::invalid_argument); // no UB, no sweep
    EXPECT_FALSE(b.best_bid().has_value());
    EXPECT_THROW(b.limit('B', 1e19, 1, 3), std::invalid_argument);   // > 2^63
    EXPECT_THROW(b.limit_ticks('B', lobmod::kMaxTicks + 1, 1, 3), std::invalid_argument);
    b.limit_ticks('B', lobmod::kMaxTicks, 1, 3);
    EXPECT_EQ(*b.best_bid(), lobmod::kMaxTicks);
}

TEST(Book, TickRoundingRuleIsHalfUpWithinTolerance) {
    LimitOrderBook b(0.5);
    b.limit('B', 99.5, 1, 1);
    EXPECT_EQ(*b.best_bid(), 199);
    b.limit('B', 100.0000000001, 1, 2); // residual 2e-10 <= 1e-9 * 200
    EXPECT_EQ(*b.best_bid(), 200);
    EXPECT_THROW(b.limit('B', 100.004, 1, 3), std::invalid_argument);
    EXPECT_THROW(b.limit('B', 0.75, 1, 3), std::invalid_argument); // exactly 1.5 ticks
}

TEST(Book, LimitTicksMatchesLimit) {
    LimitOrderBook a(0.25), c(0.25);
    a.limit('S', 25.5, 4, 1);
    c.limit_ticks('S', 102, 4, 1);
    EXPECT_EQ(*a.best_ask(), 102);
    EXPECT_EQ(*c.best_ask(), 102);
    const auto [fa, ra] = a.limit('B', 26.0, 6, 2);
    const auto [fc, rc] = c.limit_ticks('B', 104, 6, 2);
    ASSERT_EQ(fa.size(), fc.size());
    EXPECT_EQ(fa[0].price, fc[0].price);
    EXPECT_EQ(ra, 2);
    EXPECT_EQ(rc, 2);
}

TEST(Replay, PricesAreTickCounts) {
    // PT-4: Event::price is in ticks; replay never rescales it
    const auto& events = testdata::events();
    const std::vector<lobmod::Event> head(events.begin(), events.begin() + 12);
    const auto st = lobmod::replay(head, 3);
    EXPECT_DOUBLE_EQ(st.best_bid[11], 999.0);
    EXPECT_DOUBLE_EQ(st.best_ask[11], 1001.0);
    LimitOrderBook manual(0.5);
    for (const auto& e : head) {
        manual.limit(e.side, static_cast<double>(e.price) * 0.5, e.size, e.order_id);
    }
    EXPECT_EQ(*manual.best_bid(), 999);
    EXPECT_THROW(lobmod::replay(head, 0), std::invalid_argument);
}

TEST(Replay, DepthMonotoneInK) {
    // PT-11
    const auto& events = testdata::events();
    LimitOrderBook book;
    for (std::size_t i = 0; i < 500; ++i) {
        lobmod::apply_event(book, events[i]);
        ASSERT_EQ(book.depth('B', 1), book.best_bid_size());
        ASSERT_EQ(book.depth('S', 1), book.best_ask_size());
        for (char side : {'B', 'S'}) {
            for (int k = 1; k <= 5; ++k) {
                ASSERT_LE(book.depth(side, k), book.depth(side, k + 1));
            }
        }
    }
}

TEST(Book, MarketOnEmptyBookAndReuseOfCancelledId) {
    // PT-12
    LimitOrderBook b;
    const auto [f0, u0] = b.market('B', 5);
    EXPECT_TRUE(f0.empty());
    EXPECT_EQ(u0, 5);
    b.limit('B', 100, 2, 7);
    EXPECT_EQ(b.cancel(7), 2);
    const auto [f1, r1] = b.limit('S', 101, 4, 7); // id free again
    EXPECT_TRUE(f1.empty());
    EXPECT_EQ(r1, 4);
    EXPECT_TRUE(b.has_order(7));
}

TEST(Book, AggressiveSellExecutesAtRestingPrice) {
    // PT-13 (price improvement)
    LimitOrderBook b;
    b.limit('B', 100, 4, 1);
    b.limit('B', 99, 4, 2);
    b.limit('S', 100, 10, 3);
    EXPECT_EQ(b.best_bid_size(), 4); // the 99 bid untouched
    const auto [fills, rested] = b.limit('S', 95, 2, 4);
    ASSERT_EQ(fills.size(), 1u);
    EXPECT_EQ(fills[0].price, 99);
    EXPECT_EQ(fills[0].size, 2);
    EXPECT_EQ(rested, 0);
}

TEST(Book, RecordTradesFlagAndClearHistory) {
    LimitOrderBook b(1.0, /*record_trades=*/false);
    b.limit('S', 101, 5, 1);
    const auto [fills, unfilled] = b.market('B', 2);
    EXPECT_EQ(fills.size(), 1u);
    EXPECT_EQ(unfilled, 0);
    EXPECT_TRUE(b.trades().empty());
    EXPECT_TRUE(b.log().empty());
    EXPECT_EQ(b.best_ask_size(), 3);
    LimitOrderBook c;
    c.limit('S', 101, 5, 1);
    c.market('B', 2);
    EXPECT_EQ(c.trades().size(), 1u);
    EXPECT_EQ(c.log().size(), 2u);
    c.clear_history();
    EXPECT_TRUE(c.trades().empty());
    EXPECT_TRUE(c.log().empty());
    EXPECT_EQ(c.best_ask_size(), 3);
}

TEST(Replay, EventsCsvRobustness) {
    // PT-15: pinned CSV rules
    const std::string body = "seq,type,side,price,size,order_id\n1,L,B,999,10,1\n2,L,S,1001,10,2\n";
    const auto ok = lobmod::read_events_csv(scratch("ok.csv", body));
    ASSERT_EQ(ok.size(), 2u);
    EXPECT_EQ(ok[1].price, 1001);
    EXPECT_EQ(ok[1].side, 'S');
    std::string crlf;
    for (char ch : body) {
        if (ch == '\n') crlf += "\r\n"; else crlf += ch;
    }
    EXPECT_EQ(lobmod::read_events_csv(scratch("crlf.csv", crlf)).size(), 2u);
    EXPECT_EQ(lobmod::read_events_csv(scratch("trail.csv", body + "\n\n")).size(), 2u);
    EXPECT_EQ(lobmod::read_events_csv(scratch("noeol.csv", body.substr(0, body.size() - 1))).size(), 2u);
    const std::pair<const char*, std::string> bad[] = {
        {"comma.csv", body + "3,M,B,0,2,0,\n"},
        {"float.csv", body + "3,L,B,999.5,1,3\n"},
        {"space.csv", body + "3, L,B,999,1,3\n"},
        {"plus.csv", body + "3,L,B,+999,1,3\n"},
        {"type.csv", body + "3,X,B,999,1,3\n"},
        {"side.csv", body + "3,L,Q,999,1,3\n"},
        {"blank.csv", "seq,type,side,price,size,order_id\n1,L,B,999,10,1\n\n2,L,S,1001,10,2\n"},
        {"header.csv", "seq,side,type,price,size,order_id\n1,B,L,999,10,1\n"},
        {"empty.csv", ""},
        {"big.csv", body + "3,L,B,9223372036854775808,1,3\n"}, // 2^63: invalid_argument, not out_of_range
    };
    for (const auto& [name, text] : bad) {
        EXPECT_THROW(lobmod::read_events_csv(scratch(name, text)), std::invalid_argument) << name;
    }
}

TEST(Replay, EventsCsvRoundTripIsByteIdentical) {
    const auto& events = testdata::events();
    const std::string path = testing::TempDir() + "lobmod_stream.csv";
    lobmod::write_events_csv(path, events);
    const auto back = lobmod::read_events_csv(path);
    ASSERT_EQ(back.size(), events.size());
    for (std::size_t i = 0; i < events.size(); ++i) {
        ASSERT_EQ(back[i].price, events[i].price);
        ASSERT_EQ(back[i].order_id, events[i].order_id);
    }
    EXPECT_EQ(slurp(path), slurp(testdata::data_dir() + "/lob_events.csv"));
    std::remove(path.c_str());
}

TEST(Rng, SeedDomainIsPinned) {
    EXPECT_NO_THROW(lobmod::Lcg64(0));
    EXPECT_NO_THROW(lobmod::Lcg64(INT64_MAX));
    EXPECT_THROW(lobmod::Lcg64(-1), std::invalid_argument);
    EXPECT_THROW(lobmod::Lcg64(INT64_MIN), std::invalid_argument);
}

} // namespace
