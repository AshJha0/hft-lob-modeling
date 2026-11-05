package com.quant.lobmod;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.Test;

/**
 * Limit order book unit tests: matching semantics, invariants over the full
 * bundled replay, FIFO / price priority, edge cases and validation errors,
 * plus the pinned RNG and zero-intelligence generator determinism.
 */
public class BookTest {

    // ------------------------------------------------------------ replay-wide

    @Test
    public void replayInvariantsBidBelowAskEverywhere() {
        BookStates st = TestData.states();
        for (int i = 0; i < st.size(); i++) {
            double b = st.bestBid()[i];
            double a = st.bestAsk()[i];
            if (Double.isFinite(b) && Double.isFinite(a)) {
                assertTrue("bid >= ask after event " + (i + 1), b < a);
                assertEquals(0.5 * (b + a), st.mid()[i], 0.0);
                assertEquals(a - b, st.spread()[i], 0.0);
            } else {
                assertTrue(Double.isNaN(st.mid()[i]));
                assertTrue(Double.isNaN(st.spread()[i]));
            }
        }
    }

    @Test
    public void replaySizeConservation() {
        // every unit of limit size is either: executed on arrival (taker),
        // later hit as a maker, cancelled, or still resting at the end.
        LimitOrderBook book = new LimitOrderBook();
        long limitAdded = 0;
        long takerViaLimit = 0;
        long cancelled = 0;
        for (Event e : TestData.events()) {
            switch (e.etype()) {
                case 'L' -> {
                    limitAdded += e.size();
                    LimitOrderBook.ExecResult r =
                            book.limit(e.side(), e.price(), e.size(), e.orderId());
                    takerViaLimit += e.size() - r.remainder();
                }
                case 'C' -> cancelled += book.cancel(e.orderId());
                case 'M' -> book.market(e.side(), e.size());
                default -> throw new AssertionError();
            }
        }
        long makerFilled = book.trades().stream().mapToLong(Fill::size).sum();
        long resting = book.totalSize('B') + book.totalSize('S');
        assertEquals(limitAdded, takerViaLimit + makerFilled + cancelled + resting);
    }

    // -------------------------------------------------------------- semantics

    @Test
    public void fifoWithinLevel() {
        LimitOrderBook book = new LimitOrderBook();
        book.limit('S', 101, 5, 1); // first in at 101
        book.limit('S', 101, 5, 2); // second in at 101
        LimitOrderBook.ExecResult r = book.market('B', 7);
        assertEquals(2, r.fills().size());
        assertEquals(new Fill(1, 101, 5), r.fills().get(0)); // FIFO: id 1 fully first
        assertEquals(new Fill(2, 101, 2), r.fills().get(1));
        assertEquals(3, book.bestAskSize());
    }

    @Test
    public void pricePriorityBeforeTime() {
        LimitOrderBook book = new LimitOrderBook();
        book.limit('S', 102, 5, 1); // earlier but worse-priced
        book.limit('S', 101, 5, 2); // later but better-priced
        LimitOrderBook.ExecResult r = book.market('B', 6);
        assertEquals(new Fill(2, 101, 5), r.fills().get(0));
        assertEquals(new Fill(1, 102, 1), r.fills().get(1));
    }

    @Test
    public void marketWalksMultipleLevels() {
        LimitOrderBook book = new LimitOrderBook();
        book.limit('B', 99, 3, 1);
        book.limit('B', 98, 3, 2);
        book.limit('B', 97, 3, 3);
        LimitOrderBook.ExecResult r = book.market('S', 8);
        assertEquals(0, r.remainder());
        assertEquals(List.of(new Fill(1, 99, 3), new Fill(2, 98, 3), new Fill(3, 97, 2)),
                r.fills());
        assertEquals(97, book.bestBid().getAsLong());
        assertEquals(1, book.bestBidSize());
    }

    @Test
    public void marketBiggerThanBookReportsUnfilled() {
        LimitOrderBook book = new LimitOrderBook();
        book.limit('S', 101, 4, 1);
        book.limit('S', 103, 2, 2);
        LimitOrderBook.ExecResult r = book.market('B', 10);
        assertEquals(4, r.remainder()); // reported, not thrown
        assertEquals(6, r.fills().stream().mapToLong(Fill::size).sum());
        assertTrue(book.bestAsk().isEmpty());
        assertTrue(Double.isNaN(book.mid()));
    }

    @Test
    public void crossedLimitExecutesThenRests() {
        LimitOrderBook book = new LimitOrderBook();
        book.limit('S', 101, 3, 1);
        book.limit('S', 102, 3, 2);
        book.limit('S', 105, 3, 3);
        // marketable buy limit at 102: fills 101 fully, 102 fully, rests the rest
        LimitOrderBook.ExecResult r = book.limit('B', 102, 10, 4);
        assertEquals(4, r.remainder());
        assertEquals(List.of(new Fill(1, 101, 3), new Fill(2, 102, 3)), r.fills());
        assertEquals(102, book.bestBid().getAsLong()); // remainder rests at its own price
        assertEquals(105, book.bestAsk().getAsLong()); // never trades through 102
        assertTrue(book.hasOrder(4));
    }

    @Test
    public void executionsHappenAtRestingPrice() {
        LimitOrderBook book = new LimitOrderBook();
        book.limit('B', 99, 5, 1);
        LimitOrderBook.ExecResult r = book.limit('S', 95, 2, 2); // aggressive sell
        assertEquals(99, r.fills().get(0).price()); // maker's price, not 95
        assertEquals(0, r.remainder());
    }

    @Test
    public void cancelRemovesAndUnknownIdErrors() {
        LimitOrderBook book = new LimitOrderBook();
        book.limit('B', 99, 5, 7);
        assertEquals(5, book.cancel(7));
        assertFalse(book.hasOrder(7));
        assertTrue(book.bestBid().isEmpty());
        assertThrows(IllegalArgumentException.class, () -> book.cancel(7)); // already cancelled
        assertThrows(IllegalArgumentException.class, () -> book.cancel(12345)); // never existed
        book.limit('S', 101, 1, 8);
        book.market('B', 1);
        assertThrows(IllegalArgumentException.class, () -> book.cancel(8)); // fully filled
    }

    @Test
    public void emptySideGuards() {
        LimitOrderBook book = new LimitOrderBook();
        assertTrue(Double.isNaN(book.mid()));
        assertTrue(Double.isNaN(book.spread()));
        assertTrue(Double.isNaN(book.microprice()));
        assertEquals(0, book.bestBidSize());
        assertEquals(0, book.depth('B', 3));
        book.limit('B', 99, 5, 1); // one-sided book still guards
        assertTrue(Double.isNaN(book.mid()));
        assertEquals(99, book.bestBid().getAsLong());
    }

    @Test
    public void micropriceLeansTowardThinSide() {
        LimitOrderBook book = new LimitOrderBook();
        book.limit('B', 99, 9, 1);  // qb = 9
        book.limit('S', 101, 1, 2); // qa = 1
        // micro = (qa*Pb + qb*Pa)/(qa+qb) = (99 + 891)/10 = 99.0 + ... = 100.8
        assertEquals((1.0 * 99 + 9.0 * 101) / 10.0, book.microprice(), 1e-12);
        assertTrue(book.microprice() > book.mid()); // ask side thin -> price drifts up
    }

    @Test
    public void tickValidation() {
        assertThrows(IllegalArgumentException.class, () -> new LimitOrderBook(0.0));
        assertThrows(IllegalArgumentException.class, () -> new LimitOrderBook(-1.0));
        assertThrows(IllegalArgumentException.class,
                () -> new LimitOrderBook(Double.POSITIVE_INFINITY));
        LimitOrderBook book = new LimitOrderBook(0.5);
        book.limit('B', 99.5, 1, 1); // 199 ticks, valid
        assertEquals(199, book.bestBid().getAsLong());
        assertThrows(IllegalArgumentException.class, () -> book.limit('B', 99.7, 1, 2));
        assertThrows(IllegalArgumentException.class, () -> book.limit('B', -1.0, 1, 3));
        assertThrows(IllegalArgumentException.class, () -> book.limit('B', Double.NaN, 1, 4));
    }

    @Test
    public void argumentValidation() {
        LimitOrderBook book = new LimitOrderBook();
        book.limit('B', 99, 5, 1);
        assertThrows(IllegalArgumentException.class, () -> book.limit('X', 99, 5, 2));
        assertThrows(IllegalArgumentException.class, () -> book.limit('B', 99, 0, 2));
        assertThrows(IllegalArgumentException.class, () -> book.limit('B', 99, -3, 2));
        assertThrows(IllegalArgumentException.class, () -> book.limit('B', 98, 5, 1)); // dup id
        assertThrows(IllegalArgumentException.class, () -> book.market('Q', 1));
        assertThrows(IllegalArgumentException.class, () -> book.market('B', 0));
        assertThrows(IllegalArgumentException.class, () -> book.depth('B', 0));
        assertThrows(IllegalArgumentException.class, () -> book.depth('Z', 1));
    }

    // ------------------------------------------------------------------- RNG

    @Test
    public void lcg64PinnedSequence() {
        Lcg64 a = new Lcg64(42);
        Lcg64 b = new Lcg64(42);
        for (int i = 0; i < 100; i++) {
            double u = a.uniform();
            assertEquals(u, b.uniform(), 0.0); // bit-identical per seed
            assertTrue(u >= 0.0 && u < 1.0);
        }
        // first state from seed 0 is the increment itself
        assertEquals(1442695040888963407L, new Lcg64(0).nextU64());
        assertThrows(IllegalArgumentException.class, () -> new Lcg64(-1));
        Lcg64 r = new Lcg64(1);
        assertThrows(IllegalArgumentException.class, () -> r.exponential(0.0));
        assertThrows(IllegalArgumentException.class, () -> r.geometric(0.0));
        assertThrows(IllegalArgumentException.class, () -> r.geometric(1.0));
        assertThrows(IllegalArgumentException.class, () -> r.randint(0));
        for (int i = 0; i < 200; i++) {
            assertTrue(r.exponential(2.0) >= 0.0);
            long k = r.randint(7);
            assertTrue(k >= 0 && k < 7);
            assertTrue(r.geometric(0.35) >= 0);
        }
    }

    // -------------------------------------------------------------- generator

    @Test
    public void ziGeneratorReproducesBundledStream() {
        List<Event> gen = ZiGenerator.generateEvents(5000, 20240613L);
        assertEquals(TestData.events(), gen); // exact event-for-event match
    }

    @Test
    public void ziGeneratorValidation() {
        assertThrows(IllegalArgumentException.class, () -> ZiGenerator.generateEvents(11, 1));
        assertThrows(IllegalArgumentException.class, () -> ZiGenerator.generateEvents(100, -5));
    }

    @Test
    public void replayValidation() {
        assertThrows(IllegalArgumentException.class, () -> Replay.replay(List.of()));
        LimitOrderBook book = new LimitOrderBook();
        assertThrows(IllegalArgumentException.class,
                () -> Replay.applyEvent(book, new Event(1, 'Z', 'B', 100, 1, 1)));
    }

    // ------------------------------------------------------- review additions

    private static Path scratch(String name, String text) {
        try {
            Path p = Files.createTempDirectory("lobmod_b").resolve(name);
            Files.writeString(p, text);
            return p;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Test
    public void priceRoundingToZeroTicksRejected() {
        // PT-1
        LimitOrderBook b = new LimitOrderBook(1.0);
        assertThrows(IllegalArgumentException.class, () -> b.limit('B', 1e-12, 5, 1));
        LimitOrderBook b2 = new LimitOrderBook(0.01);
        assertThrows(IllegalArgumentException.class, () -> b2.limit('S', 4e-12, 1, 2));
        for (LimitOrderBook book : List.of(b, b2)) {
            assertTrue(book.bestBid().isEmpty() && book.bestAsk().isEmpty());
            assertEquals(0, book.nOrders('B') + book.nOrders('S'));
            assertTrue(book.trades().isEmpty());
        }
        assertThrows(IllegalArgumentException.class, () -> b.limitTicks('B', 0, 5, 1));
        assertThrows(IllegalArgumentException.class, () -> b.limitTicks('B', -5, 5, 1));
    }

    @Test
    public void largeTickCounts() {
        // PT-2 (CRIT-1): fine-tick products need 64-bit tick counts; overflow is an error
        LimitOrderBook b = new LimitOrderBook(1e-8);
        LimitOrderBook.ExecResult r1 = b.limit('S', 60000.0, 1, 1);
        assertTrue(r1.fills().isEmpty());
        assertEquals(1, r1.remainder());
        assertEquals(6_000_000_000_000L, b.bestAsk().getAsLong());
        LimitOrderBook.ExecResult r2 = b.limit('B', 60000.0, 1, 2);
        assertEquals(List.of(new Fill(1, 6_000_000_000_000L, 1)), r2.fills());
        assertEquals(0, r2.remainder());
        assertTrue(b.bestAsk().isEmpty() && b.bestBid().isEmpty());
        // 3e9 ticks used to narrow to a negative int and sweep the book
        b.limit('S', 30.0, 7, 5); // 3e9 ticks resting
        b.limit('S', 40.0, 7, 6); // 4e9 ticks resting
        LimitOrderBook.ExecResult r3 = b.limit('B', 30.0, 20, 7);
        assertEquals(List.of(new Fill(5, 3_000_000_000L, 7)), r3.fills()); // stops at its price
        assertEquals(13, r3.remainder());
        assertEquals(4_000_000_000L, b.bestAsk().getAsLong());
        assertEquals(3_000_000_000L, b.bestBid().getAsLong());
        assertThrows(IllegalArgumentException.class, () -> b.limit('B', 1e300, 1, 8)); // no sweep
        assertEquals(7, b.bestAskSize());
        assertThrows(IllegalArgumentException.class, () -> b.limit('B', 1e19, 1, 8));
        assertThrows(IllegalArgumentException.class,
                () -> b.limitTicks('B', LimitOrderBook.MAX_TICKS + 1, 1, 8));
        LimitOrderBook c = new LimitOrderBook(1.0);
        c.limitTicks('B', LimitOrderBook.MAX_TICKS, 1, 3);
        assertEquals(LimitOrderBook.MAX_TICKS, c.bestBid().getAsLong());
        // 64-bit sizes and ids
        c.limitTicks('B', 1, 5_000_000_000L, 9_000_000_000L);
        assertEquals(5_000_000_001L, c.totalSize('B'));
        assertEquals(5_000_000_000L, c.depth('B', 2) - c.depth('B', 1));
        assertEquals(5_000_000_000L, c.cancel(9_000_000_000L));
    }

    @Test
    public void tickRoundingRuleIsHalfUpWithinTolerance() {
        LimitOrderBook b = new LimitOrderBook(0.5);
        b.limit('B', 99.5, 1, 1);
        assertEquals(199, b.bestBid().getAsLong());
        b.limit('B', 100.0000000001, 1, 2); // residual 2e-10 <= 1e-9 * 200
        assertEquals(200, b.bestBid().getAsLong());
        assertThrows(IllegalArgumentException.class, () -> b.limit('B', 100.004, 1, 3));
        assertThrows(IllegalArgumentException.class, () -> b.limit('B', 0.75, 1, 3)); // 1.5 ticks
    }

    @Test
    public void limitTicksMatchesLimit() {
        LimitOrderBook a = new LimitOrderBook(0.25);
        LimitOrderBook c = new LimitOrderBook(0.25);
        a.limit('S', 25.5, 4, 1);
        c.limitTicks('S', 102, 4, 1);
        assertEquals(102, a.bestAsk().getAsLong());
        assertEquals(102, c.bestAsk().getAsLong());
        LimitOrderBook.ExecResult ra = a.limit('B', 26.0, 6, 2);
        LimitOrderBook.ExecResult rc = c.limitTicks('B', 104, 6, 2);
        assertEquals(ra, rc);
        assertEquals(2, ra.remainder());
    }

    @Test
    public void replayPricesAreTickCounts() {
        // PT-4: Event.price is in ticks; replay never rescales it
        List<Event> head = TestData.events().subList(0, 12);
        BookStates st = Replay.replay(head, 3);
        assertEquals(999.0, st.bestBid()[11], 0.0);
        assertEquals(1001.0, st.bestAsk()[11], 0.0);
        LimitOrderBook manual = new LimitOrderBook(0.5);
        for (Event e : head) {
            manual.limit(e.side(), e.price() * 0.5, e.size(), e.orderId());
        }
        assertEquals(999, manual.bestBid().getAsLong());
        assertThrows(IllegalArgumentException.class, () -> Replay.replay(head, 0));
    }

    @Test
    public void depthMonotoneInKOverReplay() {
        // PT-11
        LimitOrderBook book = new LimitOrderBook();
        for (Event e : TestData.events().subList(0, 500)) {
            Replay.applyEvent(book, e);
            assertEquals(book.bestBidSize(), book.depth('B', 1));
            assertEquals(book.bestAskSize(), book.depth('S', 1));
            for (char side : new char[] {'B', 'S'}) {
                for (int k = 1; k <= 5; k++) {
                    assertTrue(book.depth(side, k) <= book.depth(side, k + 1));
                }
            }
        }
    }

    @Test
    public void marketOnEmptyBookAndReuseOfCancelledId() {
        // PT-12
        LimitOrderBook b = new LimitOrderBook();
        LimitOrderBook.ExecResult r = b.market('B', 5);
        assertTrue(r.fills().isEmpty());
        assertEquals(5, r.remainder());
        assertEquals(3, b.market('S', 3).remainder());
        b.limit('B', 100, 2, 7);
        assertEquals(2, b.cancel(7));
        LimitOrderBook.ExecResult again = b.limit('S', 101, 4, 7); // id free again
        assertTrue(again.fills().isEmpty());
        assertEquals(4, again.remainder());
        assertTrue(b.hasOrder(7));
    }

    @Test
    public void crossedSellLimitStopsAtItsPrice() {
        // PT-13
        LimitOrderBook b = new LimitOrderBook();
        b.limit('B', 100, 4, 1);
        b.limit('B', 99, 4, 2);
        LimitOrderBook.ExecResult r = b.limit('S', 100, 10, 3);
        assertEquals(List.of(new Fill(1, 100, 4)), r.fills());
        assertEquals(6, r.remainder());
        assertEquals(100, b.bestAsk().getAsLong());
        assertEquals(99, b.bestBid().getAsLong());
        assertEquals(4, b.bestBidSize()); // the 99 bid was never touched
    }

    @Test
    public void recordTradesFlagAndClearHistory() {
        LimitOrderBook b = new LimitOrderBook(1.0, false);
        b.limit('S', 101, 5, 1);
        LimitOrderBook.ExecResult r = b.market('B', 2);
        assertEquals(1, r.fills().size());
        assertTrue(b.trades().isEmpty());
        assertFalse(b.recordTrades());
        assertEquals(3, b.bestAskSize()); // matching unaffected
        LimitOrderBook c = new LimitOrderBook();
        c.limit('S', 101, 5, 1);
        List<Fill> view = c.trades();
        c.market('B', 2);
        assertEquals(1, view.size()); // live unmodifiable view, O(1)
        assertThrows(UnsupportedOperationException.class, () -> view.add(new Fill(1, 1, 1)));
        c.clearHistory();
        assertTrue(c.trades().isEmpty());
        assertEquals(3, c.bestAskSize());
    }

    @Test
    public void seedDomainIsPinned() {
        new Lcg64(0);
        new Lcg64(Lcg64.MAX_SEED);
        assertThrows(IllegalArgumentException.class, () -> new Lcg64(-1));
        assertThrows(IllegalArgumentException.class, () -> new Lcg64(Long.MIN_VALUE));
        Lcg64 r = new Lcg64(3);
        assertTrue(r.randint(5_000_000_000L) < 5_000_000_000L); // 64-bit range
    }

    @Test
    public void eventsCsvRobustness() {
        // PT-15: the pinned CSV rules
        String body = "seq,type,side,price,size,order_id\n1,L,B,999,10,1\n2,L,S,1001,10,2\n";
        List<Event> ok = Replay.readEventsCsv(scratch("ok.csv", body));
        assertEquals(List.of(new Event(1, 'L', 'B', 999, 10, 1), new Event(2, 'L', 'S', 1001, 10, 2)), ok);
        assertEquals(ok, Replay.readEventsCsv(scratch("crlf.csv", body.replace("\n", "\r\n"))));
        assertEquals(ok, Replay.readEventsCsv(scratch("trail.csv", body + "\n\n")));
        assertEquals(ok, Replay.readEventsCsv(scratch("noeol.csv", body.strip())));
        String[][] bad = {
            {"comma.csv", body + "3,M,B,0,2,0,\n"},
            {"float.csv", body + "3,L,B,999.5,1,3\n"},
            {"space.csv", body + "3, L,B,999,1,3\n"},
            {"plus.csv", body + "3,L,B,+999,1,3\n"},
            {"type.csv", body + "3,X,B,999,1,3\n"},
            {"side.csv", body + "3,L,Q,999,1,3\n"},
            {"blank.csv", "seq,type,side,price,size,order_id\n1,L,B,999,10,1\n\n2,L,S,1001,10,2\n"},
            {"header.csv", "seq,side,type,price,size,order_id\n1,B,L,999,10,1\n"},
            {"empty.csv", ""},
            {"big.csv", body + "3,L,B,9223372036854775808,1,3\n"},
        };
        for (String[] c : bad) {
            Path p = scratch(c[0], c[1]);
            assertThrows(c[0], IllegalArgumentException.class, () -> Replay.readEventsCsv(p));
        }
        assertThrows(UncheckedIOException.class,
                () -> Replay.readEventsCsv(Path.of("/nonexistent/dir/x.csv")));
    }

    @Test
    public void eventsCsvRoundTripIsByteIdentical() throws IOException {
        List<Event> events = TestData.events();
        Path p = scratch("stream.csv", "");
        Replay.writeEventsCsv(p, events);
        assertEquals(events, Replay.readEventsCsv(p));
        assertArrayEquals(Files.readAllBytes(TestData.DATA.resolve("lob_events.csv")),
                Files.readAllBytes(p));
    }

    // ---- 64-bit edge rules pinned in API_SPEC 2.1/2.3 ----------------------

    private static LimitOrderBook quoted(long bid, long ask) {
        LimitOrderBook b = new LimitOrderBook();
        b.limitTicks('B', bid, 1, 1);
        b.limitTicks('S', ask, 1, 2);
        return b;
    }

    @Test
    public void midIsExactSumRoundedOnce() {
        assertEquals(5e18, quoted(5_000_000_000_000_000_000L, 5_000_000_000_000_000_001L).mid(), 0.0);
        long p53 = 1L << 53;
        long p62 = 1L << 62;
        assertEquals((double) (p53 + 2), quoted(p53 + 1, p53 + 2).mid(), 0.0);
        assertEquals((double) (p62 + 1024), quoted(p62, p62 + 1025).mid(), 0.0);
    }

    @Test
    public void spreadSubtractsAsLong() {
        long p53 = 1L << 53;
        assertEquals(1.0, quoted(p53 + 1, p53 + 2).spread(), 0.0);
    }

    @Test
    public void sideCapacityRejectedBeforeMutation() {
        long m = LimitOrderBook.MAX_TICKS;
        LimitOrderBook b = new LimitOrderBook();
        b.limitTicks('B', 100, m, 1);
        assertThrows(IllegalArgumentException.class, () -> b.limitTicks('B', 100, m, 2));
        assertThrows(IllegalArgumentException.class, () -> b.limitTicks('B', 99, 1, 2));
        assertEquals(m, b.totalSize('B'));
        assertEquals(1, b.nOrders('B'));
        assertEquals(m, b.bestBidSize());
        assertEquals(m, b.depth('B', 5));
        b.market('S', 1); // a fill frees capacity
        b.limitTicks('B', 99, 1, 2);
        assertEquals(m, b.totalSize('B'));
        b.cancel(1); // so does a cancel
        assertEquals(1, b.totalSize('B'));
        b.limitTicks('S', 200, m, 3);
        assertThrows(IllegalArgumentException.class, () -> b.limitTicks('S', 200, 1, 4));
        assertEquals(m, b.totalSize('S'));
    }

    @Test
    public void gridToleranceIsCappedBelowHalfATick() {
        LimitOrderBook b = new LimitOrderBook(0.01);
        assertThrows(IllegalArgumentException.class, () -> b.limit('B', 5000000.004, 1, 1));
        b.limit('B', 5000000.01, 1, 1);
        assertEquals(500000001L, b.bestBid().getAsLong());
    }

    @Test
    public void gridValidationStopsAt2Pow52() {
        LimitOrderBook b = new LimitOrderBook(1.0);
        assertThrows(IllegalArgumentException.class, () -> b.limit('B', 4503599627370496.0, 1, 1));
        b.limit('B', 4503599627370495.0, 1, 1);
        assertEquals(4503599627370495L, b.bestBid().getAsLong());
    }

    @Test
    public void infiniteRatioIsAnError() {
        LimitOrderBook b = new LimitOrderBook(1e-300);
        assertThrows(IllegalArgumentException.class, () -> b.limit('B', 1e308, 1, 1));
    }

    @Test
    public void geometricOutOfRangeIsAnError() {
        Lcg64 c = new Lcg64(1);
        assertThrows(IllegalArgumentException.class, () -> c.geometric(1e-300));
    }

    @Test
    public void rosenbrock4dIterationCountIsPinned() {
        NelderMead.Result r = NelderMead.minimize(x -> {
            double total = 0.0;
            for (int i = 0; i + 1 < x.length; i++) {
                double a = x[i + 1] - x[i] * x[i];
                double c = 1.0 - x[i];
                total += 100.0 * (a * a) + c * c;
            }
            return total;
        }, new double[] {-1.2, 1.0, 0.7, 1.3});
        assertEquals(308, r.nIter());
    }

    // ---------------------------------------- L3 operations and schema v2

    private static LimitOrderBook threeBids() {
        LimitOrderBook b = new LimitOrderBook(1.0);
        b.limit('B', 100, 10, 1);
        b.limit('B', 100, 20, 2);
        b.limit('B', 100, 30, 3);
        return b;
    }

    private static long[] makers(List<Fill> fills) {
        return fills.stream().mapToLong(Fill::makerId).toArray();
    }

    @Test
    public void reduceKeepsPriorityAndFullReduceRemoves() {
        LimitOrderBook b = threeBids();
        assertEquals(6, b.reduce(1, 4));
        assertEquals(56, b.totalSize('B'));
        assertArrayEquals(new long[] {1, 2}, makers(b.market('S', 7).fills()));
        assertEquals(0, b.reduce(2, 19));
        assertFalse(b.hasOrder(2));
        b.reduce(3, 30);
        assertFalse(b.bestBid().isPresent());
        assertEquals(0, b.totalSize('B'));
    }

    @Test
    public void executeNamedOrderOutOfQueueOrder() {
        LimitOrderBook b = threeBids();
        Fill f = b.execute(2, 5);
        assertEquals(new Fill(2, 100, 5), f);
        assertEquals(List.of(f), b.trades());
        assertEquals(55, b.totalSize('B'));
        b.execute(2, 15);
        assertFalse(b.hasOrder(2));
        assertArrayEquals(new long[] {1, 3}, makers(b.market('S', 40).fills()));
    }

    @Test
    public void replaceLosesPriorityAndMayReuseItsId() {
        LimitOrderBook b = threeBids();
        assertEquals(10, b.replace(1, 1, 100, 10).remainder());
        assertEquals(20, b.replaceTicks(2, 9, 100, 20).remainder());
        assertFalse(b.hasOrder(2));
        assertArrayEquals(new long[] {3, 1, 9}, makers(b.market('S', 60).fills()));
    }

    @Test
    public void crossingReplaceExecutes() {
        LimitOrderBook b = new LimitOrderBook(1.0);
        b.limit('S', 101, 5, 1);
        b.limit('B', 99, 8, 2);
        LimitOrderBook.ExecResult r = b.replace(2, 3, 101, 8);
        assertArrayEquals(new long[] {1}, makers(r.fills()));
        assertEquals(3, r.remainder());
        assertEquals(101, b.bestBid().getAsLong());
        assertEquals(0, b.totalSize('S'));
    }

    @Test
    public void l3ErrorsLeaveTheBookUnchanged() {
        LimitOrderBook b = threeBids();
        Class<IllegalArgumentException> bad = IllegalArgumentException.class;
        assertThrows(bad, () -> b.reduce(99, 1));
        assertThrows(bad, () -> b.reduce(1, 11));
        assertThrows(bad, () -> b.reduce(1, 0));
        assertThrows(bad, () -> b.execute(99, 1));
        assertThrows(bad, () -> b.execute(1, 11));
        assertThrows(bad, () -> b.execute(1, -1));
        assertThrows(bad, () -> b.replace(99, 5, 100, 1));
        assertThrows(bad, () -> b.replace(1, 2, 100, 1));
        assertThrows(bad, () -> b.replace(1, 5, 100.5, 1));
        assertThrows(bad, () -> b.replace(1, 5, 100, 0));
        assertThrows(bad, () -> b.replaceTicks(1, 5, 0, 1));
        assertEquals(60, b.totalSize('B'));
        assertEquals(3, b.nOrders('B'));
        assertArrayEquals(new long[] {1, 2, 3}, makers(b.market('S', 60).fills()));
    }

    @Test
    public void replaceCountsCapacityWithTheOldOrderGone() {
        LimitOrderBook b = new LimitOrderBook(1.0);
        b.limitTicks('B', 100, LimitOrderBook.MAX_TICKS, 1);
        b.replaceTicks(1, 2, 90, LimitOrderBook.MAX_TICKS);
        assertEquals(LimitOrderBook.MAX_TICKS, b.totalSize('B'));
        b.reduce(2, 1);
        b.limitTicks('B', 80, 1, 3);
        assertThrows(IllegalArgumentException.class, () -> b.replaceTicks(3, 4, 80, 2));
        assertEquals(LimitOrderBook.MAX_TICKS, b.totalSize('B'));
        assertTrue(b.hasOrder(3));
    }

    private static java.util.Map<String, long[]> expectedV2() throws IOException {
        java.util.Map<String, long[]> out = new java.util.HashMap<>();
        List<String> lines = Files.readAllLines(TestData.DATA.resolve("lob_events_v2_expected.csv"));
        for (String line : lines.subList(1, lines.size())) {
            String[] kv = line.split(",", 2);
            out.put(kv[0], java.util.Arrays.stream(kv[1].split(" "))
                    .mapToLong(Long::parseLong).toArray());
        }
        return out;
    }

    @Test
    public void v2FixtureReproducesExpectedBook() throws IOException {
        List<EventV2> events = Replay.readEventsV2Csv(TestData.DATA.resolve("lob_events_v2.csv"));
        java.util.Map<String, long[]> want = expectedV2();
        LimitOrderBook b = new LimitOrderBook(1.0);
        for (EventV2 e : events) {
            Replay.applyEventV2(b, e);
        }
        assertEquals(want.get("best_bid")[0], b.bestBid().getAsLong());
        assertEquals(want.get("best_ask")[0], b.bestAsk().getAsLong());
        assertEquals(want.get("best_bid_size")[0], b.bestBidSize());
        assertEquals(want.get("best_ask_size")[0], b.bestAskSize());
        assertEquals(want.get("total_bid")[0], b.totalSize('B'));
        assertEquals(want.get("total_ask")[0], b.totalSize('S'));
        assertArrayEquals(want.get("order_ids_bid"), b.orderIds('B'));
        assertArrayEquals(want.get("order_ids_ask"), b.orderIds('S'));
        assertArrayEquals(want.get("trades"), b.trades().stream()
                .flatMapToLong(f -> java.util.stream.LongStream.of(f.makerId(), f.price(), f.size()))
                .toArray());
        assertEquals(want.get("last_ts_ns")[0], events.get(events.size() - 1).tsNs());
        BookStates st = Replay.replayV2(events, 3);
        assertEquals(17, st.size());
        assertEquals(999.5, st.mid()[16], 0.0);
    }

    @Test
    public void v2ReaderLiftsV1AndRoundTrips() throws IOException {
        List<EventV2> lifted = Replay.readEventsV2Csv(TestData.DATA.resolve("lob_events.csv"));
        assertEquals(5000, lifted.size());
        assertEquals(0, lifted.get(0).tsNs());
        List<EventV2> events = Replay.readEventsV2Csv(TestData.DATA.resolve("lob_events_v2.csv"));
        Path path = Files.createTempFile("lobmod_v2", ".csv");
        Replay.writeEventsV2Csv(path, events);
        assertEquals(events, Replay.readEventsV2Csv(path));
        Files.delete(path);
    }

    @Test
    public void v2ReaderRejectsBadRows() throws IOException {
        Path path = Files.createTempFile("lobmod_v2_bad", ".csv");
        for (String row : new String[] {"1,-1,L,B,100,1,1,0", "1,0,Z,B,100,1,1,0",
            "1,0,L,B,100,1,1", "1,0,L,B,100,1,1,0,"}) {
            Files.writeString(path,
                    "seq,ts_ns,type,side,price,size,order_id,new_order_id\n" + row + "\n");
            assertThrows(IllegalArgumentException.class, () -> Replay.readEventsV2Csv(path));
        }
        Files.delete(path);
    }
}
