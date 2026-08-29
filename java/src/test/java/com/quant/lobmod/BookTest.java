package com.quant.lobmod;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

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
        long resting = (long) book.totalSize('B') + book.totalSize('S');
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
        assertEquals(97, book.bestBid().getAsInt());
        assertEquals(1, book.bestBidSize());
    }

    @Test
    public void marketBiggerThanBookReportsUnfilled() {
        LimitOrderBook book = new LimitOrderBook();
        book.limit('S', 101, 4, 1);
        book.limit('S', 103, 2, 2);
        LimitOrderBook.ExecResult r = book.market('B', 10);
        assertEquals(4, r.remainder()); // reported, not thrown
        assertEquals(6, r.fills().stream().mapToInt(Fill::size).sum());
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
        assertEquals(102, book.bestBid().getAsInt()); // remainder rests at its own price
        assertEquals(105, book.bestAsk().getAsInt()); // never trades through 102
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
        assertEquals(99, book.bestBid().getAsInt());
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
        assertEquals(199, book.bestBid().getAsInt());
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
            int k = r.randint(7);
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
}
