package com.quant.lobmod;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.OptionalLong;
import java.util.Random;
import org.junit.Test;

/**
 * The book against a deliberately naive model, the same idea as
 * python/tests/test_book_properties.py. The model keeps one flat list of
 * resting orders and finds each match by sorting it, so it shares no data
 * structures with the book.
 */
public class BookPropertyTest {

    private static final class Resting {
        final long arrival;
        final char side;
        final long price;
        final long id;
        long size;

        Resting(long arrival, char side, long price, long id, long size) {
            this.arrival = arrival;
            this.side = side;
            this.price = price;
            this.id = id;
            this.size = size;
        }
    }

    private static final class Naive {
        final List<Resting> orders = new ArrayList<>();
        long clock;

        LimitOrderBook.ExecResult execute(char side, long size, Long limit) {
            List<Resting> opp = new ArrayList<>();
            for (Resting o : orders) {
                if (o.side != side) {
                    opp.add(o);
                }
            }
            opp.sort(Comparator
                    .comparingLong((Resting o) -> side == 'B' ? o.price : -o.price)
                    .thenComparingLong(o -> o.arrival));
            List<Fill> fills = new ArrayList<>();
            long left = size;
            for (Resting o : opp) {
                if (left == 0) {
                    break;
                }
                if (limit != null && (side == 'B' ? o.price > limit : o.price < limit)) {
                    break;
                }
                long take = Math.min(left, o.size);
                o.size -= take;
                left -= take;
                fills.add(new Fill(o.id, o.price, take));
            }
            orders.removeIf(o -> o.size == 0);
            return new LimitOrderBook.ExecResult(fills, left);
        }

        LimitOrderBook.ExecResult limit(char side, long price, long size, long id) {
            LimitOrderBook.ExecResult r = execute(side, size, price);
            if (r.remainder() > 0) {
                orders.add(new Resting(++clock, side, price, id, r.remainder()));
            }
            return r;
        }

        long cancel(long id) {
            for (int i = 0; i < orders.size(); i++) {
                if (orders.get(i).id == id) {
                    return orders.remove(i).size;
                }
            }
            throw new AssertionError("model has no order " + id);
        }

        OptionalLong best(char side) {
            var prices = orders.stream().filter(o -> o.side == side).mapToLong(o -> o.price);
            return side == 'B' ? prices.max() : prices.min();
        }

        long total(char side) {
            return orders.stream().filter(o -> o.side == side).mapToLong(o -> o.size).sum();
        }
    }

    private static void assertSame(LimitOrderBook.ExecResult want, LimitOrderBook.ExecResult got) {
        assertEquals(want.fills(), got.fills());
        assertEquals(want.remainder(), got.remainder());
    }

    @Test
    public void bookMatchesNaiveModel() {
        for (int seed = 0; seed < 40; seed++) {
            Random rng = new Random(seed);
            LimitOrderBook book = new LimitOrderBook(1.0);
            Naive model = new Naive();
            long nextId = 1;
            for (int step = 0; step < 400; step++) {
                double op = rng.nextDouble();
                char side = rng.nextBoolean() ? 'B' : 'S';
                if (op < 0.55) {
                    long price = 90 + rng.nextInt(21);
                    long size = 1 + rng.nextInt(19);
                    assertSame(model.limit(side, price, size, nextId),
                            book.limitTicks(side, price, size, nextId));
                    nextId++;
                } else if (op < 0.8 && !model.orders.isEmpty()) {
                    long id = model.orders.get(rng.nextInt(model.orders.size())).id;
                    assertEquals(model.cancel(id), book.cancel(id));
                } else {
                    long size = 1 + rng.nextInt(39);
                    assertSame(model.execute(side, size, null), book.market(side, size));
                }
                assertEquals(model.best('B'), book.bestBid());
                assertEquals(model.best('S'), book.bestAsk());
                assertEquals(model.total('B'), book.totalSize('B'));
                assertEquals(model.total('S'), book.totalSize('S'));
                if (book.bestBid().isPresent() && book.bestAsk().isPresent()) {
                    assertTrue(book.bestBid().getAsLong() < book.bestAsk().getAsLong());
                }
            }
        }
    }
}
