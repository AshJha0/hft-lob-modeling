package com.quant.lobmod;

import java.util.ArrayList;
import java.util.List;
import java.util.OptionalInt;

/**
 * Zero-intelligence order-flow generator (pinned, seeded).
 *
 * <p>Smith-Farmer style flow: Poisson event-type rates around the best quotes
 * with geometric placement of passive limit orders. Everything is pinned —
 * RNG ({@link Lcg64}), draw order, placement and size formulas — so that
 * every language regenerates the identical 5000-event stream.</p>
 *
 * <p>Seed phase (events 1..12): for {@code i = 0..5} submit a buy limit at
 * {@code 999 - i} size 10 then a sell limit at {@code 1001 + i} size 10
 * (ids = seq numbers). Stochastic phase: event-type rates
 * {@code [limit_buy 1, limit_sell 1, cancel_buy 0.04*nb, cancel_sell 0.04*ns,
 * market_buy 0.25, market_sell 0.25]} picked by one uniform; see the pinned
 * per-type draw rules in the API spec. Every event is applied to an internal
 * book immediately so quotes evolve exactly as a replay would.</p>
 */
public final class ZiGenerator {

    private static final double RATE_LIMIT = 1.0;
    private static final double RATE_MARKET = 0.25;
    private static final double RATE_CANCEL_PER_ORDER = 0.04;
    private static final double P_AGGRESSIVE = 0.12;
    private static final double P_GEOMETRIC = 0.35;
    private static final double P_SWEEP = 0.02;
    private static final int REF_PRICE = 1000;

    private ZiGenerator() {
    }

    /**
     * Generates a deterministic zero-intelligence event stream.
     *
     * @param nEvents number of events to emit (&ge; 12; the first 12 seed the
     *     book)
     * @param seed LCG seed
     * @return events with seq numbered {@code 1..nEvents}
     * @throws IllegalArgumentException if {@code nEvents < 12} or the seed is
     *     negative
     */
    public static List<Event> generateEvents(int nEvents, long seed) {
        if (nEvents < 12) {
            throw new IllegalArgumentException("n_events must be an integer >= 12, got " + nEvents);
        }
        Lcg64 rng = new Lcg64(seed);
        LimitOrderBook book = new LimitOrderBook(1.0);
        List<Event> events = new ArrayList<>(nEvents);

        // -- seed phase: 6 levels each side, size 10
        for (int i = 0; i < 6; i++) {
            emit(book, events, 'L', 'B', REF_PRICE - 1 - i, 10, events.size() + 1);
            emit(book, events, 'L', 'S', REF_PRICE + 1 + i, 10, events.size() + 1);
        }

        // -- stochastic phase
        while (events.size() < nEvents) {
            int seq = events.size() + 1;
            int nb = book.nOrders('B');
            int ns = book.nOrders('S');
            double[] rates = {
                RATE_LIMIT,                     // limit buy
                RATE_LIMIT,                     // limit sell
                RATE_CANCEL_PER_ORDER * nb,     // cancel buy
                RATE_CANCEL_PER_ORDER * ns,     // cancel sell
                RATE_MARKET,                    // market buy
                RATE_MARKET,                    // market sell
            };
            double total = 0.0;
            for (double r : rates) {
                total += r;
            }
            double x = rng.uniform() * total;
            int kind = 0;
            double acc = rates[0];
            while (x >= acc && kind < 5) {
                kind++;
                acc += rates[kind];
            }

            if (kind == 0 || kind == 1) { // limit order
                boolean buy = kind == 0;
                OptionalInt oppBest = buy ? book.bestAsk() : book.bestBid();
                double u2 = rng.uniform();
                int price;
                if (u2 < P_AGGRESSIVE && oppBest.isPresent()) {
                    price = oppBest.getAsInt();
                } else {
                    int k = rng.geometric(P_GEOMETRIC);
                    int anchor = oppBest.isPresent() ? oppBest.getAsInt() : REF_PRICE;
                    price = buy ? anchor - (1 + k) : anchor + (1 + k);
                }
                price = Math.max(price, 1);
                int size = 1 + rng.randint(5);
                emit(book, events, 'L', buy ? 'B' : 'S', price, size, seq);
            } else if (kind == 2 || kind == 3) { // cancel
                char side = kind == 2 ? 'B' : 'S';
                int[] ids = book.orderIds(side);
                int oid = ids[rng.randint(ids.length)];
                emit(book, events, 'C', side, 0, 0, oid);
            } else { // market order
                char side = kind == 4 ? 'B' : 'S';
                double u3 = rng.uniform();
                int size = u3 < P_SWEEP ? 40 + rng.randint(40) : 1 + rng.randint(3);
                emit(book, events, 'M', side, 0, size, 0);
            }
        }
        return events;
    }

    private static void emit(LimitOrderBook book, List<Event> events,
            char etype, char side, int price, int size, int orderId) {
        Event event = new Event(events.size() + 1, etype, side, price, size, orderId);
        Replay.applyEvent(book, event);
        events.add(event);
    }
}
