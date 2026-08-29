package com.quant.lobmod;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;
import java.util.TreeMap;

/**
 * Limit order book with price-time (FIFO) priority on a discrete tick grid.
 *
 * <p>Prices live on a grid of integer multiples of {@code tickSize};
 * internally a price is stored as an integer number of ticks. Each price
 * level holds a FIFO queue of resting orders {@code (orderId, size)}; earlier
 * arrivals at a level fill first (time priority), better prices fill first
 * (price priority).</p>
 *
 * <p>Three event types:</p>
 * <ul>
 *   <li><b>limit add</b>: rests at its price; if it crosses the opposite best
 *       it first executes as a marketable order against all crossing levels,
 *       and only the unfilled remainder (if any) rests;</li>
 *   <li><b>cancel by id</b>: removes an order entirely; unknown id is an
 *       error;</li>
 *   <li><b>market order</b>: walks the opposite side level by level, filling
 *       FIFO within each level; if the book runs out, the unfilled remainder
 *       is <em>reported</em>, never an error.</li>
 * </ul>
 *
 * <p>Derived quantities (mid, spread, microprice) return NaN when a side is
 * empty (guarded, never a crash). Executions always happen at the resting
 * order's price.</p>
 */
public final class LimitOrderBook {

    /** Buy side marker. */
    public static final char BUY = 'B';
    /** Sell side marker. */
    public static final char SELL = 'S';

    /**
     * Result of a limit or market order submission.
     *
     * @param fills executions in order (empty if none)
     * @param remainder for a limit order the size that rested; for a market
     *     order the unfilled size (reported, not an error)
     */
    public record ExecResult(List<Fill> fills, int remainder) {
    }

    /** Mutable resting order (id, remaining size). */
    private static final class Order {
        final int id;
        int size;

        Order(int id, int size) {
            this.id = id;
            this.size = size;
        }
    }

    private final double tickSize;
    /** price ticks -> FIFO queue; natural (ascending) key order for both sides. */
    private final TreeMap<Integer, ArrayDeque<Order>> bids = new TreeMap<>();
    private final TreeMap<Integer, ArrayDeque<Order>> asks = new TreeMap<>();
    /** live order id -> (side, price ticks). */
    private final Map<Integer, long[]> orders = new HashMap<>();
    /** executed trades in execution order. */
    private final List<Fill> trades = new ArrayList<>();

    /**
     * Creates an empty book.
     *
     * @param tickSize positive finite tick; every incoming price must be an
     *     integer multiple of it (validated to 1e-9 relative precision)
     * @throws IllegalArgumentException on invalid tick size
     */
    public LimitOrderBook(double tickSize) {
        if (!(Double.isFinite(tickSize) && tickSize > 0.0)) {
            throw new IllegalArgumentException(
                    "tick_size must be positive and finite, got " + tickSize);
        }
        this.tickSize = tickSize;
    }

    /** Creates an empty book with {@code tickSize = 1}. */
    public LimitOrderBook() {
        this(1.0);
    }

    /** The book's tick size. */
    public double tickSize() {
        return tickSize;
    }

    /** All executed fills so far, in execution order (unmodifiable view). */
    public List<Fill> trades() {
        return List.copyOf(trades);
    }

    // ------------------------------------------------------------------ utils

    private int toTicks(double price) {
        if (!(Double.isFinite(price) && price > 0.0)) {
            throw new IllegalArgumentException("price must be positive and finite, got " + price);
        }
        double ratio = price / tickSize;
        long ticks = Math.round(ratio);
        if (Math.abs(ratio - ticks) > 1e-9 * Math.max(1.0, Math.abs(ratio))) {
            throw new IllegalArgumentException(
                    "price " + price + " is not a multiple of tick_size " + tickSize);
        }
        return (int) ticks;
    }

    private static char checkSide(char side) {
        if (side != BUY && side != SELL) {
            throw new IllegalArgumentException("side must be 'B' or 'S', got '" + side + "'");
        }
        return side;
    }

    private static int checkSize(int size) {
        if (size <= 0) {
            throw new IllegalArgumentException("size must be a positive integer, got " + size);
        }
        return size;
    }

    // ------------------------------------------------------------- inspection

    /** Highest bid price in ticks, or empty if the bid side has no orders. */
    public OptionalInt bestBid() {
        return bids.isEmpty() ? OptionalInt.empty() : OptionalInt.of(bids.lastKey());
    }

    /** Lowest ask price in ticks, or empty if the ask side has no orders. */
    public OptionalInt bestAsk() {
        return asks.isEmpty() ? OptionalInt.empty() : OptionalInt.of(asks.firstKey());
    }

    /** Total resting size at the best bid level (0 if the side is empty). */
    public int bestBidSize() {
        return bids.isEmpty() ? 0 : levelSize(bids.lastEntry().getValue());
    }

    /** Total resting size at the best ask level (0 if the side is empty). */
    public int bestAskSize() {
        return asks.isEmpty() ? 0 : levelSize(asks.firstEntry().getValue());
    }

    private static int levelSize(ArrayDeque<Order> queue) {
        int total = 0;
        for (Order o : queue) {
            total += o.size;
        }
        return total;
    }

    /** Mid price in ticks, {@code (bid + ask) / 2}; NaN if either side is empty. */
    public double mid() {
        if (bids.isEmpty() || asks.isEmpty()) {
            return Double.NaN;
        }
        return 0.5 * (bids.lastKey() + asks.firstKey());
    }

    /** Bid-ask spread in ticks; NaN if either side is empty. */
    public double spread() {
        if (bids.isEmpty() || asks.isEmpty()) {
            return Double.NaN;
        }
        return asks.firstKey() - (double) bids.lastKey();
    }

    /**
     * Size-weighted mid {@code (q_a * P_b + q_b * P_a) / (q_a + q_b)} with
     * best-level total sizes {@code q_b, q_a} — it leans toward the side with
     * less depth, the direction the next mid move is more likely to take.
     * NaN when either side is empty.
     */
    public double microprice() {
        if (bids.isEmpty() || asks.isEmpty()) {
            return Double.NaN;
        }
        double qb = bestBidSize();
        double qa = bestAskSize();
        return (qa * bids.lastKey() + qb * asks.firstKey()) / (qa + qb);
    }

    /**
     * Total resting size over the top {@code k} price levels on {@code side};
     * with fewer than {@code k} levels, sums what exists.
     *
     * @throws IllegalArgumentException on a bad side or non-positive {@code k}
     */
    public int depth(char side, int k) {
        checkSide(side);
        if (k <= 0) {
            throw new IllegalArgumentException("k must be a positive integer, got " + k);
        }
        TreeMap<Integer, ArrayDeque<Order>> levels = side == BUY ? bids : asks;
        Iterator<ArrayDeque<Order>> it = (side == BUY
                ? levels.descendingMap().values()
                : levels.values()).iterator();
        int total = 0;
        for (int i = 0; i < k && it.hasNext(); i++) {
            total += levelSize(it.next());
        }
        return total;
    }

    /**
     * Sorted (ascending) ids of live resting orders on {@code side} — the
     * pinned ordering used by the zero-intelligence generator's cancels.
     */
    public int[] orderIds(char side) {
        checkSide(side);
        List<Integer> ids = new ArrayList<>();
        for (Map.Entry<Integer, long[]> e : orders.entrySet()) {
            if ((char) e.getValue()[0] == side) {
                ids.add(e.getKey());
            }
        }
        int[] out = ids.stream().mapToInt(Integer::intValue).sorted().toArray();
        return out;
    }

    /** True if {@code orderId} is currently resting in the book. */
    public boolean hasOrder(int orderId) {
        return orders.containsKey(orderId);
    }

    /** Total resting size on {@code side} across all levels. */
    public int totalSize(char side) {
        checkSide(side);
        TreeMap<Integer, ArrayDeque<Order>> levels = side == BUY ? bids : asks;
        int total = 0;
        for (ArrayDeque<Order> q : levels.values()) {
            total += levelSize(q);
        }
        return total;
    }

    /** Number of live resting orders on {@code side}. */
    public int nOrders(char side) {
        checkSide(side);
        int count = 0;
        for (long[] v : orders.values()) {
            if ((char) v[0] == side) {
                count++;
            }
        }
        return count;
    }

    // -------------------------------------------------------------- execution

    /**
     * Executes up to {@code size} against the side opposite to {@code side};
     * {@code limitTicks < 0} means no price bound (market order). Consumes
     * levels best-first and orders FIFO within a level.
     */
    private ExecResult match(char side, int size, int limitTicks) {
        List<Fill> fills = new ArrayList<>();
        int remaining = size;
        TreeMap<Integer, ArrayDeque<Order>> opp = side == BUY ? asks : bids;
        while (remaining > 0 && !opp.isEmpty()) {
            int best = side == BUY ? opp.firstKey() : opp.lastKey();
            if (limitTicks >= 0) {
                if (side == BUY && best > limitTicks) {
                    break;
                }
                if (side == SELL && best < limitTicks) {
                    break;
                }
            }
            ArrayDeque<Order> queue = opp.get(best);
            while (remaining > 0 && !queue.isEmpty()) {
                Order order = queue.peekFirst();
                int take = Math.min(remaining, order.size);
                order.size -= take;
                remaining -= take;
                fills.add(new Fill(order.id, best, take));
                if (order.size == 0) {
                    queue.pollFirst();
                    orders.remove(order.id);
                }
            }
            if (queue.isEmpty()) {
                opp.remove(best);
            }
        }
        trades.addAll(fills);
        return new ExecResult(fills, remaining);
    }

    /**
     * Submits a limit order. A crossed limit (buy priced at/above best ask,
     * or sell at/below best bid) converts to a marketable execution against
     * every crossing level; the unfilled remainder rests at {@code price}
     * (appended to that level's queue tail). It never trades through its own
     * limit price.
     *
     * @param side {@code 'B'} or {@code 'S'}
     * @param price limit price (multiple of the tick size)
     * @param size positive integer quantity
     * @param orderId unique id for the resting part
     * @return fills plus the rested size (0 if fully executed)
     * @throws IllegalArgumentException on bad side/price/size or a duplicate
     *     live {@code orderId}
     */
    public ExecResult limit(char side, double price, int size, int orderId) {
        checkSide(side);
        checkSize(size);
        int ticks = toTicks(price);
        if (orders.containsKey(orderId)) {
            throw new IllegalArgumentException(
                    "order_id " + orderId + " is already live in the book");
        }
        ExecResult res = match(side, size, ticks);
        int remaining = res.remainder();
        if (remaining > 0) {
            TreeMap<Integer, ArrayDeque<Order>> levels = side == BUY ? bids : asks;
            levels.computeIfAbsent(ticks, t -> new ArrayDeque<>()).addLast(new Order(orderId, remaining));
            orders.put(orderId, new long[] {side, ticks});
        }
        return res;
    }

    /**
     * Cancels a resting order by id and returns the size removed.
     *
     * @throws IllegalArgumentException if the id is not live (never resting,
     *     already cancelled, or fully filled)
     */
    public int cancel(int orderId) {
        long[] ref = orders.remove(orderId);
        if (ref == null) {
            throw new IllegalArgumentException("cancel of unknown order_id " + orderId);
        }
        char side = (char) ref[0];
        int ticks = (int) ref[1];
        TreeMap<Integer, ArrayDeque<Order>> levels = side == BUY ? bids : asks;
        ArrayDeque<Order> queue = levels.get(ticks);
        int removed = 0;
        for (Iterator<Order> it = queue.iterator(); it.hasNext();) {
            Order o = it.next();
            if (o.id == orderId) {
                removed = o.size;
                it.remove();
                break;
            }
        }
        if (queue.isEmpty()) {
            levels.remove(ticks);
        }
        return removed;
    }

    /**
     * Submits a market order that walks the opposite side of the book. A
     * market order larger than the total opposite depth fills what is
     * available; the unfilled remainder is returned (reported), not an error.
     *
     * @return fills plus the unfilled size
     * @throws IllegalArgumentException on a bad side or non-positive size
     */
    public ExecResult market(char side, int size) {
        checkSide(side);
        checkSize(size);
        return match(side, size, -1);
    }
}
