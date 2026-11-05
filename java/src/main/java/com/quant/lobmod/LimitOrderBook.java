package com.quant.lobmod;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
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
 *
 * <p><b>64-bit everywhere.</b> Ticks, sizes and ids are {@code long}; the
 * pinned tick conversion is {@code ratio = price / tickSize}, rounded half
 * up, accepted iff {@code ratio < 2^52},
 * {@code |ratio - ticks| <= min(1e-9 * max(1, ratio), 0.01)} and
 * {@code ticks >= 1}. A price that rounds to 0 ticks or
 * would overflow is an error — it can never become a negative price or an
 * unbounded market order (the price bound of a match is an explicit
 * flag, not a sentinel value). {@link #limitTicks} takes integer tick
 * counts directly (the replay path).</p>
 *
 * <p><b>Memory.</b> {@link #trades()} grows for the life of the book;
 * construct with {@code recordTrades = false} or call
 * {@link #clearHistory()} for long sessions.</p>
 */
public final class LimitOrderBook {

    /** Buy side marker. */
    public static final char BUY = 'B';
    /** Sell side marker. */
    public static final char SELL = 'S';
    /** Largest accepted tick count / size (fits a signed 64-bit integer). */
    public static final long MAX_TICKS = 9_000_000_000_000_000_000L;
    /** At or above this ratio doubles have no fractional bits (2^52). */
    private static final double MAX_GRID_RATIO = 4503599627370496.0;
    /** Absolute cap on the grid residual, in ticks. */
    private static final double GRID_TOL_CAP = 0.01;

    /**
     * Result of a limit or market order submission.
     *
     * @param fills executions in order (empty if none)
     * @param remainder for a limit order the size that rested; for a market
     *     order the unfilled size (reported, not an error)
     */
    public record ExecResult(List<Fill> fills, long remainder) {
    }

    /** Mutable resting order (id, remaining size). */
    private static final class Order {
        final long id;
        long size;

        Order(long id, long size) {
            this.id = id;
            this.size = size;
        }
    }

    /** Where a live order rests. */
    private record Location(char side, long ticks) {
    }

    private final double tickSize;
    private final boolean recordTrades;
    /** price ticks -> FIFO queue; natural (ascending) key order for both sides. */
    private final TreeMap<Long, ArrayDeque<Order>> bids = new TreeMap<>();
    private final TreeMap<Long, ArrayDeque<Order>> asks = new TreeMap<>();
    /** live order id -> (side, price ticks). */
    private final Map<Long, Location> orders = new HashMap<>();
    /** Running total resting size per side (never above MAX_TICKS). */
    private long bidTotal;
    private long askTotal;
    /** executed trades in execution order. */
    private final List<Fill> trades = new ArrayList<>();
    private final List<Fill> tradesView = Collections.unmodifiableList(trades);

    /**
     * Creates an empty book.
     *
     * @param tickSize positive finite tick; every incoming price must be an
     *     integer multiple of it (see the pinned tick conversion)
     * @param recordTrades keep every {@link Fill} in {@link #trades()};
     *     {@code false} disables recording (matching is unaffected)
     * @throws IllegalArgumentException on invalid tick size
     */
    public LimitOrderBook(double tickSize, boolean recordTrades) {
        if (!(Double.isFinite(tickSize) && tickSize > 0.0)) {
            throw new IllegalArgumentException(
                    "tick_size must be positive and finite, got " + tickSize);
        }
        this.tickSize = tickSize;
        this.recordTrades = recordTrades;
    }

    /** Creates an empty book that records fills. */
    public LimitOrderBook(double tickSize) {
        this(tickSize, true);
    }

    /** Creates an empty book with {@code tickSize = 1}. */
    public LimitOrderBook() {
        this(1.0, true);
    }

    /** The book's tick size. */
    public double tickSize() {
        return tickSize;
    }

    /** Whether fills are being recorded. */
    public boolean recordTrades() {
        return recordTrades;
    }

    /**
     * All executed fills so far, in execution order — an unmodifiable
     * <em>view</em> (O(1); it reflects later fills).
     */
    public List<Fill> trades() {
        return tradesView;
    }

    /** Drops the accumulated {@link #trades()} (book unchanged). */
    public void clearHistory() {
        trades.clear();
    }

    // ------------------------------------------------------------------ utils

    /**
     * Pinned tick conversion: ratio below {@code 2^52}, round half up, grid
     * residual within {@code min(1e-9 * max(1, ratio), 0.01)}, result at
     * least 1.
     */
    private long toTicks(double price) {
        if (!(Double.isFinite(price) && price > 0.0)) {
            throw new IllegalArgumentException("price must be positive and finite, got " + price);
        }
        double ratio = price / tickSize;
        if (ratio >= MAX_GRID_RATIO) {
            throw new IllegalArgumentException("price " + price + " is " + ratio
                    + " ticks, too large to validate on the grid (limit 2^52);"
                    + " use limitTicks for counts up to MAX_TICKS");
        }
        double ticks = Math.floor(ratio);
        if (ratio - ticks >= 0.5) {
            ticks += 1.0;
        }
        if (Math.abs(ratio - ticks) > Math.min(1e-9 * Math.max(1.0, ratio), GRID_TOL_CAP)) {
            throw new IllegalArgumentException(
                    "price " + price + " is not a multiple of tick_size " + tickSize);
        }
        if (ticks < 1.0) {
            throw new IllegalArgumentException(
                    "price " + price + " rounds to 0 ticks; prices must be >= 1 tick");
        }
        return (long) ticks; // in [1, 2^52]: exact, no narrowing
    }

    private static long checkTicks(long ticks) {
        if (ticks < 1) {
            throw new IllegalArgumentException("price_ticks must be >= 1, got " + ticks);
        }
        if (ticks > MAX_TICKS) {
            throw new IllegalArgumentException("price_ticks " + ticks + " exceeds MAX_TICKS");
        }
        return ticks;
    }

    private static char checkSide(char side) {
        if (side != BUY && side != SELL) {
            throw new IllegalArgumentException("side must be 'B' or 'S', got '" + side + "'");
        }
        return side;
    }

    private static long checkSize(long size) {
        if (size <= 0 || size > MAX_TICKS) {
            throw new IllegalArgumentException(
                    "size must be a positive integer <= 9e18, got " + size);
        }
        return size;
    }

    // ------------------------------------------------------------- inspection

    /** Highest bid price in ticks, or empty if the bid side has no orders. */
    public OptionalLong bestBid() {
        return bids.isEmpty() ? OptionalLong.empty() : OptionalLong.of(bids.lastKey());
    }

    /** Lowest ask price in ticks, or empty if the ask side has no orders. */
    public OptionalLong bestAsk() {
        return asks.isEmpty() ? OptionalLong.empty() : OptionalLong.of(asks.firstKey());
    }

    /** Total resting size at the best bid level (0 if the side is empty). */
    public long bestBidSize() {
        return bids.isEmpty() ? 0 : levelSize(bids.lastEntry().getValue());
    }

    /** Total resting size at the best ask level (0 if the side is empty). */
    public long bestAskSize() {
        return asks.isEmpty() ? 0 : levelSize(asks.firstEntry().getValue());
    }

    private static long levelSize(ArrayDeque<Order> queue) {
        long total = 0;
        for (Order o : queue) {
            total += o.size;
        }
        return total;
    }

    /**
     * Mid price in ticks; NaN if either side is empty. Pinned: the exact
     * integer {@code bid + ask} rounded once to the nearest double, then
     * halved. The sum (at most 1.8e19) is held as an unsigned 64-bit value;
     * above 2^63 it is halved with the lost bit kept as a sticky bit, so the
     * single rounding stays correct.
     */
    public double mid() {
        if (bids.isEmpty() || asks.isEmpty()) {
            return Double.NaN;
        }
        long sum = bids.lastKey() + asks.firstKey();
        return sum >= 0 ? 0.5 * (double) sum : (double) ((sum >>> 1) | (sum & 1L));
    }

    /** Bid-ask spread in ticks; NaN if either side is empty. */
    public double spread() {
        if (bids.isEmpty() || asks.isEmpty()) {
            return Double.NaN;
        }
        return (double) (asks.firstKey() - bids.lastKey());
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
    public long depth(char side, int k) {
        checkSide(side);
        if (k <= 0) {
            throw new IllegalArgumentException("k must be a positive integer, got " + k);
        }
        TreeMap<Long, ArrayDeque<Order>> levels = side == BUY ? bids : asks;
        Iterator<ArrayDeque<Order>> it = (side == BUY
                ? levels.descendingMap().values()
                : levels.values()).iterator();
        long total = 0;
        for (int i = 0; i < k && it.hasNext(); i++) {
            total += levelSize(it.next());
        }
        return total;
    }

    /**
     * Sorted (ascending) ids of live resting orders on {@code side} — the
     * pinned ordering used by the zero-intelligence generator's cancels.
     */
    public long[] orderIds(char side) {
        checkSide(side);
        List<Long> ids = new ArrayList<>();
        for (Map.Entry<Long, Location> e : orders.entrySet()) {
            if (e.getValue().side() == side) {
                ids.add(e.getKey());
            }
        }
        return ids.stream().mapToLong(Long::longValue).sorted().toArray();
    }

    /** True if {@code orderId} is currently resting in the book. */
    public boolean hasOrder(long orderId) {
        return orders.containsKey(orderId);
    }

    /** Total resting size on {@code side} across all levels (O(1)). */
    public long totalSize(char side) {
        checkSide(side);
        return side == BUY ? bidTotal : askTotal;
    }

    /** Number of live resting orders on {@code side}. */
    public long nOrders(char side) {
        checkSide(side);
        long count = 0;
        for (Location v : orders.values()) {
            if (v.side() == side) {
                count++;
            }
        }
        return count;
    }

    // -------------------------------------------------------------- execution

    /**
     * Executes up to {@code size} against the side opposite to {@code side}.
     * The price bound is an explicit flag: {@code bounded == false} is a
     * market order (walks everything); {@code bounded == true} never trades
     * through {@code limitTicks}. No sentinel value exists, so an invalid
     * tick count can never be mistaken for "no bound". Consumes levels
     * best-first and orders FIFO within a level.
     */
    private ExecResult match(char side, long size, boolean bounded, long limitTicks) {
        List<Fill> fills = new ArrayList<>();
        long remaining = size;
        TreeMap<Long, ArrayDeque<Order>> opp = side == BUY ? asks : bids;
        while (remaining > 0 && !opp.isEmpty()) {
            long best = side == BUY ? opp.firstKey() : opp.lastKey();
            if (bounded) {
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
                long take = Math.min(remaining, order.size);
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
        if (side == BUY) {
            askTotal -= size - remaining;
        } else {
            bidTotal -= size - remaining;
        }
        if (recordTrades) {
            trades.addAll(fills);
        }
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
     * @param price limit price in currency units (multiple of the tick size)
     * @param size positive integer quantity
     * @param orderId unique id for the resting part
     * @return fills plus the rested size (0 if fully executed)
     * @throws IllegalArgumentException on bad side/price/size, an off-grid
     *     price, a price at or above {@code 2^52} ticks, a duplicate live
     *     {@code orderId}, or a size taking the side's total above
     *     {@code MAX_TICKS}
     */
    public ExecResult limit(char side, double price, long size, long orderId) {
        checkSide(side);
        checkSize(size);
        return limitTicks(side, toTicks(price), size, orderId);
    }

    /**
     * Submits a limit order priced directly in ticks (no float conversion);
     * identical semantics to {@link #limit}. {@code priceTicks} must lie in
     * {@code [1, MAX_TICKS]}.
     *
     * @throws IllegalArgumentException as {@link #limit}
     */
    public ExecResult limitTicks(char side, long priceTicks, long size, long orderId) {
        checkSide(side);
        checkSize(size);
        long ticks = checkTicks(priceTicks);
        if (orders.containsKey(orderId)) {
            throw new IllegalArgumentException(
                    "order_id " + orderId + " is already live in the book");
        }
        if (size > MAX_TICKS - totalSize(side)) {
            throw new IllegalArgumentException("size " + size + " would take side " + side
                    + " total resting size above MAX_TICKS");
        }
        ExecResult res = match(side, size, true, ticks);
        long remaining = res.remainder();
        if (remaining > 0) {
            TreeMap<Long, ArrayDeque<Order>> levels = side == BUY ? bids : asks;
            levels.computeIfAbsent(ticks, t -> new ArrayDeque<>()).addLast(new Order(orderId, remaining));
            orders.put(orderId, new Location(side, ticks));
            if (side == BUY) {
                bidTotal += remaining;
            } else {
                askTotal += remaining;
            }
        }
        return res;
    }

    /**
     * Cancels a resting order by id and returns the size removed.
     *
     * @throws IllegalArgumentException if the id is not live (never resting,
     *     already cancelled, or fully filled)
     */
    public long cancel(long orderId) {
        Location ref = orders.remove(orderId);
        if (ref == null) {
            throw new IllegalArgumentException("cancel of unknown order_id " + orderId);
        }
        char side = ref.side();
        long ticks = ref.ticks();
        TreeMap<Long, ArrayDeque<Order>> levels = side == BUY ? bids : asks;
        ArrayDeque<Order> queue = levels.get(ticks);
        long removed = 0;
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
        if (side == BUY) {
            bidTotal -= removed;
        } else {
            askTotal -= removed;
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
    public ExecResult market(char side, long size) {
        checkSide(side);
        checkSize(size);
        return match(side, size, false, 0L);
    }
}
