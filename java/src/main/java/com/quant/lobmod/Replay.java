package com.quant.lobmod;

import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Event stream CSV i/o and deterministic book replay.
 *
 * <p>An event stream is an ordered list of {@link Event} rows. Replaying a
 * stream through a fresh {@link LimitOrderBook} is fully deterministic, so
 * recorded golden book snapshots must be reproduced exactly by every language
 * implementation. After <em>each</em> event the replay records the
 * top-of-book state into {@link BookStates}.</p>
 *
 * <p>{@link Event#price()} is a <em>tick count</em>, never a currency price:
 * {@link #replay(List, int)} takes no tick size and drives a
 * {@code tickSize = 1} book through {@link LimitOrderBook#limitTicks}. CSV
 * parsing follows the pinned rules in {@link Csv}; {@code type} must be one
 * of L/C/M and {@code side} one of B/S at parse time.</p>
 */
public final class Replay {

    private static final String HEADER = "seq,type,side,price,size,order_id";

    private Replay() {
    }

    private static char parseCode(String s, String allowed, String what, String context) {
        if (s.length() == 1 && allowed.indexOf(s.charAt(0)) >= 0) {
            return s.charAt(0);
        }
        throw new IllegalArgumentException("invalid " + what + " '" + s + "' in " + context);
    }

    /**
     * Reads an event stream CSV with the pinned header
     * {@code seq,type,side,price,size,order_id} (rules: {@link Csv}).
     *
     * @throws IllegalArgumentException on a malformed header or row
     * @throws UncheckedIOException on i/o failure
     */
    public static List<Event> readEventsCsv(Path path) {
        List<String> lines = Csv.readCsvLines(path, HEADER);
        List<Event> events = new ArrayList<>(lines.size());
        for (int i = 0; i < lines.size(); i++) {
            String line = lines.get(i);
            String context = "event row " + (i + 1);
            String[] parts = line.split(",", -1);
            if (parts.length != 6) {
                throw new IllegalArgumentException("malformed " + context
                        + ": expected 6 fields, got " + parts.length + ": '" + line + "'");
            }
            events.add(new Event(
                    Csv.parseInt64Strict(parts[0], context),
                    parseCode(parts[1], "LCM", "event type", context),
                    parseCode(parts[2], "BS", "side", context),
                    Csv.parseInt64Strict(parts[3], context),
                    Csv.parseInt64Strict(parts[4], context),
                    Csv.parseInt64Strict(parts[5], context)));
        }
        return events;
    }

    /**
     * Writes an event stream CSV with the pinned header ({@code \n} line
     * endings).
     *
     * @throws UncheckedIOException on i/o failure
     */
    public static void writeEventsCsv(Path path, List<Event> events) {
        StringBuilder sb = new StringBuilder(32 * (events.size() + 1));
        sb.append(HEADER).append('\n');
        for (Event e : events) {
            sb.append(e.seq()).append(',').append(e.etype()).append(',').append(e.side())
                    .append(',').append(e.price()).append(',').append(e.size()).append(',')
                    .append(e.orderId()).append('\n');
        }
        Csv.writeText(path, sb.toString());
    }

    /**
     * Applies one event to the book.
     *
     * @throws IllegalArgumentException on an unknown event type or any
     *     book-level validation error (e.g. cancel of an unknown id)
     */
    public static void applyEvent(LimitOrderBook book, Event event) {
        switch (event.etype()) {
            case 'L' -> book.limitTicks(event.side(), event.price(), event.size(), event.orderId());
            case 'C' -> book.cancel(event.orderId());
            case 'M' -> book.market(event.side(), event.size());
            default -> throw new IllegalArgumentException(
                    "unknown event type '" + event.etype() + "'");
        }
    }

    /**
     * Replays {@code events} through a fresh {@code tickSize = 1} book (event
     * prices are tick counts), recording the top-of-book state after each
     * event.
     *
     * @param events the ordered event stream (non-empty)
     * @param depthK number of levels for the depth arrays (positive)
     * @return per-event {@link BookStates} arrays of length
     *     {@code events.size()}
     * @throws IllegalArgumentException if the stream is empty, {@code depthK}
     *     is not positive, or an event is invalid
     */
    public static BookStates replay(List<Event> events, int depthK) {
        if (depthK <= 0) {
            throw new IllegalArgumentException("depth_k must be a positive integer, got " + depthK);
        }
        if (events.isEmpty()) {
            throw new IllegalArgumentException("event stream is empty");
        }
        int n = events.size();
        BookStates st = new BookStates(new double[n], new double[n], new double[n],
                new double[n], new double[n], new double[n], new double[n],
                new double[n], new double[n]);
        LimitOrderBook book = new LimitOrderBook(1.0, false);
        for (int i = 0; i < n; i++) {
            applyEvent(book, events.get(i));
            st.bestBid()[i] = book.bestBid().isPresent() ? book.bestBid().getAsLong() : Double.NaN;
            st.bestAsk()[i] = book.bestAsk().isPresent() ? book.bestAsk().getAsLong() : Double.NaN;
            st.bidSize()[i] = book.bestBidSize();
            st.askSize()[i] = book.bestAskSize();
            st.bidDepthK()[i] = book.depth(LimitOrderBook.BUY, depthK);
            st.askDepthK()[i] = book.depth(LimitOrderBook.SELL, depthK);
            st.mid()[i] = book.mid();
            st.microprice()[i] = book.microprice();
            st.spread()[i] = book.spread();
        }
        return st;
    }

    /** Replays with the default {@code depthK = 3}. */
    public static BookStates replay(List<Event> events) {
        return replay(events, 3);
    }
}
