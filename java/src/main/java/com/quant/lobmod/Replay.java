package com.quant.lobmod;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
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
 */
public final class Replay {

    private static final String HEADER = "seq,type,side,price,size,order_id";

    private Replay() {
    }

    /**
     * Reads an event stream CSV with the pinned header
     * {@code seq,type,side,price,size,order_id}.
     *
     * @throws IllegalArgumentException on a malformed header or row
     * @throws UncheckedIOException on i/o failure
     */
    public static List<Event> readEventsCsv(Path path) {
        List<String> lines;
        try {
            lines = Files.readAllLines(path);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        if (lines.isEmpty() || !lines.get(0).strip().equals(HEADER)) {
            throw new IllegalArgumentException(
                    "unexpected event CSV header: " + (lines.isEmpty() ? "<empty>" : lines.get(0)));
        }
        List<Event> events = new ArrayList<>(lines.size() - 1);
        for (int i = 1; i < lines.size(); i++) {
            String line = lines.get(i).strip();
            if (line.isEmpty()) {
                continue;
            }
            String[] parts = line.split(",", -1);
            if (parts.length != 6 || parts[1].length() != 1 || parts[2].length() != 1) {
                throw new IllegalArgumentException("malformed event row: " + line);
            }
            try {
                events.add(new Event(
                        Integer.parseInt(parts[0]),
                        parts[1].charAt(0),
                        parts[2].charAt(0),
                        Integer.parseInt(parts[3]),
                        Integer.parseInt(parts[4]),
                        Integer.parseInt(parts[5])));
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("malformed event row: " + line, e);
            }
        }
        return events;
    }

    /**
     * Applies one event to the book.
     *
     * @throws IllegalArgumentException on an unknown event type or any
     *     book-level validation error (e.g. cancel of an unknown id)
     */
    public static void applyEvent(LimitOrderBook book, Event event) {
        switch (event.etype()) {
            case 'L' -> book.limit(event.side(), event.price(), event.size(), event.orderId());
            case 'C' -> book.cancel(event.orderId());
            case 'M' -> book.market(event.side(), event.size());
            default -> throw new IllegalArgumentException(
                    "unknown event type '" + event.etype() + "'");
        }
    }

    /**
     * Replays {@code events} through a fresh book, recording the top-of-book
     * state after each event.
     *
     * @param events the ordered event stream (non-empty)
     * @param depthK number of levels for the depth arrays
     * @param tickSize book tick size
     * @return per-event {@link BookStates} arrays of length
     *     {@code events.size()}
     * @throws IllegalArgumentException if the stream is empty or an event is
     *     invalid
     */
    public static BookStates replay(List<Event> events, int depthK, double tickSize) {
        if (events.isEmpty()) {
            throw new IllegalArgumentException("event stream is empty");
        }
        int n = events.size();
        BookStates st = new BookStates(new double[n], new double[n], new double[n],
                new double[n], new double[n], new double[n], new double[n],
                new double[n], new double[n]);
        LimitOrderBook book = new LimitOrderBook(tickSize);
        for (int i = 0; i < n; i++) {
            applyEvent(book, events.get(i));
            st.bestBid()[i] = book.bestBid().isPresent() ? book.bestBid().getAsInt() : Double.NaN;
            st.bestAsk()[i] = book.bestAsk().isPresent() ? book.bestAsk().getAsInt() : Double.NaN;
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

    /** Replays with the default {@code depthK = 3} and {@code tickSize = 1}. */
    public static BookStates replay(List<Event> events) {
        return replay(events, 3, 1.0);
    }
}
