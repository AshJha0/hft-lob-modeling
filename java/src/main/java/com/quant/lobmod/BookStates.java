package com.quant.lobmod;

/**
 * Top-of-book state after each replayed event (arrays of length nEvents).
 *
 * <p>Sides that are empty at an event produce NaN in the price-derived arrays
 * and 0 in the size arrays for that row. These arrays are the single input to
 * the feature pipeline ({@link Features}).</p>
 *
 * @param bestBid best bid price in ticks (NaN if none)
 * @param bestAsk best ask price in ticks (NaN if none)
 * @param bidSize total size at the best bid level (0 if none)
 * @param askSize total size at the best ask level (0 if none)
 * @param bidDepthK summed size over the top-k bid levels
 * @param askDepthK summed size over the top-k ask levels
 * @param mid {@code (bid + ask) / 2} in ticks (NaN guarded)
 * @param microprice size-weighted mid (NaN guarded)
 * @param spread {@code ask - bid} in ticks (NaN guarded)
 */
public record BookStates(
        double[] bestBid,
        double[] bestAsk,
        double[] bidSize,
        double[] askSize,
        double[] bidDepthK,
        double[] askDepthK,
        double[] mid,
        double[] microprice,
        double[] spread) {

    /** Number of recorded events. */
    public int size() {
        return mid.length;
    }
}
