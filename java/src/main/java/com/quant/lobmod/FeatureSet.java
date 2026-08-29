package com.quant.lobmod;

/**
 * Feature matrix with aligned labels and trading metadata.
 *
 * @param x (n, 5) feature matrix in {@link Features#FEATURE_NAMES} order
 * @param y (n,) integer labels in {0 = down, 1 = flat, 2 = up}
 * @param eventIndex (n,) 0-based index into the {@link BookStates} arrays
 * @param fwdRet (n,) realized {@code mid_(i+H) - mid_i} (for PnL evaluation)
 * @param spread (n,) spread at decision time (for cost modelling)
 */
public record FeatureSet(double[][] x, int[] y, int[] eventIndex, double[] fwdRet,
        double[] spread) {

    /** Number of emitted feature rows. */
    public int size() {
        return y.length;
    }
}
