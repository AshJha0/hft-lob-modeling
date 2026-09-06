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

    private void checkRows(int[] idx) {
        for (int i : idx) {
            if (i < 0 || i >= size()) {
                throw new IllegalArgumentException(
                        "row index " + i + " out of range [0, " + size() + ")");
            }
        }
    }

    /**
     * The selected feature rows (e.g. a {@link Features.Split} index array)
     * as a fresh matrix.
     *
     * @throws IllegalArgumentException on an out-of-range index
     */
    public double[][] xRows(int[] idx) {
        checkRows(idx);
        double[][] out = new double[idx.length][];
        for (int i = 0; i < idx.length; i++) {
            out[i] = x[idx[i]].clone();
        }
        return out;
    }

    /** The selected labels. */
    public int[] yRows(int[] idx) {
        checkRows(idx);
        int[] out = new int[idx.length];
        for (int i = 0; i < idx.length; i++) {
            out[i] = y[idx[i]];
        }
        return out;
    }

    /** The selected forward returns. */
    public double[] fwdRetRows(int[] idx) {
        checkRows(idx);
        double[] out = new double[idx.length];
        for (int i = 0; i < idx.length; i++) {
            out[i] = fwdRet[idx[i]];
        }
        return out;
    }

    /** The selected decision-time spreads. */
    public double[] spreadRows(int[] idx) {
        checkRows(idx);
        double[] out = new double[idx.length];
        for (int i = 0; i < idx.length; i++) {
            out[i] = spread[idx[i]];
        }
        return out;
    }
}
