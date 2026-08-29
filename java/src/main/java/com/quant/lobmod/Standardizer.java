package com.quant.lobmod;

/**
 * Pinned per-feature standardization fitted on the training split only.
 *
 * <p>Per column: {@code mean = sum/n} and population standard deviation
 * {@code std = sqrt(sum((x-mean)^2)/n)} (divisor n — pinned); a std below
 * 1e-12 is replaced by 1. Train AND test are transformed with these
 * training-split moments.</p>
 */
public final class Standardizer {

    private static final double STD_FLOOR = 1e-12;

    private double[] mean;
    private double[] std;

    /**
     * Computes population mean/std per column.
     *
     * @return this, for chaining
     * @throws IllegalArgumentException on empty or non-finite input
     */
    public Standardizer fit(double[][] x) {
        checkMatrix(x);
        int n = x.length;
        int d = x[0].length;
        mean = new double[d];
        std = new double[d];
        for (int j = 0; j < d; j++) {
            double sum = 0.0;
            for (double[] row : x) {
                sum += row[j];
            }
            mean[j] = sum / n;
            double sq = 0.0;
            for (double[] row : x) {
                double c = row[j] - mean[j];
                sq += c * c;
            }
            double s = Math.sqrt(sq / n); // population (divisor n) — pinned
            std[j] = s < STD_FLOOR ? 1.0 : s;
        }
        return this;
    }

    /**
     * Applies {@code (x - mean) / std}; requires {@link #fit} first.
     *
     * @return a newly allocated standardized copy
     * @throws IllegalArgumentException if not fitted or the column count
     *     differs from the fitted data
     */
    public double[][] transform(double[][] x) {
        if (mean == null || std == null) {
            throw new IllegalArgumentException("Standardizer.transform called before fit");
        }
        checkMatrix(x);
        if (x[0].length != mean.length) {
            throw new IllegalArgumentException(
                    "expected " + mean.length + " columns, got " + x[0].length);
        }
        double[][] out = new double[x.length][mean.length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < mean.length; j++) {
                out[i][j] = (x[i][j] - mean[j]) / std[j];
            }
        }
        return out;
    }

    /** Fitted per-column means (copy); requires {@link #fit} first. */
    public double[] mean() {
        if (mean == null) {
            throw new IllegalArgumentException("Standardizer not fitted");
        }
        return mean.clone();
    }

    /** Fitted per-column stds (copy, floored at 1); requires {@link #fit} first. */
    public double[] std() {
        if (std == null) {
            throw new IllegalArgumentException("Standardizer not fitted");
        }
        return std.clone();
    }

    static void checkMatrix(double[][] x) {
        if (x == null || x.length == 0 || x[0].length == 0) {
            throw new IllegalArgumentException("X must be a non-empty 2-D array");
        }
        int d = x[0].length;
        for (double[] row : x) {
            if (row.length != d) {
                throw new IllegalArgumentException("X must be rectangular");
            }
            for (double v : row) {
                if (!Double.isFinite(v)) {
                    throw new IllegalArgumentException("X must be finite");
                }
            }
        }
    }
}
