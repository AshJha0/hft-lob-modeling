package com.quant.lobmod;

/**
 * Classification metrics and the honest cost-aware PnL of trading a signal.
 */
public final class Metrics {

    /**
     * Signal PnL outcome.
     *
     * @param gross sum of {@code pos * fwd_ret} (mid-to-mid, ticks)
     * @param net gross minus a half-spread cost per non-flat signal
     * @param nTrades number of non-flat signals
     */
    public record Pnl(double gross, double net, int nTrades) {
    }

    private Metrics() {
    }

    /**
     * Fraction of matching labels.
     *
     * @throws IllegalArgumentException on mismatched or empty arrays
     */
    public static double accuracy(int[] yTrue, int[] yPred) {
        if (yTrue == null || yPred == null || yTrue.length != yPred.length || yTrue.length == 0) {
            throw new IllegalArgumentException(
                    "y_true and y_pred must be equal-length non-empty 1-D arrays");
        }
        int hits = 0;
        for (int i = 0; i < yTrue.length; i++) {
            if (yTrue[i] == yPred[i]) {
                hits++;
            }
        }
        return (double) hits / yTrue.length;
    }

    /**
     * (nClasses x nClasses) counts, rows = true class, cols = predicted.
     *
     * @throws IllegalArgumentException on labels outside {@code [0, nClasses)}
     *     or mismatched lengths
     */
    public static int[][] confusionMatrix(int[] yTrue, int[] yPred, int nClasses) {
        if (yTrue == null || yPred == null || yTrue.length != yPred.length) {
            throw new IllegalArgumentException("y_true and y_pred must have equal length");
        }
        checkLabels(yTrue, nClasses);
        checkLabels(yPred, nClasses);
        int[][] cm = new int[nClasses][nClasses];
        for (int i = 0; i < yTrue.length; i++) {
            cm[yTrue[i]][yPred[i]]++;
        }
        return cm;
    }

    private static void checkLabels(int[] y, int nClasses) {
        for (int v : y) {
            if (v < 0 || v >= nClasses) {
                throw new IllegalArgumentException("labels must be in [0, " + nClasses + ")");
            }
        }
    }

    /**
     * Most frequent label; ties resolve to the smallest label (pinned).
     *
     * @throws IllegalArgumentException on an empty array or negative labels
     */
    public static int majorityClass(int[] y) {
        if (y == null || y.length == 0) {
            throw new IllegalArgumentException("y must be a non-empty 1-D array");
        }
        int max = 0;
        for (int v : y) {
            if (v < 0) {
                throw new IllegalArgumentException("labels must be non-negative, got " + v);
            }
            max = Math.max(max, v);
        }
        int[] counts = new int[max + 1];
        for (int v : y) {
            counts[v]++;
        }
        int best = 0;
        for (int v = 1; v <= max; v++) {
            if (counts[v] > counts[best]) { // strict: ties keep the smallest label
                best = v;
            }
        }
        return best;
    }

    /**
     * PnL of trading the predicted direction with half-spread costs.
     *
     * <p>Position per row: +1 for class 2 (up), -1 for class 0 (down), 0 for
     * flat. Gross PnL adds {@code pos * fwd_ret} (mid-to-mid, ticks per unit
     * size); net PnL subtracts {@code |pos| * spread / 2} per non-flat
     * signal, charged on entry only (the exit is marked at mid). This is a
     * signal diagnostic, not a tradable strategy: positions of consecutive
     * rows overlap; no fees, latency or fill probability.</p>
     *
     * @throws IllegalArgumentException on mismatched, empty, or non-finite
     *     inputs, or a label outside {@code {0, 1, 2}}
     */
    public static Pnl signalPnl(int[] yPred, double[] fwdRet, double[] spread) {
        if (yPred == null || fwdRet == null || spread == null
                || yPred.length != fwdRet.length || yPred.length != spread.length) {
            throw new IllegalArgumentException(
                    "y_pred, fwd_ret, spread must be equal-length 1-D arrays");
        }
        if (yPred.length == 0) {
            throw new IllegalArgumentException("empty prediction array");
        }
        for (int i = 0; i < yPred.length; i++) {
            if (yPred[i] < 0 || yPred[i] > 2) {
                throw new IllegalArgumentException("y_pred labels must be in {0, 1, 2}");
            }
            if (!Double.isFinite(fwdRet[i]) || !Double.isFinite(spread[i])) {
                throw new IllegalArgumentException("fwd_ret and spread must be finite");
            }
        }
        double gross = 0.0;
        double cost = 0.0;
        int nTrades = 0;
        for (int i = 0; i < yPred.length; i++) {
            int pos = yPred[i] - 1; // {0,1,2} -> {-1,0,+1}
            gross += pos * fwdRet[i];
            cost += Math.abs(pos) * spread[i] * 0.5;
            if (pos != 0) {
                nTrades++;
            }
        }
        return new Pnl(gross, gross - cost, nTrades);
    }
}
