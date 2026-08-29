package com.quant.lobmod;

import java.util.ArrayList;
import java.util.List;

/**
 * Order-flow features and direction labels from replayed book states.
 *
 * <p><b>Order-flow imbalance (OFI)</b>, Cont-Kukanov-Stoikov style. For
 * consecutive book states {@code n-1 -> n} with best bid price/size
 * {@code (Pb, qb)} and best ask {@code (Pa, qa)}, the pinned per-event
 * contribution is</p>
 *
 * <pre>
 *   e_n =   (Pb_n &gt;= Pb_(n-1) ? qb_n     : 0)
 *         - (Pb_n &lt;= Pb_(n-1) ? qb_(n-1) : 0)
 *         - (Pa_n &lt;= Pa_(n-1) ? qa_n     : 0)
 *         + (Pa_n &gt;= Pa_(n-1) ? qa_(n-1) : 0)
 * </pre>
 *
 * <p>(bid improvements / additions push up, ask improvements / additions push
 * down; equal prices contribute the size <em>change</em> on that side).
 * {@code e_0 = NaN}; any event whose own or previous state has an empty side
 * yields NaN. The OFI over a trailing window of {@code w} events ending at
 * event {@code i} is the sum of {@code e_(i-w+1) .. e_i}.</p>
 *
 * <p><b>Feature vector</b> (pinned column order) at event {@code i}:
 * {@code [OFI_i, QI_i = (qb-qa)/(qb+qa), spread_i, depth_ratio_i =
 * bid_depth_k/ask_depth_k, ret_i = mid_i - mid_(i-ret_window)]}.</p>
 *
 * <p><b>Label</b> (classes pinned {@code 0 = down, 1 = flat, 2 = up}):
 * {@code d = mid_(i+H) - mid_i}; up if {@code d > epsilon}, down if
 * {@code d < -epsilon}, flat otherwise. Rows are emitted only where every
 * input is defined (see the API spec's emission rule) and keep replay
 * order.</p>
 */
public final class Features {

    /** Pinned feature column order. */
    public static final String[] FEATURE_NAMES = {
        "ofi", "queue_imbalance", "spread", "depth_ratio", "ret",
    };

    /** Pinned label classes. */
    public static final int LABEL_DOWN = 0;
    /** Flat (dead-zone) class. */
    public static final int LABEL_FLAT = 1;
    /** Upward move class. */
    public static final int LABEL_UP = 2;

    /**
     * Chronological split: index arrays for train and test.
     *
     * @param train first {@code floor(train_frac * n)} row indices
     * @param test the remaining row indices
     */
    public record Split(int[] train, int[] test) {
    }

    private Features() {
    }

    /**
     * Per-event order-flow contributions {@code e_n} (class docstring
     * formula); {@code e_0 = NaN} (no predecessor), NaN wherever any of the
     * four involved best prices is NaN.
     */
    public static double[] ofiSeries(BookStates states) {
        int n = states.size();
        double[] pb = states.bestBid();
        double[] qb = states.bidSize();
        double[] pa = states.bestAsk();
        double[] qa = states.askSize();
        double[] e = new double[n];
        e[0] = Double.NaN;
        for (int i = 1; i < n; i++) {
            if (!Double.isFinite(pb[i]) || !Double.isFinite(pb[i - 1])
                    || !Double.isFinite(pa[i]) || !Double.isFinite(pa[i - 1])) {
                e[i] = Double.NaN;
                continue;
            }
            double contrib = 0.0;
            if (pb[i] >= pb[i - 1]) {
                contrib += qb[i];
            }
            if (pb[i] <= pb[i - 1]) {
                contrib -= qb[i - 1];
            }
            if (pa[i] <= pa[i - 1]) {
                contrib -= qa[i];
            }
            if (pa[i] >= pa[i - 1]) {
                contrib += qa[i - 1];
            }
            e[i] = contrib;
        }
        return e;
    }

    /**
     * OFI over the {@code window} events ending at {@code eventIndex}
     * (0-based).
     *
     * @throws IllegalArgumentException if the window does not fit before the
     *     event or arguments are invalid
     */
    public static double ofiAt(BookStates states, int eventIndex, int window) {
        if (window <= 0) {
            throw new IllegalArgumentException("window must be a positive integer, got " + window);
        }
        int n = states.size();
        if (eventIndex < 0 || eventIndex >= n) {
            throw new IllegalArgumentException(
                    "event_index " + eventIndex + " out of range [0, " + n + ")");
        }
        if (eventIndex - window < 0) {
            throw new IllegalArgumentException(
                    "window " + window + " does not fit before event " + eventIndex);
        }
        double[] e = ofiSeries(states);
        double sum = 0.0;
        for (int i = eventIndex - window + 1; i <= eventIndex; i++) {
            sum += e[i];
        }
        return sum;
    }

    /**
     * Builds the feature matrix and labels from replayed book states; rows
     * with any undefined input are skipped (class docstring rules).
     *
     * @throws IllegalArgumentException if no valid feature row can be built
     */
    public static FeatureSet computeFeatures(BookStates states, FeatureConfig cfg) {
        int n = states.size();
        double[] e = ofiSeries(states);
        int start = Math.max(cfg.ofiWindow(), cfg.retWindow());
        List<double[]> rows = new ArrayList<>(); // {i, ofi, qi, spread, depth_ratio, ret, label, d}
        for (int i = start; i < n - cfg.horizon(); i++) {
            boolean windowOk = true;
            double ofi = 0.0;
            for (int m = i - cfg.ofiWindow() + 1; m <= i; m++) {
                if (!Double.isFinite(e[m])) {
                    windowOk = false;
                    break;
                }
                ofi += e[m];
            }
            if (!windowOk) {
                continue;
            }
            double qb = states.bidSize()[i];
            double qa = states.askSize()[i];
            double midNow = states.mid()[i];
            double midPast = states.mid()[i - cfg.retWindow()];
            double midFwd = states.mid()[i + cfg.horizon()];
            double spread = states.spread()[i];
            double bd = states.bidDepthK()[i];
            double ad = states.askDepthK()[i];
            if (!Double.isFinite(midNow) || !Double.isFinite(midPast)
                    || !Double.isFinite(midFwd) || !Double.isFinite(spread)) {
                continue;
            }
            if (qb + qa <= 0.0 || ad <= 0.0) {
                continue;
            }
            double qi = (qb - qa) / (qb + qa);
            double depthRatio = bd / ad;
            double ret = midNow - midPast;
            double d = midFwd - midNow;
            int label;
            if (d > cfg.epsilon()) {
                label = LABEL_UP;
            } else if (d < -cfg.epsilon()) {
                label = LABEL_DOWN;
            } else {
                label = LABEL_FLAT;
            }
            rows.add(new double[] {i, ofi, qi, spread, depthRatio, ret, label, d});
        }
        if (rows.isEmpty()) {
            throw new IllegalArgumentException("no valid feature rows could be built from the states");
        }
        int m = rows.size();
        double[][] x = new double[m][5];
        int[] y = new int[m];
        int[] eventIndex = new int[m];
        double[] fwdRet = new double[m];
        double[] spread = new double[m];
        for (int r = 0; r < m; r++) {
            double[] row = rows.get(r);
            System.arraycopy(row, 1, x[r], 0, 5);
            eventIndex[r] = (int) row[0];
            y[r] = (int) row[6];
            fwdRet[r] = row[7];
            spread[r] = row[3];
        }
        return new FeatureSet(x, y, eventIndex, fwdRet, spread);
    }

    /**
     * Pinned chronological split: the first {@code floor(trainFrac * n)} rows
     * train, the rest test.
     *
     * @throws IllegalArgumentException if the split would leave either side
     *     empty or the arguments are invalid
     */
    public static Split chronologicalSplit(int n, double trainFrac) {
        if (n <= 1) {
            throw new IllegalArgumentException("n must be an integer > 1, got " + n);
        }
        if (!(trainFrac > 0.0 && trainFrac < 1.0)) {
            throw new IllegalArgumentException("train_frac must be in (0, 1), got " + trainFrac);
        }
        int nTrain = (int) Math.floor(trainFrac * n);
        if (nTrain == 0 || nTrain == n) {
            throw new IllegalArgumentException(
                    "split leaves an empty side: n=" + n + ", train_frac=" + trainFrac);
        }
        int[] train = new int[nTrain];
        int[] test = new int[n - nTrain];
        for (int i = 0; i < nTrain; i++) {
            train[i] = i;
        }
        for (int i = nTrain; i < n; i++) {
            test[i - nTrain] = i;
        }
        return new Split(train, test);
    }
}
