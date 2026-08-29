package com.quant.lobmod;

import java.util.Arrays;

/**
 * Goodness of fit: Kolmogorov-Smirnov test of Exp(1) residuals.
 *
 * <p>By the time-rescaling theorem (see {@link Hawkes}), a correctly
 * specified Hawkes model turns its inter-arrival compensator increments into
 * iid Exp(1) draws. This class tests that natively:</p>
 * <ul>
 *   <li>KS statistic against the Exp(1) CDF {@code F(x) = 1 - exp(-x)}:
 *       {@code D = max_i max(i/n - F(x_(i)), F(x_(i)) - (i-1)/n)} over the
 *       sorted sample;</li>
 *   <li>asymptotic p-value with the Stephens small-sample correction
 *       (Numerical Recipes pinned form):
 *       {@code lambda = (sqrt(n) + 0.12 + 0.11/sqrt(n)) * D} and the
 *       Kolmogorov tail series
 *       {@code 2 * sum_(j=1..100) (-1)^(j-1) exp(-2 j^2 lambda^2)}, clipped
 *       to [0, 1]. The series is pinned at exactly 100 terms.</li>
 * </ul>
 *
 * <p>With very small n the asymptotic p-value is only indicative — callers
 * should treat it as such (documented, not an error).</p>
 */
public final class Gof {

    /**
     * KS test outcome.
     *
     * @param stat the KS statistic D
     * @param pValue the asymptotic p-value
     */
    public record KsResult(double stat, double pValue) {
    }

    private Gof() {
    }

    /**
     * KS statistic of {@code sample} against the Exp(1) CDF.
     *
     * @throws IllegalArgumentException on an empty sample or non-finite /
     *     negative values
     */
    public static double ksStatisticExp(double[] sample) {
        if (sample == null || sample.length == 0) {
            throw new IllegalArgumentException("sample must be a non-empty 1-D array");
        }
        for (double v : sample) {
            if (!Double.isFinite(v) || v < 0.0) {
                throw new IllegalArgumentException("sample values must be finite and non-negative");
            }
        }
        double[] xs = sample.clone();
        Arrays.sort(xs);
        int n = xs.length;
        double dPlus = Double.NEGATIVE_INFINITY;
        double dMinus = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            double cdf = -Math.expm1(-xs[i]); // 1 - exp(-x), accurately
            dPlus = Math.max(dPlus, (i + 1.0) / n - cdf);
            dMinus = Math.max(dMinus, cdf - (double) i / n);
        }
        return Math.max(dPlus, dMinus);
    }

    /**
     * Asymptotic Kolmogorov p-value for statistic {@code d} on {@code n}
     * points (pinned Stephens-corrected 100-term series, clipped to [0, 1];
     * exactly 1 when {@code d = 0}).
     *
     * @throws IllegalArgumentException if {@code d} is outside [0, 1] or
     *     {@code n} is not positive
     */
    public static double ksPvalue(double d, int n) {
        if (!(Double.isFinite(d) && d >= 0.0 && d <= 1.0)) {
            throw new IllegalArgumentException("d must be in [0, 1], got " + d);
        }
        if (n <= 0) {
            throw new IllegalArgumentException("n must be a positive integer, got " + n);
        }
        if (d == 0.0) {
            return 1.0;
        }
        double sqrtN = Math.sqrt(n);
        double lam = (sqrtN + 0.12 + 0.11 / sqrtN) * d;
        double total = 0.0;
        for (int j = 1; j <= 100; j++) {
            double term = Math.exp(-2.0 * j * j * lam * lam);
            total += j % 2 == 1 ? term : -term;
        }
        double p = 2.0 * total;
        return Math.min(1.0, Math.max(0.0, p));
    }

    /** KS test of {@code sample} against Exp(1): statistic plus p-value. */
    public static KsResult ksTestExp(double[] sample) {
        double d = ksStatisticExp(sample);
        return new KsResult(d, ksPvalue(d, sample.length));
    }
}
