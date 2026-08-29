package com.quant.lobmod;

import java.util.ArrayList;
import java.util.List;
import java.util.function.ToDoubleFunction;

/**
 * Hawkes processes with exponential kernels: likelihood, MLE, simulation.
 *
 * <p><b>Univariate model.</b> Conditional intensity
 * {@code lambda(t) = mu + alpha * sum_(t_i < t) exp(-beta (t - t_i))} with
 * background rate {@code mu > 0}, jump size {@code alpha >= 0} and decay
 * {@code beta > 0}. The branching ratio {@code alpha / beta} is the expected
 * number of children per event; the process is stationary iff it is
 * &lt; 1.</p>
 *
 * <p><b>Pinned O(N) likelihood recursion</b> (Ozaki 1979): with
 * {@code R_1 = 0}, {@code R_i = exp(-beta (t_i - t_(i-1))) (1 + R_(i-1))},</p>
 *
 * <pre>
 *   ll = sum_i log(mu + alpha R_i) - mu T
 *        + (alpha/beta) * sum_i expm1(-beta (T - t_i))
 * </pre>
 *
 * <p><b>Bivariate model</b> (pinned parameterization): two event types with
 * full 2x2 excitation {@code alpha} and a scalar shared {@code beta}; the
 * same recursion runs on the merged stream with per-type decayed sums.</p>
 *
 * <p><b>Time-rescaling theorem.</b> Under the true model the compensator
 * increments {@code tau_i = Lambda(t_i) - Lambda(t_(i-1))} are iid Exp(1),
 * which is what {@link Gof} tests.</p>
 *
 * <p>Recoverable diagnostics (near-critical / non-stationary branching) are
 * reported in the fit result's {@code warning} field, never thrown.</p>
 */
public final class Hawkes {

    private static final double PENALTY = 1e10;

    private Hawkes() {
    }

    // --------------------------------------------------------------- results

    /**
     * Univariate MLE result.
     *
     * @param mu fitted background rate
     * @param alpha fitted jump size
     * @param beta fitted decay
     * @param loglik maximized log-likelihood
     * @param branching {@code alpha / beta}
     * @param converged Nelder-Mead tolerance convergence flag
     * @param nIter Nelder-Mead iterations
     * @param warning stationarity diagnostic ("near-critical" /
     *     "non-stationary" message) or {@code null} when unremarkable
     */
    public record UniFit(double mu, double alpha, double beta, double loglik,
            double branching, boolean converged, int nIter, String warning) {
    }

    /**
     * Bivariate MLE result.
     *
     * @param mu length-2 background rates
     * @param alpha 2x2 excitation matrix
     * @param beta shared decay
     * @param loglik maximized log-likelihood
     * @param branching 2x2 branching matrix {@code alpha / beta}
     * @param spectralRadius stationarity measure (stationary iff &lt; 1)
     * @param converged Nelder-Mead tolerance convergence flag (second stage)
     * @param nIter total Nelder-Mead iterations over both pinned stages
     * @param warning stationarity diagnostic or {@code null}
     */
    public record BivFit(double[] mu, double[][] alpha, double beta, double loglik,
            double[][] branching, double spectralRadius, boolean converged, int nIter,
            String warning) {
    }

    /**
     * A simulated bivariate sample.
     *
     * @param times strictly increasing event times
     * @param types aligned event types in {0, 1}
     */
    public record BivSample(double[] times, int[] types) {
    }

    // ------------------------------------------------------------ validation

    private static void checkTimes(double[] t, double T) {
        if (t == null || t.length == 0) {
            throw new IllegalArgumentException("times must be a non-empty 1-D array");
        }
        for (double v : t) {
            if (!Double.isFinite(v)) {
                throw new IllegalArgumentException("times must be finite");
            }
        }
        if (t[0] <= 0.0) {
            throw new IllegalArgumentException("times must be strictly positive");
        }
        for (int i = 1; i < t.length; i++) {
            if (t[i] - t[i - 1] <= 0.0) {
                throw new IllegalArgumentException("times must be strictly increasing");
            }
        }
        if (!(Double.isFinite(T) && T > 0.0)) {
            throw new IllegalArgumentException("T must be positive and finite, got " + T);
        }
        if (t[t.length - 1] > T) {
            throw new IllegalArgumentException(
                    "last event time " + t[t.length - 1] + " exceeds horizon T=" + T);
        }
    }

    private static void checkUniParams(double mu, double alpha, double beta) {
        if (!(Double.isFinite(mu) && mu > 0.0)) {
            throw new IllegalArgumentException("mu must be positive and finite, got " + mu);
        }
        if (!(Double.isFinite(alpha) && alpha >= 0.0)) {
            throw new IllegalArgumentException("alpha must be non-negative and finite, got " + alpha);
        }
        if (!(Double.isFinite(beta) && beta > 0.0)) {
            throw new IllegalArgumentException("beta must be positive and finite, got " + beta);
        }
    }

    private static void checkBivParams(double[] mu, double[][] alpha, double beta) {
        if (mu == null || mu.length != 2 || !Double.isFinite(mu[0]) || !Double.isFinite(mu[1])
                || mu[0] <= 0.0 || mu[1] <= 0.0) {
            throw new IllegalArgumentException("mu must be two positive finite values");
        }
        if (alpha == null || alpha.length != 2) {
            throw new IllegalArgumentException("alpha must be a 2x2 non-negative finite matrix");
        }
        for (double[] row : alpha) {
            if (row == null || row.length != 2) {
                throw new IllegalArgumentException("alpha must be a 2x2 non-negative finite matrix");
            }
            for (double v : row) {
                if (!Double.isFinite(v) || v < 0.0) {
                    throw new IllegalArgumentException(
                            "alpha must be a 2x2 non-negative finite matrix");
                }
            }
        }
        if (!(Double.isFinite(beta) && beta > 0.0)) {
            throw new IllegalArgumentException("beta must be positive and finite, got " + beta);
        }
    }

    private static void checkTypes(int[] types, int n) {
        if (types == null || types.length != n) {
            throw new IllegalArgumentException("types must have the same length as times");
        }
        for (int k : types) {
            if (k != 0 && k != 1) {
                throw new IllegalArgumentException("types must be 0 or 1");
            }
        }
    }

    // -------------------------------------------------------- log-likelihood

    /** Exact univariate log-likelihood via the pinned O(N) recursion. */
    public static double uniLoglik(double[] times, double T, double mu, double alpha, double beta) {
        checkTimes(times, T);
        checkUniParams(mu, alpha, beta);
        double ll = Math.log(mu); // R_1 = 0
        double r = 0.0;
        for (int i = 1; i < times.length; i++) {
            double w = Math.exp(-beta * (times[i] - times[i - 1]));
            r = w * (1.0 + r);
            ll += Math.log(mu + alpha * r);
        }
        ll -= mu * T;
        double tail = 0.0;
        for (double t : times) {
            tail += Math.expm1(-beta * (T - t));
        }
        ll += (alpha / beta) * tail;
        return ll;
    }

    /**
     * Reference O(N^2) log-likelihood (direct double sum). Used by tests to
     * validate the recursion; not for production use.
     */
    public static double uniLoglikBrute(double[] times, double T, double mu, double alpha,
            double beta) {
        checkTimes(times, T);
        checkUniParams(mu, alpha, beta);
        double ll = 0.0;
        for (int i = 0; i < times.length; i++) {
            double excitation = 0.0;
            for (int j = 0; j < i; j++) {
                excitation += Math.exp(-beta * (times[i] - times[j]));
            }
            ll += Math.log(mu + alpha * excitation);
        }
        ll -= mu * T;
        double tail = 0.0;
        for (double t : times) {
            tail += Math.expm1(-beta * (T - t));
        }
        ll += (alpha / beta) * tail;
        return ll;
    }

    /** Bivariate log-likelihood (full 2x2 alpha, scalar shared beta), O(N). */
    public static double bivLoglik(double[] times, int[] types, double T, double[] mu,
            double[][] alpha, double beta) {
        checkTimes(times, T);
        checkTypes(types, times.length);
        checkBivParams(mu, alpha, beta);
        double s0 = 0.0;
        double s1 = 0.0;
        double prev = 0.0;
        double ll = 0.0;
        for (int i = 0; i < times.length; i++) {
            double w = Math.exp(-beta * (times[i] - prev));
            s0 *= w;
            s1 *= w;
            int m = types[i];
            double lam = mu[m] + alpha[m][0] * s0 + alpha[m][1] * s1;
            ll += Math.log(lam);
            if (m == 0) {
                s0 += 1.0;
            } else {
                s1 += 1.0;
            }
            prev = times[i];
        }
        ll -= (mu[0] + mu[1]) * T;
        double[] tail = new double[2];
        for (int i = 0; i < times.length; i++) {
            tail[types[i]] += Math.expm1(-beta * (T - times[i]));
        }
        for (int n = 0; n < 2; n++) {
            ll += ((alpha[0][n] + alpha[1][n]) / beta) * tail[n];
        }
        return ll;
    }

    /** Reference O(N^2) bivariate log-likelihood (tests / validation only). */
    public static double bivLoglikBrute(double[] times, int[] types, double T, double[] mu,
            double[][] alpha, double beta) {
        checkTimes(times, T);
        checkTypes(types, times.length);
        checkBivParams(mu, alpha, beta);
        double ll = 0.0;
        for (int i = 0; i < times.length; i++) {
            double lam = mu[types[i]];
            for (int n = 0; n < 2; n++) {
                double sum = 0.0;
                for (int j = 0; j < i; j++) {
                    if (types[j] == n) {
                        sum += Math.exp(-beta * (times[i] - times[j]));
                    }
                }
                lam += alpha[types[i]][n] * sum;
            }
            ll += Math.log(lam);
        }
        ll -= (mu[0] + mu[1]) * T;
        double[] tail = new double[2];
        for (int i = 0; i < times.length; i++) {
            tail[types[i]] += Math.expm1(-beta * (T - times[i]));
        }
        for (int n = 0; n < 2; n++) {
            ll += ((alpha[0][n] + alpha[1][n]) / beta) * tail[n];
        }
        return ll;
    }

    // -------------------------------------------------------------- residuals

    /**
     * Univariate time-rescaling residuals
     * {@code tau_i = Lambda(t_i) - Lambda(t_(i-1))}.
     *
     * <p>With {@code S_i = sum_(j&lt;i) exp(-beta (t_i - t_j))} (same recursion
     * as the likelihood), the compensator at an event time is
     * {@code Lambda(t_i) = mu t_i + (alpha/beta) ((i-1) - S_i)}; under the
     * true model the increments are iid Exp(1).</p>
     */
    public static double[] uniResiduals(double[] times, double T, double mu, double alpha,
            double beta) {
        checkTimes(times, T);
        checkUniParams(mu, alpha, beta);
        int n = times.length;
        double[] lamInt = new double[n];
        double s = 0.0;
        for (int i = 0; i < n; i++) {
            if (i > 0) {
                s = Math.exp(-beta * (times[i] - times[i - 1])) * (s + 1.0);
            }
            lamInt[i] = mu * times[i] + (alpha / beta) * (i - s);
        }
        double[] tau = new double[n];
        double prev = 0.0;
        for (int i = 0; i < n; i++) {
            tau[i] = lamInt[i] - prev;
            prev = lamInt[i];
        }
        return tau;
    }

    /**
     * Per-type time-rescaling residuals for the bivariate model: walks the
     * merged stream keeping decayed sums and prior event counts per source
     * type; increments between consecutive same-type events are iid Exp(1)
     * under the model.
     *
     * @return {@code {tau_type0, tau_type1}}
     */
    public static double[][] bivResiduals(double[] times, int[] types, double T, double[] mu,
            double[][] alpha, double beta) {
        checkTimes(times, T);
        checkTypes(types, times.length);
        checkBivParams(mu, alpha, beta);
        double[] s = {0.0, 0.0};
        double[] c = {0.0, 0.0};
        double prevT = 0.0;
        List<List<Double>> lamInt = List.of(new ArrayList<>(), new ArrayList<>());
        for (int i = 0; i < times.length; i++) {
            double w = Math.exp(-beta * (times[i] - prevT));
            s[0] *= w;
            s[1] *= w;
            int m = types[i];
            double val = mu[m] * times[i];
            for (int n = 0; n < 2; n++) {
                val += (alpha[m][n] / beta) * (c[n] - s[n]);
            }
            lamInt.get(m).add(val);
            s[m] += 1.0;
            c[m] += 1.0;
            prevT = times[i];
        }
        double[][] out = new double[2][];
        for (int m = 0; m < 2; m++) {
            List<Double> v = lamInt.get(m);
            out[m] = new double[v.size()];
            double prev = 0.0;
            for (int i = 0; i < v.size(); i++) {
                out[m][i] = v.get(i) - prev;
                prev = v.get(i);
            }
        }
        return out;
    }

    // ------------------------------------------------------------- simulation

    /**
     * Simulates a univariate Hawkes process on {@code [0, T]} by Ogata
     * thinning (pinned; RNG = {@code Lcg64(seed)}, exactly two uniforms per
     * loop iteration).
     *
     * @throws IllegalArgumentException on bad params, a non-stationary
     *     {@code alpha/beta >= 1}, or bad {@code T}
     */
    public static double[] simulateUni(double mu, double alpha, double beta, double T, long seed) {
        checkUniParams(mu, alpha, beta);
        if (alpha / beta >= 1.0) {
            throw new IllegalArgumentException(
                    "cannot simulate non-stationary process: alpha/beta = " + (alpha / beta));
        }
        if (!(Double.isFinite(T) && T > 0.0)) {
            throw new IllegalArgumentException("T must be positive and finite, got " + T);
        }
        Lcg64 rng = new Lcg64(seed);
        double t = 0.0;
        double s = 0.0;
        List<Double> out = new ArrayList<>();
        while (true) {
            double lamBar = mu + alpha * s;      // intensity just after last step
            double w = rng.exponential(lamBar);  // 1st uniform
            t += w;
            if (t > T) {
                break;
            }
            s *= Math.exp(-beta * w);
            double lamT = mu + alpha * s;
            if (rng.uniform() * lamBar <= lamT) { // 2nd uniform
                out.add(t);
                s += 1.0;
            }
        }
        return out.stream().mapToDouble(Double::doubleValue).toArray();
    }

    /**
     * Simulates a bivariate Hawkes process by Ogata thinning (pinned). Same
     * scheme as {@link #simulateUni} on the total intensity; an accepted
     * candidate with {@code d = u * lambda_bar} is type 0 iff
     * {@code d <= lambda_0(t)}, else type 1.
     *
     * @throws IllegalArgumentException on bad params, a branching spectral
     *     radius &ge; 1, or bad {@code T}
     */
    public static BivSample simulateBiv(double[] mu, double[][] alpha, double beta, double T,
            long seed) {
        checkBivParams(mu, alpha, beta);
        double disc = (alpha[0][0] - alpha[1][1]) * (alpha[0][0] - alpha[1][1])
                + 4.0 * alpha[0][1] * alpha[1][0];
        double rad = 0.5 * (alpha[0][0] + alpha[1][1] + Math.sqrt(disc)) / beta;
        if (rad >= 1.0) {
            throw new IllegalArgumentException(
                    "cannot simulate non-stationary process: spectral radius " + rad);
        }
        if (!(Double.isFinite(T) && T > 0.0)) {
            throw new IllegalArgumentException("T must be positive and finite, got " + T);
        }
        Lcg64 rng = new Lcg64(seed);
        double t = 0.0;
        double[] s = {0.0, 0.0};
        List<Double> times = new ArrayList<>();
        List<Integer> types = new ArrayList<>();
        while (true) {
            double lam0 = mu[0] + alpha[0][0] * s[0] + alpha[0][1] * s[1];
            double lam1 = mu[1] + alpha[1][0] * s[0] + alpha[1][1] * s[1];
            double lamBar = lam0 + lam1;
            double w = rng.exponential(lamBar);
            t += w;
            if (t > T) {
                break;
            }
            double decay = Math.exp(-beta * w);
            s[0] *= decay;
            s[1] *= decay;
            lam0 = mu[0] + alpha[0][0] * s[0] + alpha[0][1] * s[1];
            lam1 = mu[1] + alpha[1][0] * s[0] + alpha[1][1] * s[1];
            double d = rng.uniform() * lamBar;
            if (d <= lam0 + lam1) {
                int typ = d <= lam0 ? 0 : 1;
                times.add(t);
                types.add(typ);
                s[typ] += 1.0;
            }
        }
        return new BivSample(
                times.stream().mapToDouble(Double::doubleValue).toArray(),
                types.stream().mapToInt(Integer::intValue).toArray());
    }

    // -------------------------------------------------------------------- MLE

    private static String branchingWarning(double value) {
        if (value >= 1.0) {
            return String.format(
                    "fitted Hawkes process is non-stationary (branching %.4f >= 1)", value);
        }
        if (value >= 0.9) {
            return String.format(
                    "fitted Hawkes process is near-critical (branching %.4f >= 0.9)", value);
        }
        return null;
    }

    /**
     * Univariate MLE via Nelder-Mead on log-transformed parameters.
     *
     * <p>Pinned setup: minimize {@code -ll(exp(x))} over
     * {@code x = (log mu, log alpha, log beta)} from
     * {@code x0 = (log(0.5 N / T), log 0.5, 0)} with the pinned NM defaults.
     * The log transform enforces positivity without constraints. A
     * near-critical (&ge; 0.9) or non-stationary (&ge; 1) fitted branching
     * ratio is reported in {@code warning}; the fit still returns.</p>
     */
    public static UniFit fitUni(double[] times, double T) {
        checkTimes(times, T);
        final double[] t = times.clone();

        ToDoubleFunction<double[]> objective = x -> {
            double mu = Math.exp(x[0]);
            double alpha = Math.exp(x[1]);
            double beta = Math.exp(x[2]);
            if (!Double.isFinite(mu) || !Double.isFinite(alpha) || !Double.isFinite(beta)
                    || mu <= 0.0 || beta <= 0.0) {
                return PENALTY;
            }
            double val = uniLoglik(t, T, mu, alpha, beta);
            return Double.isFinite(val) ? -val : PENALTY;
        };

        double[] x0 = {Math.log(0.5 * t.length / T), Math.log(0.5), 0.0};
        NelderMead.Result res = NelderMead.minimize(objective, x0);
        double mu = Math.exp(res.x()[0]);
        double alpha = Math.exp(res.x()[1]);
        double beta = Math.exp(res.x()[2]);
        double branching = alpha / beta;
        return new UniFit(mu, alpha, beta, -res.fval(), branching, res.converged(),
                res.nIter(), branchingWarning(branching));
    }

    /**
     * Spectral radius of a 2x2 matrix via the pinned closed form: with trace
     * {@code tr} and determinant {@code det}, the eigenvalues are
     * {@code (tr +/- sqrt(tr^2 - 4 det)) / 2}; when the discriminant is
     * negative the complex pair has modulus {@code sqrt(det)}.
     *
     * @throws IllegalArgumentException on a non-finite or non-2x2 matrix
     */
    public static double spectralRadius2x2(double[][] matrix) {
        if (matrix == null || matrix.length != 2 || matrix[0].length != 2
                || matrix[1].length != 2) {
            throw new IllegalArgumentException("matrix must be a finite 2x2 array");
        }
        for (double[] row : matrix) {
            for (double v : row) {
                if (!Double.isFinite(v)) {
                    throw new IllegalArgumentException("matrix must be a finite 2x2 array");
                }
            }
        }
        double tr = matrix[0][0] + matrix[1][1];
        double det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        double disc = tr * tr - 4.0 * det;
        if (disc >= 0.0) {
            double r = Math.sqrt(disc);
            return Math.max(Math.abs(0.5 * (tr + r)), Math.abs(0.5 * (tr - r)));
        }
        return Math.sqrt(det);
    }

    /**
     * Bivariate MLE (full 2x2 alpha, scalar beta) via pinned Nelder-Mead.
     *
     * <p>Pinned setup: parameters {@code x = (log mu0, log mu1, log a00,
     * log a01, log a10, log a11, log beta)}; start {@code mu_m = 0.5 N_m / T},
     * all {@code a = 0.25}, {@code beta = 1}; two chained Nelder-Mead stages
     * (the second restarts from the first stage's optimum) — a deterministic
     * polish that guards against premature simplex collapse in 7 dimensions.
     * Reports {@code warning} when the branching spectral radius is &ge; 0.9
     * (near-critical) or &ge; 1 (non-stationary).</p>
     *
     * @throws IllegalArgumentException if either event type is absent
     */
    public static BivFit fitBiv(double[] times, int[] types, double T) {
        checkTimes(times, T);
        checkTypes(types, times.length);
        int n0 = 0;
        for (int k : types) {
            if (k == 0) {
                n0++;
            }
        }
        int n1 = times.length - n0;
        if (n0 == 0 || n1 == 0) {
            throw new IllegalArgumentException(
                    "both event types must be present to fit the bivariate model");
        }
        final double[] t = times.clone();
        final int[] k = types.clone();

        ToDoubleFunction<double[]> objective = x -> {
            double[] p = new double[7];
            for (int i = 0; i < 7; i++) {
                p[i] = Math.exp(x[i]);
                if (!Double.isFinite(p[i])) {
                    return PENALTY;
                }
            }
            if (p[0] <= 0.0 || p[1] <= 0.0 || p[6] <= 0.0) {
                return PENALTY; // exp underflow guard: keep the objective pure
            }
            double[] mu = {p[0], p[1]};
            double[][] alpha = {{p[2], p[3]}, {p[4], p[5]}};
            double val = bivLoglik(t, k, T, mu, alpha, p[6]);
            return Double.isFinite(val) ? -val : PENALTY;
        };

        double[] x0 = {
            Math.log(0.5 * n0 / T), Math.log(0.5 * n1 / T),
            Math.log(0.25), Math.log(0.25), Math.log(0.25), Math.log(0.25),
            0.0,
        };
        NelderMead.Result res1 = NelderMead.minimize(objective, x0);
        NelderMead.Result res2 = NelderMead.minimize(objective, res1.x());
        double[] p = new double[7];
        for (int i = 0; i < 7; i++) {
            p[i] = Math.exp(res2.x()[i]);
        }
        double[] mu = {p[0], p[1]};
        double[][] alpha = {{p[2], p[3]}, {p[4], p[5]}};
        double beta = p[6];
        double[][] branching = {
            {alpha[0][0] / beta, alpha[0][1] / beta},
            {alpha[1][0] / beta, alpha[1][1] / beta},
        };
        double rad = spectralRadius2x2(branching);
        return new BivFit(mu, alpha, beta, -res2.fval(), branching, rad,
                res2.converged(), res1.nIter() + res2.nIter(), branchingWarning(rad));
    }
}
