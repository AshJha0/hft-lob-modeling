package com.quant.lobmod;

import java.util.Arrays;
import java.util.function.ToDoubleFunction;

/**
 * Native Nelder-Mead simplex minimizer (pinned scipy-compatible configuration).
 *
 * <p>Deterministic, derivative-free minimization of {@code f: R^n -> double}.
 * The configuration is pinned so every language reproduces the same
 * iterates:</p>
 * <ul>
 *   <li>Coefficients: reflection 1, expansion 2, contraction 0.5,
 *       shrink 0.5.</li>
 *   <li>Initial simplex (scipy's rule): vertex 0 is {@code x0}; vertex
 *       {@code i} perturbs coordinate {@code i-1} — multiply by 1.05 if
 *       nonzero, else set it to 0.00025.</li>
 *   <li>Iteration: order vertices by {@code f} (stable sort, key
 *       {@code (f_i, index)}), reflect the worst through the centroid of the
 *       rest, then expand / outside-contract / inside-contract / shrink
 *       following the Lagarias et al. (1998) decision tree.</li>
 *   <li>Termination checked <em>before</em> an iteration: max abs coordinate
 *       spread to the best vertex {@code < xatol} AND max abs {@code f}
 *       spread {@code < fatol}, or {@code maxiter} iterations.</li>
 * </ul>
 *
 * <p>Objectives must be pure and never throw: out-of-domain points should
 * return a large penalty value.</p>
 */
public final class NelderMead {

    /**
     * Nelder-Mead outcome.
     *
     * @param x best vertex found
     * @param fval objective value at {@code x}
     * @param nIter iterations performed
     * @param converged true if the tolerance test (not the iteration cap)
     *     stopped it
     */
    public record Result(double[] x, double fval, int nIter, boolean converged) {
    }

    private NelderMead() {
    }

    /**
     * Minimizes {@code f} from {@code x0} with the pinned defaults
     * {@code xatol = 1e-8}, {@code fatol = 1e-10}, {@code maxiter = 5000}
     * (used by all Hawkes fits).
     */
    public static Result minimize(ToDoubleFunction<double[]> f, double[] x0) {
        return minimize(f, x0, 1e-8, 1e-10, 5000);
    }

    /**
     * Minimizes {@code f} from {@code x0} with the pinned Nelder-Mead scheme.
     *
     * @param f pure objective; return a penalty for out-of-domain points
     * @param x0 finite, non-empty starting point
     * @param xatol coordinate-spread tolerance (&gt; 0)
     * @param fatol f-spread tolerance (&gt; 0)
     * @param maxiter iteration cap (&ge; 1)
     * @throws IllegalArgumentException on an empty or non-finite {@code x0}
     *     or non-positive tolerances
     */
    public static Result minimize(ToDoubleFunction<double[]> f, double[] x0,
            double xatol, double fatol, int maxiter) {
        int n = x0.length;
        if (n == 0) {
            throw new IllegalArgumentException("x0 must be non-empty");
        }
        for (double v : x0) {
            if (!Double.isFinite(v)) {
                throw new IllegalArgumentException("x0 must be finite");
            }
        }
        if (!(xatol > 0.0 && fatol > 0.0 && maxiter >= 1)) {
            throw new IllegalArgumentException("tolerances must be positive and maxiter >= 1");
        }

        // initial simplex (scipy rule)
        double[][] simplex = new double[n + 1][];
        simplex[0] = x0.clone();
        for (int i = 0; i < n; i++) {
            double[] v = x0.clone();
            v[i] = v[i] != 0.0 ? v[i] * 1.05 : 0.00025;
            simplex[i + 1] = v;
        }
        double[] fvals = new double[n + 1];
        for (int i = 0; i <= n; i++) {
            fvals[i] = f.applyAsDouble(simplex[i]);
        }

        int nIter = 0;
        boolean converged = false;
        while (nIter < maxiter) {
            sortSimplex(simplex, fvals);

            double xspread = 0.0;
            for (int i = 1; i <= n; i++) {
                for (int j = 0; j < n; j++) {
                    xspread = Math.max(xspread, Math.abs(simplex[i][j] - simplex[0][j]));
                }
            }
            double fspread = 0.0;
            for (int i = 1; i <= n; i++) {
                fspread = Math.max(fspread, Math.abs(fvals[i] - fvals[0]));
            }
            if (xspread < xatol && fspread < fatol) {
                converged = true;
                break;
            }
            nIter++;

            double[] centroid = new double[n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    centroid[j] += simplex[i][j];
                }
            }
            for (int j = 0; j < n; j++) {
                centroid[j] /= n;
            }
            double[] worst = simplex[n];
            double[] xr = new double[n];
            for (int j = 0; j < n; j++) {
                xr[j] = centroid[j] + (centroid[j] - worst[j]);
            }
            double fr = f.applyAsDouble(xr);

            if (fr < fvals[0]) {
                // try expansion
                double[] xe = new double[n];
                for (int j = 0; j < n; j++) {
                    xe[j] = centroid[j] + 2.0 * (centroid[j] - worst[j]);
                }
                double fe = f.applyAsDouble(xe);
                if (fe < fr) {
                    simplex[n] = xe;
                    fvals[n] = fe;
                } else {
                    simplex[n] = xr;
                    fvals[n] = fr;
                }
            } else if (fr < fvals[n - 1]) {
                simplex[n] = xr;
                fvals[n] = fr;
            } else {
                double[] xc = new double[n];
                double fc;
                boolean accept;
                if (fr < fvals[n]) {
                    // outside contraction
                    for (int j = 0; j < n; j++) {
                        xc[j] = centroid[j] + 0.5 * (xr[j] - centroid[j]);
                    }
                    fc = f.applyAsDouble(xc);
                    accept = fc <= fr;
                } else {
                    // inside contraction
                    for (int j = 0; j < n; j++) {
                        xc[j] = centroid[j] - 0.5 * (centroid[j] - worst[j]);
                    }
                    fc = f.applyAsDouble(xc);
                    accept = fc < fvals[n];
                }
                if (accept) {
                    simplex[n] = xc;
                    fvals[n] = fc;
                } else {
                    // shrink toward the best vertex
                    for (int i = 1; i <= n; i++) {
                        for (int j = 0; j < n; j++) {
                            simplex[i][j] = simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]);
                        }
                        fvals[i] = f.applyAsDouble(simplex[i]);
                    }
                }
            }
        }

        sortSimplex(simplex, fvals);
        return new Result(simplex[0].clone(), fvals[0], nIter, converged);
    }

    /** Stable ascending sort of the simplex vertices by {@code (f, index)}. */
    private static void sortSimplex(double[][] simplex, double[] fvals) {
        int m = fvals.length;
        Integer[] order = new Integer[m];
        for (int i = 0; i < m; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> {
            int c = Double.compare(fvals[a], fvals[b]);
            return c != 0 ? c : Integer.compare(a, b);
        });
        double[][] s2 = new double[m][];
        double[] f2 = new double[m];
        for (int i = 0; i < m; i++) {
            s2[i] = simplex[order[i]];
            f2[i] = fvals[order[i]];
        }
        System.arraycopy(s2, 0, simplex, 0, m);
        System.arraycopy(f2, 0, fvals, 0, m);
    }
}
