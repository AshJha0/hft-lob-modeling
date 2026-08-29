package com.quant.lobmod;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * KS-vs-Exp(1) tests: statistic definition, pinned p-value behaviour, tiny
 * samples, monotonicity in D, and validation errors — plus the pinned
 * Nelder-Mead minimizer on standard objectives.
 */
public class GofTest {

    @Test
    public void statisticOfPerfectQuantilesIsSmall() {
        // exponential quantiles at plotting positions (i - 0.5)/n
        int n = 1000;
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = -Math.log1p(-((i + 0.5) / n));
        }
        double d = Gof.ksStatisticExp(x);
        assertTrue("D should be ~1/(2n), got " + d, d <= 0.5 / n + 1e-12);
        assertTrue(Gof.ksTestExp(x).pValue() > 0.99); // 0.9938 with the pinned 100-term series
    }

    @Test
    public void tinySamplesWork() {
        double d1 = Gof.ksStatisticExp(new double[] {0.7});
        assertTrue(d1 >= 0.0 && d1 <= 1.0);
        double p1 = Gof.ksPvalue(d1, 1);
        assertTrue(p1 >= 0.0 && p1 <= 1.0);
        Gof.KsResult r3 = Gof.ksTestExp(new double[] {0.1, 0.9, 2.5});
        assertTrue(r3.stat() >= 0.0 && r3.stat() <= 1.0);
        assertTrue(r3.pValue() >= 0.0 && r3.pValue() <= 1.0);
    }

    @Test
    public void pvalueBoundsAndMonotonicityInD() {
        assertEquals(1.0, Gof.ksPvalue(0.0, 50), 0.0); // exactly 1 at D = 0
        double prev = 1.0;
        for (double d = 0.01; d <= 1.0; d += 0.01) { // property-style grid sweep
            double p = Gof.ksPvalue(d, 50);
            assertTrue("p out of [0,1]: " + p, p >= 0.0 && p <= 1.0);
            assertTrue("p must decrease as D grows", p <= prev + 1e-15);
            prev = p;
        }
        assertTrue(Gof.ksPvalue(0.9, 50) < 1e-10);
    }

    @Test
    public void validationErrors() {
        assertThrows(IllegalArgumentException.class, () -> Gof.ksStatisticExp(new double[0]));
        assertThrows(IllegalArgumentException.class,
                () -> Gof.ksStatisticExp(new double[] {0.5, -0.1}));
        assertThrows(IllegalArgumentException.class,
                () -> Gof.ksStatisticExp(new double[] {Double.NaN}));
        assertThrows(IllegalArgumentException.class, () -> Gof.ksPvalue(-0.1, 10));
        assertThrows(IllegalArgumentException.class, () -> Gof.ksPvalue(1.5, 10));
        assertThrows(IllegalArgumentException.class, () -> Gof.ksPvalue(0.5, 0));
    }

    // ------------------------------------------------------------ Nelder-Mead

    @Test
    public void nelderMeadMinimizesQuadratic() {
        NelderMead.Result res = NelderMead.minimize(
                x -> (x[0] - 1.0) * (x[0] - 1.0) + 2.0 * (x[1] + 3.0) * (x[1] + 3.0),
                new double[] {0.0, 0.0});
        assertTrue(res.converged());
        assertEquals(1.0, res.x()[0], 1e-6);
        assertEquals(-3.0, res.x()[1], 1e-6);
        assertEquals(0.0, res.fval(), 1e-10);
    }

    @Test
    public void nelderMeadMinimizesRosenbrock() {
        NelderMead.Result res = NelderMead.minimize(
                x -> {
                    double a = 1.0 - x[0];
                    double b = x[1] - x[0] * x[0];
                    return a * a + 100.0 * b * b;
                },
                new double[] {-1.2, 1.0});
        assertTrue(res.converged());
        assertEquals(1.0, res.x()[0], 1e-5);
        assertEquals(1.0, res.x()[1], 1e-5);
    }

    @Test
    public void nelderMeadIterationCapAndValidation() {
        NelderMead.Result res = NelderMead.minimize(
                x -> x[0] * x[0], new double[] {5.0}, 1e-8, 1e-10, 3);
        assertEquals(3, res.nIter());
        assertTrue(!res.converged()); // stopped by the cap, not the tolerance
        assertThrows(IllegalArgumentException.class,
                () -> NelderMead.minimize(x -> 0.0, new double[0]));
        assertThrows(IllegalArgumentException.class,
                () -> NelderMead.minimize(x -> 0.0, new double[] {Double.NaN}));
        assertThrows(IllegalArgumentException.class,
                () -> NelderMead.minimize(x -> 0.0, new double[] {1.0}, -1.0, 1e-10, 100));
        assertThrows(IllegalArgumentException.class,
                () -> NelderMead.minimize(x -> 0.0, new double[] {1.0}, 1e-8, 1e-10, 0));
    }
}
