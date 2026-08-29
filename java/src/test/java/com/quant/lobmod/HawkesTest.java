package com.quant.lobmod;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import org.junit.Test;

/**
 * Hawkes process tests: recursion vs brute force, simulation determinism
 * (including exact reproduction of the bundled samples), refits close to the
 * data-generating parameters, near-critical warnings, residual properties
 * and validation errors.
 */
public class HawkesTest {

    private static double[] smallTimes() {
        // deterministic small sample on [0, 10]
        return new double[] {0.5, 1.1, 1.15, 2.4, 3.9, 4.0, 6.5, 9.8};
    }

    @Test
    public void uniRecursionMatchesBruteForce() {
        double[] t = smallTimes();
        for (double beta : new double[] {0.5, 1.0, 2.5}) {
            double ll = Hawkes.uniLoglik(t, 10.0, 0.7, 0.4, beta);
            assertEquals(Hawkes.uniLoglikBrute(t, 10.0, 0.7, 0.4, beta), ll, 1e-10);
        }
        // and on the full bundled sample
        assertEquals(Hawkes.uniLoglikBrute(TestData.uniTimes(), 1000.0, 0.5, 1.0, 1.6),
                Hawkes.uniLoglik(TestData.uniTimes(), 1000.0, 0.5, 1.0, 1.6), 1e-10);
    }

    @Test
    public void bivRecursionMatchesBruteForce() {
        double[] t = smallTimes();
        int[] k = {0, 1, 0, 0, 1, 1, 0, 1};
        double[] mu = {0.3, 0.4};
        double[][] a = {{0.5, 0.2}, {0.1, 0.6}};
        assertEquals(Hawkes.bivLoglikBrute(t, k, 10.0, mu, a, 1.7),
                Hawkes.bivLoglik(t, k, 10.0, mu, a, 1.7), 1e-10);
        assertEquals(Hawkes.bivLoglikBrute(TestData.bivTimes(), TestData.bivTypes(), 600.0,
                        new double[] {0.3, 0.3}, new double[][] {{0.6, 0.3}, {0.25, 0.5}}, 2.0),
                Hawkes.bivLoglik(TestData.bivTimes(), TestData.bivTypes(), 600.0,
                        new double[] {0.3, 0.3}, new double[][] {{0.6, 0.3}, {0.25, 0.5}}, 2.0),
                1e-10);
    }

    @Test
    public void simulateUniReproducesBundledSample() {
        // dgp_params.json: mu=0.5, alpha=1.0, beta=1.6, T=1000, seed=101;
        // the CSV is %.17g so doubles round-trip exactly.
        double[] sim = Hawkes.simulateUni(0.5, 1.0, 1.6, 1000.0, 101);
        assertArrayEquals(TestData.uniTimes(), sim, 0.0);
        assertEquals(1211, sim.length);
    }

    @Test
    public void simulateBivReproducesBundledSample() {
        Hawkes.BivSample sim = Hawkes.simulateBiv(new double[] {0.3, 0.3},
                new double[][] {{0.6, 0.3}, {0.25, 0.5}}, 2.0, 600.0, 33);
        assertArrayEquals(TestData.bivTimes(), sim.times(), 0.0);
        assertArrayEquals(TestData.bivTypes(), sim.types());
        assertEquals(611, sim.times().length);
    }

    @Test
    public void simulationDeterministicPerSeed() {
        double[] a = Hawkes.simulateUni(0.4, 0.5, 1.0, 200.0, 7);
        double[] b = Hawkes.simulateUni(0.4, 0.5, 1.0, 200.0, 7);
        double[] c = Hawkes.simulateUni(0.4, 0.5, 1.0, 200.0, 8);
        assertArrayEquals(a, b, 0.0);
        assertTrue(a.length > 0 && !Arrays.equals(a, c));
        for (int i = 1; i < a.length; i++) {
            assertTrue(a[i] > a[i - 1]); // strictly increasing, within horizon
        }
        assertTrue(a[a.length - 1] <= 200.0);
    }

    @Test
    public void refitRecoversDgpParamsLoosely() {
        Hawkes.UniFit fit = TestData.uniFit();
        assertTrue(fit.converged());
        assertEquals(0.5, fit.mu(), 0.15);   // true mu
        assertEquals(1.0, fit.alpha(), 0.2); // true alpha
        assertEquals(1.6, fit.beta(), 0.3);  // true beta
        assertNull(fit.warning());           // branching 0.58: unremarkable
        Hawkes.BivFit fitB = TestData.bivFit();
        assertTrue(fitB.spectralRadius() < 1.0);
        assertEquals(0.3, fitB.mu()[0], 0.15);
        assertEquals(0.3, fitB.mu()[1], 0.15);
        assertEquals(2.0, fitB.beta(), 0.5);
        assertNull(fitB.warning());
    }

    @Test
    public void nearCriticalFitWarnsButReturns() {
        // pinned near-critical DGP: branching 0.95
        double[] t = Hawkes.simulateUni(0.2, 1.9, 2.0, 150.0, 999);
        assertTrue(t.length > 50);
        Hawkes.UniFit fit = Hawkes.fitUni(t, 150.0);
        assertTrue("branching should be near-critical, got " + fit.branching(),
                fit.branching() >= 0.9);
        assertNotNull(fit.warning());
        assertTrue(fit.warning().contains("near-critical")
                || fit.warning().contains("non-stationary"));
    }

    @Test
    public void residualsAtTrueParamsBehaveLikeExp1() {
        double[] tau = Hawkes.uniResiduals(TestData.uniTimes(), 1000.0, 0.5, 1.0, 1.6);
        double sum = 0.0;
        for (double v : tau) {
            assertTrue("compensator must be nondecreasing", v >= 0.0);
            sum += v;
        }
        assertEquals(1.0, sum / tau.length, 0.1); // Exp(1) mean
        Gof.KsResult ks = Gof.ksTestExp(tau);
        assertTrue("true params should not be rejected, p=" + ks.pValue(), ks.pValue() > 0.05);
        // wrong params get rejected
        double[] tauWrong = Hawkes.uniResiduals(TestData.uniTimes(), 1000.0, 1.5, 0.1, 3.0);
        assertTrue(Gof.ksTestExp(tauWrong).pValue() < 0.05);
    }

    @Test
    public void bivResidualsAtTrueParamsBehaveLikeExp1() {
        double[][] taus = Hawkes.bivResiduals(TestData.bivTimes(), TestData.bivTypes(), 600.0,
                new double[] {0.3, 0.3}, new double[][] {{0.6, 0.3}, {0.25, 0.5}}, 2.0);
        assertEquals(TestData.bivTimes().length, taus[0].length + taus[1].length);
        for (double[] tau : taus) {
            double sum = 0.0;
            for (double v : tau) {
                assertTrue(v >= 0.0);
                sum += v;
            }
            assertEquals(1.0, sum / tau.length, 0.15);
            assertTrue(Gof.ksTestExp(tau).pValue() > 0.05);
        }
    }

    @Test
    public void spectralRadiusClosedForm() {
        assertEquals(2.0, Hawkes.spectralRadius2x2(new double[][] {{2, 0}, {0, 1}}), 1e-15);
        // complex pair: rotation matrix has |lambda| = sqrt(det) = 1
        assertEquals(1.0, Hawkes.spectralRadius2x2(new double[][] {{0, -1}, {1, 0}}), 1e-15);
        // dominant negative eigenvalue is picked by absolute value
        assertEquals(3.0, Hawkes.spectralRadius2x2(new double[][] {{-3, 0}, {0, 1}}), 1e-15);
        assertThrows(IllegalArgumentException.class,
                () -> Hawkes.spectralRadius2x2(new double[][] {{Double.NaN, 0}, {0, 1}}));
    }

    @Test
    public void validationErrors() {
        double[] ok = {1.0, 2.0};
        assertThrows(IllegalArgumentException.class,
                () -> Hawkes.uniLoglik(new double[0], 10.0, 0.5, 0.5, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> Hawkes.uniLoglik(new double[] {2.0, 1.0}, 10.0, 0.5, 0.5, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> Hawkes.uniLoglik(new double[] {0.0, 1.0}, 10.0, 0.5, 0.5, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> Hawkes.uniLoglik(new double[] {1.0, 11.0}, 10.0, 0.5, 0.5, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> Hawkes.uniLoglik(new double[] {1.0, Double.NaN}, 10.0, 0.5, 0.5, 1.0));
        assertThrows(IllegalArgumentException.class, () -> Hawkes.uniLoglik(ok, 10.0, 0.0, 0.5, 1.0));
        assertThrows(IllegalArgumentException.class, () -> Hawkes.uniLoglik(ok, 10.0, 0.5, -0.1, 1.0));
        assertThrows(IllegalArgumentException.class, () -> Hawkes.uniLoglik(ok, 10.0, 0.5, 0.5, 0.0));
        assertThrows(IllegalArgumentException.class, () -> Hawkes.uniLoglik(ok, -1.0, 0.5, 0.5, 1.0));
        // non-stationary simulation refused
        assertThrows(IllegalArgumentException.class,
                () -> Hawkes.simulateUni(0.5, 2.0, 1.0, 10.0, 1));
        assertThrows(IllegalArgumentException.class,
                () -> Hawkes.simulateBiv(new double[] {0.5, 0.5},
                        new double[][] {{2.0, 0.0}, {0.0, 2.0}}, 1.0, 10.0, 1));
        // bivariate types misaligned / invalid / single-type
        assertThrows(IllegalArgumentException.class,
                () -> Hawkes.bivLoglik(ok, new int[] {0}, 10.0, new double[] {0.5, 0.5},
                        new double[][] {{0.1, 0.1}, {0.1, 0.1}}, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> Hawkes.bivLoglik(ok, new int[] {0, 2}, 10.0, new double[] {0.5, 0.5},
                        new double[][] {{0.1, 0.1}, {0.1, 0.1}}, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> Hawkes.fitBiv(ok, new int[] {0, 0}, 10.0));
    }
}
