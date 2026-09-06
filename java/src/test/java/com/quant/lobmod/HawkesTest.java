package com.quant.lobmod;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.function.ToDoubleFunction;
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
        assertTrue(fitB.converged());
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
        assertTrue(fit.converged());
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
            assertTrue("compensator must be strictly increasing at events", v > 0.0);
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
                assertTrue(v > 0.0);
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

    // ------------------------------------------------------- review additions

    private static Path scratch(String name, String text) {
        try {
            Path p = Files.createTempDirectory("lobmod_h").resolve(name);
            Files.writeString(p, text);
            return p;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Test
    public void fitObjectivesArePureUnderUnderflow() {
        // PT-6
        double[] t = TestData.bivTimes();
        int[] k = TestData.bivTypes();
        ToDoubleFunction<double[]> obj = Hawkes.bivFitObjective(t, k, 600.0);
        assertEquals(1e10, obj.applyAsDouble(new double[] {-800, -800, 0, 0, 0, 0, -800}), 0.0);
        assertEquals(1e10, obj.applyAsDouble(new double[] {-800, 0, 0, 0, 0, 0, 0}), 0.0);
        assertEquals(1e10, obj.applyAsDouble(new double[] {0, 0, 0, 0, 0, 0, -800}), 0.0);
        assertEquals(1e10, obj.applyAsDouble(new double[] {0, 0, 0, 0, 0, 0, 800}), 0.0);
        double mu0 = 0.5 * 323 / 600.0;
        double mu1 = 0.5 * 288 / 600.0;
        double[] x0 = {Math.log(mu0), Math.log(mu1), Math.log(0.25), Math.log(0.25),
            Math.log(0.25), Math.log(0.25), 0.0};
        double want = -Hawkes.bivLoglik(t, k, 600.0, new double[] {mu0, mu1},
                new double[][] {{0.25, 0.25}, {0.25, 0.25}}, 1.0);
        assertEquals(want, obj.applyAsDouble(x0), 1e-12);
        NelderMead.Result res = NelderMead.minimize(obj,
                new double[] {-800, -800, 0, 0, 0, 0, -800}, 1e-8, 1e-10, 5);
        assertTrue(Double.isFinite(res.fval()));
        double[] tu = TestData.uniTimes();
        ToDoubleFunction<double[]> uobj = Hawkes.uniFitObjective(tu, 1000.0);
        assertEquals(1e10, uobj.applyAsDouble(new double[] {-800, 0, 0}), 0.0);
        assertEquals(1e10, uobj.applyAsDouble(new double[] {0, 0, -800}), 0.0);
        assertEquals(-Hawkes.uniLoglik(tu, 1000.0, 1.0, 0.0, 1.0),
                uobj.applyAsDouble(new double[] {0, -800, 0}), 1e-12);
        Hawkes.BivFit capped = Hawkes.fitBiv(t, k, 600.0, 1);
        assertFalse(capped.converged());
        assertEquals(2, capped.nIter()); // two stages x 1 iteration
        Hawkes.UniFit cappedU = Hawkes.fitUni(tu, 1000.0, 1);
        assertFalse(cappedU.converged());
        assertEquals(1, cappedU.nIter());
        assertThrows(IllegalArgumentException.class, () -> Hawkes.fitUni(tu, 1000.0, 0));
    }

    @Test
    public void poissonLimitOfFit() {
        // PT-8: alpha = 0 is a homogeneous Poisson process; the MLE must find it
        double T = 2000.0;
        double[] t = Hawkes.simulateUni(1.0, 0.0, 1.0, T, 5);
        assertTrue(t.length > 1500);
        Hawkes.UniFit fit = Hawkes.fitUni(t, T);
        assertTrue(fit.converged());
        assertTrue(fit.branching() < 0.05);
        double n = t.length;
        double poissonLl = Hawkes.uniLoglik(t, T, n / T, 0.0, 1.0);
        assertTrue(fit.loglik() >= poissonLl - 1e-6);
        assertEquals(n * Math.log(n / T) - n, poissonLl, 1e-9);
        assertEquals(Hawkes.uniLoglik(t, T, 0.7, 0.0, 1.0),
                Hawkes.uniLoglik(t, T, 0.7, 0.0, 3.3), 0.0);
    }

    @Test
    public void singleEventPaths() {
        // PT-9
        double mu = 0.5;
        double alpha = 0.8;
        double beta = 1.5;
        double[] t = {0.7};
        double expected = Math.log(mu) - mu + (alpha / beta) * Math.expm1(-beta * 0.3);
        assertEquals(expected, Hawkes.uniLoglik(t, 1.0, mu, alpha, beta), 1e-15);
        assertEquals(expected, Hawkes.uniLoglikBrute(t, 1.0, mu, alpha, beta), 1e-15);
        double[] tau = Hawkes.uniResiduals(t, 1.0, mu, alpha, beta);
        assertEquals(1, tau.length);
        assertEquals(mu * 0.7, tau[0], 1e-15);
        Gof.KsResult ks = Gof.ksTestExp(tau);
        assertTrue(ks.stat() > 0.0 && ks.stat() < 1.0 && ks.pValue() > 0.0 && ks.pValue() <= 1.0);
        Hawkes.UniFit fit = Hawkes.fitUni(t, 1.0);
        assertTrue(Double.isFinite(fit.loglik()) && fit.mu() > 0.0 && fit.nIter() >= 1);
        double[] tT = {1.0}; // event exactly at the horizon: expm1(0) = 0
        assertEquals(Math.log(mu) - mu, Hawkes.uniLoglik(tT, 1.0, mu, alpha, beta), 1e-15);
        assertEquals(mu, Hawkes.uniResiduals(tT, 1.0, mu, alpha, beta)[0], 1e-15);
        assertThrows(IllegalArgumentException.class, () -> Hawkes.fitBiv(tT, new int[] {0}, 1.0));
        assertEquals(Math.log(0.2) - 0.5, Hawkes.bivLoglik(tT, new int[] {0}, 1.0,
                new double[] {0.2, 0.3}, new double[][] {{0.1, 0.1}, {0.1, 0.1}}, 2.0), 1e-15);
    }

    @Test
    public void timesCsvRoundTripAndRobustness() {
        double[] tu = TestData.uniTimes();
        Path p = scratch("u.csv", "");
        Hawkes.writeTimesCsv(p, tu);
        assertArrayEquals(tu, Hawkes.readTimesCsv(p), 0.0);
        Path q = scratch("b.csv", "");
        Hawkes.writeMarkedTimesCsv(q, TestData.bivTimes(), TestData.bivTypes());
        Hawkes.BivSample back = Hawkes.readMarkedTimesCsv(q);
        assertArrayEquals(TestData.bivTimes(), back.times(), 0.0);
        assertArrayEquals(TestData.bivTypes(), back.types());

        String body = "t\n0.5\n1.25e0\n";
        double[] want = {0.5, 1.25};
        assertArrayEquals(want, Hawkes.readTimesCsv(scratch("crlf.csv", body.replace("\n", "\r\n"))), 0.0);
        assertArrayEquals(want, Hawkes.readTimesCsv(scratch("trail.csv", body + "\n")), 0.0);
        String[][] bad = {
            {"nan.csv", body + "nan\n"}, {"inf.csv", body + "Infinity\n"},
            {"two.csv", body + "1.5,0\n"}, {"word.csv", body + "abc\n"},
            {"hdr.csv", "time\n0.5\n"}, {"blank.csv", "t\n0.5\n\n1.0\n"},
            {"space.csv", "t\n 0.5\n"},
        };
        for (String[] c : bad) {
            Path bp = scratch(c[0], c[1]);
            assertThrows(c[0], IllegalArgumentException.class, () -> Hawkes.readTimesCsv(bp));
        }
        String mbody = "t,type\n0.5,0\n1.0,1\n";
        Hawkes.BivSample m = Hawkes.readMarkedTimesCsv(scratch("m.csv", mbody));
        assertArrayEquals(new double[] {0.5, 1.0}, m.times(), 0.0);
        assertArrayEquals(new int[] {0, 1}, m.types());
        String[][] mbad = {
            {"type2.csv", mbody + "1.5,2\n"}, {"typef.csv", mbody + "1.5,1.0\n"},
            {"one.csv", mbody + "1.5\n"}, {"three.csv", mbody + "1.5,1,\n"},
            {"mhdr.csv", "type,t\n0,0.5\n"},
        };
        for (String[] c : mbad) {
            Path bp = scratch(c[0], c[1]);
            assertThrows(c[0], IllegalArgumentException.class, () -> Hawkes.readMarkedTimesCsv(bp));
        }
        Path x = scratch("x.csv", "");
        assertThrows(IllegalArgumentException.class,
                () -> Hawkes.writeTimesCsv(x, new double[] {1.0, Double.NaN}));
        assertThrows(IllegalArgumentException.class,
                () -> Hawkes.writeMarkedTimesCsv(x, new double[] {1.0, 2.0}, new int[] {0, 2}));
    }
}
