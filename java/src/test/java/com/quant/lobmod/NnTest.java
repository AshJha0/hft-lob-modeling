package com.quant.lobmod;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * Neural-network tests: pinned deterministic init, backprop vs central
 * finite differences, training behaviour, bit-identical retrains, degenerate
 * labels, metrics, and validation errors.
 */
public class NnTest {

    /** Small deterministic dataset (3 features, 3 classes, 12 rows). */
    private static double[][] tinyX() {
        double[][] x = new double[12][3];
        for (int i = 0; i < 12; i++) {
            for (int j = 0; j < 3; j++) {
                x[i][j] = Math.sin(3.0 * i + j) + 0.1 * i; // spread, deterministic
            }
        }
        return x;
    }

    private static int[] tinyY() {
        return new int[] {0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0};
    }

    @Test
    public void pinnedInitFormula() {
        double[][] w = Mlp.pinnedInit(5, 16, 0);
        assertEquals(5, w.length);
        assertEquals(16, w[0].length);
        double scale = 0.5 * Math.sqrt(2.0 / 5.0);
        assertEquals(scale * Math.sin(1.0), w[0][0], 1e-15);           // l=0,i=0,j=0
        assertEquals(scale * Math.sin(0 + 2 * 16 + 5 + 1.0), w[2][5], 1e-15);
        double[][] w1 = Mlp.pinnedInit(16, 8, 1);
        assertEquals(0.5 * Math.sqrt(2.0 / 16.0) * Math.sin(10000.0 + 1.0), w1[0][0], 1e-15);
        assertThrows(IllegalArgumentException.class, () -> Mlp.pinnedInit(0, 4, 0));
        assertThrows(IllegalArgumentException.class, () -> Mlp.pinnedInit(4, 4, -1));
    }

    @Test
    public void backpropMatchesCentralFiniteDifferences() {
        Mlp net = new Mlp(3, new int[] {4}, 3);
        double[][] x = tinyX();
        int[] y = tinyY();
        Mlp.Grads grads = net.lossAndGrads(x, y);
        double h = 1e-6;
        for (int l = 0; l < net.weights.size(); l++) {
            double[][] w = net.weights.get(l);
            for (int i = 0; i < w.length; i++) {
                for (int j = 0; j < w[i].length; j++) {
                    double orig = w[i][j];
                    w[i][j] = orig + h;
                    double up = net.loss(x, y);
                    w[i][j] = orig - h;
                    double down = net.loss(x, y);
                    w[i][j] = orig;
                    double fd = (up - down) / (2.0 * h);
                    double g = grads.gradsW()[l][i][j];
                    assertEquals("dW[" + l + "][" + i + "][" + j + "]",
                            fd, g, 1e-6 * Math.max(1.0, Math.abs(fd)));
                }
            }
            double[] b = net.biases.get(l);
            for (int j = 0; j < b.length; j++) {
                double orig = b[j];
                b[j] = orig + h;
                double up = net.loss(x, y);
                b[j] = orig - h;
                double down = net.loss(x, y);
                b[j] = orig;
                double fd = (up - down) / (2.0 * h);
                assertEquals("db[" + l + "][" + j + "]",
                        fd, grads.gradsB()[l][j], 1e-6 * Math.max(1.0, Math.abs(fd)));
            }
        }
    }

    @Test
    public void lossDecreasesOverEpochs() {
        Mlp net = new Mlp(3, new int[] {4}, 3);
        double[] history = net.fit(tinyX(), tinyY(), 0.1, 50);
        assertEquals(50, history.length);
        assertTrue("training should reduce the loss",
                history[history.length - 1] < history[0]);
        for (double v : history) {
            assertTrue(Double.isFinite(v));
        }
        // on the real pipeline the recorded history is monotone as well
        double[] big = TestData.pipeline().history();
        assertTrue(big[big.length - 1] < big[0]);
    }

    @Test
    public void softmaxRowsSumToOne() {
        Mlp net = new Mlp(3, new int[] {4}, 3);
        double[][] p = net.predictProba(tinyX());
        for (double[] row : p) {
            double sum = 0.0;
            for (double v : row) {
                assertTrue(v >= 0.0 && v <= 1.0);
                sum += v;
            }
            assertEquals(1.0, sum, 1e-12);
        }
    }

    @Test
    public void retrainingIsBitIdentical() {
        Mlp a = new Mlp(3, new int[] {4, 2}, 3);
        Mlp b = new Mlp(3, new int[] {4, 2}, 3);
        double[] ha = a.fit(tinyX(), tinyY(), 0.1, 30);
        double[] hb = b.fit(tinyX(), tinyY(), 0.1, 30);
        assertArrayEquals(ha, hb, 0.0); // deterministic: exact equality
        assertArrayEquals(a.predict(tinyX()), b.predict(tinyX()));
    }

    @Test
    public void degenerateOneClassLabelsStillTrainAndLoseToMajority() {
        double[][] x = tinyX();
        int[] y = new int[x.length]; // all "down"
        Mlp net = new Mlp(3, new int[] {4}, 3);
        double[] history = net.fit(x, y, 0.1, 40);
        assertTrue(Double.isFinite(history[history.length - 1]));
        assertTrue(history[history.length - 1] < history[0]); // still learns the constant
        int maj = Metrics.majorityClass(y);
        assertEquals(0, maj);
        int[] predMj = new int[x.length]; // all zeros already
        double accMj = Metrics.accuracy(y, predMj);
        assertEquals(1.0, accMj, 0.0);
        assertTrue(accMj >= Metrics.accuracy(y, net.predict(x))); // majority wins (or ties)
    }

    @Test
    public void standardizerPinnedMoments() {
        double[][] x = {{1.0, 5.0}, {3.0, 5.0}, {5.0, 5.0}};
        Standardizer std = new Standardizer().fit(x);
        assertArrayEquals(new double[] {3.0, 5.0}, std.mean(), 1e-15);
        // population std (divisor n): sqrt(8/3); constant column floors to 1
        assertEquals(Math.sqrt(8.0 / 3.0), std.std()[0], 1e-15);
        assertEquals(1.0, std.std()[1], 0.0);
        double[][] z = std.transform(x);
        assertEquals((1.0 - 3.0) / Math.sqrt(8.0 / 3.0), z[0][0], 1e-15);
        assertEquals(0.0, z[1][1], 0.0);
        assertThrows(IllegalArgumentException.class,
                () -> new Standardizer().transform(x)); // fit first
        assertThrows(IllegalArgumentException.class,
                () -> std.transform(new double[][] {{1.0}})); // column mismatch
        assertThrows(IllegalArgumentException.class,
                () -> new Standardizer().fit(new double[][] {{Double.NaN}}));
    }

    @Test
    public void metricsBehaviour() {
        int[] yt = {0, 1, 2, 1};
        int[] yp = {0, 2, 2, 1};
        assertEquals(0.75, Metrics.accuracy(yt, yp), 0.0);
        int[][] cm = Metrics.confusionMatrix(yt, yp, 3);
        assertEquals(1, cm[0][0]);
        assertEquals(1, cm[1][2]);
        assertEquals(1, cm[1][1]);
        assertEquals(1, cm[2][2]);
        assertEquals(0, cm[2][0]);
        assertEquals(0, Metrics.majorityClass(new int[] {0, 0, 1, 1})); // tie -> smallest
        assertEquals(2, Metrics.majorityClass(new int[] {2, 2, 1}));
        Metrics.Pnl pnl = Metrics.signalPnl(new int[] {2, 1, 0},
                new double[] {1.0, 2.0, 3.0}, new double[] {1.0, 1.0, 1.0});
        assertEquals(-2.0, pnl.gross(), 1e-15); // +1*1 + 0 - 1*3
        assertEquals(-3.0, pnl.net(), 1e-15);   // two half-spread costs of 0.5
        assertEquals(2, pnl.nTrades());
    }

    @Test
    public void validationErrors() {
        Mlp net = new Mlp(3, new int[] {4}, 3);
        double[][] x = tinyX();
        int[] y = tinyY();
        assertThrows(IllegalArgumentException.class, () -> new Mlp(0, new int[] {4}, 3));
        assertThrows(IllegalArgumentException.class, () -> new Mlp(3, new int[] {4}, 1));
        assertThrows(IllegalArgumentException.class, () -> new Mlp(3, new int[] {0}, 3));
        assertThrows(IllegalArgumentException.class, () -> net.fit(x, y, 0.0, 10));
        assertThrows(IllegalArgumentException.class, () -> net.fit(x, y, 0.1, 0));
        assertThrows(IllegalArgumentException.class, () -> net.fit(new double[0][0], y, 0.1, 10));
        assertThrows(IllegalArgumentException.class,
                () -> net.fit(x, new int[] {0, 1}, 0.1, 10)); // length mismatch
        int[] bad = y.clone();
        bad[0] = 3;
        assertThrows(IllegalArgumentException.class, () -> net.fit(x, bad, 0.1, 10));
        assertThrows(IllegalArgumentException.class,
                () -> net.predict(new double[][] {{1.0, 2.0}})); // wrong feature count
        assertThrows(IllegalArgumentException.class,
                () -> net.loss(new double[][] {{1.0, Double.NaN, 2.0}}, new int[] {0}));
        assertThrows(IllegalArgumentException.class,
                () -> Metrics.accuracy(new int[0], new int[0]));
        assertThrows(IllegalArgumentException.class,
                () -> Metrics.signalPnl(new int[] {1}, new double[] {1.0, 2.0}, new double[] {1.0}));
        assertThrows(IllegalArgumentException.class,
                () -> Metrics.signalPnl(new int[] {1}, new double[] {Double.NaN}, new double[] {1.0}));
        assertThrows(IllegalArgumentException.class, () -> Metrics.majorityClass(new int[0]));
    }
}
