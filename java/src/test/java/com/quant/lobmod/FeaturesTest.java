package com.quant.lobmod;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import org.junit.Test;

/**
 * Feature/label pipeline tests: the pinned OFI formula, row emission rules,
 * label classes, the chronological split, and config validation.
 */
public class FeaturesTest {

    /** Book states with constant sizes and a chosen mid path (spread 2). */
    private static BookStates statesFromMids(double[] mids) {
        int n = mids.length;
        double[] bid = new double[n];
        double[] ask = new double[n];
        double[] qb = new double[n];
        double[] qa = new double[n];
        double[] bd = new double[n];
        double[] ad = new double[n];
        double[] micro = new double[n];
        double[] spread = new double[n];
        for (int i = 0; i < n; i++) {
            bid[i] = mids[i] - 1.0;
            ask[i] = mids[i] + 1.0;
            qb[i] = 10.0;
            qa[i] = 10.0;
            bd[i] = 30.0;
            ad[i] = 30.0;
            micro[i] = mids[i];
            spread[i] = 2.0;
        }
        return new BookStates(bid, ask, qb, qa, bd, ad, mids.clone(), micro, spread);
    }

    private static FeatureConfig tinyConfig() {
        return new FeatureConfig(1, 1, 1, 0.25, 3, new int[] {}, 0.1, 10, 0.7);
    }

    @Test
    public void ofiPerEventFormula() {
        // equal best prices contribute the size change on each side
        BookStates st = new BookStates(
                new double[] {100, 100}, new double[] {102, 102},
                new double[] {5, 8}, new double[] {7, 4},
                new double[] {5, 8}, new double[] {7, 4},
                new double[] {101, 101}, new double[] {101, 101}, new double[] {2, 2});
        double[] e = Features.ofiSeries(st);
        assertTrue(Double.isNaN(e[0])); // no predecessor
        assertEquals(8 - 5 - 4 + 7, e[1], 1e-12);
        assertEquals(e[1], Features.ofiAt(st, 1, 1), 1e-12);
    }

    @Test
    public void ofiNanWhenSideEmpty() {
        BookStates st = new BookStates(
                new double[] {100, Double.NaN, 100}, new double[] {102, 102, 102},
                new double[] {5, 0, 5}, new double[] {7, 7, 7},
                new double[] {5, 0, 5}, new double[] {7, 7, 7},
                new double[] {101, Double.NaN, 101}, new double[] {101, Double.NaN, 101},
                new double[] {2, Double.NaN, 2});
        double[] e = Features.ofiSeries(st);
        assertTrue(Double.isNaN(e[1])); // empty side at event 1
        assertTrue(Double.isNaN(e[2])); // previous state had an empty side
    }

    @Test
    public void ofiWindowValidation() {
        BookStates st = TestData.states();
        assertThrows(IllegalArgumentException.class, () -> Features.ofiAt(st, 100, 0));
        assertThrows(IllegalArgumentException.class, () -> Features.ofiAt(st, -1, 10));
        assertThrows(IllegalArgumentException.class, () -> Features.ofiAt(st, st.size(), 10));
        assertThrows(IllegalArgumentException.class, () -> Features.ofiAt(st, 5, 10)); // no fit
    }

    @Test
    public void labelsFollowPinnedClasses() {
        // mids: at i the label looks at mid[i+1] - mid[i] with eps = 0.25
        double[] mids = {100.0, 100.0, 101.0, 100.0, 100.0};
        FeatureSet fs = Features.computeFeatures(statesFromMids(mids), tinyConfig());
        assertArrayEquals(new int[] {1, 2, 3}, fs.eventIndex());
        assertArrayEquals(new int[] {Features.LABEL_UP, Features.LABEL_DOWN, Features.LABEL_FLAT},
                fs.y());
        // forward return and spread carried for PnL
        assertArrayEquals(new double[] {1.0, -1.0, 0.0}, fs.fwdRet(), 1e-12);
        assertArrayEquals(new double[] {2.0, 2.0, 2.0}, fs.spread(), 1e-12);
        // feature columns: [ofi, qi, spread, depth_ratio, ret]
        assertEquals(0.0, fs.x()[0][1], 1e-12);  // balanced book -> QI 0
        assertEquals(2.0, fs.x()[0][2], 1e-12);  // spread
        assertEquals(1.0, fs.x()[0][3], 1e-12);  // symmetric depth ratio
        assertEquals(0.0, fs.x()[0][4], 1e-12);  // ret = mid_1 - mid_0
        assertEquals(1.0, fs.x()[1][4], 1e-12);  // ret = mid_2 - mid_1
    }

    @Test
    public void noValidRowsIsAnError() {
        double[] nan = new double[5];
        Arrays.fill(nan, Double.NaN);
        BookStates st = new BookStates(nan, nan, new double[5], new double[5],
                new double[5], new double[5], nan, nan, nan);
        assertThrows(IllegalArgumentException.class,
                () -> Features.computeFeatures(st, tinyConfig()));
    }

    @Test
    public void bundledPipelineRowCountsAndSplit() {
        TestData.Pipeline p = TestData.pipeline();
        assertEquals(4348, p.fs().size());
        assertEquals(3043, p.split().train().length); // floor(0.7 * 4348)
        assertEquals(1305, p.split().test().length);
        // rows keep replay order and the split is chronological
        int[] idx = p.fs().eventIndex();
        for (int i = 1; i < idx.length; i++) {
            assertTrue(idx[i] > idx[i - 1]);
        }
        assertEquals(0, p.split().train()[0]);
        assertEquals(3043, p.split().test()[0]);
    }

    @Test
    public void chronologicalSplitValidation() {
        Features.Split s = Features.chronologicalSplit(10, 0.7);
        assertEquals(7, s.train().length);
        assertEquals(3, s.test().length);
        assertThrows(IllegalArgumentException.class, () -> Features.chronologicalSplit(1, 0.7));
        assertThrows(IllegalArgumentException.class, () -> Features.chronologicalSplit(10, 0.0));
        assertThrows(IllegalArgumentException.class, () -> Features.chronologicalSplit(10, 1.0));
        assertThrows(IllegalArgumentException.class, () -> Features.chronologicalSplit(10, 0.05));
    }

    @Test
    public void configValidation() {
        FeatureConfig cfg = TestData.config();
        assertEquals(50, cfg.ofiWindow());
        assertEquals(0.25, cfg.epsilon(), 0.0);
        assertArrayEquals(new int[] {16, 8}, cfg.hidden());
        assertThrows(IllegalArgumentException.class,
                () -> new FeatureConfig(0, 20, 10, 0.25, 3, new int[] {16}, 0.1, 800, 0.7));
        assertThrows(IllegalArgumentException.class,
                () -> new FeatureConfig(50, 20, 10, -0.1, 3, new int[] {16}, 0.1, 800, 0.7));
        assertThrows(IllegalArgumentException.class,
                () -> new FeatureConfig(50, 20, 10, 0.25, 3, new int[] {16}, 0.0, 800, 0.7));
        assertThrows(IllegalArgumentException.class,
                () -> new FeatureConfig(50, 20, 10, 0.25, 3, new int[] {16}, 0.1, 800, 1.5));
        assertThrows(IllegalArgumentException.class,
                () -> new FeatureConfig(50, 20, 10, 0.25, 3, new int[] {0}, 0.1, 800, 0.7));
    }
}
