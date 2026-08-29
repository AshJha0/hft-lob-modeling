package com.quant.lobmod;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import org.junit.Test;

/**
 * Golden-value suite: recomputes every case in data/golden/golden.json and
 * compares within its pinned tolerance. The same cases are asserted by the
 * Python, C++ and Rust suites, which is what pins cross-language agreement.
 */
public class GoldenTest {

    private static void checkSnapshot(String name, int seq) {
        BookStates st = TestData.states();
        double tol = TestData.tol(name);
        int i = seq - 1;
        assertEquals(TestData.expect(name, "best_bid"), st.bestBid()[i], tol);
        assertEquals(TestData.expect(name, "best_ask"), st.bestAsk()[i], tol);
        assertEquals(TestData.expect(name, "bid_size"), st.bidSize()[i], tol);
        assertEquals(TestData.expect(name, "ask_size"), st.askSize()[i], tol);
        assertEquals(TestData.expect(name, "bid_depth_k"), st.bidDepthK()[i], tol);
        assertEquals(TestData.expect(name, "ask_depth_k"), st.askDepthK()[i], tol);
        assertEquals(TestData.expect(name, "mid"), st.mid()[i], tol);
    }

    @Test
    public void bookSnapshots() {
        checkSnapshot("book_snapshot_event_1000", 1000);
        checkSnapshot("book_snapshot_event_3000", 3000);
        checkSnapshot("book_snapshot_event_5000", 5000);
    }

    @Test
    public void microprice() {
        assertEquals(TestData.expect("microprice_event_2500", "microprice"),
                TestData.states().microprice()[2500 - 1], TestData.tol("microprice_event_2500"));
    }

    @Test
    public void ofi() {
        double ofi = Features.ofiAt(TestData.states(), 2200 - 1, 100);
        assertEquals(TestData.expect("ofi_event_2200_w100", "ofi"), ofi,
                TestData.tol("ofi_event_2200_w100"));
    }

    @Test
    public void hawkesUniLoglikPinned() {
        double ll = Hawkes.uniLoglik(TestData.uniTimes(), 1000.0, 0.4, 0.6, 1.2);
        assertEquals(TestData.expect("hawkes_uni_loglik_pinned", "loglik"), ll,
                TestData.tol("hawkes_uni_loglik_pinned"));
        // and the recursion agrees with brute force on the full bundled sample
        assertEquals(Hawkes.uniLoglikBrute(TestData.uniTimes(), 1000.0, 0.4, 0.6, 1.2), ll, 1e-10);
    }

    @Test
    public void hawkesUniFit() {
        Hawkes.UniFit fit = TestData.uniFit();
        double tol = TestData.tol("hawkes_uni_fit");
        assertEquals(TestData.expect("hawkes_uni_fit", "mu"), fit.mu(), tol);
        assertEquals(TestData.expect("hawkes_uni_fit", "alpha"), fit.alpha(), tol);
        assertEquals(TestData.expect("hawkes_uni_fit", "beta"), fit.beta(), tol);
        assertEquals(TestData.expect("hawkes_uni_branching", "branching"), fit.branching(),
                TestData.tol("hawkes_uni_branching"));
    }

    @Test
    public void hawkesUniKsFitted() {
        Hawkes.UniFit fit = TestData.uniFit();
        double[] tau = Hawkes.uniResiduals(TestData.uniTimes(), 1000.0,
                fit.mu(), fit.alpha(), fit.beta());
        Gof.KsResult ks = Gof.ksTestExp(tau);
        double tol = TestData.tol("hawkes_uni_ks_fitted");
        assertEquals(TestData.expect("hawkes_uni_ks_fitted", "ks_stat"), ks.stat(), tol);
        assertEquals(TestData.expect("hawkes_uni_ks_fitted", "p_value"), ks.pValue(), tol);
    }

    @Test
    public void hawkesBivLoglikPinned() {
        double ll = Hawkes.bivLoglik(TestData.bivTimes(), TestData.bivTypes(), 600.0,
                new double[] {0.25, 0.25}, new double[][] {{0.5, 0.2}, {0.2, 0.5}}, 1.8);
        assertEquals(TestData.expect("hawkes_biv_loglik_pinned", "loglik"), ll,
                TestData.tol("hawkes_biv_loglik_pinned"));
    }

    @Test
    public void hawkesBivFit() {
        Hawkes.BivFit fit = TestData.bivFit();
        double tol = TestData.tol("hawkes_biv_fit_alpha");
        assertEquals(TestData.expect("hawkes_biv_fit_alpha", "a00"), fit.alpha()[0][0], tol);
        assertEquals(TestData.expect("hawkes_biv_fit_alpha", "a01"), fit.alpha()[0][1], tol);
        assertEquals(TestData.expect("hawkes_biv_fit_alpha", "a10"), fit.alpha()[1][0], tol);
        assertEquals(TestData.expect("hawkes_biv_fit_alpha", "a11"), fit.alpha()[1][1], tol);
        assertEquals(TestData.expect("hawkes_biv_spectral_radius", "spectral_radius"),
                fit.spectralRadius(), TestData.tol("hawkes_biv_spectral_radius"));
    }

    @Test
    public void nnLosses() {
        double[] history = TestData.pipeline().history();
        assertEquals(TestData.expect("nn_loss_epoch1", "loss"), history[0],
                TestData.tol("nn_loss_epoch1"));
        assertEquals(TestData.expect("nn_loss_final", "loss"), history[history.length - 1],
                TestData.tol("nn_loss_final"));
    }

    @Test
    public void holdoutAccuracies() {
        TestData.Pipeline p = TestData.pipeline();
        int[] predNn = p.net().predict(p.xTest());
        assertEquals(TestData.expect("nn_holdout_accuracy", "accuracy"),
                Metrics.accuracy(p.yTest(), predNn), TestData.tol("nn_holdout_accuracy"));
        int[] predLg = p.logistic().predict(p.xTest());
        assertEquals(TestData.expect("logistic_accuracy", "accuracy"),
                Metrics.accuracy(p.yTest(), predLg), TestData.tol("logistic_accuracy"));
        int maj = Metrics.majorityClass(p.yTrain());
        int[] predMj = new int[p.yTest().length];
        Arrays.fill(predMj, maj);
        assertEquals(TestData.expect("majority_accuracy", "accuracy"),
                Metrics.accuracy(p.yTest(), predMj), TestData.tol("majority_accuracy"));
    }

    @Test
    public void signalPnl() {
        TestData.Pipeline p = TestData.pipeline();
        int[] pred = p.net().predict(p.xTest());
        Metrics.Pnl pnl = Metrics.signalPnl(pred,
                TestData.doubles(p.fs().fwdRet(), p.split().test()),
                TestData.doubles(p.fs().spread(), p.split().test()));
        double tol = TestData.tol("signal_pnl_after_costs");
        assertEquals(TestData.expect("signal_pnl_after_costs", "gross"), pnl.gross(), tol);
        assertEquals(TestData.expect("signal_pnl_after_costs", "net"), pnl.net(), tol);
        assertEquals(TestData.expect("signal_pnl_after_costs", "n_trades"), pnl.nTrades(), tol);
        assertTrue("half-spread costs kill the naive signal", pnl.net() < 0.0);
    }

    @Test
    public void allGoldenCasesCovered() {
        assertEquals(18, TestData.golden().size());
    }
}
