package com.quant.lobmod;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * lobmod end-to-end demo.
 *
 * <p>Replays the bundled 5000-event LOB stream, fits univariate and bivariate
 * Hawkes models to the bundled order-flow data with goodness-of-fit residual
 * analysis, and trains the deterministic MLP direction classifier against
 * logistic-regression and majority baselines, closing with an honest look at
 * whether the signal survives trading costs.</p>
 *
 * <p>Run: {@code ./demo.sh} (data directory defaults to {@code ../data}).</p>
 */
public final class Demo {

    private static final String LINE = "-".repeat(68);

    private Demo() {
    }

    private static void section(String title) {
        System.out.printf("%n%s%n%s%n%s%n", LINE, title, LINE);
    }

    /**
     * Entry point.
     *
     * @param args optional: the data directory (default {@code ../data})
     */
    public static void main(String[] args) {
        Path data = Path.of(args.length > 0 ? args[0] : "../data");
        System.out.println("lobmod demo - high-frequency LOB modeling (Java)");

        // ------------------------------------------------------------ replay
        section("1. Limit order book replay (data/lob_events.csv)");
        List<Event> events = Replay.readEventsCsv(data.resolve("lob_events.csv"));
        FeatureConfig cfg = FeatureConfig.load(data.resolve("features_config.json"));
        BookStates states = Replay.replay(events, cfg.depthK());
        LimitOrderBook book = new LimitOrderBook();
        for (Event e : events) {
            Replay.applyEvent(book, e);
        }
        long nL = events.stream().filter(e -> e.etype() == 'L').count();
        long nC = events.stream().filter(e -> e.etype() == 'C').count();
        long nM = events.stream().filter(e -> e.etype() == 'M').count();
        long volume = book.trades().stream().mapToLong(Fill::size).sum();
        double midMin = Double.POSITIVE_INFINITY;
        double midMax = Double.NEGATIVE_INFINITY;
        int nanMid = 0;
        double spreadSum = 0.0;
        int spreadCount = 0;
        for (int i = 0; i < states.size(); i++) {
            double m = states.mid()[i];
            if (Double.isNaN(m)) {
                nanMid++;
            } else {
                midMin = Math.min(midMin, m);
                midMax = Math.max(midMax, m);
            }
            if (Double.isFinite(states.spread()[i])) {
                spreadSum += states.spread()[i];
                spreadCount++;
            }
        }
        System.out.printf("events: %d  (limit %d, cancel %d, market %d)%n",
                events.size(), nL, nC, nM);
        System.out.printf("trades: %d fills, %d units traded%n", book.trades().size(), volume);
        System.out.printf("mid range: [%.1f, %.1f] ticks; mean spread %.3f ticks%n",
                midMin, midMax, spreadSum / spreadCount);
        System.out.printf("events with an empty side (mid = NaN, guarded): %d%n", nanMid);
        System.out.printf("%n  snapshot     best_bid  best_ask     mid   micro    depth3(B/S)%n");
        for (int seq : new int[] {1000, 3000, 5000}) {
            int i = seq - 1;
            System.out.printf("  event %5d   %7.0f  %8.0f  %7.1f %8.2f   %.0f/%.0f%n",
                    seq, states.bestBid()[i], states.bestAsk()[i], states.mid()[i],
                    states.microprice()[i], states.bidDepthK()[i], states.askDepthK()[i]);
        }

        // ------------------------------------------------- Hawkes univariate
        section("2. Univariate Hawkes fit (data/hawkes_uni.csv)");
        Map<String, Object> dgp = readJson(data.resolve("dgp_params.json"));
        double[] tUni = Hawkes.readTimesCsv(data.resolve("hawkes_uni.csv"));
        @SuppressWarnings("unchecked")
        Map<String, Object> dgpUni = (Map<String, Object>) dgp.get("hawkes_uni");
        double uniT = (Double) dgpUni.get("T");
        Hawkes.UniFit fitU = Hawkes.fitUni(tUni, uniT);
        double sumGap = 0.0;
        for (int i = 1; i < tUni.length; i++) {
            double gap = tUni[i] - tUni[i - 1];
            sumGap += gap;
        }
        double meanGap = sumGap / (tUni.length - 1);
        double sqGap = 0.0;
        for (int i = 1; i < tUni.length; i++) {
            double dev = tUni[i] - tUni[i - 1] - meanGap;
            sqGap += dev * dev;
        }
        double gapCv = Math.sqrt(sqGap / (tUni.length - 1)) / meanGap;
        System.out.printf("%d events on [0, %.0f]  (gap CV %.2f vs 1 for Poisson)%n",
                tUni.length, uniT, gapCv);
        System.out.printf("%n  param        true    fitted%n");
        System.out.printf("  %-8s %8.3f  %8.4f%n", "mu", (Double) dgpUni.get("mu"), fitU.mu());
        System.out.printf("  %-8s %8.3f  %8.4f%n", "alpha", (Double) dgpUni.get("alpha"), fitU.alpha());
        System.out.printf("  %-8s %8.3f  %8.4f%n", "beta", (Double) dgpUni.get("beta"), fitU.beta());
        System.out.printf("  branching ratio alpha/beta = %.4f  (%s; %.0f%% of events are "
                + "endogenous offspring)%n",
                fitU.branching(), fitU.branching() < 1 ? "stationary" : "NON-STATIONARY",
                100 * fitU.branching());
        System.out.printf("  loglik %.2f, Nelder-Mead converged=%s (%d iters)%n",
                fitU.loglik(), fitU.converged(), fitU.nIter());

        section("3. Goodness of fit - time-rescaling residuals + KS");
        double[] tau = Hawkes.uniResiduals(tUni, uniT, fitU.mu(), fitU.alpha(), fitU.beta());
        Gof.KsResult ks = Gof.ksTestExp(tau);
        System.out.println("time-rescaled inter-arrivals should be iid Exp(1) under the model:");
        System.out.printf("  fitted params : KS stat %.4f, p-value %.4f -> %s%n",
                ks.stat(), ks.pValue(),
                ks.pValue() > 0.05 ? "no rejection (test is weak with fitted parameters)" : "REJECT");
        double[] tauWrong = Hawkes.uniResiduals(tUni, uniT, 1.5, 0.1, 3.0);
        Gof.KsResult ksWrong = Gof.ksTestExp(tauWrong);
        System.out.printf("  wrong params  : KS stat %.4f, p-value %.2e -> REJECT (sanity)%n",
                ksWrong.stat(), ksWrong.pValue());

        // -------------------------------------------------- Hawkes bivariate
        section("4. Bivariate Hawkes: buy/sell market-order cross-excitation");
        Hawkes.BivSample bivSample = Hawkes.readMarkedTimesCsv(data.resolve("hawkes_biv.csv"));
        double[] tBiv = bivSample.times();
        int[] kBiv = bivSample.types();
        int nBuy = 0;
        for (int k : kBiv) {
            if (k == 0) {
                nBuy++;
            }
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> dgpBiv = (Map<String, Object>) dgp.get("hawkes_biv");
        double bivT = (Double) dgpBiv.get("T");
        Hawkes.BivFit fitB = Hawkes.fitBiv(tBiv, kBiv, bivT);
        System.out.printf("%d events (%d buys, %d sells) on [0, %.0f]%n",
                tBiv.length, nBuy, tBiv.length - nBuy, bivT);
        double[][] a = fitB.alpha();
        double[][] br = fitB.branching();
        System.out.printf("%n  excitation alpha (fitted)      branching alpha/beta%n");
        System.out.printf("    buy <-buy  %6.3f            %6.3f%n", a[0][0], br[0][0]);
        System.out.printf("    buy <-sell %6.3f            %6.3f%n", a[0][1], br[0][1]);
        System.out.printf("    sell<-buy  %6.3f            %6.3f%n", a[1][0], br[1][0]);
        System.out.printf("    sell<-sell %6.3f            %6.3f%n", a[1][1], br[1][1]);
        System.out.printf("  mu = %.4f (buy), %.4f (sell); beta = %.4f%n",
                fitB.mu()[0], fitB.mu()[1], fitB.beta());
        System.out.printf("  spectral radius of branching matrix = %.4f (%s)%n",
                fitB.spectralRadius(),
                fitB.spectralRadius() < 1 ? "stationary" : "NON-STATIONARY");
        System.out.println("  interpretation: self terms exceed cross terms because the simulation was");
        System.out.println("  parameterised that way; the fit recovers them. This checks the estimator");
        System.out.println("  and says nothing about real order flow.");
        double[][] taus = Hawkes.bivResiduals(tBiv, kBiv, bivT, fitB.mu(), fitB.alpha(), fitB.beta());
        String[] names = {"buy", "sell"};
        for (int m = 0; m < 2; m++) {
            Gof.KsResult kk = Gof.ksTestExp(taus[m]);
            System.out.printf("  KS residuals (%-4s): stat %.4f, p %.4f -> %s%n",
                    names[m], kk.stat(), kk.pValue(), kk.pValue() > 0.05 ? "no rejection (test is weak with fitted parameters)" : "REJECT");
        }

        // --------------------------------------------- features + learning
        section("5. Direction prediction: MLP vs logistic vs majority");
        FeatureSet fs = Features.computeFeatures(states, cfg);
        Features.Split split = Features.chronologicalSplit(fs.size(), cfg.trainFrac());
        double[][] xTrainRaw = take(fs.x(), split.train());
        double[][] xTestRaw = take(fs.x(), split.test());
        Standardizer std = new Standardizer().fit(xTrainRaw);
        double[][] xTrain = std.transform(xTrainRaw);
        double[][] xTest = std.transform(xTestRaw);
        int[] yTrain = takeInt(fs.y(), split.train());
        int[] yTest = takeInt(fs.y(), split.test());
        System.out.printf("%d samples (features: %s); H=%d events, eps=%s%n",
                fs.size(), String.join(", ", Features.FEATURE_NAMES),
                cfg.horizon(), cfg.epsilon());
        System.out.printf("class counts (down/flat/up): train %s, test %s%n",
                countsOf(yTrain), countsOf(yTest));
        System.out.printf("chronological split: %d train / %d test; MLP (16, 8), lr=%s, "
                + "%d full-batch epochs%n",
                split.train().length, split.test().length, cfg.learningRate(), cfg.epochs());

        Mlp net = new Mlp(5, cfg.hidden(), 3);
        double[] hist = net.fit(xTrain, yTrain, cfg.learningRate(), cfg.epochs());
        Mlp logistic = new Mlp(5, new int[0], 3);
        logistic.fit(xTrain, yTrain, cfg.learningRate(), cfg.epochs());
        int maj = Metrics.majorityClass(yTrain);
        int[] predNn = net.predict(xTest);
        int[] predLg = logistic.predict(xTest);
        int[] predMj = new int[yTest.length];
        java.util.Arrays.fill(predMj, maj);
        System.out.printf("%ntraining loss: %.4f (epoch 1) -> %.4f (epoch %d)%n",
                hist[0], hist[hist.length - 1], cfg.epochs());
        System.out.printf("%n  model                holdout accuracy%n");
        System.out.printf("  MLP (16,8)               %.4f%n", Metrics.accuracy(yTest, predNn));
        System.out.printf("  logistic regression      %.4f%n", Metrics.accuracy(yTest, predLg));
        System.out.printf("  majority class (%d)       %.4f%n", maj, Metrics.accuracy(yTest, predMj));
        int[][] cm = Metrics.confusionMatrix(yTest, predNn, 3);
        System.out.printf("%n  MLP confusion matrix (rows true, cols predicted; "
                + "0=down 1=flat 2=up)%n");
        for (int r = 0; r < 3; r++) {
            System.out.printf("    %d | %5d %5d %5d%n", r, cm[r][0], cm[r][1], cm[r][2]);
        }

        section("6. Does the signal survive costs?");
        Metrics.Pnl pnl = Metrics.signalPnl(predNn,
                takeDouble(fs.fwdRet(), split.test()), takeDouble(fs.spread(), split.test()));
        System.out.printf("trading the MLP signal on the holdout (%d trades, mid-to-mid, "
                + "half-spread cost per trade):%n", pnl.nTrades());
        System.out.printf("  gross PnL %+.1f ticks   net PnL %+.1f ticks%n",
                pnl.gross(), pnl.net());
        System.out.println("note: on this split the MLP edges both baselines and gross PnL is positive,");
        System.out.println("but across 100 seeds logistic beats the MLP by 0.59 pts (SE 0.16). The structure");
        System.out.println("is generator mechanics (spread alone gives 0.693 of the 0.704), and paying half");
        System.out.println("the spread on every trade erases the gross PnL. See research/results.md.");

        System.out.printf("%ndemo complete.%n");
    }

    // ---------------------------------------------------------------- helpers

    @SuppressWarnings("unchecked")
    private static Map<String, Object> readJson(Path path) {
        try {
            return (Map<String, Object>) MiniJson.parse(Files.readString(path));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static double[][] take(double[][] x, int[] idx) {
        double[][] out = new double[idx.length][];
        for (int i = 0; i < idx.length; i++) {
            out[i] = x[idx[i]].clone();
        }
        return out;
    }

    private static int[] takeInt(int[] x, int[] idx) {
        int[] out = new int[idx.length];
        for (int i = 0; i < idx.length; i++) {
            out[i] = x[idx[i]];
        }
        return out;
    }

    private static double[] takeDouble(double[] x, int[] idx) {
        double[] out = new double[idx.length];
        for (int i = 0; i < idx.length; i++) {
            out[i] = x[idx[i]];
        }
        return out;
    }

    private static String countsOf(int[] y) {
        int[] counts = new int[3];
        for (int v : y) {
            counts[v]++;
        }
        return "[" + counts[0] + ", " + counts[1] + ", " + counts[2] + "]";
    }
}
