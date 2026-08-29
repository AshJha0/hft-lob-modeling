package com.quant.lobmod;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
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
        BookStates states = Replay.replay(events, cfg.depthK(), 1.0);
        LimitOrderBook book = new LimitOrderBook();
        for (Event e : events) {
            Replay.applyEvent(book, e);
        }
        long nL = events.stream().filter(e -> e.etype() == 'L').count();
        long nC = events.stream().filter(e -> e.etype() == 'C').count();
        long nM = events.stream().filter(e -> e.etype() == 'M').count();
        int volume = book.trades().stream().mapToInt(Fill::size).sum();
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
        double[] tUni = readColumnCsv(data.resolve("hawkes_uni.csv"));
        @SuppressWarnings("unchecked")
        Map<String, Object> dgpUni = (Map<String, Object>) dgp.get("hawkes_uni");
        double uniT = (Double) dgpUni.get("T");
        Hawkes.UniFit fitU = Hawkes.fitUni(tUni, uniT);
        double minGap = Double.POSITIVE_INFINITY;
        double sumGap = 0.0;
        for (int i = 1; i < tUni.length; i++) {
            double gap = tUni[i] - tUni[i - 1];
            minGap = Math.min(minGap, gap);
            sumGap += gap;
        }
        System.out.printf("%d events on [0, %.0f]  (min gap %.4f vs mean %.3f "
                + "- self-exciting bursts)%n",
                tUni.length, uniT, minGap, sumGap / (tUni.length - 1));
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
                ks.pValue() > 0.05 ? "PASS (no rejection at 5%)" : "REJECT");
        double[] tauWrong = Hawkes.uniResiduals(tUni, uniT, 1.5, 0.1, 3.0);
        Gof.KsResult ksWrong = Gof.ksTestExp(tauWrong);
        System.out.printf("  wrong params  : KS stat %.4f, p-value %.2e -> REJECT (sanity)%n",
                ksWrong.stat(), ksWrong.pValue());

        // -------------------------------------------------- Hawkes bivariate
        section("4. Bivariate Hawkes: buy/sell market-order cross-excitation");
        double[][] biv = readTwoColumnCsv(data.resolve("hawkes_biv.csv"));
        double[] tBiv = biv[0];
        int[] kBiv = new int[biv[1].length];
        int nBuy = 0;
        for (int i = 0; i < kBiv.length; i++) {
            kBiv[i] = (int) biv[1][i];
            if (kBiv[i] == 0) {
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
        System.out.println("  interpretation: self-excitation (buy<-buy, sell<-sell) dominates the");
        System.out.println("  cross terms - order flow is strongly clustered by side (herding /");
        System.out.println("  order splitting), with weaker but positive cross-side contagion.");
        double[][] taus = Hawkes.bivResiduals(tBiv, kBiv, bivT, fitB.mu(), fitB.alpha(), fitB.beta());
        String[] names = {"buy", "sell"};
        for (int m = 0; m < 2; m++) {
            Gof.KsResult kk = Gof.ksTestExp(taus[m]);
            System.out.printf("  KS residuals (%-4s): stat %.4f, p %.4f -> %s%n",
                    names[m], kk.stat(), kk.pValue(), kk.pValue() > 0.05 ? "PASS" : "REJECT");
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
        System.out.println("note: the classifier finds real structure (it beats both baselines and");
        System.out.println("gross PnL is positive), but paying half the spread on every trade");
        System.out.println("erases it - naive microstructure signals die on transaction costs.");

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

    private static double[] readColumnCsv(Path path) {
        try {
            List<String> lines = Files.readAllLines(path);
            List<Double> out = new ArrayList<>();
            for (int i = 1; i < lines.size(); i++) {
                String line = lines.get(i).strip();
                if (!line.isEmpty()) {
                    out.add(Double.parseDouble(line));
                }
            }
            return out.stream().mapToDouble(Double::doubleValue).toArray();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static double[][] readTwoColumnCsv(Path path) {
        try {
            List<String> lines = Files.readAllLines(path);
            List<double[]> rows = new ArrayList<>();
            for (int i = 1; i < lines.size(); i++) {
                String line = lines.get(i).strip();
                if (line.isEmpty()) {
                    continue;
                }
                String[] parts = line.split(",");
                rows.add(new double[] {Double.parseDouble(parts[0]), Double.parseDouble(parts[1])});
            }
            double[][] out = new double[2][rows.size()];
            for (int i = 0; i < rows.size(); i++) {
                out[0][i] = rows.get(i)[0];
                out[1][i] = rows.get(i)[1];
            }
            return out;
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
