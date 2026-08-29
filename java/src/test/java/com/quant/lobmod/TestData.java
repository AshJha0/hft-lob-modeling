package com.quant.lobmod;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Lazily-computed shared fixtures for the test suite: bundled data, golden
 * cases, the replayed book states, the Hawkes fits and the full NN pipeline.
 * Everything is deterministic, so computing each fixture once keeps the suite
 * fast without affecting results.
 */
final class TestData {

    static final Path DATA = Path.of("..", "data");

    private static Map<String, Map<String, Object>> golden;
    private static List<Event> events;
    private static BookStates states;
    private static double[] uniTimes;
    private static double[] bivTimes;
    private static int[] bivTypes;
    private static Hawkes.UniFit uniFit;
    private static Hawkes.BivFit bivFit;
    private static Pipeline pipeline;

    /** Everything downstream of features: split, standardized data, trained models. */
    record Pipeline(FeatureSet fs, Features.Split split, double[][] xTrain, double[][] xTest,
            int[] yTrain, int[] yTest, Mlp net, Mlp logistic, double[] history) {
    }

    private TestData() {
    }

    /** Golden cases by name; each value holds "inputs", "expect", "tol". */
    static synchronized Map<String, Map<String, Object>> golden() {
        if (golden == null) {
            Object parsed = MiniJson.parse(readString(DATA.resolve("golden").resolve("golden.json")));
            @SuppressWarnings("unchecked")
            Map<String, Object> root = (Map<String, Object>) parsed;
            @SuppressWarnings("unchecked")
            List<Object> cases = (List<Object>) root.get("cases");
            Map<String, Map<String, Object>> byName = new LinkedHashMap<>();
            for (Object c : cases) {
                @SuppressWarnings("unchecked")
                Map<String, Object> m = (Map<String, Object>) c;
                byName.put((String) m.get("name"), m);
            }
            golden = byName;
        }
        return golden;
    }

    /** Expected value {@code key} of golden case {@code name}. */
    static double expect(String name, String key) {
        @SuppressWarnings("unchecked")
        Map<String, Object> e = (Map<String, Object>) golden().get(name).get("expect");
        return (Double) e.get(key);
    }

    /** Tolerance of golden case {@code name}. */
    static double tol(String name) {
        return (Double) golden().get(name).get("tol");
    }

    static synchronized List<Event> events() {
        if (events == null) {
            events = Replay.readEventsCsv(DATA.resolve("lob_events.csv"));
        }
        return events;
    }

    static synchronized BookStates states() {
        if (states == null) {
            states = Replay.replay(events(), 3, 1.0);
        }
        return states;
    }

    static FeatureConfig config() {
        return FeatureConfig.load(DATA.resolve("features_config.json"));
    }

    static synchronized double[] uniTimes() {
        if (uniTimes == null) {
            List<String> lines = readLines(DATA.resolve("hawkes_uni.csv"));
            uniTimes = lines.stream().skip(1).map(String::strip).filter(s -> !s.isEmpty())
                    .mapToDouble(Double::parseDouble).toArray();
        }
        return uniTimes;
    }

    static synchronized double[] bivTimes() {
        loadBiv();
        return bivTimes;
    }

    static synchronized int[] bivTypes() {
        loadBiv();
        return bivTypes;
    }

    private static void loadBiv() {
        if (bivTimes == null) {
            List<String> lines = readLines(DATA.resolve("hawkes_biv.csv"));
            int n = 0;
            for (int i = 1; i < lines.size(); i++) {
                if (!lines.get(i).strip().isEmpty()) {
                    n++;
                }
            }
            bivTimes = new double[n];
            bivTypes = new int[n];
            int r = 0;
            for (int i = 1; i < lines.size(); i++) {
                String line = lines.get(i).strip();
                if (line.isEmpty()) {
                    continue;
                }
                String[] parts = line.split(",");
                bivTimes[r] = Double.parseDouble(parts[0]);
                bivTypes[r] = Integer.parseInt(parts[1]);
                r++;
            }
        }
    }

    static synchronized Hawkes.UniFit uniFit() {
        if (uniFit == null) {
            uniFit = Hawkes.fitUni(uniTimes(), 1000.0);
        }
        return uniFit;
    }

    static synchronized Hawkes.BivFit bivFit() {
        if (bivFit == null) {
            bivFit = Hawkes.fitBiv(bivTimes(), bivTypes(), 600.0);
        }
        return bivFit;
    }

    static synchronized Pipeline pipeline() {
        if (pipeline == null) {
            FeatureConfig cfg = config();
            FeatureSet fs = Features.computeFeatures(states(), cfg);
            Features.Split split = Features.chronologicalSplit(fs.size(), cfg.trainFrac());
            double[][] xTrainRaw = rows(fs.x(), split.train());
            double[][] xTestRaw = rows(fs.x(), split.test());
            Standardizer std = new Standardizer().fit(xTrainRaw);
            double[][] xTrain = std.transform(xTrainRaw);
            double[][] xTest = std.transform(xTestRaw);
            int[] yTrain = ints(fs.y(), split.train());
            int[] yTest = ints(fs.y(), split.test());
            Mlp net = new Mlp(5, cfg.hidden(), 3);
            double[] history = net.fit(xTrain, yTrain, cfg.learningRate(), cfg.epochs());
            Mlp logistic = new Mlp(5, new int[0], 3);
            logistic.fit(xTrain, yTrain, cfg.learningRate(), cfg.epochs());
            pipeline = new Pipeline(fs, split, xTrain, xTest, yTrain, yTest, net, logistic, history);
        }
        return pipeline;
    }

    static double[][] rows(double[][] x, int[] idx) {
        double[][] out = new double[idx.length][];
        for (int i = 0; i < idx.length; i++) {
            out[i] = x[idx[i]].clone();
        }
        return out;
    }

    static int[] ints(int[] x, int[] idx) {
        int[] out = new int[idx.length];
        for (int i = 0; i < idx.length; i++) {
            out[i] = x[idx[i]];
        }
        return out;
    }

    static double[] doubles(double[] x, int[] idx) {
        double[] out = new double[idx.length];
        for (int i = 0; i < idx.length; i++) {
            out[i] = x[idx[i]];
        }
        return out;
    }

    private static String readString(Path p) {
        try {
            return Files.readString(p);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static List<String> readLines(Path p) {
        try {
            return Files.readAllLines(p);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
