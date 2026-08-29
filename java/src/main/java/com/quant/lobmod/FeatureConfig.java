package com.quant.lobmod;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * Pinned feature / label / NN hyperparameters
 * ({@code data/features_config.json}).
 *
 * @param ofiWindow trailing OFI window length in events (50)
 * @param retWindow lookback for the short-horizon return feature (20)
 * @param horizon label horizon H in events (10)
 * @param epsilon dead-zone half-width for the flat class in ticks (0.25)
 * @param depthK levels per side in the depth-ratio feature (3)
 * @param hidden MLP hidden layer sizes ([16, 8])
 * @param learningRate full-batch gradient-descent step (0.1)
 * @param epochs training epochs, i.e. full-batch steps (800)
 * @param trainFrac chronological train fraction (0.7)
 */
public record FeatureConfig(
        int ofiWindow,
        int retWindow,
        int horizon,
        double epsilon,
        int depthK,
        int[] hidden,
        double learningRate,
        int epochs,
        double trainFrac) {

    /**
     * Validates all fields.
     *
     * @throws IllegalArgumentException on any out-of-range value
     */
    public FeatureConfig {
        checkPositive("ofi_window", ofiWindow);
        checkPositive("ret_window", retWindow);
        checkPositive("horizon", horizon);
        checkPositive("depth_k", depthK);
        checkPositive("epochs", epochs);
        if (!(Double.isFinite(epsilon) && epsilon >= 0.0)) {
            throw new IllegalArgumentException("epsilon must be >= 0, got " + epsilon);
        }
        if (!(Double.isFinite(learningRate) && learningRate > 0.0)) {
            throw new IllegalArgumentException("learning_rate must be positive, got " + learningRate);
        }
        if (!(trainFrac > 0.0 && trainFrac < 1.0)) {
            throw new IllegalArgumentException("train_frac must be in (0, 1), got " + trainFrac);
        }
        for (int h : hidden) {
            if (h <= 0) {
                throw new IllegalArgumentException("hidden sizes must be positive integers");
            }
        }
    }

    private static void checkPositive(String name, int v) {
        if (v <= 0) {
            throw new IllegalArgumentException(name + " must be a positive integer, got " + v);
        }
    }

    /**
     * Loads a config from its pinned JSON file.
     *
     * @throws UncheckedIOException on i/o failure
     * @throws IllegalArgumentException on invalid values
     */
    public static FeatureConfig load(Path path) {
        String text;
        try {
            text = Files.readString(path);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> raw = (Map<String, Object>) MiniJson.parse(text);
        @SuppressWarnings("unchecked")
        List<Object> hiddenRaw = (List<Object>) raw.get("hidden");
        int[] hidden = hiddenRaw.stream().mapToInt(h -> (int) (double) (Double) h).toArray();
        return new FeatureConfig(
                (int) (double) (Double) raw.get("ofi_window"),
                (int) (double) (Double) raw.get("ret_window"),
                (int) (double) (Double) raw.get("horizon"),
                (Double) raw.get("epsilon"),
                (int) (double) (Double) raw.get("depth_k"),
                hidden,
                (Double) raw.get("learning_rate"),
                (int) (double) (Double) raw.get("epochs"),
                (Double) raw.get("train_frac"));
    }
}
