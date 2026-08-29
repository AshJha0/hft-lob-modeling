package com.quant.lobmod;

import java.util.ArrayList;
import java.util.List;

/**
 * Feed-forward neural network from scratch — fully deterministic, no RNG.
 *
 * <p>Architecture (pinned for the prediction task): input -&gt; 16 ReLU -&gt;
 * 8 ReLU -&gt; 3-way softmax. The same class with an empty {@code hidden}
 * array <em>is</em> the multinomial logistic-regression baseline
 * (input -&gt; softmax).</p>
 *
 * <p><b>Determinism.</b> Cross-language golden reproducibility requires that
 * training involve no randomness at all:</p>
 * <ul>
 *   <li><b>Pinned init</b> (no RNG): for layer {@code l} (0-based) with shape
 *       {@code (n_in, n_out)},
 *       {@code W_l[i][j] = 0.5 * sqrt(2 / n_in) * sin(l*10000 + i*n_out + j + 1)}
 *       (radians, plain double {@code sin}), {@code b_l = 0}. The sin of
 *       distinct integers gives well-spread values in [-1, 1]; the
 *       {@code sqrt(2/n_in)} factor is He-style scaling for ReLU units.</li>
 *   <li><b>Full-batch gradient descent</b>: exactly one gradient step per
 *       epoch on the entire training set — no shuffling, no minibatches, no
 *       momentum, no regularization.</li>
 * </ul>
 *
 * <p><b>Forward / backward equations</b> (activations are row vectors,
 * {@code Z = A W + b}; softmax is max-shifted per row):</p>
 *
 * <pre>
 *   G_L     = (P - onehot(y)) / n
 *   dW_l    = A_(l-1)^T G_l ;  db_l = column sums of G_l
 *   G_(l-1) = (G_l W_l^T) .* 1{Z_(l-1) &gt; 0}
 * </pre>
 *
 * <p>Loss is mean cross-entropy with the picked probability floored at
 * 1e-300 before the log so a confident wrong answer cannot produce
 * {@code -inf}.</p>
 */
public final class Mlp {

    private static final double LOG_FLOOR = 1e-300;

    /** Layer sizes including input and output. */
    private final int[] sizes;
    /** Weight matrices, shape (nIn, nOut) each; package-private for tests. */
    final List<double[][]> weights = new ArrayList<>();
    /** Bias vectors, length nOut each; package-private for tests. */
    final List<double[]> biases = new ArrayList<>();

    /**
     * Builds a network with the pinned deterministic initialization.
     *
     * @param nInputs input dimension (&gt; 0)
     * @param hidden hidden layer sizes; empty = logistic regression
     * @param nClasses number of output classes (&ge; 2)
     * @throws IllegalArgumentException on invalid sizes
     */
    public Mlp(int nInputs, int[] hidden, int nClasses) {
        if (nInputs <= 0 || nClasses < 2) {
            throw new IllegalArgumentException("n_inputs must be positive and n_classes >= 2");
        }
        for (int h : hidden) {
            if (h <= 0) {
                throw new IllegalArgumentException("hidden sizes must be positive integers");
            }
        }
        sizes = new int[hidden.length + 2];
        sizes[0] = nInputs;
        System.arraycopy(hidden, 0, sizes, 1, hidden.length);
        sizes[sizes.length - 1] = nClasses;
        for (int l = 0; l < sizes.length - 1; l++) {
            weights.add(pinnedInit(sizes[l], sizes[l + 1], l));
            biases.add(new double[sizes[l + 1]]);
        }
    }

    /**
     * Deterministic sin-based weight matrix (class docstring formula).
     *
     * @throws IllegalArgumentException on non-positive dims or a negative
     *     layer index
     */
    public static double[][] pinnedInit(int nIn, int nOut, int layerIndex) {
        if (nIn <= 0 || nOut <= 0 || layerIndex < 0) {
            throw new IllegalArgumentException("n_in, n_out must be positive and layer_index >= 0");
        }
        double scale = 0.5 * Math.sqrt(2.0 / nIn);
        double[][] w = new double[nIn][nOut];
        for (int i = 0; i < nIn; i++) {
            for (int j = 0; j < nOut; j++) {
                w[i][j] = scale * Math.sin(layerIndex * 10000.0 + i * (double) nOut + j + 1.0);
            }
        }
        return w;
    }

    /** Layer sizes including input and output (copy). */
    public int[] sizes() {
        return sizes.clone();
    }

    // ------------------------------------------------------------------ checks

    private void checkFeatures(double[][] x) {
        Standardizer.checkMatrix(x);
        if (x[0].length != sizes[0]) {
            throw new IllegalArgumentException(
                    "expected " + sizes[0] + " features, got " + x[0].length);
        }
    }

    private void checkLabels(int[] y, int nRows) {
        if (y == null || y.length != nRows) {
            throw new IllegalArgumentException("y must have length " + nRows);
        }
        int nClasses = sizes[sizes.length - 1];
        for (int v : y) {
            if (v < 0 || v >= nClasses) {
                throw new IllegalArgumentException("labels must be in [0, " + nClasses + ")");
            }
        }
    }

    // ----------------------------------------------------------------- forward

    /** Row-wise softmax with the pinned max-shift for overflow safety. */
    static double[][] softmax(double[][] z) {
        double[][] p = new double[z.length][z[0].length];
        for (int i = 0; i < z.length; i++) {
            double max = Double.NEGATIVE_INFINITY;
            for (double v : z[i]) {
                max = Math.max(max, v);
            }
            double sum = 0.0;
            for (int j = 0; j < z[i].length; j++) {
                p[i][j] = Math.exp(z[i][j] - max);
                sum += p[i][j];
            }
            for (int j = 0; j < z[i].length; j++) {
                p[i][j] /= sum;
            }
        }
        return p;
    }

    /** {@code A W + b} for row-major activations. */
    private static double[][] affine(double[][] a, double[][] w, double[] b) {
        int n = a.length;
        int nIn = w.length;
        int nOut = w[0].length;
        double[][] z = new double[n][nOut];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < nOut; j++) {
                double sum = 0.0;
                for (int k = 0; k < nIn; k++) {
                    sum += a[i][k] * w[k][j];
                }
                z[i][j] = sum + b[j]; // dot product first, then bias — matches A@W + b
            }
        }
        return z;
    }

    /** Forward pass: pre-activations per layer, activations, probabilities. */
    private ForwardPass forward(double[][] x) {
        int nLayers = weights.size();
        double[][][] zs = new double[nLayers][][];
        double[][][] activations = new double[nLayers][][];
        double[][] a = x;
        for (int l = 0; l < nLayers; l++) {
            activations[l] = a;
            double[][] z = affine(a, weights.get(l), biases.get(l));
            zs[l] = z;
            if (l < nLayers - 1) {
                double[][] relu = new double[z.length][z[0].length];
                for (int i = 0; i < z.length; i++) {
                    for (int j = 0; j < z[0].length; j++) {
                        relu[i][j] = Math.max(z[i][j], 0.0);
                    }
                }
                a = relu;
            }
        }
        return new ForwardPass(zs, activations, softmax(zs[nLayers - 1]));
    }

    private record ForwardPass(double[][][] zs, double[][][] activations, double[][] probs) {
    }

    /**
     * Class probabilities, shape (n, nClasses); rows sum to 1.
     *
     * @throws IllegalArgumentException on invalid input
     */
    public double[][] predictProba(double[][] x) {
        checkFeatures(x);
        return forward(x).probs();
    }

    /** Argmax class per row (ties resolve to the lowest class index). */
    public int[] predict(double[][] x) {
        double[][] p = predictProba(x);
        int[] out = new int[p.length];
        for (int i = 0; i < p.length; i++) {
            int best = 0;
            for (int j = 1; j < p[i].length; j++) {
                if (p[i][j] > p[i][best]) {
                    best = j;
                }
            }
            out[i] = best;
        }
        return out;
    }

    /**
     * Mean cross-entropy {@code -(1/n) sum log(max(P[i, y_i], 1e-300))}.
     *
     * @throws IllegalArgumentException on invalid inputs
     */
    public double loss(double[][] x, int[] y) {
        checkFeatures(x);
        checkLabels(y, x.length);
        double[][] probs = forward(x).probs();
        return meanCrossEntropy(probs, y);
    }

    private static double meanCrossEntropy(double[][] probs, int[] y) {
        double sum = 0.0;
        for (int i = 0; i < y.length; i++) {
            sum += Math.log(Math.max(probs[i][y[i]], LOG_FLOOR));
        }
        return -sum / y.length;
    }

    // ---------------------------------------------------------------- backward

    /**
     * Backprop outcome: loss plus per-layer gradients.
     *
     * @param loss mean cross-entropy at the current parameters
     * @param gradsW d(loss)/dW_l per layer
     * @param gradsB d(loss)/db_l per layer
     */
    public record Grads(double loss, double[][][] gradsW, double[][] gradsB) {
    }

    /**
     * Loss plus gradients via backprop — exactly the equations in the class
     * docstring.
     *
     * @throws IllegalArgumentException on invalid inputs
     */
    public Grads lossAndGrads(double[][] x, int[] y) {
        checkFeatures(x);
        checkLabels(y, x.length);
        int n = x.length;
        int nLayers = weights.size();
        ForwardPass fp = forward(x);
        double loss = meanCrossEntropy(fp.probs(), y);

        // G_L = (P - onehot(y)) / n
        int nClasses = sizes[sizes.length - 1];
        double[][] g = new double[n][nClasses];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < nClasses; j++) {
                g[i][j] = (fp.probs()[i][j] - (y[i] == j ? 1.0 : 0.0)) / n;
            }
        }
        double[][][] gradsW = new double[nLayers][][];
        double[][] gradsB = new double[nLayers][];
        for (int l = nLayers - 1; l >= 0; l--) {
            double[][] act = fp.activations()[l];
            int nIn = sizes[l];
            int nOut = sizes[l + 1];
            double[][] dw = new double[nIn][nOut];
            double[] db = new double[nOut];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < nOut; j++) {
                    double gij = g[i][j];
                    db[j] += gij;
                    for (int kk = 0; kk < nIn; kk++) {
                        dw[kk][j] += act[i][kk] * gij;
                    }
                }
            }
            gradsW[l] = dw;
            gradsB[l] = db;
            if (l > 0) {
                double[][] wl = weights.get(l);
                double[][] zPrev = fp.zs()[l - 1];
                double[][] gPrev = new double[n][nIn];
                for (int i = 0; i < n; i++) {
                    for (int kk = 0; kk < nIn; kk++) {
                        if (zPrev[i][kk] > 0.0) { // ReLU gate
                            double sum = 0.0;
                            for (int j = 0; j < nOut; j++) {
                                sum += g[i][j] * wl[kk][j];
                            }
                            gPrev[i][kk] = sum;
                        }
                    }
                }
                g = gPrev;
            }
        }
        return new Grads(loss, gradsW, gradsB);
    }

    /**
     * Full-batch gradient descent (pinned schedule): one gradient step per
     * epoch on the whole training set, then the training loss is recorded —
     * {@code history[0]} is the golden "loss after epoch 1".
     *
     * @return the per-epoch loss history (length {@code epochs})
     * @throws IllegalArgumentException on invalid inputs or hyperparameters
     */
    public double[] fit(double[][] x, int[] y, double learningRate, int epochs) {
        if (!(Double.isFinite(learningRate) && learningRate > 0.0)) {
            throw new IllegalArgumentException("learning_rate must be positive, got " + learningRate);
        }
        if (epochs <= 0) {
            throw new IllegalArgumentException("epochs must be a positive integer, got " + epochs);
        }
        checkFeatures(x);
        checkLabels(y, x.length);
        double[] history = new double[epochs];
        for (int epoch = 0; epoch < epochs; epoch++) {
            Grads grads = lossAndGrads(x, y);
            for (int l = 0; l < weights.size(); l++) {
                double[][] w = weights.get(l);
                double[][] dw = grads.gradsW()[l];
                for (int i = 0; i < w.length; i++) {
                    for (int j = 0; j < w[i].length; j++) {
                        w[i][j] -= learningRate * dw[i][j];
                    }
                }
                double[] b = biases.get(l);
                double[] db = grads.gradsB()[l];
                for (int j = 0; j < b.length; j++) {
                    b[j] -= learningRate * db[j];
                }
            }
            history[epoch] = loss(x, y);
        }
        return history;
    }
}
