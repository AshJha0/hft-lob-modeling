#ifndef LOBMOD_NN_HPP
#define LOBMOD_NN_HPP

/// \file nn.hpp
/// \brief Feed-forward neural network from scratch — fully deterministic,
///        no RNG — plus baselines and evaluation metrics.
///
/// Architecture for the prediction task: input -> 16 ReLU -> 8 ReLU ->
/// 3-way softmax; the same class with empty `hidden` *is* the multinomial
/// logistic-regression baseline.  Weights W_l have shape (n_in, n_out) and
/// activations are row vectors (Z = A W + b).
///
/// Determinism (pinned):
/// * init (no RNG): W_l[i][j] = 0.5 sqrt(2/n_in) sin(l*10000 + i*n_out +
///   j + 1) (radians, plain double sin), b_l = 0;
/// * full-batch gradient descent: exactly one update per epoch on the whole
///   training set (no shuffling / minibatches / momentum / regularization);
/// * standardization: training-split population moments (divisor n), std
///   below 1e-12 replaced by 1.
///
/// Backward pass (exact pinned equations): G_L = (P - onehot(y)) / n;
/// dW_l = A_{l-1}^T G_l; db_l = column sums of G_l;
/// G_{l-1} = (G_l W_l^T) .* [Z_{l-1} > 0].

#include <cstddef>
#include <tuple>
#include <vector>

namespace lobmod {

/// Row-major dense matrix of doubles.
struct Matrix {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<double> a; ///< rows*cols entries, row major

    Matrix() = default;
    Matrix(std::size_t r, std::size_t c, double fill = 0.0)
        : rows(r), cols(c), a(r * c, fill) {}

    double& operator()(std::size_t i, std::size_t j) { return a[i * cols + j]; }
    double operator()(std::size_t i, std::size_t j) const { return a[i * cols + j]; }
};

/// Deterministic sin-based weight matrix (file docs formula).
/// \throws std::invalid_argument on non-positive sizes or a negative layer.
Matrix pinned_init(std::size_t n_in, std::size_t n_out, int layer_index);

/// Row-wise softmax with the pinned max-shift for overflow safety.
Matrix softmax(const Matrix& z);

/// Pinned per-feature standardization fitted on the training split only.
class Standardizer {
public:
    /// Compute population mean/std per column; std below 1e-12 becomes 1.
    /// \throws std::invalid_argument on empty or non-finite input.
    Standardizer& fit(const Matrix& X);

    /// Apply (x - mean) / std with the fitted moments.
    /// \throws std::invalid_argument before fit or on a column mismatch.
    Matrix transform(const Matrix& X) const;

    const std::vector<double>& mean() const { return mean_; }
    const std::vector<double>& std_dev() const { return std_; }

private:
    std::vector<double> mean_;
    std::vector<double> std_;
};

/// Deterministic MLP classifier (see file docs).
class MLP {
public:
    /// \param n_inputs input dimension (> 0).
    /// \param hidden hidden layer sizes; empty = logistic regression.
    /// \param n_classes number of output classes (>= 2).
    /// \throws std::invalid_argument on bad sizes.
    MLP(std::size_t n_inputs, const std::vector<int>& hidden,
        std::size_t n_classes);

    /// Class probabilities, shape (n, n_classes); rows sum to 1.
    Matrix predict_proba(const Matrix& X) const;

    /// Argmax class per row; ties resolve to the lowest class index.
    std::vector<int> predict(const Matrix& X) const;

    /// Mean cross-entropy -(1/n) sum log P[i, y_i], probability floored at
    /// 1e-300 before the log.
    double loss(const Matrix& X, const std::vector<int>& y) const;

    /// Loss plus gradients dW_l, db_l via backprop (exact pinned equations).
    std::tuple<double, std::vector<Matrix>, std::vector<std::vector<double>>>
    loss_and_grads(const Matrix& X, const std::vector<int>& y) const;

    /// Full-batch gradient descent (pinned schedule): one update per epoch,
    /// recording the loss on the whole training set after each update
    /// (history[0] is the golden "loss after epoch 1").
    /// \throws std::invalid_argument on bad inputs or hyperparameters.
    std::vector<double> fit(const Matrix& X, const std::vector<int>& y,
                            double learning_rate, int epochs);

    const std::vector<Matrix>& weights() const { return weights_; }
    const std::vector<std::vector<double>>& biases() const { return biases_; }
    /// Mutable parameter access (finite-difference checks, custom inits).
    std::vector<Matrix>& weights() { return weights_; }
    /// Mutable bias access (finite-difference checks, custom inits).
    std::vector<std::vector<double>>& biases() { return biases_; }
    const std::vector<std::size_t>& sizes() const { return sizes_; }

private:
    void forward(const Matrix& X, std::vector<Matrix>& zs,
                 std::vector<Matrix>& activations, Matrix& probs) const;

    std::vector<std::size_t> sizes_;
    std::vector<Matrix> weights_;
    std::vector<std::vector<double>> biases_;
};

/// Fraction of matching labels.
/// \throws std::invalid_argument on mismatched or empty inputs.
double accuracy(const std::vector<int>& y_true, const std::vector<int>& y_pred);

/// (n_classes x n_classes) counts; rows = true class, cols = predicted.
std::vector<std::vector<int>> confusion_matrix(const std::vector<int>& y_true,
                                               const std::vector<int>& y_pred,
                                               int n_classes);

/// Most frequent label; ties resolve to the smallest label (pinned).
int majority_class(const std::vector<int>& y);

/// PnL of trading the predicted direction with half-spread costs: position
/// y_pred - 1 in {-1, 0, +1}; gross = sum pos * fwd_ret; net subtracts
/// |pos| * spread / 2 per signal.  Returns (gross, net, n_trades).
/// \throws std::invalid_argument on mismatched or non-finite inputs.
std::tuple<double, double, int> signal_pnl(const std::vector<int>& y_pred,
                                           const std::vector<double>& fwd_ret,
                                           const std::vector<double>& spread);

} // namespace lobmod

#endif // LOBMOD_NN_HPP
