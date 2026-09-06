#include "lobmod/nn.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>

namespace lobmod {

namespace {

constexpr double kStdFloor = 1e-12;
constexpr double kLogFloor = 1e-300;

void check_matrix(const Matrix& X) {
    if (X.rows == 0 || X.cols == 0) {
        throw std::invalid_argument("X must be a non-empty 2-D array");
    }
    for (double v : X.a) {
        if (!std::isfinite(v)) throw std::invalid_argument("X must be finite");
    }
}

void check_labels(const std::vector<int>& y, std::size_t n_rows,
                  std::size_t n_classes) {
    if (y.size() != n_rows) {
        throw std::invalid_argument("y must have length " + std::to_string(n_rows) +
                                    ", got " + std::to_string(y.size()));
    }
    for (int v : y) {
        if (v < 0 || static_cast<std::size_t>(v) >= n_classes) {
            throw std::invalid_argument("labels must be in [0, " +
                                        std::to_string(n_classes) + ")");
        }
    }
}

/// C = A * B (plain sequential triple loop; sizes here are tiny).
Matrix matmul(const Matrix& A, const Matrix& B) {
    Matrix C(A.rows, B.cols, 0.0);
    for (std::size_t i = 0; i < A.rows; ++i) {
        for (std::size_t k = 0; k < A.cols; ++k) {
            const double aik = A(i, k);
            for (std::size_t j = 0; j < B.cols; ++j) {
                C(i, j) += aik * B(k, j);
            }
        }
    }
    return C;
}

} // namespace

Matrix pinned_init(std::size_t n_in, std::size_t n_out, int layer_index) {
    if (n_in == 0 || n_out == 0 || layer_index < 0) {
        throw std::invalid_argument(
            "n_in, n_out must be positive and layer_index >= 0");
    }
    Matrix w(n_in, n_out);
    const double scale = 0.5 * std::sqrt(2.0 / static_cast<double>(n_in));
    for (std::size_t i = 0; i < n_in; ++i) {
        for (std::size_t j = 0; j < n_out; ++j) {
            const double arg = static_cast<double>(
                static_cast<long long>(layer_index) * 10000 +
                static_cast<long long>(i) * static_cast<long long>(n_out) +
                static_cast<long long>(j) + 1);
            w(i, j) = scale * std::sin(arg);
        }
    }
    return w;
}

Matrix softmax(const Matrix& z) {
    Matrix p(z.rows, z.cols);
    for (std::size_t i = 0; i < z.rows; ++i) {
        double zmax = z(i, 0);
        for (std::size_t j = 1; j < z.cols; ++j) zmax = std::max(zmax, z(i, j));
        double total = 0.0;
        for (std::size_t j = 0; j < z.cols; ++j) {
            p(i, j) = std::exp(z(i, j) - zmax);
            total += p(i, j);
        }
        for (std::size_t j = 0; j < z.cols; ++j) p(i, j) /= total;
    }
    return p;
}

Standardizer& Standardizer::fit(const Matrix& X) {
    check_matrix(X);
    mean_.assign(X.cols, 0.0);
    std_.assign(X.cols, 0.0);
    const double n = static_cast<double>(X.rows);
    for (std::size_t i = 0; i < X.rows; ++i) {
        for (std::size_t j = 0; j < X.cols; ++j) mean_[j] += X(i, j);
    }
    for (std::size_t j = 0; j < X.cols; ++j) mean_[j] /= n;
    for (std::size_t i = 0; i < X.rows; ++i) {
        for (std::size_t j = 0; j < X.cols; ++j) {
            const double d = X(i, j) - mean_[j];
            std_[j] += d * d;
        }
    }
    for (std::size_t j = 0; j < X.cols; ++j) {
        std_[j] = std::sqrt(std_[j] / n); // population std (divisor n) — pinned
        if (std_[j] < kStdFloor) std_[j] = 1.0;
    }
    return *this;
}

Matrix Standardizer::transform(const Matrix& X) const {
    if (mean_.empty()) {
        throw std::invalid_argument("Standardizer::transform called before fit");
    }
    check_matrix(X);
    if (X.cols != mean_.size()) {
        throw std::invalid_argument("expected " + std::to_string(mean_.size()) +
                                    " columns, got " + std::to_string(X.cols));
    }
    Matrix out(X.rows, X.cols);
    for (std::size_t i = 0; i < X.rows; ++i) {
        for (std::size_t j = 0; j < X.cols; ++j) {
            out(i, j) = (X(i, j) - mean_[j]) / std_[j];
        }
    }
    return out;
}

MLP::MLP(std::size_t n_inputs, const std::vector<int>& hidden,
         std::size_t n_classes) {
    if (n_inputs == 0 || n_classes < 2) {
        throw std::invalid_argument("n_inputs must be positive and n_classes >= 2");
    }
    for (int h : hidden) {
        if (h <= 0) {
            throw std::invalid_argument("hidden sizes must be positive integers");
        }
    }
    sizes_.push_back(n_inputs);
    for (int h : hidden) sizes_.push_back(static_cast<std::size_t>(h));
    sizes_.push_back(n_classes);
    for (std::size_t l = 0; l + 1 < sizes_.size(); ++l) {
        weights_.push_back(
            pinned_init(sizes_[l], sizes_[l + 1], static_cast<int>(l)));
        biases_.emplace_back(sizes_[l + 1], 0.0);
    }
}

void MLP::forward(const Matrix& X, std::vector<Matrix>& zs,
                  std::vector<Matrix>& activations, Matrix& probs) const {
    zs.clear();
    activations.clear();
    activations.push_back(X); // A_0 = X
    const std::size_t n_layers = weights_.size();
    for (std::size_t l = 0; l < n_layers; ++l) {
        Matrix z = matmul(activations.back(), weights_[l]);
        for (std::size_t i = 0; i < z.rows; ++i) {
            for (std::size_t j = 0; j < z.cols; ++j) z(i, j) += biases_[l][j];
        }
        zs.push_back(z);
        if (l < n_layers - 1) {
            Matrix a = z;
            for (double& v : a.a) v = std::max(v, 0.0); // ReLU
            activations.push_back(std::move(a));
        }
    }
    probs = softmax(zs.back());
}

Matrix MLP::predict_proba(const Matrix& X) const {
    check_matrix(X);
    if (X.cols != sizes_.front()) {
        throw std::invalid_argument("expected " + std::to_string(sizes_.front()) +
                                    " features, got " + std::to_string(X.cols));
    }
    std::vector<Matrix> zs, activations;
    Matrix probs;
    forward(X, zs, activations, probs);
    return probs;
}

std::vector<int> MLP::predict(const Matrix& X) const {
    const Matrix probs = predict_proba(X);
    std::vector<int> out(probs.rows);
    for (std::size_t i = 0; i < probs.rows; ++i) {
        int best = 0;
        for (std::size_t j = 1; j < probs.cols; ++j) {
            if (probs(i, j) > probs(i, static_cast<std::size_t>(best))) {
                best = static_cast<int>(j); // strict >: ties keep the lowest
            }
        }
        out[i] = best;
    }
    return out;
}

double MLP::loss(const Matrix& X, const std::vector<int>& y) const {
    check_matrix(X);
    check_labels(y, X.rows, sizes_.back());
    const Matrix probs = predict_proba(X);
    double total = 0.0;
    for (std::size_t i = 0; i < X.rows; ++i) {
        total += std::log(
            std::max(probs(i, static_cast<std::size_t>(y[i])), kLogFloor));
    }
    return -total / static_cast<double>(X.rows);
}

std::tuple<double, std::vector<Matrix>, std::vector<std::vector<double>>>
MLP::loss_and_grads(const Matrix& X, const std::vector<int>& y) const {
    check_matrix(X);
    check_labels(y, X.rows, sizes_.back());
    const std::size_t n = X.rows;
    std::vector<Matrix> zs, activations;
    Matrix probs;
    forward(X, zs, activations, probs);
    double total = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        total += std::log(
            std::max(probs(i, static_cast<std::size_t>(y[i])), kLogFloor));
    }
    const double loss_val = -total / static_cast<double>(n);

    // G_L = (P - onehot(y)) / n
    Matrix g = probs;
    for (std::size_t i = 0; i < n; ++i) {
        g(i, static_cast<std::size_t>(y[i])) -= 1.0;
    }
    for (double& v : g.a) v /= static_cast<double>(n);

    const std::size_t n_layers = weights_.size();
    std::vector<Matrix> grads_w(n_layers);
    std::vector<std::vector<double>> grads_b(n_layers);
    for (std::size_t li = n_layers; li-- > 0;) {
        // dW_l = A_{l-1}^T G_l
        const Matrix& a = activations[li];
        Matrix dw(a.cols, g.cols, 0.0);
        for (std::size_t i = 0; i < a.rows; ++i) {
            for (std::size_t k = 0; k < a.cols; ++k) {
                const double aik = a(i, k);
                for (std::size_t j = 0; j < g.cols; ++j) {
                    dw(k, j) += aik * g(i, j);
                }
            }
        }
        grads_w[li] = std::move(dw);
        std::vector<double> db(g.cols, 0.0);
        for (std::size_t i = 0; i < g.rows; ++i) {
            for (std::size_t j = 0; j < g.cols; ++j) db[j] += g(i, j);
        }
        grads_b[li] = std::move(db);
        if (li > 0) {
            // G_{l-1} = (G_l W_l^T) .* [Z_{l-1} > 0]
            const Matrix& w = weights_[li];
            Matrix gprev(g.rows, w.rows, 0.0);
            for (std::size_t i = 0; i < g.rows; ++i) {
                for (std::size_t j = 0; j < w.rows; ++j) {
                    double s = 0.0;
                    for (std::size_t k = 0; k < g.cols; ++k) {
                        s += g(i, k) * w(j, k);
                    }
                    gprev(i, j) = zs[li - 1](i, j) > 0.0 ? s : 0.0;
                }
            }
            g = std::move(gprev);
        }
    }
    return {loss_val, std::move(grads_w), std::move(grads_b)};
}

std::vector<double> MLP::fit(const Matrix& X, const std::vector<int>& y,
                             double learning_rate, int epochs) {
    if (!(std::isfinite(learning_rate) && learning_rate > 0.0)) {
        throw std::invalid_argument("learning_rate must be positive, got " +
                                    std::to_string(learning_rate));
    }
    if (epochs <= 0) {
        throw std::invalid_argument("epochs must be a positive integer, got " +
                                    std::to_string(epochs));
    }
    check_matrix(X);
    if (X.cols != sizes_.front()) {
        throw std::invalid_argument("expected " + std::to_string(sizes_.front()) +
                                    " features, got " + std::to_string(X.cols));
    }
    check_labels(y, X.rows, sizes_.back());
    std::vector<double> history;
    history.reserve(static_cast<std::size_t>(epochs));
    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto [l, grads_w, grads_b] = loss_and_grads(X, y);
        (void)l; // pinned schedule records the loss *after* the update
        for (std::size_t li = 0; li < weights_.size(); ++li) {
            for (std::size_t idx = 0; idx < weights_[li].a.size(); ++idx) {
                weights_[li].a[idx] -= learning_rate * grads_w[li].a[idx];
            }
            for (std::size_t j = 0; j < biases_[li].size(); ++j) {
                biases_[li][j] -= learning_rate * grads_b[li][j];
            }
        }
        history.push_back(loss(X, y));
    }
    return history;
}

double accuracy(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
    if (y_true.empty() || y_true.size() != y_pred.size()) {
        throw std::invalid_argument(
            "y_true and y_pred must be equal-length non-empty arrays");
    }
    std::size_t hits = 0;
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == y_pred[i]) ++hits;
    }
    return static_cast<double>(hits) / static_cast<double>(y_true.size());
}

std::vector<std::vector<int>> confusion_matrix(const std::vector<int>& y_true,
                                               const std::vector<int>& y_pred,
                                               int n_classes) {
    if (n_classes <= 0) {
        throw std::invalid_argument("n_classes must be positive");
    }
    check_labels(y_true, y_true.size(), static_cast<std::size_t>(n_classes));
    check_labels(y_pred, y_true.size(), static_cast<std::size_t>(n_classes));
    std::vector<std::vector<int>> cm(
        static_cast<std::size_t>(n_classes),
        std::vector<int>(static_cast<std::size_t>(n_classes), 0));
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        cm[static_cast<std::size_t>(y_true[i])]
          [static_cast<std::size_t>(y_pred[i])] += 1;
    }
    return cm;
}

int majority_class(const std::vector<int>& y) {
    if (y.empty()) {
        throw std::invalid_argument("y must be a non-empty array");
    }
    std::map<int, std::size_t> counts;
    for (int v : y) ++counts[v];
    int best = counts.begin()->first;
    std::size_t best_count = counts.begin()->second;
    for (const auto& [label, count] : counts) {
        if (count > best_count) { // strict >: ties keep the smallest label
            best = label;
            best_count = count;
        }
    }
    return best;
}

std::tuple<double, double, int> signal_pnl(const std::vector<int>& y_pred,
                                           const std::vector<double>& fwd_ret,
                                           const std::vector<double>& spread) {
    if (y_pred.empty()) {
        throw std::invalid_argument("empty prediction array");
    }
    if (y_pred.size() != fwd_ret.size() || y_pred.size() != spread.size()) {
        throw std::invalid_argument(
            "y_pred, fwd_ret, spread must be equal-length arrays");
    }
    for (std::size_t i = 0; i < y_pred.size(); ++i) {
        if (y_pred[i] < 0 || y_pred[i] > 2) {
            throw std::invalid_argument("y_pred labels must be in {0, 1, 2}");
        }
        if (!(std::isfinite(fwd_ret[i]) && std::isfinite(spread[i]))) {
            throw std::invalid_argument("fwd_ret and spread must be finite");
        }
    }
    double gross = 0.0;
    double cost = 0.0;
    int n_trades = 0;
    for (std::size_t i = 0; i < y_pred.size(); ++i) {
        const int pos = y_pred[i] - 1; // {0,1,2} -> {-1,0,+1}
        gross += static_cast<double>(pos) * fwd_ret[i];
        cost += std::abs(pos) * spread[i] * 0.5;
        if (pos != 0) ++n_trades;
    }
    return {gross, gross - cost, n_trades};
}

} // namespace lobmod
