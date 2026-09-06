/// Neural network: pinned init, gradient check vs central finite
/// differences, softmax properties, deterministic retraining, degenerate
/// labels, metrics, and validation errors.

#include <cmath>
#include <stdexcept>
#include <gtest/gtest.h>

#include "lobmod/nn.hpp"
#include "test_data.hpp"

namespace {

using lobmod::Matrix;
using lobmod::MLP;

/// Small deterministic dataset (3 classes, 4 features, 9 rows).
struct TinyData {
    Matrix X{9, 4};
    std::vector<int> y;
    TinyData() {
        for (std::size_t i = 0; i < 9; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                X(i, j) = std::sin(0.7 * static_cast<double>(i * 4 + j) + 0.3);
            }
        }
        y = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    }
};

TEST(Nn, PinnedInitFormula) {
    const Matrix w = lobmod::pinned_init(5, 16, 0);
    ASSERT_EQ(w.rows, 5u);
    ASSERT_EQ(w.cols, 16u);
    const double scale = 0.5 * std::sqrt(2.0 / 5.0);
    EXPECT_DOUBLE_EQ(w(0, 0), scale * std::sin(1.0));
    EXPECT_DOUBLE_EQ(w(2, 3), scale * std::sin(2.0 * 16 + 3 + 1));
    const Matrix w1 = lobmod::pinned_init(16, 8, 1);
    EXPECT_DOUBLE_EQ(w1(0, 0), 0.5 * std::sqrt(2.0 / 16.0) * std::sin(10001.0));
    EXPECT_THROW(lobmod::pinned_init(0, 4, 0), std::invalid_argument);
    EXPECT_THROW(lobmod::pinned_init(4, 4, -1), std::invalid_argument);
}

TEST(Nn, SoftmaxRowsSumToOne) {
    const TinyData d;
    MLP net(4, {5}, 3);
    const Matrix p = net.predict_proba(d.X);
    ASSERT_EQ(p.rows, 9u);
    ASSERT_EQ(p.cols, 3u);
    for (std::size_t i = 0; i < p.rows; ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < p.cols; ++j) {
            ASSERT_GT(p(i, j), 0.0);
            s += p(i, j);
        }
        ASSERT_NEAR(s, 1.0, 1e-12);
    }
    // softmax must survive huge logits thanks to the pinned max shift
    Matrix big(1, 3);
    big(0, 0) = 1000.0;
    big(0, 1) = 999.0;
    big(0, 2) = -1000.0;
    const Matrix q = lobmod::softmax(big);
    EXPECT_NEAR(q(0, 0) + q(0, 1) + q(0, 2), 1.0, 1e-12);
    EXPECT_GT(q(0, 0), q(0, 1));
}

TEST(Nn, GradientCheckVsCentralDifferences) {
    const TinyData d;
    MLP net(4, {3}, 3);
    const auto [loss0, grads_w, grads_b] = net.loss_and_grads(d.X, d.y);
    EXPECT_GT(loss0, 0.0);
    const double h = 1e-6;
    // every weight of every layer against a central difference
    for (std::size_t l = 0; l < net.weights().size(); ++l) {
        for (std::size_t idx = 0; idx < net.weights()[l].a.size(); ++idx) {
            MLP plus = net, minus = net;
            plus.weights()[l].a[idx] += h;
            minus.weights()[l].a[idx] -= h;
            const double fd =
                (plus.loss(d.X, d.y) - minus.loss(d.X, d.y)) / (2.0 * h);
            const double an = grads_w[l].a[idx];
            const double denom = std::max({1.0, std::abs(fd), std::abs(an)});
            ASSERT_NEAR(an / denom, fd / denom, 1e-6)
                << "layer " << l << " weight " << idx;
        }
        for (std::size_t j = 0; j < net.biases()[l].size(); ++j) {
            MLP plus = net, minus = net;
            plus.biases()[l][j] += h;
            minus.biases()[l][j] -= h;
            const double fd =
                (plus.loss(d.X, d.y) - minus.loss(d.X, d.y)) / (2.0 * h);
            const double an = grads_b[l][j];
            const double denom = std::max({1.0, std::abs(fd), std::abs(an)});
            ASSERT_NEAR(an / denom, fd / denom, 1e-6)
                << "layer " << l << " bias " << j;
        }
    }
}

TEST(Nn, LossDecreasesUnderTraining) {
    const TinyData d;
    MLP net(4, {6}, 3);
    const double initial = net.loss(d.X, d.y);
    const auto hist = net.fit(d.X, d.y, 0.05, 200);
    ASSERT_EQ(hist.size(), 200u);
    EXPECT_LT(hist.front(), initial);
    for (std::size_t i = 1; i < hist.size(); ++i) {
        ASSERT_LE(hist[i], hist[i - 1] + 1e-12) << "epoch " << i;
    }
    // real pipeline history also decreases start to finish
    const auto& big = testdata::pipeline().history;
    EXPECT_LT(big.back(), big.front());
    EXPECT_LT(big[99], big[0]);
    EXPECT_LT(big.back(), big[99]);
}

TEST(Nn, RetrainingIsBitIdentical) {
    const TinyData d;
    MLP a(4, {5, 3}, 3), b(4, {5, 3}, 3);
    const auto ha = a.fit(d.X, d.y, 0.1, 50);
    const auto hb = b.fit(d.X, d.y, 0.1, 50);
    ASSERT_EQ(ha.size(), hb.size());
    for (std::size_t i = 0; i < ha.size(); ++i) {
        ASSERT_EQ(ha[i], hb[i]); // bit-identical, not just close
    }
    for (std::size_t l = 0; l < a.weights().size(); ++l) {
        for (std::size_t idx = 0; idx < a.weights()[l].a.size(); ++idx) {
            ASSERT_EQ(a.weights()[l].a[idx], b.weights()[l].a[idx]);
        }
    }
    const auto pa = a.predict(d.X);
    const auto pb = b.predict(d.X);
    EXPECT_EQ(pa, pb);
}

TEST(Nn, DegenerateOneClassLabelsTrainAndLoseToMajority) {
    const TinyData d;
    const std::vector<int> flat(d.y.size(), 1); // all flat
    MLP net(4, {5}, 3);
    const auto hist = net.fit(d.X, flat, 0.1, 100);
    for (double v : hist) ASSERT_TRUE(std::isfinite(v));
    EXPECT_LT(hist.back(), hist.front()); // still learns the constant class
    const int maj = lobmod::majority_class(flat);
    EXPECT_EQ(maj, 1);
    const std::vector<int> pred_mj(flat.size(), maj);
    EXPECT_DOUBLE_EQ(lobmod::accuracy(flat, pred_mj), 1.0);
    EXPECT_LE(lobmod::accuracy(flat, net.predict(d.X)), 1.0); // majority wins/ties
}

TEST(Nn, StandardizerPinnedMoments) {
    Matrix X(4, 2);
    const double col0[4] = {1.0, 2.0, 3.0, 4.0};
    for (std::size_t i = 0; i < 4; ++i) {
        X(i, 0) = col0[i];
        X(i, 1) = 7.0; // constant column: std floored to 1
    }
    lobmod::Standardizer sc;
    sc.fit(X);
    EXPECT_DOUBLE_EQ(sc.mean()[0], 2.5);
    EXPECT_DOUBLE_EQ(sc.std_dev()[0], std::sqrt(1.25)); // population divisor n
    EXPECT_DOUBLE_EQ(sc.std_dev()[1], 1.0);
    const Matrix Z = sc.transform(X);
    EXPECT_DOUBLE_EQ(Z(0, 1), 0.0); // (7 - 7) / 1
    EXPECT_NEAR(Z(3, 0), 1.5 / std::sqrt(1.25), 1e-15);
    lobmod::Standardizer unfit;
    EXPECT_THROW(unfit.transform(X), std::invalid_argument);
    Matrix bad(1, 3);
    EXPECT_THROW(sc.transform(bad), std::invalid_argument);
}

TEST(Nn, PredictionTiesGoToLowestClass) {
    Matrix z(1, 3, 0.0); // all-equal logits
    const Matrix p = lobmod::softmax(z);
    EXPECT_DOUBLE_EQ(p(0, 0), p(0, 2));
    MLP net(2, {}, 3);
    // zero out the logistic layer so probabilities tie exactly
    for (double& v : net.weights()[0].a) v = 0.0;
    Matrix X(1, 2);
    X(0, 0) = 0.3;
    X(0, 1) = -0.4;
    EXPECT_EQ(net.predict(X)[0], 0);
}

TEST(Nn, MetricsAndPnl) {
    const std::vector<int> y_true = {0, 1, 2, 2, 1};
    const std::vector<int> y_pred = {0, 2, 2, 0, 1};
    EXPECT_DOUBLE_EQ(lobmod::accuracy(y_true, y_pred), 0.6);
    const auto cm = lobmod::confusion_matrix(y_true, y_pred, 3);
    EXPECT_EQ(cm[0][0], 1);
    EXPECT_EQ(cm[1][2], 1);
    EXPECT_EQ(cm[2][2], 1);
    EXPECT_EQ(cm[2][0], 1);
    EXPECT_EQ(cm[1][1], 1);
    EXPECT_EQ(lobmod::majority_class({2, 2, 0, 0, 1}), 0); // tie -> smallest
    EXPECT_EQ(lobmod::majority_class({2, 2, 0, 1}), 2);
    // PnL: pos = pred - 1; gross = sum pos*fwd; net subtracts half spreads
    const auto [gross, net, n_trades] = lobmod::signal_pnl(
        {2, 0, 1}, {1.5, -2.0, 3.0}, {1.0, 2.0, 1.0});
    EXPECT_DOUBLE_EQ(gross, 1.5 + 2.0);
    EXPECT_DOUBLE_EQ(net, 3.5 - (0.5 + 1.0));
    EXPECT_EQ(n_trades, 2);
    EXPECT_THROW(lobmod::signal_pnl({}, {}, {}), std::invalid_argument);
    EXPECT_THROW(lobmod::signal_pnl({1}, {1.0, 2.0}, {1.0}),
                 std::invalid_argument);
    EXPECT_THROW(lobmod::signal_pnl({1}, {std::nan("")}, {1.0}),
                 std::invalid_argument);
    EXPECT_THROW(lobmod::accuracy({}, {}), std::invalid_argument);
    EXPECT_THROW(lobmod::majority_class({}), std::invalid_argument);
}

TEST(Nn, ValidationErrors) {
    const TinyData d;
    EXPECT_THROW(MLP(0, {4}, 3), std::invalid_argument);
    EXPECT_THROW(MLP(4, {0}, 3), std::invalid_argument);
    EXPECT_THROW(MLP(4, {4}, 1), std::invalid_argument);
    MLP net(4, {4}, 3);
    EXPECT_THROW(net.fit(d.X, d.y, 0.0, 10), std::invalid_argument);
    EXPECT_THROW(net.fit(d.X, d.y, -1.0, 10), std::invalid_argument);
    EXPECT_THROW(net.fit(d.X, d.y, 0.1, 0), std::invalid_argument);
    std::vector<int> bad_labels = d.y;
    bad_labels[0] = 3; // out of range
    EXPECT_THROW(net.fit(d.X, bad_labels, 0.1, 10), std::invalid_argument);
    bad_labels[0] = -1;
    EXPECT_THROW(net.loss(d.X, bad_labels), std::invalid_argument);
    Matrix wrong(9, 3); // wrong feature count
    EXPECT_THROW(net.predict_proba(wrong), std::invalid_argument);
    Matrix nonfinite = d.X;
    nonfinite(0, 0) = std::nan("");
    EXPECT_THROW(net.loss(nonfinite, d.y), std::invalid_argument);
}

// ------------------------------------------------------------ review additions

TEST(Nn, SignalPnlRejectsInvalidLabels) {
    // PT-5
    EXPECT_THROW(lobmod::signal_pnl({3}, {1.0}, {1.0}), std::invalid_argument);
    EXPECT_THROW(lobmod::signal_pnl({-1}, {1.0}, {1.0}), std::invalid_argument);
    const auto [gross, net, n] = lobmod::signal_pnl({2, 0, 1}, {1.0, -2.0, 5.0}, {1.0, 1.0, 1.0});
    EXPECT_DOUBLE_EQ(gross, 3.0);
    EXPECT_DOUBLE_EQ(net, 2.0);
    EXPECT_EQ(n, 2);
}

double fd_worst(MLP& net, const Matrix& x, const std::vector<int>& y) {
    const auto [loss0, grads_w, grads_b] = net.loss_and_grads(x, y);
    (void)loss0;
    const double h = 1e-6;
    double worst = 0.0;
    for (std::size_t l = 0; l < net.weights().size(); ++l) {
        for (std::size_t idx = 0; idx < net.weights()[l].a.size(); ++idx) {
            MLP plus = net, minus = net;
            plus.weights()[l].a[idx] += h;
            minus.weights()[l].a[idx] -= h;
            const double fd = (plus.loss(x, y) - minus.loss(x, y)) / (2.0 * h);
            const double an = grads_w[l].a[idx];
            worst = std::max(worst, std::abs(fd - an) / std::max({1.0, std::abs(fd), std::abs(an)}));
        }
        for (std::size_t j = 0; j < net.biases()[l].size(); ++j) {
            MLP plus = net, minus = net;
            plus.biases()[l][j] += h;
            minus.biases()[l][j] -= h;
            const double fd = (plus.loss(x, y) - minus.loss(x, y)) / (2.0 * h);
            const double an = grads_b[l][j];
            worst = std::max(worst, std::abs(fd - an) / std::max({1.0, std::abs(fd), std::abs(an)}));
        }
    }
    return worst;
}

TEST(Nn, GradientCheckLogisticHeadAndTwoClassNet) {
    // PT-14: the l == 0 early exit of backprop (no hidden layer) and a 2-class head
    Matrix x(12, 5);
    std::vector<int> y(12);
    for (std::size_t i = 0; i < 12; ++i) {
        for (std::size_t j = 0; j < 5; ++j) {
            x(i, j) = std::sin(0.7 * static_cast<double>(i * 5 + j + 1));
        }
        const double score = x(i, 0) + 0.5 * x(i, 1) - 0.8 * x(i, 2);
        y[i] = score > 0.4 ? 2 : (score < -0.4 ? 0 : 1);
    }
    MLP logistic(5, {}, 3);
    EXPECT_LT(fd_worst(logistic, x, y), 1e-6);
    Matrix x2(12, 4);
    std::vector<int> y2(12);
    for (std::size_t i = 0; i < 12; ++i) {
        for (std::size_t j = 0; j < 4; ++j) x2(i, j) = x(i, j);
        y2[i] = y[i] > 0 ? 1 : 0;
    }
    MLP two(4, {3}, 2);
    EXPECT_LT(fd_worst(two, x2, y2), 1e-6);
}

} // namespace
