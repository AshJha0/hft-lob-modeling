"""Neural net: gradient checks, determinism, training behavior, metrics."""

import numpy as np
import pytest

import lobmod as lm
from lobmod.nn import pinned_init


def toy_data(n=40, d=5):
    """Deterministic, linearly-structured 3-class toy problem (no RNG)."""
    x = np.array([[np.sin(0.7 * (i * d + j + 1)) for j in range(d)] for i in range(n)])
    score = x[:, 0] + 0.5 * x[:, 1] - 0.8 * x[:, 2]
    y = np.where(score > 0.4, 2, np.where(score < -0.4, 0, 1))
    return x, y


def test_pinned_init_reproducible_and_bounded():
    w1 = pinned_init(5, 16, 0)
    w2 = pinned_init(5, 16, 0)
    assert np.array_equal(w1, w2)
    scale = 0.5 * np.sqrt(2.0 / 5.0)
    assert np.max(np.abs(w1)) <= scale + 1e-15
    assert not np.array_equal(w1, pinned_init(5, 16, 1))  # layer index matters
    assert w1[0, 0] == pytest.approx(scale * np.sin(1.0), abs=1e-15)


def test_gradient_check_vs_finite_differences():
    """Backprop gradient matches central differences to 1e-6 on a tiny net."""
    x, y = toy_data(12)
    net = lm.MLP(5, (4, 3), 3)
    _, gw, gb = net.loss_and_grads(x, y)
    h = 1e-6
    worst = 0.0
    for l in range(len(net.weights)):
        rows, cols = net.weights[l].shape
        for (i, j) in [(0, 0), (rows // 2, cols // 2), (rows - 1, cols - 1)]:
            w0 = net.weights[l][i, j]
            net.weights[l][i, j] = w0 + h
            lp = net.loss(x, y)
            net.weights[l][i, j] = w0 - h
            lo = net.loss(x, y)
            net.weights[l][i, j] = w0
            fd = (lp - lo) / (2 * h)
            worst = max(worst, abs(fd - gw[l][i, j]) / max(1e-8, abs(fd)))
        for j in (0, net.biases[l].size - 1):
            b0 = net.biases[l][j]
            net.biases[l][j] = b0 + h
            lp = net.loss(x, y)
            net.biases[l][j] = b0 - h
            lo = net.loss(x, y)
            net.biases[l][j] = b0
            fd = (lp - lo) / (2 * h)
            worst = max(worst, abs(fd - gb[l][j]) / max(1e-8, abs(fd)))
    assert worst < 1e-6


def test_softmax_rows_sum_to_one():
    x, y = toy_data(30)
    net = lm.MLP(5, (16, 8), 3)
    p = net.predict_proba(x)
    assert p.shape == (30, 3)
    assert np.all(p > 0)
    assert np.allclose(np.sum(p, axis=1), 1.0, atol=1e-12)
    # overflow safety of the pinned max-shift
    p2 = lm.softmax(np.array([[1000.0, 0.0, -1000.0]]))
    assert np.isfinite(p2).all() and p2[0, 0] == pytest.approx(1.0)


def test_loss_decreases_over_epochs():
    x, y = toy_data(60)
    net = lm.MLP(5, (16, 8), 3)
    hist = net.fit(x, y, 0.2, 200)
    assert hist[-1] < hist[0]
    assert hist[-1] < 0.7 * hist[0]
    # loss history is finite throughout
    assert np.all(np.isfinite(hist))


def test_deterministic_retrain_bitwise_identical():
    x, y = toy_data(50)
    a = lm.MLP(5, (16, 8), 3)
    ha = a.fit(x, y, 0.1, 100)
    b = lm.MLP(5, (16, 8), 3)
    hb = b.fit(x, y, 0.1, 100)
    assert ha == hb  # exact float equality — full determinism
    for wa, wb in zip(a.weights, b.weights):
        assert np.array_equal(wa, wb)


def test_logistic_learns_separable_toy():
    x, y = toy_data(80)
    logistic = lm.MLP(5, (), 3)
    logistic.fit(x, y, 0.5, 400)
    assert lm.accuracy(y, logistic.predict(x)) > 0.85


def test_degenerate_single_class_labels():
    """All-flat labels: training still runs and the majority baseline wins."""
    x, _ = toy_data(40)
    y = np.full(40, lm.LABEL_FLAT)
    net = lm.MLP(5, (16, 8), 3)
    hist = net.fit(x, y, 0.1, 50)
    assert np.all(np.isfinite(hist)) and hist[-1] < hist[0]
    maj = lm.majority_class(y)
    assert maj == lm.LABEL_FLAT
    acc_maj = lm.accuracy(y, np.full(40, maj))
    assert acc_maj == 1.0
    assert lm.accuracy(y, net.predict(x)) <= acc_maj


def test_standardizer_pinned_semantics():
    x = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    std = lm.Standardizer().fit(x)
    assert std.mean == pytest.approx([3.0, 5.0])
    # population std (divisor n); zero-variance column guarded to 1
    assert std.std[0] == pytest.approx(np.sqrt(8.0 / 3.0))
    assert std.std[1] == 1.0
    z = std.transform(x)
    assert np.mean(z[:, 0]) == pytest.approx(0.0, abs=1e-15)
    assert z[:, 1] == pytest.approx([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="before fit"):
        lm.Standardizer().transform(x)


def test_confusion_matrix_and_accuracy():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 0, 2])
    cm = lm.confusion_matrix(y_true, y_pred, 3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert cm.sum() == 6
    assert lm.accuracy(y_true, y_pred) == pytest.approx(4.0 / 6.0)


def test_majority_class_tie_break():
    assert lm.majority_class(np.array([2, 2, 0, 0, 1])) == 0  # pinned: smallest


def test_signal_pnl_arithmetic():
    pred = np.array([2, 0, 1, 2])          # long, short, flat, long
    fwd = np.array([1.0, -2.0, 5.0, -0.5])
    spr = np.array([1.0, 1.0, 1.0, 3.0])
    gross, net, n = lm.signal_pnl(pred, fwd, spr)
    assert gross == pytest.approx(1.0 + 2.0 + 0.0 - 0.5)
    assert net == pytest.approx(gross - (0.5 + 0.5 + 1.5))
    assert n == 3


def test_nn_validation_errors():
    x, y = toy_data(10)
    net = lm.MLP(5, (4,), 3)
    with pytest.raises(ValueError, match="learning_rate"):
        net.fit(x, y, -0.1, 10)
    with pytest.raises(ValueError, match="epochs"):
        net.fit(x, y, 0.1, 0)
    with pytest.raises(ValueError, match="labels"):
        net.fit(x, np.full(10, 7), 0.1, 10)
    with pytest.raises(ValueError, match="finite"):
        net.fit(np.full((10, 5), np.nan), y, 0.1, 10)
    with pytest.raises(ValueError, match="features"):
        net.predict(np.zeros((3, 4)))
    with pytest.raises(ValueError):
        lm.MLP(0, (4,), 3)
    with pytest.raises(ValueError):
        lm.MLP(5, (4, -1), 3)
    with pytest.raises(ValueError):
        lm.signal_pnl(np.array([1]), np.array([1.0, 2.0]), np.array([1.0]))
