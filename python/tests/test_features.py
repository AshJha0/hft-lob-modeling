"""Feature pipeline: OFI hand checks, labels, NaN handling, split."""

import math

import numpy as np
import pytest

import lobmod as lm
from lobmod.replay import BookStates


def states_from(rows):
    """rows: list of (pb, qb, pa, qa); depth arrays mirror best sizes."""
    n = len(rows)
    pb = np.array([r[0] for r in rows], dtype=float)
    qb = np.array([r[1] for r in rows], dtype=float)
    pa = np.array([r[2] for r in rows], dtype=float)
    qa = np.array([r[3] for r in rows], dtype=float)
    mid = 0.5 * (pb + pa)
    return BookStates(
        best_bid=pb, best_ask=pa, bid_size=qb, ask_size=qa,
        bid_depth_k=qb.copy(), ask_depth_k=qa.copy(),
        mid=mid, microprice=mid.copy(), spread=pa - pb,
    )


def test_ofi_hand_computed_cases():
    st = states_from([
        (100, 5, 101, 5),
        (100, 8, 101, 5),   # bid size +3 at same price: e = 8 - 5 = +3
        (101, 2, 102, 4),   # bid improves (+2), ask backs off (+5): e = 2 + 5 = +7
        (100, 6, 102, 9),   # bid worse (-2), ask size +5 at same price: e = -2 - 9 + 4 = -7
    ])
    e = lm.ofi_series(st)
    assert math.isnan(e[0])
    assert e[1] == pytest.approx(3.0)
    assert e[2] == pytest.approx(7.0)
    assert e[3] == pytest.approx(-7.0)
    assert lm.ofi_at(st, 3, 3) == pytest.approx(3.0)  # 3 + 7 - 7
    assert lm.ofi_at(st, 2, 2) == pytest.approx(10.0)


def test_ofi_equal_prices_track_size_changes():
    st = states_from([(100, 5, 101, 7), (100, 9, 101, 3)])
    # bid: +9 -5 = +4 ; ask: -3 +7 = +4 -> e = 8
    assert lm.ofi_series(st)[1] == pytest.approx(8.0)


def test_ofi_nan_propagation():
    st = states_from([(100, 5, 101, 5), (100, 5, 101, 5)])
    st.best_ask[1] = math.nan
    assert math.isnan(lm.ofi_series(st)[1])


def test_ofi_at_validation():
    st = states_from([(100, 5, 101, 5)] * 10)
    with pytest.raises(ValueError, match="window"):
        lm.ofi_at(st, 5, 0)
    with pytest.raises(ValueError, match="does not fit"):
        lm.ofi_at(st, 2, 5)
    with pytest.raises(ValueError, match="out of range"):
        lm.ofi_at(st, 10, 2)


def cfg_small(**over):
    base = dict(ofi_window=2, ret_window=2, horizon=2, epsilon=0.25, depth_k=1,
                hidden=(4,), learning_rate=0.1, epochs=5, train_frac=0.5)
    base.update(over)
    return lm.FeatureConfig(**base)


def test_label_dead_zone_boundaries():
    # mid path: flat, then +1 tick jump, then -1 tick
    rows = [(100, 5, 101, 5)] * 3 + [(102, 5, 103, 5)] * 3 + [(100, 5, 101, 5)] * 3
    st = states_from(rows)
    fs = lm.compute_features(st, cfg_small())
    # every |mid change| is 0 or 2 ticks vs epsilon 0.25
    for i, y in zip(fs.event_index, fs.y):
        d = st.mid[i + 2] - st.mid[i]
        if d > 0.25:
            assert y == lm.LABEL_UP
        elif d < -0.25:
            assert y == lm.LABEL_DOWN
        else:
            assert y == lm.LABEL_FLAT
    # large epsilon swallows everything into the dead zone
    fs_flat = lm.compute_features(st, cfg_small(epsilon=10.0))
    assert np.all(fs_flat.y == lm.LABEL_FLAT)


def test_rows_with_empty_side_are_skipped():
    rows = [(100, 5, 101, 5)] * 12
    st = states_from(rows)
    st.best_ask[6] = math.nan
    st.mid[6] = math.nan
    st.spread[6] = math.nan
    fs = lm.compute_features(st, cfg_small())
    # no emitted row may use state 6 as its own index or inside its OFI window
    assert 6 not in fs.event_index
    assert 7 not in fs.event_index  # OFI window of size 2 touches e_7 = NaN
    full = lm.compute_features(states_from(rows), cfg_small())
    assert len(full) > len(fs)


def test_feature_vector_contents(states, cfg):
    fs = lm.compute_features(states, cfg)
    assert fs.X.shape == (len(fs), 5)
    assert np.all(np.isfinite(fs.X))
    i = fs.event_index[0]
    qb, qa = states.bid_size[i], states.ask_size[i]
    assert fs.X[0, 1] == pytest.approx((qb - qa) / (qb + qa))
    assert fs.X[0, 2] == pytest.approx(states.spread[i])
    assert fs.X[0, 3] == pytest.approx(states.bid_depth_k[i] / states.ask_depth_k[i])
    assert fs.X[0, 4] == pytest.approx(states.mid[i] - states.mid[i - cfg.ret_window])
    assert fs.fwd_ret[0] == pytest.approx(states.mid[i + cfg.horizon] - states.mid[i])
    assert set(np.unique(fs.y)).issubset({0, 1, 2})


def test_chronological_split_pinned():
    tr, te = lm.chronological_split(10, 0.7)
    assert tr.tolist() == list(range(7))
    assert te.tolist() == [7, 8, 9]
    tr, te = lm.chronological_split(4348, 0.7)
    assert tr.size == 3043 and te.size == 1305  # floor(0.7 * 4348)
    with pytest.raises(ValueError):
        lm.chronological_split(1, 0.7)
    with pytest.raises(ValueError):
        lm.chronological_split(10, 1.0)


def test_feature_config_validation():
    with pytest.raises(ValueError, match="ofi_window"):
        cfg_small(ofi_window=0)
    with pytest.raises(ValueError, match="epsilon"):
        cfg_small(epsilon=-1.0)
    with pytest.raises(ValueError, match="learning_rate"):
        cfg_small(learning_rate=0.0)
    with pytest.raises(ValueError, match="train_frac"):
        cfg_small(train_frac=1.5)
    with pytest.raises(ValueError, match="hidden"):
        cfg_small(hidden=(4, 0))


# ----------------------------------------------------------- review additions

def test_ofi_at_window_boundary():
    """PT-3: error iff event_index < window; event_index == window is valid."""
    st = states_from([(100, 5, 101, 5)] * 10)
    with pytest.raises(ValueError, match="does not fit"):
        lm.ofi_at(st, 3, 4)
    assert lm.ofi_at(st, 4, 4) == 0.0  # sums e_1..e_4, all defined and zero
    assert math.isfinite(lm.ofi_at(st, 4, 4))
    with pytest.raises(ValueError, match="does not fit"):
        lm.ofi_at(st, 9, 10)
    assert lm.ofi_at(st, 9, 9) == 0.0
    with pytest.raises(ValueError, match="window"):
        lm.ofi_at(st, 4, 2.0)
    assert lm.ofi_at(st, np.int64(4), np.int64(4)) == 0.0
