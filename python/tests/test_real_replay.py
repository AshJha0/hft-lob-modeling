"""research/real_replay.py on hand-built fixtures; it never needs a real file."""
import gzip
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from test_feeds import add, directory, executed, framed, sys_event

_PATH = Path(__file__).resolve().parents[2] / 'research' / 'real_replay.py'
_spec = importlib.util.spec_from_file_location('real_replay', _PATH)
rr = importlib.util.module_from_spec(_spec)
sys.modules['real_replay'] = rr
_spec.loader.exec_module(rr)

T = 10 * 3600 * 10 ** 9


def test_aggregate_signs_merges_one_sweep_into_one_order():
    t, s = rr.aggregate_signs([5, 5, 5, 6, 6, 7], [1, 1, 1, 1, -1, -1])
    assert t.tolist() == [5, 6, 6, 7]
    assert s.tolist() == [1, 1, -1, -1]


def test_aggregate_signs_keeps_distinct_times_and_empty_input():
    t, s = rr.aggregate_signs([1, 2, 3], [1, 1, 1])
    assert t.tolist() == [1, 2, 3] and s.tolist() == [1, 1, 1]
    assert rr.aggregate_signs([], [])[0].size == 0
    with pytest.raises(ValueError):
        rr.aggregate_signs([1, 2], [1])


def test_aggregation_removes_mechanical_sign_memory():
    rng = np.random.default_rng(1)
    orders = rng.choice([-1, 1], 20000)
    fills = rng.integers(1, 6, orders.size)
    s = np.repeat(orders, fills)
    t = np.repeat(np.arange(orders.size), fills)
    assert rr.acf(s, (1,))['1'] > 0.4
    assert abs(rr.acf(rr.aggregate_signs(t, s)[1], (1,))['1']) < 0.03


def test_lean_book_tracks_best_through_deletes():
    b = rr.LeanBook()
    b.add(1, 0, 100, 5)
    b.add(2, 0, 101, 5)
    b.add(3, 0, 101, 2)
    assert b.best(0) == 101 and b.levels[0][101] == [7, 2]
    b.reduce(2, None)
    b.reduce(3, 2)
    assert b.best(0) == 100
    b.add(4, 0, 101, 1)
    assert b.best(0) == 101
    assert b.verify() == 'ok'


def test_one_pass_extract_on_a_fixture(tmp_path):
    msgs = [sys_event(), directory(), directory('MSFT', 2),
            add(1, b'B', 100, 1000000, ts=T), add(2, b'S', 300, 1000200, ts=T + 1),
            add(9, b'B', 7, 500000, name='MSFT', locate=2, ts=T + 1),
            add(3, b'B', 50, 999900, ts=T + 2),
            executed(2, 100, ts=T + 3), executed(2, 100, match=8, ts=T + 3),
            executed(1, 40, match=9, ts=T + 4)]
    path = tmp_path / 'f.gz'
    with gzip.open(path, 'wb') as fh:
        fh.write(framed(*msgs).getvalue())
    s = rr.main(path, tmp_path, ['AAPL'])['symbols']['AAPL']
    assert s['touch'] == {'at_touch': 3} and s['verify'] == 'ok' and s['crossed_states'] == 0
    assert s['fills_rth'] == 3 and s['orders_rth'] == 2
    a = np.load(tmp_path / 'extract' / 'AAPL.npz')
    assert a['tr_sign'].tolist() == [1, 1, -1]
    # the second add arrives with a two-sided book absent, so only later events are logged
    assert a['ev_code'].tolist() == [0, 5, 5, 4]
    assert a['ev_dist'].tolist() == [1, 0, 0, 0]
    assert a['grid'].shape == (int(6.5 * 3600 * 10), 6)
