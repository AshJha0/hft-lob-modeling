"""Local studies on the per-symbol extracts written by research/real_replay.py.

  python research/calibrate.py hawkes DATA_DIR [--workers 4] [SYMBOL ...]
  python research/calibrate.py fit DATA_DIR [--workers 5] [SYMBOL ...]
  python research/calibrate.py fit2 DATA_DIR [--workers 5] [SYMBOL ...]
  python research/calibrate.py fit3 DATA_DIR [--workers 5] [SYMBOL ...]
  python research/calibrate.py heldout DATA_DIR [--workers 5] [SYMBOL ...]

Both read DATA_DIR/extract/*.npz and write results into DATA_DIR only: every number
they produce derives from licensed data and stays out of the repository.

``hawkes``: aggressive orders from 10:00 to 15:30 (first half of that if there are
more than MAX_HAWKES_N), spline baseline with 30-minute knots.  Single exponential,
K=2 and power-law kernels, flat-baseline controls, then a look at the fast end
of the kernel: gaps, a fixed-rate spectrum reaching 10 microseconds, and a refit
after merging same-side orders closer than 1 ms.

``fit``: fit on the morning window, judge against
research/calibration_criteria.md in sample and on the afternoon window.

``fit2``: model v2, fitted and judged on the whole of 10:00 to 15:30, with morning
and afternoon rows as in-sample diagnostics.  ``fit3``: model v3, same windows, with row C7
beside the others.  ``heldout``: the fit3 parameters
against DATA_DIR/heldout/extract, once; it refuses to overwrite an earlier result.
"""
import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'python' / 'src'))
sys.path.insert(0, str(ROOT / 'research'))

from lobmod.gof import ks_test_exp
from lobmod.research.kernels import FlatBaseline, fit_fixed_mixture, fit_power_law, fit_residuals, fit_sum_exp
from lobmod.research.seasonality import LinearSplineBaseline
from lobmod.sim.calibrate import (
    MEAN_LIMIT_LOTS, REFINED_V2, REFINED_V3, book_fit, build_stats, evaluate, mean_stats, moment_fit, refine, simulate_stats,
    evaluate_gaps, structural_fit,
)
from real_replay import BUCKET_NS, TICK, aggregate_signs

SYMBOLS = ('AAPL', 'MSFT', 'INTC', 'NVDA', 'TSLA', 'SPY', 'F', 'CSX', 'ZION', 'JBLU')
H = 3600 * 10 ** 9
START, MIDDAY, END = 10 * H, int(12.75 * H), int(15.5 * H)
OPEN_NS = int(9.5 * H)
GRID_NS = 10 ** 8
MAX_HAWKES_N = 300000
FAST_RATES = tuple(np.geomspace(1e-3, 1e5, 25))
FIT_SEEDS, CHECK_SEEDS = (1, 2), (101, 102, 103, 104)
FIT_T, CHECK_T, WARMUP, BUDGET = 600.0, 3600.0, 60.0, 60


def load(data, sym):
    with np.load(Path(data) / 'extract' / f'{sym}.npz') as z:
        return {k: z[k] for k in z.files}


def order_times(a, lo, hi):
    """Aggressive orders in a window; one point per nanosecond, as Hawkes needs."""
    ts, sg = aggregate_signs(a['tr_ts'], a['tr_sign'])
    m = (ts >= lo) & (ts < hi)
    ts, sg = ts[m], sg[m]
    keep = np.ones(ts.size, bool)
    keep[1:] = ts[1:] != ts[:-1]
    return ts[keep], sg[keep]


def merge_bursts(ts, sg, gap_ns):
    """Same-side orders closer than ``gap_ns`` to the previous one on that side become one."""
    keep = np.zeros(ts.size, bool)
    for s in (-1, 1):
        idx = np.nonzero(sg == s)[0]
        if idx.size:
            keep[idx[np.r_[True, np.diff(ts[idx]) >= gap_ns]]] = True
    return ts[keep]


def describe(fit, t, T, base):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = fit_residuals(t, T, fit, base)
    ks, _ = ks_test_exp(r)
    return {'loglik': fit.loglik, 'aic': fit.aic, 'bic': fit.bic, 'branching': fit.branching,
            'weights': fit.weights.tolist(), 'rates': fit.betas.tolist() if fit.betas.size <= 3 else None,
            'shape': list(fit.shape), 'converged': fit.converged, 'residual_ks': float(ks),
            'residual_acf1': float(np.corrcoef(r[:-1], r[1:])[0, 1])}


def hawkes_one(job):
    data, sym = job
    t0 = time.time()
    a = load(data, sym)
    ts, sg = order_times(a, START, END)
    hi = END
    if ts.size > MAX_HAWKES_N:
        hi = MIDDAY
        ts, sg = order_times(a, START, hi)
    T = (hi - START) / 1e9
    t = (ts - START) / 1e9
    t = t[t > 0]
    gaps = np.diff(ts)
    mean_gap = T / max(t.size, 1)
    out = {'n': int(t.size), 'T': T, 'gaps': {
        f'{int(g * 1e6)}us': {'share': float(np.mean(gaps < g * 1e9)), 'poisson': float(1 - np.exp(-g / mean_gap))}
        for g in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)}}
    if t.size < 500:
        return sym, {**out, 'skipped': 'too few orders'}
    J = max(1, round(T / 1800))
    season, flat = LinearSplineBaseline(T, J), FlatBaseline(T)
    fits = {}
    for name, fn, base in (
            ('flat_K1', lambda: fit_sum_exp(t, T, 1, flat, n_starts=1), flat),
            ('seasonal_K1', lambda: fit_sum_exp(t, T, 1, season, n_starts=1), season),
            ('seasonal_K2', lambda: fit_sum_exp(t, T, 2, season, n_starts=1), season),
            ('seasonal_power', lambda: fit_power_law(t, T, season), season),
            ('seasonal_power_fast', lambda: fit_power_law(t, T, season, rates=FAST_RATES), season)):
        fits[name] = describe(fn(), t, T, base)
        print(f'{sym} {name} {time.time() - t0:.0f}s', flush=True)
    mix = fit_fixed_mixture(t, T, FAST_RATES, season)
    out['spectrum'] = {'rates': list(FAST_RATES), 'weights': mix.weights.tolist(), 'branching': mix.branching,
                       'loglik': mix.loglik, 'bic': mix.bic}
    b = merge_bursts(ts, sg, 10 ** 6)
    tb = (b - START) / 1e9
    tb = tb[tb > 0]
    out['merged_1ms'] = {'n': int(tb.size), **describe(fit_sum_exp(tb, T, 2, season, n_starts=1), tb, T, season)}
    out['fits'] = fits
    out['seconds'] = round(time.time() - t0)
    print(f'{sym} done {out["seconds"]}s', flush=True)
    return sym, out


def real_stats(a, lo, hi, lot):
    g = a['grid']
    t = OPEN_NS + np.arange(g.shape[0], dtype=np.int64) * GRID_NS
    g = g[(t >= lo) & (t < hi) & (g[:, 0] > 0)]
    m = (a['ev_ts'] >= lo) & (a['ev_ts'] < hi)
    cls = a['ev_code'][m] // 2
    ots, sg = aggregate_signs(*(x[(a['tr_ts'] >= lo) & (a['tr_ts'] < hi)] for x in (a['tr_ts'], a['tr_sign'])))
    k = a['bucket']
    i = np.nonzero((k[1:] == k[:-1] + 1) & (k[:-1] * BUCKET_NS >= lo) & (k[1:] * BUCKET_NS < hi))[0]
    return build_stats((g[:, 1] - g[:, 0]) / TICK, np.concatenate([g[:, 2], g[:, 3]]) / lot,
                       int(np.sum(cls == 0)), int(np.sum(cls == 1)), int(np.sum(cls == 2)), (hi - lo) / 1e9, sg,
                       a['ofi'][i + 1], (a['mid_end'][i + 1] - a['mid_end'][i]) / TICK, ots / 1e9)


def simulated(flow, seeds, T):
    return mean_stats([simulate_stats(flow, T, s, WARMUP) for s in seeds])


def fit_one(job):
    data, sym = job
    t0 = time.time()
    a = load(data, sym)
    m = (a['ev_ts'] >= START) & (a['ev_ts'] < MIDDAY) & (a['ev_code'] // 2 == 0)
    lot = float(a['ev_size'][m].mean()) / MEAN_LIMIT_LOTS
    fit_real = real_stats(a, START, MIDDAY, lot)
    ts, _ = order_times(a, START, MIDDAY)
    t = (ts - START) / 1e9
    T = (MIDDAY - START) / 1e9
    h = fit_sum_exp(t[t > 0], T, 1, n_starts=1)
    start = moment_fit(fit_real, branching=min(h.branching, 0.9), beta=float(h.betas[0]))

    def sim(flow):
        try:
            return simulated(flow, FIT_SEEDS, FIT_T)
        except ValueError:
            return {**fit_real, 'event_rate': 100.0 * fit_real['event_rate']}

    res = refine(fit_real, start, sim, BUDGET)
    print(f'{sym} refined {res.start_value:.1f} -> {res.value:.1f} in {time.time() - t0:.0f}s', flush=True)
    paths = simulated(res.flow, CHECK_SEEDS, CHECK_T)
    check_real = real_stats(a, MIDDAY, END, lot)      # first read here, after fitting
    return sym, {'lot_shares': lot, 'hawkes_K1': {'branching': h.branching, 'beta': float(h.betas[0])},
                 'flow': asdict(res.flow), 'distance_start': res.start_value, 'distance': res.value,
                 'real_fit': fit_real, 'real_check': check_real, 'sim': paths,
                 'in_sample': evaluate(paths, fit_real), 'within_day': evaluate(paths, check_real),
                 'seconds': round(time.time() - t0)}


def activity_ratio(a, lo, hi, width_ns=300 * 10 ** 9):
    """Busy over quiet event rate: 90th over 10th percentile of five-minute counts."""
    n = np.histogram(a['ev_ts'], np.arange(lo, hi + 1, width_ns))[0]
    return float(np.clip(np.percentile(n, 90) / max(np.percentile(n, 10), 1.0), 1.0, 6.0))


def fit2_one(job):
    data, sym = job
    t0 = time.time()
    a = load(data, sym)
    m = (a['ev_ts'] >= START) & (a['ev_ts'] < END) & (a['ev_code'] // 2 == 0)
    lot = float(a['ev_size'][m].mean()) / MEAN_LIMIT_LOTS
    real = real_stats(a, START, END, lot)
    ts, _ = order_times(a, START, END)
    t = (ts - START) / 1e9
    h = fit_sum_exp(t[t > 0], (END - START) / 1e9, 1, n_starts=1)
    start = structural_fit(real, moment_fit(real, branching=min(h.branching, 0.9), beta=float(h.betas[0])),
                           activity_ratio(a, START, END))

    def sim(flow):
        try:
            return simulated(flow, FIT_SEEDS, FIT_T)
        except ValueError:
            return {**real, 'event_rate': 100.0 * real['event_rate']}

    res = refine(real, start, sim, BUDGET, REFINED_V2)
    print(f'{sym} refined {res.start_value:.1f} -> {res.value:.1f} in {time.time() - t0:.0f}s', flush=True)
    paths = simulated(res.flow, CHECK_SEEDS, CHECK_T)
    morning, afternoon = real_stats(a, START, MIDDAY, lot), real_stats(a, MIDDAY, END, lot)
    return sym, {'lot_shares': lot, 'flow': asdict(res.flow), 'distance': res.value, 'real': real, 'sim': paths,
                 'in_sample': evaluate(paths, real), 'morning': evaluate(paths, morning),
                 'afternoon': evaluate(paths, afternoon), 'seconds': round(time.time() - t0)}


def fit3_one(job):
    data, sym = job
    t0 = time.time()
    a = load(data, sym)
    m = (a['ev_ts'] >= START) & (a['ev_ts'] < END) & (a['ev_code'] // 2 == 0)
    lot = float(a['ev_size'][m].mean()) / MEAN_LIMIT_LOTS
    real = real_stats(a, START, END, lot)
    ts, _ = order_times(a, START, END)
    t = (ts - START) / 1e9
    h = fit_sum_exp(t[t > 0], (END - START) / 1e9, 1, n_starts=1)
    gaps = np.diff(t)
    short = gaps[gaps < 1e-3]
    start = book_fit(real, moment_fit(real, branching=min(h.branching, 0.9), beta=float(h.betas[0])),
                     activity_ratio(a, START, END), real['gap_shares'][1],
                     float(short.mean()) if short.size else 2e-4)

    def sim(flow):
        try:
            return simulated(flow, FIT_SEEDS, FIT_T)
        except ValueError:
            return {**real, 'event_rate': 100.0 * real['event_rate']}

    res = refine(real, start, sim, BUDGET, REFINED_V3)
    print(f'{sym} refined {res.start_value:.1f} -> {res.value:.1f} in {time.time() - t0:.0f}s', flush=True)
    paths = simulated(res.flow, CHECK_SEEDS, CHECK_T)
    morning, afternoon = real_stats(a, START, MIDDAY, lot), real_stats(a, MIDDAY, END, lot)
    return sym, {'lot_shares': lot, 'flow': asdict(res.flow), 'distance': res.value, 'real': real, 'sim': paths,
                 'in_sample': evaluate(paths, real), 'c7': evaluate_gaps(paths, real), 'morning': evaluate(paths, morning),
                 'afternoon': evaluate(paths, afternoon), 'seconds': round(time.time() - t0)}


def heldout_one(job):
    data, sym, fitted = job
    a = load(Path(data) / 'heldout', sym)
    real = real_stats(a, START, END, fitted['lot_shares'])
    return sym, {'real': real, 'held_out': evaluate(fitted['sim'], real), 'c7': evaluate_gaps(fitted['sim'], real)}


def verdict(res, names=('in_sample', 'within_day')):
    out = {}
    for name in names:
        rows = {s: r[name] for s, r in res.items()}
        passed = [s for s, r in rows.items() if all(v['passed'] for v in r.values())]
        out[name] = {'symbols_passed': passed, 'model_pass': len(passed) >= 8,
                     'row_passes': {k: sum(r[k]['passed'] for r in rows.values()) for k in next(iter(rows.values()))}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('hawkes', 'fit', 'fit2', 'fit3', 'heldout'))
    ap.add_argument('data')
    ap.add_argument('symbols', nargs='*', default=list(SYMBOLS))
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    if a.mode == 'heldout':
        target = Path(a.data) / 'heldout_results.json'
        if target.exists():
            raise SystemExit('held-out result exists: the test runs once')
        fitted = json.loads((Path(a.data) / 'calibration_v3_results.json').read_text())['symbols']
        with ProcessPoolExecutor(a.workers) as ex:
            res = dict(ex.map(heldout_one, [(a.data, s, fitted[s]) for s in a.symbols]))
        target.write_text(json.dumps({'symbols': res, 'verdict': verdict(res, ('held_out',))}, indent=1, default=str))
        print('written', target.name, flush=True)
        return
    fn = {'hawkes': hawkes_one, 'fit': fit_one, 'fit2': fit2_one, 'fit3': fit3_one}[a.mode]
    with ProcessPoolExecutor(a.workers) as ex:
        res = dict(ex.map(fn, [(a.data, s) for s in a.symbols]))
    if a.mode != 'hawkes':
        names = ('in_sample', 'morning', 'afternoon') if a.mode in ('fit2', 'fit3') else ('in_sample', 'within_day')
        res = {'symbols': res, 'verdict': verdict(res, names)}
    name = {'hawkes': 'hawkes_results.json', 'fit': 'calibration_results.json',
            'fit2': 'calibration_v2_results.json',
            'fit3': 'calibration_v3_results.json'}[a.mode]
    (Path(a.data) / name).write_text(json.dumps(res, indent=1, default=str))
    print('written', name, flush=True)


if __name__ == '__main__':
    main()
