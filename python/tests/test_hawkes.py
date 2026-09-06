"""Hawkes: recursion vs brute force, MLE recovery, residuals, edge cases."""

import math
import os
import warnings

import numpy as np
import pytest

import lobmod as lm

from conftest import DATA_DIR


def small_sample():
    return np.array([0.5, 1.2, 1.35, 2.8, 3.1, 3.15, 4.9, 7.2, 7.5, 9.0]), 10.0


def test_uni_recursion_matches_brute_force():
    t, T = small_sample()
    for mu, a, b in [(0.5, 0.8, 1.5), (1.0, 0.1, 0.5), (0.2, 2.0, 4.0)]:
        assert lm.uni_loglik(t, T, mu, a, b) == pytest.approx(
            lm.uni_loglik_brute(t, T, mu, a, b), abs=1e-10
        )


def test_uni_alpha_zero_is_poisson():
    t, T = small_sample()
    # alpha = 0 -> homogeneous Poisson: ll = N log mu - mu T
    assert lm.uni_loglik(t, T, 0.7, 0.0, 1.0) == pytest.approx(
        t.size * math.log(0.7) - 0.7 * T, abs=1e-12
    )


def test_biv_recursion_matches_brute_force():
    t, T = small_sample()
    k = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1])
    for mu, a, b in [
        ([0.3, 0.4], [[0.5, 0.2], [0.1, 0.6]], 2.0),
        ([0.2, 0.2], [[0.9, 0.0], [0.0, 0.9]], 1.1),
    ]:
        assert lm.biv_loglik(t, k, T, mu, a, b) == pytest.approx(
            lm.biv_loglik_brute(t, k, T, mu, a, b), abs=1e-10
        )


def test_simulation_deterministic_and_sorted():
    a = lm.simulate_uni(0.5, 1.0, 1.6, 100.0, 42)
    b = lm.simulate_uni(0.5, 1.0, 1.6, 100.0, 42)
    assert np.array_equal(a, b)
    assert np.all(np.diff(a) > 0)
    assert a[0] > 0 and a[-1] <= 100.0
    c = lm.simulate_uni(0.5, 1.0, 1.6, 100.0, 43)
    assert not np.array_equal(a, c)


def test_simulated_refit_close(uni_times, uni_fit, dgp):
    """Loose parameter recovery on the bundled sample (known DGP)."""
    true = dgp["hawkes_uni"]
    assert uni_fit.converged
    assert uni_fit.mu == pytest.approx(true["mu"], rel=0.35)
    assert uni_fit.alpha == pytest.approx(true["alpha"], rel=0.35)
    assert uni_fit.beta == pytest.approx(true["beta"], rel=0.35)
    assert abs(uni_fit.branching - true["alpha"] / true["beta"]) < 0.15
    assert uni_fit.branching < 1.0
    # MLE must beat the true parameters in-sample
    T = true["T"]
    assert uni_fit.loglik >= lm.uni_loglik(uni_times, T, true["mu"], true["alpha"], true["beta"])


def test_biv_refit_close(biv_fit, dgp):
    true = np.asarray(dgp["hawkes_biv"]["alpha"])
    assert biv_fit.converged
    assert np.all(np.abs(biv_fit.alpha - true) < 0.2)
    assert abs(biv_fit.beta - dgp["hawkes_biv"]["beta"]) < 0.5
    assert np.all(np.abs(biv_fit.mu - np.asarray(dgp["hawkes_biv"]["mu"])) < 0.12)
    assert biv_fit.spectral_radius < 1.0
    # buy/sell self-excitation should dominate cross terms in this DGP
    assert biv_fit.alpha[0, 0] > biv_fit.alpha[1, 0]
    assert biv_fit.alpha[1, 1] > biv_fit.alpha[0, 1]


def test_near_critical_fit_succeeds_with_warning():
    t = lm.simulate_uni(0.2, 1.9, 2.0, 150.0, 999)  # branching 0.95
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = lm.fit_uni(t, 150.0)
    assert fit.converged
    assert 0.85 < fit.branching < 1.05
    assert any("near-critical" in str(w.message) or "non-stationary" in str(w.message)
               for w in caught)


def test_hawkes_data_contains_burst(uni_times):
    """Self-excitation must show up as clustering: the minimum inter-arrival
    gap is far below the mean gap (a burst), and local event counts spike."""
    gaps = np.diff(uni_times)
    assert gaps.min() < 0.05 * gaps.mean()
    counts, _ = np.histogram(uni_times, bins=100)
    assert counts.max() >= 3 * counts.mean()


def test_residuals_exp1_properties(uni_times, dgp):
    true = dgp["hawkes_uni"]
    tau = lm.uni_residuals(uni_times, true["T"], true["mu"], true["alpha"], true["beta"])
    assert tau.shape == uni_times.shape
    assert np.all(tau > 0)  # compensator strictly monotone at event times
    assert tau.mean() == pytest.approx(1.0, abs=0.1)  # Exp(1) mean


def test_ks_true_params_fails_to_reject(uni_times, dgp):
    true = dgp["hawkes_uni"]
    tau = lm.uni_residuals(uni_times, true["T"], true["mu"], true["alpha"], true["beta"])
    _, p = lm.ks_test_exp(tau)
    assert p > 0.05  # correct model: no rejection


def test_ks_wrong_params_rejects(uni_times, dgp):
    tau = lm.uni_residuals(uni_times, dgp["hawkes_uni"]["T"], 1.5, 0.1, 3.0)
    _, p = lm.ks_test_exp(tau)
    assert p < 0.01


def test_biv_residuals_properties(biv_data, dgp):
    t, k = biv_data
    true = dgp["hawkes_biv"]
    taus = lm.biv_residuals(t, k, true["T"], true["mu"], true["alpha"], true["beta"])
    assert len(taus) == 2
    for m, tau in enumerate(taus):
        assert tau.size == int(np.sum(k == m))
        assert np.all(tau > 0)
        assert tau.mean() == pytest.approx(1.0, abs=0.15)
        _, p = lm.ks_test_exp(tau)
        assert p > 0.05


def test_spectral_radius_closed_form():
    assert lm.spectral_radius_2x2([[0.5, 0.0], [0.0, 0.25]]) == pytest.approx(0.5)
    # rotation-like matrix with complex eigenvalues: radius = sqrt(det)
    assert lm.spectral_radius_2x2([[0.0, -1.0], [1.0, 0.0]]) == pytest.approx(1.0)
    m = [[0.3, 0.15], [0.125, 0.25]]
    assert lm.spectral_radius_2x2(m) == pytest.approx(
        max(abs(np.linalg.eigvals(m))), abs=1e-12
    )


def test_validation_errors():
    t, T = small_sample()
    with pytest.raises(ValueError, match="increasing"):
        lm.uni_loglik(np.array([1.0, 0.5]), 2.0, 0.5, 0.5, 1.0)
    with pytest.raises(ValueError, match="mu"):
        lm.uni_loglik(t, T, -0.1, 0.5, 1.0)
    with pytest.raises(ValueError, match="alpha"):
        lm.uni_loglik(t, T, 0.5, -0.5, 1.0)
    with pytest.raises(ValueError, match="beta"):
        lm.uni_loglik(t, T, 0.5, 0.5, 0.0)
    with pytest.raises(ValueError, match="exceeds horizon"):
        lm.uni_loglik(t, 5.0, 0.5, 0.5, 1.0)
    with pytest.raises(ValueError, match="non-empty"):
        lm.uni_loglik(np.array([]), 1.0, 0.5, 0.5, 1.0)
    with pytest.raises(ValueError, match="non-stationary"):
        lm.simulate_uni(0.5, 2.0, 1.0, 10.0, 1)
    with pytest.raises(ValueError, match="types"):
        lm.biv_loglik(t, np.zeros(3), T, [0.3, 0.3], [[0.1, 0.1], [0.1, 0.1]], 1.0)
    with pytest.raises(ValueError, match="both event types"):
        lm.fit_biv(t, np.zeros(t.size, dtype=int), T)


def test_nelder_mead_on_quadratics():
    """Property-style: pinned NM finds the minimum of shifted quadratics."""
    for shift in (-2.0, 0.0, 1.5, 3.0):
        res = lm.nelder_mead(lambda x: (x[0] - shift) ** 2 + (x[1] + shift) ** 2 + 1.0,
                             [0.1, 0.1])
        assert res.converged
        assert res.x[0] == pytest.approx(shift, abs=1e-5)
        assert res.x[1] == pytest.approx(-shift, abs=1e-5)
        assert res.fval == pytest.approx(1.0, abs=1e-9)


def test_nelder_mead_matches_scipy(uni_times, uni_fit, dgp):
    """Cross-check the native optimizer against scipy on the real objective."""
    from scipy.optimize import minimize

    T = dgp["hawkes_uni"]["T"]

    def nll(x):
        mu, a, b = np.exp(x)
        return -lm.uni_loglik(uni_times, T, mu, a, b)

    res = minimize(nll, [math.log(0.5 * uni_times.size / T), math.log(0.5), 0.0],
                   method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000})
    mu, a, b = np.exp(res.x)
    assert uni_fit.mu == pytest.approx(mu, rel=1e-4)
    assert uni_fit.alpha == pytest.approx(a, rel=1e-4)
    assert uni_fit.beta == pytest.approx(b, rel=1e-4)


# ----------------------------------------------------------- review additions

def test_fit_objectives_are_pure_under_underflow(biv_data, uni_times):
    """PT-6: exp underflow of mu/beta must return the 1e10 penalty, never raise."""
    t, k = biv_data
    obj = lm.biv_fit_objective(t, k, 600.0)
    assert obj([-800.0, -800.0, 0.0, 0.0, 0.0, 0.0, -800.0]) == 1e10
    assert obj([-800.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) == 1e10   # mu0 underflow only
    assert obj([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -800.0]) == 1e10   # beta underflow only
    assert obj([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 800.0]) == 1e10    # exp overflow
    x0 = [math.log(0.5 * 323 / 600.0), math.log(0.5 * 288 / 600.0)] + [math.log(0.25)] * 4 + [0.0]
    assert obj(x0) == pytest.approx(-lm.biv_loglik(t, k, 600.0, [0.5 * 323 / 600.0, 0.5 * 288 / 600.0],
                                                    [[0.25, 0.25], [0.25, 0.25]], 1.0))
    # driving Nelder-Mead through the underflow region is harmless
    res = lm.nelder_mead(obj, [-800.0, -800.0, 0.0, 0.0, 0.0, 0.0, -800.0], maxiter=5)
    assert math.isfinite(res.fval)
    uobj = lm.uni_fit_objective(uni_times, 1000.0)
    assert uobj([-800.0, 0.0, 0.0]) == 1e10 and uobj([0.0, 0.0, -800.0]) == 1e10
    assert uobj([0.0, -800.0, 0.0]) == pytest.approx(-lm.uni_loglik(uni_times, 1000.0, 1.0, 0.0, 1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        capped = lm.fit_biv(t, k, 600.0, maxiter=1)
        capped_u = lm.fit_uni(uni_times, 1000.0, maxiter=1)
    assert capped.converged is False and capped.n_iter == 2   # two stages x 1 iteration
    assert capped_u.converged is False and capped_u.n_iter == 1
    with pytest.raises(ValueError, match="maxiter"):
        lm.fit_uni(uni_times, 1000.0, maxiter=0)


def test_bundled_streams_regenerate_bitwise(events, uni_times, biv_data, dgp):
    """PT-7: every bundled dataset is exactly its pinned generator output."""
    assert lm.generate_events(5000, 20240613) == events
    u = dgp["hawkes_uni"]
    sim = lm.simulate_uni(u["mu"], u["alpha"], u["beta"], u["T"], u["seed"])
    assert sim.size == 1211 and np.array_equal(sim, uni_times)
    b = dgp["hawkes_biv"]
    t, k = biv_data
    st, sk = lm.simulate_biv(b["mu"], b["alpha"], b["beta"], b["T"], b["seed"])
    assert st.size == 611 and np.array_equal(st, t) and np.array_equal(sk, k)


def test_poisson_limit_of_fit():
    """PT-8: alpha = 0 is a homogeneous Poisson process; the MLE must find it."""
    t = lm.simulate_uni(1.0, 0.0, 1.0, 2000.0, 5)
    assert t.size > 1500
    T = 2000.0
    fit = lm.fit_uni(t, T)
    assert fit.converged
    assert fit.branching < 0.05
    poisson_ll = lm.uni_loglik(t, T, t.size / T, 0.0, 1.0)  # Poisson MLE mu = N/T
    assert fit.loglik >= poisson_ll - 1e-6
    assert poisson_ll == pytest.approx(t.size * math.log(t.size / T) - t.size, abs=1e-9)
    # with alpha = 0 the likelihood does not depend on beta at all
    assert lm.uni_loglik(t, T, 0.7, 0.0, 1.0) == lm.uni_loglik(t, T, 0.7, 0.0, 3.3)


def test_single_event_paths():
    """PT-9: N = 1 and t_N = T boundaries."""
    mu, alpha, beta = 0.5, 0.8, 1.5
    t = np.array([0.7])
    expected = math.log(mu) - mu * 1.0 + (alpha / beta) * math.expm1(-beta * 0.3)
    assert lm.uni_loglik(t, 1.0, mu, alpha, beta) == pytest.approx(expected, abs=1e-15)
    assert lm.uni_loglik_brute(t, 1.0, mu, alpha, beta) == pytest.approx(expected, abs=1e-15)
    tau = lm.uni_residuals(t, 1.0, mu, alpha, beta)
    assert tau.shape == (1,) and tau[0] == pytest.approx(mu * 0.7)  # Lambda(t_1) = mu t_1
    d, p = lm.ks_test_exp(tau)
    assert 0.0 < d < 1.0 and 0.0 < p <= 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = lm.fit_uni(t, 1.0)
    assert math.isfinite(fit.loglik) and fit.mu > 0.0 and fit.n_iter >= 1
    # event exactly at the horizon: expm1(0) = 0
    tT = np.array([1.0])
    assert lm.uni_loglik(tT, 1.0, mu, alpha, beta) == pytest.approx(math.log(mu) - mu)
    assert lm.uni_residuals(tT, 1.0, mu, alpha, beta)[0] == pytest.approx(mu)
    kT = np.array([0])
    with pytest.raises(ValueError, match="both event types"):
        lm.fit_biv(tT, kT, 1.0)
    ll = lm.biv_loglik(tT, kT, 1.0, [0.2, 0.3], [[0.1, 0.1], [0.1, 0.1]], 2.0)
    assert ll == pytest.approx(math.log(0.2) - 0.5)


def test_simulation_candidate_cap(monkeypatch):
    import lobmod.hawkes as H
    monkeypatch.setattr(H, "MAX_CANDIDATES", 3)
    with pytest.raises(ValueError, match="MAX_CANDIDATES"):
        lm.simulate_uni(0.5, 0.2, 1.0, 100.0, 1)
    with pytest.raises(ValueError, match="MAX_CANDIDATES"):
        lm.simulate_biv([0.3, 0.3], [[0.1, 0.0], [0.0, 0.1]], 1.0, 100.0, 1)
    monkeypatch.setattr(H, "MAX_CANDIDATES", 50_000_000)
    assert lm.simulate_uni(0.5, 0.2, 1.0, 10.0, 1).size > 0


def test_types_must_be_integral(uni_times):
    t = uni_times[:4]
    with pytest.raises(ValueError, match="types"):
        lm.biv_loglik(t, [0.9, 1.2, 0.0, 1.0], 1000.0, [0.3, 0.3], [[0.1, 0.1], [0.1, 0.1]], 1.0)
    with pytest.raises(ValueError, match="types"):
        lm.biv_loglik(t, [True, False, True, False], 1000.0, [0.3, 0.3], [[0.1, 0.1], [0.1, 0.1]], 1.0)
    # exact-integer floats are fine
    a = lm.biv_loglik(t, [0.0, 1.0, 0.0, 1.0], 1000.0, [0.3, 0.3], [[0.1, 0.1], [0.1, 0.1]], 1.0)
    b = lm.biv_loglik(t, [0, 1, 0, 1], 1000.0, [0.3, 0.3], [[0.1, 0.1], [0.1, 0.1]], 1.0)
    assert a == b


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_bytes(text.encode())
    return str(p)


def test_times_csv_round_trip_and_robustness(tmp_path, uni_times, biv_data):
    p = str(tmp_path / "u.csv")
    lm.write_times_csv(p, uni_times)
    assert np.array_equal(lm.read_times_csv(p), uni_times)
    with open(p, "rb") as fh, open(os.path.join(DATA_DIR, "hawkes_uni.csv"), "rb") as ref:
        assert fh.read() == ref.read()  # %.17g: byte-identical to the bundled file
    t, k = biv_data
    q = str(tmp_path / "b.csv")
    lm.write_marked_times_csv(q, t, k)
    rt, rk = lm.read_marked_times_csv(q)
    assert np.array_equal(rt, t) and np.array_equal(rk, k)
    with open(q, "rb") as fh, open(os.path.join(DATA_DIR, "hawkes_biv.csv"), "rb") as ref:
        assert fh.read() == ref.read()
    # robustness: CRLF and trailing blank lines accepted, everything else rejected
    body = "t\n0.5\n1.25e0\n"
    assert lm.read_times_csv(_write(tmp_path, "crlf.csv", body.replace("\n", "\r\n"))).tolist() == [0.5, 1.25]
    assert lm.read_times_csv(_write(tmp_path, "trail.csv", body + "\n")).tolist() == [0.5, 1.25]
    for name, text in {
        "nan.csv": body + "nan\n",
        "inf.csv": body + "inf\n",
        "two.csv": body + "1.5,0\n",
        "word.csv": body + "abc\n",
        "hdr.csv": "time\n0.5\n",
        "blank.csv": "t\n0.5\n\n1.0\n",
        "space.csv": "t\n 0.5\n",
    }.items():
        with pytest.raises(ValueError):
            lm.read_times_csv(_write(tmp_path, name, text))
    mbody = "t,type\n0.5,0\n1.0,1\n"
    rt, rk = lm.read_marked_times_csv(_write(tmp_path, "m.csv", mbody))
    assert rt.tolist() == [0.5, 1.0] and rk.tolist() == [0, 1]
    for name, text in {
        "type2.csv": mbody + "1.5,2\n",
        "typef.csv": mbody + "1.5,1.0\n",
        "one.csv": mbody + "1.5\n",
        "three.csv": mbody + "1.5,1,\n",
        "hdr.csv": "type,t\n0,0.5\n",
    }.items():
        with pytest.raises(ValueError):
            lm.read_marked_times_csv(_write(tmp_path, name, text))
    with pytest.raises(ValueError, match="finite"):
        lm.write_times_csv(str(tmp_path / "x.csv"), [1.0, math.nan])
    with pytest.raises(ValueError, match="types"):
        lm.write_marked_times_csv(str(tmp_path / "y.csv"), [1.0, 2.0], [0, 2])
