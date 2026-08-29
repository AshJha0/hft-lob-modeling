# P13 — High-Frequency LOB Modeling (`lobmod`)

A four-language, cross-verified laboratory for high-frequency market microstructure:
a price-time-priority **limit order book** simulator with deterministic replay, a pinned
**zero-intelligence order-flow generator**, **Hawkes processes** (univariate and
bivariate mutually-exciting, exponential kernel) with exact MLE, Ogata thinning and
time-rescaling goodness of fit, and an **order-flow feature pipeline** feeding a
fully deterministic **neural network built from scratch** (no frameworks, no RNG) —
closed out by an honest accounting of whether the predicted signal survives
transaction costs. (Spoiler from the bundled data: gross +45.0 ticks, net −139.5.)

Every numerical path is pinned by `API_SPEC.md` and verified against
`data/golden/golden.json` (18 golden cases) in all four languages.

## Feature matrix

| Component | Python | C++ | Rust | Java |
|---|---|---|---|---|
| Limit order book (price-time priority, partial fills, market orders walking the book) | `lobmod.book` | `lobmod::LimitOrderBook` | `lobmod::LimitOrderBook` | `com.quant.lobmod.LimitOrderBook` |
| Deterministic event replay + golden book snapshots | `lobmod.replay` | `lobmod::replay` | `lobmod::replay` | `Replay` |
| Zero-intelligence flow generator (seeded, pinned) | `lobmod.zi` | `lobmod::generate_events` | `lobmod::generate_events` | `ZiGenerator` |
| Pinned 64-bit LCG (single RNG for the whole project) | `lobmod.rng.Lcg64` | `lobmod::Lcg64` | `lobmod::Lcg64` | `Lcg64` |
| Hawkes log-likelihood, O(N) recursion + brute-force check | `lobmod.hawkes` | `lobmod::uni_loglik` … | `lobmod::uni_loglik` … | `Hawkes` |
| Ogata thinning simulation (uni + bivariate) | `simulate_uni/biv` | ditto | ditto | ditto |
| MLE via native Nelder-Mead (log-transformed params) | `fit_uni`, `fit_biv` | ditto | ditto | ditto |
| Time-rescaling residuals + KS test vs Exp(1) | `lobmod.gof` | `lobmod::ks_test_exp` | `lobmod::ks_test_exp` | `Gof` |
| OFI / queue-imbalance features, 3-class direction labels | `lobmod.features` | `lobmod::compute_features` | `lobmod::compute_features` | `Features` |
| MLP + logistic + majority baselines (deterministic, no RNG) | `lobmod.nn` | `lobmod::Mlp` | `lobmod::Mlp` | `Mlp` |
| Signal PnL with half-spread costs | `signal_pnl` | ditto | ditto | ditto |
| Golden-value test suite (18 cases) | pytest | GoogleTest | cargo test | JUnit 4 |

## Directory layout

```
13-hft-lob-modeling/
  README.md, LEARN.md, COOKBOOK.md, API_SPEC.md
  docs/
    ARCHITECTURE.md          # design, data flow, Mermaid diagrams
    diagrams/*.mmd           # raw Mermaid sources
  data/
    lob_events.csv           # 5000-event deterministic ZI stream (seed 20240613)
    hawkes_uni.csv           # 1211 univariate Hawkes event times (seed 101)
    hawkes_biv.csv           # 611 bivariate events, t + type (seed 33)
    dgp_params.json          # true DGP parameters + pinned eval points
    features_config.json     # H, epsilon, windows, NN hyperparameters
    generate_data.py         # regenerates everything (fixed seeds)
    golden/golden.json       # 18 cross-language golden cases
  python/   src/lobmod/*.py, tests/, demo.py, requirements.txt
  cpp/      CMakeLists.txt, include/lobmod/*.hpp, src/, tests/, build.sh
  rust/     Cargo.toml, src/lib.rs, src/bin/demo.rs, tests/
  java/     src/main/java/com/quant/lobmod/*.java, src/test/..., build.sh, test.sh, demo.sh
```

## Build, run, test

### Python (3.11, numpy/scipy stack)

```bash
cd python
PYTHONPATH=src python3 demo.py     # end-to-end demo
PYTHONPATH=src pytest -q           # tests incl. golden suite
```

### C++ (C++17, CMake, GoogleTest)

```bash
cd cpp
./build.sh    # cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j2
./build/demo                                  # demo
ctest --test-dir build --output-on-failure    # tests
```

### Rust (edition 2021)

```bash
cd rust
cargo run --bin demo   # demo
cargo test             # unit + integration + golden tests
```

### Java (21, JUnit 4 at /usr/share/java/junit4.jar)

```bash
cd java
./build.sh    # javac everything into out/
./demo.sh     # runs the demo main
./test.sh     # JUnitCore on all *Test classes
```

## Demo output (Python, truncated)

```
lobmod demo — high-frequency LOB modeling

--------------------------------------------------------------------
1. Limit order book replay (data/lob_events.csv)
--------------------------------------------------------------------
events: 5000  (limit 2774, cancel 1522, market 704)
trades: 1595 fills, 2895 units traded
mid range: [964.0, 1008.5] ticks; mean spread 1.233 ticks
events with an empty side (mid = NaN, guarded): 61

  snapshot     best_bid  best_ask     mid   micro    depth3(B/S)
  event  1000       999      1000    999.5   999.33   21/27
  event  3000      1002      1003   1002.5  1002.29   13/29
  event  5000      1003      1004   1003.5  1003.86   19/14

--------------------------------------------------------------------
2. Univariate Hawkes fit (data/hawkes_uni.csv)
--------------------------------------------------------------------
1211 events on [0, 1000]  (min gap 0.0005 vs mean 0.822 — self-exciting bursts)

  param        true    fitted
  mu          0.500    0.5052
  alpha       1.000    0.9795
  beta        1.600    1.6792
  branching ratio alpha/beta = 0.5833  (stationary; 58% of events are endogenous offspring)
  loglik -711.77, Nelder-Mead converged=True (172 iters)

--------------------------------------------------------------------
3. Goodness of fit — time-rescaling residuals + KS
--------------------------------------------------------------------
  fitted params : KS stat 0.0172, p-value 0.8632 -> PASS (no rejection at 5%)
  wrong params  : KS stat 0.1149, p-value 2.05e-14 -> REJECT (sanity)

--------------------------------------------------------------------
4. Bivariate Hawkes: buy/sell market-order cross-excitation
--------------------------------------------------------------------
611 events (323 buys, 288 sells) on [0, 600]

  excitation alpha (fitted)      branching alpha/beta
    buy <-buy   0.727             0.345
    buy <-sell  0.303             0.144
    sell<-buy   0.295             0.140
    sell<-sell  0.585             0.278
  mu = 0.2838 (buy), 0.2716 (sell); beta = 2.1084
  spectral radius of branching matrix = 0.4568 (stationary)
  ...
--------------------------------------------------------------------
5. Direction prediction: MLP vs logistic vs majority
--------------------------------------------------------------------
4348 samples ...; chronological split: 3043 train / 1305 test
training loss: 1.0806 (epoch 1) -> 0.6760 (epoch 800)

  model                holdout accuracy
  MLP (16,8)               0.7042
  logistic regression      0.6996
  majority class (1)       0.6582

--------------------------------------------------------------------
6. Does the signal survive costs?
--------------------------------------------------------------------
trading the MLP signal on the holdout (177 trades, mid-to-mid, half-spread cost per trade):
  gross PnL +45.0 ticks   net PnL -139.5 ticks
note: the classifier finds real structure ... but paying half the spread on
every trade erases it — naive microstructure signals die on transaction costs.

demo complete.
```

## Cross-language golden testing

`data/golden/golden.json` holds 18 named cases (flat JSON schema:
`{"cases": [{"name", "inputs", "expect", "tol"}]}`) covering book snapshots after
events 1000/3000/5000, the microprice at event 2500 (1e-12), OFI over a pinned
window (1e-10), Hawkes univariate/bivariate log-likelihoods at pinned parameters
(1e-8), fitted parameters and branching diagnostics, the KS statistic and p-value
of fitted residuals (1e-6), NN training losses after epoch 1 and 800 (1e-8),
holdout accuracies for MLP/logistic/majority (1e-10), and the signal PnL after
costs (1e-8).

Every language's test suite loads this file and asserts agreement within
tolerance. This works because *everything* is pinned: one shared LCG (never a
language's built-in RNG), fixed iteration orders, an exact Nelder-Mead recipe,
a deterministic sin-based weight init, and full-batch gradient descent with no
shuffling. Any drift in any port breaks a golden case rather than passing
silently. `data/generate_data.py` additionally cross-checks the O(N) likelihood
recursions against brute-force O(N²) sums (1e-10) and backprop against central
finite differences (rel. 1e-6) before it will write goldens.

## License

Educational use.
