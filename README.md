# High-Frequency LOB Modeling (`lobmod`)

**This is a synthetic research and teaching lab.** The bundled data is simulated,
and so are the published results, apart from one IEX replay
([IEX replay](#iex-replay)), which holds the only real-data numbers published here.
Book-only timings exist in
[docs/LATENCY.md](docs/LATENCY.md), but nothing here measures wire-to-wire
latency, the simulator is not calibrated to any market (an attempt failed), and
this is not a trading system. See [Real data](#real-data), [Limits](#limits) and
[Known results](#known-results).

Status: v1.3.0. Changes are listed in [CHANGELOG.md](CHANGELOG.md).

A four-language, cross-verified laboratory for high-frequency market microstructure:
a price-time-priority **limit order book** simulator with deterministic replay, a pinned
**zero-intelligence order-flow generator**, **Hawkes processes** (univariate and
bivariate mutually-exciting, exponential kernel) with exact MLE, Ogata thinning and
time-rescaling goodness of fit, and an **order-flow feature pipeline** feeding a
fully deterministic **neural network built from scratch** (no frameworks, no RNG) —
closed out by an honest accounting of whether the predicted signal survives
transaction costs. (Spoiler from the bundled data: gross +45.0 ticks, net −139.5.)
Around that pinned core sit a Python research package, an execution simulator,
Python ITCH 5.0 and IEX parsers, and a preallocated FastBook with a binary ITCH
decoder in C++ and Rust.

Every numerical path is pinned by `API_SPEC.md` and verified against
`data/golden/golden.json` (18 golden cases) in all four languages, plus
per-language suites of edge-case, invariant and cross-language parity tests
(current counts: see CI).

## Feature matrix

| Component | Python | C++ | Rust | Java |
|---|---|---|---|---|
| Limit order book (price-time priority, partial fills, market orders walking the book; `reduce`, `execute`, `replace`) | `lobmod.book` | `lobmod::LimitOrderBook` | `lobmod::LimitOrderBook` | `com.quant.lobmod.LimitOrderBook` |
| Deterministic event replay + golden book snapshots | `lobmod.replay` | `lobmod::replay` | `lobmod::replay` | `Replay` |
| Schema v2 events: nanosecond timestamps, types `R`, `U`, `E`; a v1 file also reads as v2 (`read_/write_events_v2_csv`, `replay_v2`) | `lobmod.replay` | `lobmod::…` (`replay.hpp`) | `lobmod::replay::…` | `Replay`, `EventV2` |
| Event / event-time CSV i/o with pinned strict parsing (`read_/write_events_csv`, `read_/write_times_csv`, `read_/write_marked_times_csv`) | `lobmod.replay`, `lobmod.hawkes` | `lobmod::…` (`replay.hpp`, `hawkes.hpp`) | `lobmod::…` | `Replay`, `Hawkes` |
| Zero-intelligence flow generator (seeded, pinned) | `lobmod.zi` | `lobmod::generate_events` | `lobmod::generate_events` | `ZiGenerator` |
| Pinned 64-bit LCG (single RNG for the whole project) | `lobmod.rng.Lcg64` | `lobmod::Lcg64` | `lobmod::Lcg64` | `Lcg64` |
| Hawkes log-likelihood, O(N) recursion + brute-force check | `lobmod.hawkes` | `lobmod::uni_loglik` … | `lobmod::uni_loglik` … | `Hawkes` |
| Ogata thinning simulation (uni + bivariate) | `simulate_uni/biv` | ditto | ditto | ditto |
| MLE via native Nelder-Mead (log-transformed params) | `fit_uni`, `fit_biv` | ditto | ditto | ditto |
| Time-rescaling residuals + KS test vs Exp(1) | `lobmod.gof` | `lobmod::ks_test_exp` | `lobmod::ks_test_exp` | `Gof` |
| OFI / queue-imbalance features, 3-class direction labels | `lobmod.features` | `lobmod::compute_features` | `lobmod::compute_features` | `Features` |
| MLP + logistic + majority baselines (deterministic, no RNG) | `lobmod.nn` | `lobmod::Mlp` | `lobmod::Mlp` | `Mlp` |
| Signal PnL with half-spread costs | `signal_pnl` | ditto | ditto | ditto |
| Golden-value test suite (18 cases) + edge-case suite | pytest | GoogleTest | cargo test | JUnit 4 |
| Preallocated FastBook with `rest`, `reduce`, `replace`, tested against the reference book | none | `lobmod::fast::FastBook` (`cpp/fast/`) | `lobmod::fast_book::FastBook` | none |
| Book and decoder benchmarks (`bench/run_all.sh`) | none | `fast_book_bench`, `itch_bench` | `bench`, `itch_bench` binaries | `java/bench.sh`, reference book only |
| ITCH 5.0 parser and book-integrity checker, tested on fixtures and replayed on real sample files | `lobmod.feeds` | none | none | none |
| IEX HIST reader (pcap, pcapng, DEEP, DEEP+), tested on hand-built fixtures and run once on one real DEEP+ file, see [IEX replay](#iex-replay) | `lobmod.feeds.iex`, `.pcap` | none | none | none |
| Binary ITCH 5.0 decoder and FastBook mirror, allocation free | none | `cpp/fast/itch_decode.hpp` | `lobmod::itch_decode` | none |
| Hawkes standard errors, LR tests, bootstrap goodness of fit, walk-forward evaluation, taker backtest, passive maker | `lobmod.research` | none | none | none |
| Multi-exponential and power-law kernels, seasonal baselines, cost-aware labels, boosted trees | `lobmod.research.kernels`, `.seasonality`, `.labels`, `.baselines` | none | none | none |
| Execution simulator: queues, latency, informed flow, fees, attribution; calibration checks | `lobmod.sim` | none | none | none |
| libFuzzer harnesses for book, CSV and ITCH decoder, 60 seconds each in CI | none | `fuzz/` | none | none |

Only the rows down to the golden suite, plus the book operations and v2 schema,
are in the four-language contract. The rows below it have no golden values.

## Directory layout

```
hft-lob-modeling/
  README.md, LEARN.md, COOKBOOK.md, API_SPEC.md, CHANGELOG.md, LICENSE (MIT)
  CITATION.cff, CONTRIBUTING.md, SECURITY.md
  pyproject.toml, requirements.lock   # installable package, hashed lock file
  docs/
    ARCHITECTURE.md          # design, data flow, Mermaid diagrams
    LATENCY.md               # book-only timings: reference book and FastBook
    GITHUB_PAGES.md          # the docs site (live at https://ashjha0.github.io/hft-lob-modeling/) and how to publish it
    index.html               # the docs site landing page
    diagrams/*.mmd           # raw Mermaid sources
  data/
    lob_events.csv           # 5000-event deterministic ZI stream (seed 20240613)
    lob_events_v2.csv        # 17-event v2 fixture, with its expected final state
    hawkes_uni.csv           # 1211 univariate Hawkes event times (seed 101)
    hawkes_biv.csv           # 611 bivariate events, t + type (seed 33)
    dgp_params.json          # true DGP parameters + pinned eval points
    features_config.json     # H, epsilon, windows, NN hyperparameters
    generate_data.py         # regenerates everything (fixed seeds)
    golden/golden.json       # 18 cross-language golden cases
  python/   src/lobmod/*.py (pinned core), tests/, demo.py, requirements.txt
            src/lobmod/feeds/     # ITCH 5.0 and IEX parsers, capture reader, book-integrity checker
            src/lobmod/research/  # inference, evaluation, kernels, labels, backtest
            src/lobmod/sim/       # execution simulator, accounting, calibration checks
  cpp/      CMakeLists.txt, include/lobmod/*.hpp, src/, tests/, build.sh
            fast/                 # FastBook, ITCH decoder, benchmarks
  rust/     Cargo.toml, src/lib.rs, src/fast_book.rs, src/itch_decode.rs,
            src/bin/{demo,bench,itch_bench}.rs, tests/
  java/     src/main/java/com/quant/lobmod/*.java, src/test/..., src/bench/...,
            build.sh, test.sh, demo.sh, bench.sh
  research/ run_all.py, run_models.py, run_sim.py and their generated results*.md
            real_replay.py, iex_replay.py   # replay a file you supply
            calibrate.py, calibration_criteria.md, control_362.py, results_control.md
  bench/    run_all.sh, every latency benchmark in one go
  fuzz/     libFuzzer harnesses for the C++ book, CSV code and ITCH decoder
```

## Build, run, test

All commands are for a fresh Linux clone (the `.sh` scripts are invoked with
`bash`, so their executable bit does not matter).

### Python (3.11; numpy is required, scipy only by three cross-check tests)

```bash
cd python
pip install -r requirements.txt          # numpy, scipy, pytest
PYTHONPATH=src python3 demo.py           # end-to-end demo
PYTHONPATH=src python3 -m pytest -q      # incl. the golden suite
```

Or install the package from the repository root with `pip install -e ".[test]"`.
`requirements.lock` pins exact versions with hashes.

### C++ (C++17, CMake >= 3.16, g++ 13, GoogleTest)

```bash
cd cpp
bash build.sh    # cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j2
./build/demo                                  # demo
ctest --test-dir build --output-on-failure    # GoogleTest suite and FastBook tests
```

### Rust (edition 2021, cargo 1.95)

```bash
cd rust
cargo run --release --bin demo   # demo
cargo test --release             # unit, integration, property and golden tests
```

`cargo build --release` produces zero warnings; the dev profile also builds
with `-O2` because the 800-epoch golden pipeline is too slow unoptimized.

### Java (21; JUnit 4 at /usr/share/java/junit4.jar and hamcrest at /usr/share/java/hamcrest.jar)

```bash
cd java
bash build.sh    # javac -Xlint:all -Werror everything into out/
bash demo.sh     # runs the demo main
bash test.sh     # JUnitCore on all *Test classes
```

### Benchmarks and research scripts

```bash
# Everything at once, on an idle machine; method and caveats are in docs/LATENCY.md
bash bench/run_all.sh

# Or book-only timings by hand, Linux or WSL
cmake -S cpp/fast -B build-fast -DCMAKE_BUILD_TYPE=Release
cmake --build build-fast -j
build-fast/fast_book_bench 2000000 20000 5
(cd rust && cargo run --release --bin bench -- 2000000 20000 5)

# Regenerate the research tables from the repository root
PYTHONPATH=python/src python3 research/run_all.py      # research/results.md
PYTHONPATH=python/src python3 research/run_models.py   # research/results_models.md
PYTHONPATH=python/src python3 research/run_sim.py      # research/results_sim.md

sh fuzz/run.sh 60     # needs clang; seconds per harness, see fuzz/README.md
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
1211 events on [0, 1000]  (gap CV 1.63 vs 1 for Poisson)

  param        true    fitted
  mu          0.500    0.5052
  alpha       1.000    0.9795
  beta        1.600    1.6792
  branching ratio alpha/beta = 0.5833  (stationary; 58% of events are endogenous offspring)
  loglik -711.77, Nelder-Mead converged=True (173 iters)

--------------------------------------------------------------------
3. Goodness of fit — time-rescaling residuals + KS
--------------------------------------------------------------------
  fitted params : KS stat 0.0172, p-value 0.8632 -> no rejection (test is weak with fitted parameters)
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
note: on this split the MLP edges both baselines and gross PnL is positive,
but across 100 seeds logistic beats the MLP by 0.59 pts (SE 0.16). The structure
is generator mechanics (spread alone gives 0.693 of the 0.704), and paying half
the spread on every trade erases the gross PnL. See research/results.md.

demo complete.
```

## New in v1.3.0

These sit beside the pinned core. Only the book operations and the v2 event schema are in the four-language contract.

- **L3 book operations and v2 events.** `reduce`, `execute`, `replace` and nanosecond timestamps in all four ports (`API_SPEC.md`). v1 files and goldens are unchanged.
- **ITCH 5.0 parser** (`lobmod.feeds.itch50`, Python). Tested on fixtures built by hand and replayed on real sample files: see [Real data](#real-data).
- **IEX reader** (`lobmod.feeds.iex`, `lobmod.feeds.pcap`, Python). Reads HIST captures, DEEP+ order by order and DEEP by price level, and checks sequence numbers. Tested on fixtures built from the public specifications and run once on one real DEEP+ file: see [IEX replay](#iex-replay). The DEEP path has not met a real file.
- **Binary ITCH decoder** (C++ `cpp/fast/itch_decode.hpp`, Rust `itch_decode`). Reads a message without allocating, and `apply` mirrors it into a FastBook. Fuzzed in CI. Not a feed handler: there is no socket or capture path.
- **Execution simulator** (`lobmod.sim`, Python). Hawkes-driven reactive flow, informed traders, queue position, latency, fees and P&L attribution. It is uncalibrated: the parameters are assumed, a calibration attempt failed, and its P&L levels mean nothing. See [research/results_sim.md](research/results_sim.md).
- **Richer Hawkes models** (`lobmod.research`). Multi-timescale and power-law kernels and seasonal baselines. See [research/results_models.md](research/results_models.md).
- **FastBook** (C++ `cpp/fast/`, Rust `fast_book`). A preallocated book with O(1) cancel, plus `rest`, `reduce` and `replace` so it can mirror a feed, tested against the reference book. On an idle laptop under WSL2 it beats the reference book on add, cancel and throughput, book only: [docs/LATENCY.md](docs/LATENCY.md). This is not wire-to-wire latency.
- **Repository basics.** `pyproject.toml`, a hashed `requirements.lock`, `CITATION.cff`, `CONTRIBUTING.md`, `SECURITY.md`, Dependabot and an advisory clang-tidy job.

## Real data

The parser and book were replayed on public ITCH 5.0 sample files for ten symbols with no parser or book anomalies and every execution at the touch. That is internal consistency, not comparison with a venue snapshot. Raw data is not redistributed. Nasdaq-derived numbers do not appear in this repository.

Calibration of `lobmod.sim` to real data was attempted against criteria fixed in advance ([research/calibration_criteria.md](research/calibration_criteria.md)) and FAILED: it did not meet the criteria in sample, parameters did not carry across the day, and the held-out day was not run. The simulator remains uncalibrated and its P&L levels mean nothing.

Hawkes on real trades, qualitative only: a seasonal two-timescale model is preferred over a single exponential, but the fast rate hits its bound and branching is not resolved, so no value is quoted. Sub-millisecond burst structure cannot be attributed to a mechanism from this feed.

The cap gain of +362 in the old maker study is inconclusive
([research/results_control.md](research/results_control.md)): mean reversion is
only partly supported as its explanation.

## IEX replay

Data provided for free by IEX. By accessing or using IEX Historical Data, you agree to the IEX Historical Data Terms of Use. Terms: <https://www.iex.io/products/market-data-connectivity/hist-terms>.

`research/iex_replay.py` was run once on one real file, `20251212_IEXTP1_DPLS1.0.pcap.gz`: 335.5M packets and 477.6M messages, with books kept for AAPL, MSFT and INTC. No sequence gaps or parser anomalies were reported, every displayed execution was at the touch, and the book verified empty at the end of the day.

| symbol | median spread (cents) | OFI R² | sign ACF, lag 1, aggregated |
|---|---|---|---|
| AAPL | 7 | 0.018 | 0.174 |
| MSFT | 41 | 0.023 | 0.551 |
| INTC | 1 | 0.250 | 0.296 |

These describe IEX's own displayed book, which is thin and is not the NBBO. That is why the AAPL and MSFT spreads are wide and their OFI R² low. DEEP+ shows displayed orders only, and many IEX fills are not against displayed orders. One day and three symbols, on one venue. Details and an outside price check are in [research/results_iex.md](research/results_iex.md).

## Testing

Each language runs three layers: the 18 golden cases below, the invariant and
statistical tests required by `API_SPEC.md` §10, and an edge-case suite that
was added after an external review (tick-0 and 64-bit tick handling, CSV
robustness, pure optimizer objectives, Poisson and single-event limits,
small-λ KS accuracy, label validation, `ofi_at` boundary, depth monotonicity,
and byte-identical regeneration of every bundled file from its seed). Python
also has suites for the feed parser, research tools and simulator. Rust, Java
and Python check the book against a naive model with property tests, and the
C++ and Rust FastBook tests compare it with the reference book on randomized
scripts. CI runs the three fuzz harnesses for 60 seconds each from an empty
corpus, so it catches shallow crashes rather than proving much. Python coverage
is reported, not enforced. Current counts are in CI, which runs all four suites
on every push.

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

One caveat on portability: the accuracy goldens (cases 15–17, tol 1e-10) and
the PnL golden (case 18, tol 1e-8) are argmax-discrete — a single flipped
holdout prediction moves accuracy by 1/1305 ≈ 7.7e-4. They pass bit-for-bit
on this glibc/x86-64 box in all four languages, but a libm whose `sin`/`exp`
differ by 1 ulp (e.g. a JVM intrinsic on another platform) could legitimately
flip a near-tie after 800 epochs. If that happens, widen those tolerances to
1/n_test rather than "fixing" the port.

## Real-world usage notes

What a practitioner gets, precisely:

- **Units and conventions.** Book prices are integer *ticks* (`tick_size`
  converts a currency price once, at the API boundary, and is otherwise never
  used); `Event.price` in the CSV schema is a tick count; sizes and ids are
  signed 64-bit integers in every port (nothing is narrowed or wrapped);
  Hawkes times and `T` share one arbitrary unit (seconds in the bundled data)
  with `mu`, `beta` per unit time and `alpha` per event; PnL is in ticks per
  unit size, mid-to-mid, half a spread charged on entry only.
- **Parameter bounds.** Prices must round to `[1, 9e18]` ticks within a
  relative grid tolerance of `1e-9`, capped at 0.01 tick (round half up;
  `limit()` refuses ratios from `2^52`, use `limit_ticks`); sizes `1..9e18`,
  with total resting size per side also capped at `9e18`; seeds
  `[0, 2^63−1]`; Hawkes `mu > 0`, `alpha ≥ 0`, `beta > 0`, simulation refuses
  branching `≥ 1`; `ofi_at(i, w)` needs `i ≥ w`; feature windows, lookback and
  horizon are counted in events, not clock time.
- **What is validated (and rejected with the language's invalid-input error):**
  NaN/inf/negative/zero where invalid, empty inputs, unsorted or duplicate
  event times, off-grid or out-of-range prices (never reinterpreted as a
  market order or a negative price), duplicate live order ids, unknown cancel
  ids, labels outside `{0,1,2}`, malformed CSV (strict field grammar, CRLF
  tolerated, only trailing blank lines allowed). Non-integral floats are never
  truncated to integers. Fit objectives are total functions (penalty `1e10`,
  never an exception), `converged=false` means the iteration cap stopped the
  search, and every loop has a documented bound (simulators: 5·10⁷
  candidates).
- **Where the round 2 features live.** Nanosecond timestamps and
  `reduce`/`replace`/`execute` are in the book and v2 schema of all four ports.
  Fees, latency, queue-position tracking and fill statistics are in `lobmod.sim`,
  and the taker backtest has a fee and an event-count delay. Standard errors for
  the Hawkes MLE are in `lobmod.research`. None of these is in the golden
  contract except the book operations and v2 schema. Feature windows, lookback
  and horizon in the pinned pipeline are still counted in events.
- **Still absent.** Auctions, halts as book states (the parser reports trading
  actions but the book ignores them), iceberg and hidden orders in the book,
  multi-venue books, calendars and sessions, day-count conventions, tie-breaking
  of duplicate timestamps in real data (reject and pre-process), and any
  calibrated simulator or model: the one attempt failed.
- **Memory.** `trades` (and Python/C++ `log`) grow for the life of a book;
  pass `record_trades=False` / call `clear_history()` when replaying a full
  day.
- **What is verified.** Every formula is pinned in `API_SPEC.md`, reproduced
  by four ports to the golden tolerances, and the
  reference mathematics (likelihood recursions, backprop, KS series) is
  cross-checked against brute force, finite differences and scipy in the test
  suites. The ports all come from one spec, so agreement shows port fidelity.
  It is not independent proof that the spec is right. This is a reference and
  teaching implementation, not an exchange-grade matching engine or a trading
  system.

## Limits

- Bundled data is simulated, and the IEX results are the only real-data numbers
  published. Apart from those one-venue, one-day figures, nothing here is evidence
  about real markets; a parameter recovered from the bundled data is one the
  generator put there. The real-file replays check our code, not a market.
  Nasdaq-derived numbers are not published.
- Book-only timings exist ([docs/LATENCY.md](docs/LATENCY.md), an idle laptop
  under WSL2). There is no wire-to-wire figure, and no live feed handler, venue
  or order gateway is involved. Java has no FastBook and no GC-free path, and was
  not in the idle benchmark run.
- The simulator is uncalibrated: every parameter is assumed, calibration failed
  ([Real data](#real-data)), and its P&L levels mean nothing. Its agents also get a free fair-value feed. Without that feed
  the latency effect is not distinguishable from zero (2 ms minus 100 ms:
  −10.8 [−32.1, 10.5] ticks, 40 seeds, [research/results_sim.md](research/results_sim.md)).
  Positive maker P&L is not an edge.
- The zero-intelligence generator is inspired by Smith et al. (2003) but differs
  from it (geometric placement off the opposite best, no event clock).
- KS on fitted parameters is a weak test: in the v1.2.0 review, 0/400 rejections
  at 5% under the true model and 3–5% against wrong kernels. Use the bootstrap in `lobmod.research`.
- `microprice` is the size-weighted mid, not Stoikov's micro-price.
- The single-split MLP lead is noise. Across 60 streams MLP minus logistic is
  −0.53 points [−0.95, −0.12] ([research/results.md](research/results.md)), an
  earlier 100-seed run gave −0.59 (SE 0.16), and on the bundled split spread alone
  gives 0.693 of the 0.704.

## Known results

All simulated except [research/results_iex.md](research/results_iex.md). Multi-seed model comparison, bootstrap goodness of fit, the
executable backtest and the maker simulator are in
[research/results.md](research/results.md). Kernels, seasonality, cost-aware
labels and boosted trees are in
[research/results_models.md](research/results_models.md). The execution
simulator is in [research/results_sim.md](research/results_sim.md), and book
timings are in [docs/LATENCY.md](docs/LATENCY.md).

## References

Works the code implements or the documentation relies on:

- Smith, E., Farmer, J. D., Gillemot, L., Krishnamurthy, S. (2003). "Statistical theory of the continuous double auction." *Quantitative Finance* 3(6), 481–514. https://doi.org/10.1088/1469-7688/3/6/307 — inspiration for the zero-intelligence generator (`zi`), which differs from their model.
- Stoikov, S. (2018). "The micro-price: a high-frequency estimator of future prices." *Quantitative Finance* 18(12), 1959–1966. https://doi.org/10.1080/14697688.2018.1489139 — cited for contrast: Stoikov's micro-price is a limit of expected future mids; `microprice` here is the plain size-weighted mid.
- Cont, R., Kukanov, A., Stoikov, S. (2014). "The price impact of order book events." *Journal of Financial Econometrics* 12(1), 47–88. https://doi.org/10.1093/jjfinec/nbt003 — order-flow imbalance, eq. (4) (`features.ofi_series`).
- Hawkes, A. G. (1971). "Spectra of some self-exciting and mutually exciting point processes." *Biometrika* 58(1), 83–90. https://doi.org/10.1093/biomet/58.1.83 — the Hawkes process.
- Ozaki, T. (1979). "Maximum likelihood estimation of Hawkes' self-exciting point processes." *Annals of the Institute of Statistical Mathematics* 31, 145–155. https://doi.org/10.1007/BF02480272 — the O(N) likelihood recursion.
- Ogata, Y. (1978). "The asymptotic behaviour of maximum likelihood estimators for stationary point processes." *Annals of the Institute of Statistical Mathematics* 30, 243–261. https://doi.org/10.1007/BF02480216 — MLE for stationary point processes.
- Ogata, Y. (1981). "On Lewis' simulation method for point processes." *IEEE Transactions on Information Theory* 27(1), 23–31. https://doi.org/10.1109/TIT.1981.1056305 — thinning simulation (`simulate_uni/biv`).
- Bacry, E., Mastromatteo, I., Muzy, J.-F. (2015). "Hawkes processes in finance." *Market Microstructure and Liquidity* 1(1), 1550005. https://doi.org/10.1142/S2382626615500057 (arXiv:1502.04592) — multivariate Hawkes, branching matrix and stationarity (`fit_biv`, `spectral_radius_2x2`).
- Brown, E. N., Barbieri, R., Ventura, V., Kass, R. E., Frank, L. M. (2002). "The time-rescaling theorem and its application to neural spike train data analysis." *Neural Computation* 14(2), 325–346. https://doi.org/10.1162/08997660252741149 — residual construction (`uni_residuals`, `biv_residuals`).
- Meyer, P.-A. (1971). "Démonstration simplifiée d'un théorème de Knight." *Séminaire de Probabilités V*, Lecture Notes in Mathematics 191, 191–195. Springer. https://doi.org/10.1007/BFb0058859 — multivariate time-rescaling.
- Stephens, M. A. (1970). "Use of the Kolmogorov–Smirnov, Cramér–von Mises and related statistics without extensive tables." *Journal of the Royal Statistical Society, Series B* 32(1), 115–122. https://doi.org/10.1111/j.2517-6161.1970.tb00821.x — the small-sample correction in `ks_pvalue`.
- Press, W. H., Teukolsky, S. A., Vetterling, W. T., Flannery, B. P. (2007). *Numerical Recipes: The Art of Scientific Computing*, 3rd ed., §14.3.3 (`probks`, `KSdist`; eqs. 14.3.19–14.3.20). Cambridge University Press. ISBN 978-0-521-88068-8 — the two Kolmogorov series forms.
- Lagarias, J. C., Reeds, J. A., Wright, M. H., Wright, P. E. (1998). "Convergence properties of the Nelder–Mead simplex method in low dimensions." *SIAM Journal on Optimization* 9(1), 112–147. https://doi.org/10.1137/S1052623496303470 — the simplex decision tree (`nelder_mead`).
- Nelder, J. A., Mead, R. (1965). "A simplex method for function minimization." *The Computer Journal* 7(4), 308–313. https://doi.org/10.1093/comjnl/7.4.308.
- Virtanen, P. et al. (2020). "SciPy 1.0: fundamental algorithms for scientific computing in Python." *Nature Methods* 17, 261–272. https://doi.org/10.1038/s41592-019-0686-2 — the initial-simplex and acceptance conventions of `scipy.optimize.minimize(method="Nelder-Mead")` that the optimizer pins.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, §5.3 (error backpropagation). Springer. ISBN 978-0-387-31073-2 — the MLP gradients (`nn`).
- He, K., Zhang, X., Ren, S., Sun, J. (2015). "Delving deep into rectifiers: surpassing human-level performance on ImageNet classification." *Proceedings of the IEEE ICCV*, 1026–1034. https://doi.org/10.1109/ICCV.2015.123 (arXiv:1502.01852) — source of the √(2/n_in) term; the pinned init is smaller than He (variance about 8x below).
- Avellaneda, M., Stoikov, S. (2008). "High-frequency trading in a limit order book." *Quantitative Finance* 8(3), 217–224. https://doi.org/10.1080/14697680701381228 — the quoting formula of `sim.AvellanedaStoikov`, used with a rolling horizon.
- Hardiman, S. J., Bercot, N., Bouchaud, J.-P. (2013). "Critical reflexivity in financial markets: a Hawkes process analysis." *European Physical Journal B* 86, 442. https://doi.org/10.1140/epjb/e2013-40107-3 — approximating a power-law kernel with exponentials (`research.kernels`).
- Kyle, A. S. (1985). "Continuous auctions and insider trading." *Econometrica* 53(6), 1315. https://doi.org/10.2307/1913210 — LEARN section 10.8.
- Glosten, L. R., Milgrom, P. R. (1985). "Bid, ask and transaction prices in a specialist market with heterogeneously informed traders." *Journal of Financial Economics* 14(1), 71–100. https://doi.org/10.1016/0304-405X(85)90044-3 — LEARN section 10.9.
- Tóth, B., Lempérière, Y., Deremble, C., de Lataillade, J., et al. (2011). "Anomalous price impact and the critical nature of liquidity in financial markets." *Physical Review X* 1(2). https://doi.org/10.1103/PhysRevX.1.021006 — square-root impact, LEARN section 10.10.
- Lee, C. M. C., Ready, M. J. (1991). "Inferring trade direction from intraday data." *Journal of Finance* 46(2), 733–746. https://doi.org/10.1111/j.1540-6261.1991.tb02683.x — LEARN section 10.11.
- Ho, T., Stoll, H. R. (1981). "Optimal dealer pricing under transactions and return uncertainty." *Journal of Financial Economics* 9(1), 47–73. https://doi.org/10.1016/0304-405X(81)90020-9 — inventory and dealer quotes.
- Knuth, D. E. (1997). *The Art of Computer Programming, Vol. 2: Seminumerical Algorithms*, 3rd ed., §3.3.4 (MMIX LCG constants). Addison-Wesley. ISBN 978-0-201-89684-8 — the pinned 64-bit LCG (`Lcg64`).

## License

MIT — see `LICENSE`.
