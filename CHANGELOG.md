# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [1.5.0] - 2026-11-06

### Added

- Simulator model v3: `FlowParams` fields `book_feedback`, `burst_prob` and `burst_gap`, all off by default, so earlier paths are unchanged.
- `lobmod.sim.calibrate.book_fit`, `evaluate_gaps`, `REFINED_V3` and row C7, on short gaps between aggressive orders. `build_stats` takes optional order times.
- `fit3` mode in `research/calibrate.py`. `heldout` now reads the v3 fit.

### Changed

- `research/calibration_criteria.md` logs the v3 design and checkpoint, written before any v3 fit, and its result. Tolerances and the pass mark are unchanged.

### Results

- **Failed:** calibration of model v3, against the same criteria. Order flow explained simulated prices better on liquid symbols, but no symbol passed every row in sample, and most failed the new gap row. The checkpoint was missed, so the held-out day was not read or tested. The simulator remains uncalibrated.

### Known limits

- Gap shares below 10 ms are not reproduced for most symbols.
- Price formation on thin symbols remains poor.

## [1.4.0] - 2026-11-06

### Added

- Simulator model v2: `FlowParams` fields `sign_persist`, `impact`, `active_mult`, `active_share` and `active_dwell`, all off by default, so earlier paths are unchanged.
- `lobmod.sim.calibrate.structural_fit` and `REFINED_V2`; `refine` takes the names to search.
- `fit2` and `heldout` modes in `research/calibrate.py`. `heldout` refuses to run a second time.

### Changed

- `research/calibration_criteria.md` logs the v2 design, written before any v2 fit, and its result. Tolerances and the pass mark are unchanged.
- GitHub Actions: checkout v7, setup-java v6, setup-python v7, upload-artifact v7. Rust: serde_json 1.0.154.

### Results

- **Failed:** calibration of model v2, against the same criteria. Sign memory improved, but the model missed the pass mark in sample and stopped at a checkpoint set before the fit. The held-out day was extracted locally and not read or tested. The simulator remains uncalibrated.

### Known limits

- Order timing below one millisecond is not modelled: the fitted Hawkes decay rate sits on its upper bound.

## [1.3.0] - 2026-11-05

### Added

- `reduce`, `execute`, `replace` and a v2 event schema with nanosecond timestamps in all four ports.
- `lobmod.feeds.itch50` parser and book-integrity checker. Replayed on public ITCH 5.0 sample files for ten symbols with no parser or book anomalies and every execution at the touch; internal consistency, not comparison with a venue snapshot. Raw data is not redistributed.
- `lobmod.feeds.iex` and `lobmod.feeds.pcap`: IEX HIST capture reader for DEEP and DEEP+ with sequence checks, fixture-tested and run once on one real DEEP+ file.
- `research/real_replay.py` and `research/iex_replay.py` for replaying a file you supply.
- Binary ITCH 5.0 decoder in C++ and Rust with `apply` into FastBook, and `itch_bench` in both.
- FastBook `rest`, `reduce` and `replace`; a single id-map walk on a non-crossing add; a Rust allocation test.
- `lobmod.sim.calibrate`, `research/calibrate.py` and `research/calibration_criteria.md`, with the criteria fixed before any fit.
- `research/results_iex.md`: one IEX DEEP+ day for AAPL, MSFT and INTC, the only real-data numbers published. Data provided for free by IEX. By accessing or using IEX Historical Data, you agree to the IEX Historical Data Terms of Use. Terms: <https://www.iex.io/products/market-data-connectivity/hist-terms>.
- `research/control_362.py` and `research/results_control.md`.
- `bench/run_all.sh` and a Java reference-book benchmark (`java/bench.sh`).
- `pyproject.toml`, `requirements.lock`, `CITATION.cff`, `CONTRIBUTING.md`, `SECURITY.md`, Dependabot, an advisory clang-tidy job, reported Python coverage and a package-install check in CI.
- LEARN: one order end to end, market making, Kyle, Glosten-Milgrom, square-root impact, Lee-Ready, how to read a backtest honestly. COOKBOOK recipes 21 to 25. Six new diagrams.
- `lobmod.sim` execution simulator with latency, queues, informed flow and P&L attribution.
- Multi-exponential and power-law Hawkes kernels, seasonal baselines, cost-aware labels and a boosted-tree baseline in `lobmod.research`.
- `FastBook` in C++ and Rust, benchmarks and `docs/LATENCY.md`.
- Naive-model property tests in Rust and Java; three libFuzzer harnesses in `fuzz/` (book, CSV, ITCH decoder), each run for 60 seconds in CI.

### Changed

- `BookIntegrityChecker` tracks the remaining size of held orders.
- Documentation no longer attributes most of the old maker cap gain to mean reversion.
- `docs/LATENCY.md` now reports idle-machine runs, in which FastBook is faster than the reference book on passive add as well as cancel. Java was not in this run.
- `research/results.md`, `results_models.md` and `results_sim.md` regenerated with larger runs (60 streams, 30 streams, 40 seeds).

### Research outcomes

- **Failed:** calibration of `lobmod.sim` to real data, against criteria fixed in advance. It did not meet the criteria in sample, parameters did not carry across the day, and the held-out day was not run. The simulator remains uncalibrated and its P&L levels mean nothing.
- **Inconclusive:** the +362 control. Mean reversion is only partly supported as the explanation.
- **Not done:** a GC-free Java path; Java has no FastBook.
- Hawkes on real trades, qualitative only: a seasonal two-timescale model is preferred over a single exponential, but the fast rate hits its bound and branching is not resolved. No value is quoted.

## [1.2.0] - 2026-11-05

### Added

- `.gitattributes` keeping bundled data byte-exact on every platform.
- CI for all four ports, a sanitizer build, a Windows Python job and a docs check.
- README "Limits" and "Known results" sections; LEARN "Topics not covered".
- `lobmod.research` (Python only, outside the golden contract): Hawkes standard
  errors, likelihood-ratio tests, parametric-bootstrap goodness of fit, residual
  dependence tests, purged walk-forward evaluation, multi-seed comparison, an
  executable taker backtest and a passive maker simulator.
- `research/run_all.py` and generated `research/results.md`.
- `LOBMOD_SANITIZE` CMake option; `JUNIT_CP` for the Java scripts.

### Changed

- Demos report the gap coefficient of variation instead of the minimum gap, and
  "no rejection (test is weak with fitted parameters)" instead of "PASS".
- Rust release builds keep overflow checks on, so dev and release agree.
- Grid tolerance is capped at 0.01 tick; `limit()` refuses ratios from 2^52.
- Adds that would take a side's resting size above 9e18 are rejected. The check
  uses the submitted size, so a fully marketable order can be refused on a
  near-full side.

### Fixed

- Documentation corrections: `microprice` is the size-weighted mid and not
  Stoikov's micro-price; the pinned init is below He scaling; KS on fitted
  parameters is weak, not slightly conservative; adverse selection is a
  passive-fill cost; claims about real markets and independent verification
  removed.
- `mid()` no longer overflows at legal prices, and all ports agree on its rounding.
- Size sums can no longer wrap; Java spread subtracts before converting.
- Off-grid prices above about 5e8 ticks are rejected instead of silently rounded.
- Python uses plain left-to-right sums, matching the other ports (the bundled
  univariate fit now reports 173 iterations in Python too).
- `geometric` overflow is a range error in every port; Python raises `ValueError`
  on price overflow, rejects `

` line endings and reads CSV as UTF-8.

## [1.1.0] - 2026-09-06

- Review release: correctness fixes, edge-case tests, references and docs site.
