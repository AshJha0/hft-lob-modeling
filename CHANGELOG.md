# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
