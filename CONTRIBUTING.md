# Contributing

Issues and pull requests are welcome. This page covers what a change has to do
to pass review.

## The contract comes first

`API_SPEC.md` pins the formulas, constants and iteration order, and all four
languages reproduce `data/golden/golden.json` from the bundled CSVs. A change to
behaviour in one language needs the same change in the others, a spec update,
and a deliberate regeneration of the golden file if numbers move. Files under
`data/` are byte-exact, so do not reformat them.

## Running the checks

```sh
cd python && PYTHONPATH=src python -m pytest -q
cd rust && cargo fmt --check && cargo clippy --all-targets && cargo test --release
cd cpp && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build
cd java && bash build.sh && bash test.sh
sh fuzz/run.sh 60        # needs clang
```

Finish these locally before pushing. Python installs with
`pip install -e ".[test]"` from the repository root, and `requirements.lock`
pins exact versions with hashes for a reproducible install.

## Standards

- FastBook and the ITCH decoder must not allocate after construction. In Rust,
  `rust/tests/alloc_free.rs` enforces this.
- Changes to FastBook keep the differential tests against the reference book
  passing.
- A performance claim needs a before and after from `bench/run_all.sh` on an idle
  machine, stated with its method. Say so when a change made no difference.
- A research result states what was run and on what data, including results
  that did not support the idea.
- Documentation states the plain idea first, then the formal statement, then a
  worked example.

## Pull requests

Keep a pull request to one purpose and update `CHANGELOG.md`. All required CI
jobs must pass. The `clang-tidy` job is advisory, but do read what it reports.
