# Fuzzing

Three libFuzzer harnesses for the C++ code. They are not part of the CMake build
because they need clang. The `fuzz` job in CI runs each for 60 seconds on every
push and pull request, starting from an empty corpus, and keeps any `crash-*`
file as a build artifact.

| Harness | What it feeds | What counts as a finding |
|---|---|---|
| `fuzz_book.cpp` | four bytes per operation, into the reference book and FastBook together | any difference in result, fills or quotes; any sanitizer report |
| `fuzz_csv.cpp` | raw bytes, into the field parsers and through `read_events_csv` into `replay` | any exception other than `std::invalid_argument`; any sanitizer report |
| `fuzz_itch.cpp` | raw bytes as ITCH frames, through the decoder and `apply` into a small FastBook | a pointer outside the message, a broken size or order count; any sanitizer report |

`fuzz_book.cpp` deliberately reaches the cases the seeded tests rarely hit:
sizes at or below zero, reused and duplicate ids, prices outside the band, and
a 64-order pool that fills up. When FastBook refuses for one of its own limits
(`OutOfBand`, `PoolFull`), the operation is not sent to the reference book,
which has neither limit.

Build and run from the repository root on Linux or WSL:

```sh
sh fuzz/run.sh 60        # seconds per harness; corpus and binaries go to /tmp
```

A crash leaves a `crash-*` file in the working directory. Replay it with
`/tmp/lobmod_fuzz/fuzz_book crash-<hash>`.

No corpus is stored in the repository, so each run starts from nothing unless
`/tmp/lobmod_fuzz` survives from an earlier one. A short run shows that the
harnesses work and that nothing shallow is wrong. It does not show that the
code is free of bugs.
