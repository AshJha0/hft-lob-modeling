# Book latency: reference book and FastBook

This page compares the reference book with FastBook on one machine. The numbers
cover the book only: one call into the book, in a loop, with its input already
in memory. There is no network, parsing, feed handling, risk check or order
gateway here, so nothing on this page is a wire-to-wire figure.

## What was measured

FastBook (`cpp/fast/fast_book.hpp`, `rust/src/fast_book.rs`) has the same
matching rules as the reference book inside a fixed price band. Its deliberate
differences are listed at the top of those files. The main ones are that a limit
outside the band returns an error code, and that a full pool refuses every
limit.

Both benchmarks build the same kind of script from the same seeded generator
and parameters:

- 20,000 resting orders before timing starts, up to 500 ticks from the touch and
  concentrated near it.
- 2,000,000 timed operations: about 52% passive adds, 43% cancels of random live
  orders, 3% marketable limits and 2% market orders. The add rate adapts to hold
  the live count near the target, and the mid drifts by one tick every 1,000
  operations.
- A band of 131,072 ticks and a pool of 131,072 orders. A much larger pool
  makes the id map larger and slows FastBook: an early run with a pool of 2^20
  was visibly worse, so pool size belongs in any sizing decision.

Method:

- Each operation is timed with fenced `rdtsc`/`rdtscp`, calibrated against the
  monotonic clock. The median cost of an empty timed section (14 to 19 ns here)
  is subtracted, clamping at zero. Fences stop the operation overlapping its
  neighbours, so these latencies differ from what a pipelined loop would see.
- Percentiles are per operation type. The tables give the median over 5 repeats
  of each percentile, and the maximum over all 5 repeats.
- Throughput comes from a separate pass with no per-operation timer.
- Both books see identical scripts, and a checksum of all fills must match or
  the benchmark exits non-zero. It matched in every run.
- The reference book runs with `record_trades = false`. Its timings include the
  `Vec` or `std::vector` of fills that its API returns; FastBook writes to a sink
  instead. That difference is part of what is being compared.
- The process is pinned to CPU 2 on Linux. It is not pinned on other systems.

The script is synthetic and the loop keeps the book warm in cache, because
nothing else competes between calls. Real flow has different shapes, and a book
that has gone cold between messages will be slower.

## Machine and caveats

Intel Core 5 120U laptop, Windows 11 host, WSL2 kernel 6.18, g++ 15.2 at `-O3`,
rustc 1.99.1 in the release profile. The Rust release profile keeps
`overflow-checks = true`, and FastBook in Rust is safe code with bounds checks.

Treat these as rough figures:

- The laptop was running other work during the runs, with no frequency locking
  and no core isolation. WSL2 runs under a hypervisor, so pinning a virtual CPU
  does not pin a physical one.
- Between two back-to-back runs the same cell moved by up to about 30% for
  FastBook's percentiles, and reverse-cancel FastBook cells moved by a factor of
  three. Small differences below mean nothing.
- The maximum columns, around 1 to 7 ms for both books, look like scheduler or
  hypervisor preemption rather than book behaviour. They do not rank the books.
  This was not confirmed by tracing.
- WSL and native Windows numbers are not comparable, and only WSL was measured.
  Nor do these numbers predict server hardware.

## Results

Two complete runs, shown as run 1 / run 2. All latencies in ns.

### C++

| Book | Op | p50 | p99 | p99.9 |
|---|---|---|---|---|
| reference | add | 112 / 110 | 377 / 357 | 910 / 762 |
| reference | cancel | 273 / 271 | 867 / 798 | 3069 / 2476 |
| reference | marketable limit | 284 / 285 | 1274 / 1188 | 3517 / 2807 |
| reference | market | 309 / 312 | 1491 / 1375 | 8312 / 3264 |
| FastBook | add | 131 / 109 | 440 / 375 | 845 / 696 |
| FastBook | cancel | 156 / 122 | 523 / 456 | 1120 / 768 |
| FastBook | marketable limit | 182 / 137 | 740 / 597 | 1981 / 1058 |
| FastBook | market | 145 / 110 | 764 / 609 | 1706 / 991 |

Untimed throughput: reference 4.26 / 4.47 M ops/s, FastBook 21.74 / 24.62 M ops/s.

### Rust

| Book | Op | p50 | p99 | p99.9 |
|---|---|---|---|---|
| reference | add | 89 / 86 | 323 / 351 | 751 / 668 |
| reference | cancel | 187 / 179 | 574 / 587 | 2352 / 1115 |
| reference | marketable limit | 249 / 242 | 1040 / 1047 | 3324 / 2296 |
| reference | market | 250 / 238 | 1249 / 1212 | 4669 / 3492 |
| FastBook | add | 115 / 113 | 431 / 413 | 778 / 850 |
| FastBook | cancel | 134 / 128 | 505 / 512 | 956 / 1032 |
| FastBook | marketable limit | 169 / 152 | 768 / 758 | 3536 / 1954 |
| FastBook | market | 141 / 134 | 786 / 769 | 1365 / 1490 |

Untimed throughput: reference 6.13 / 6.06 M ops/s, FastBook 14.94 / 20.68 M ops/s.

### Reverse cancel, worst case

N orders rest at one price and are cancelled newest first, so the reference book
scans its whole queue each time. Mean ns per cancel, one pass per run.

| Orders | C++ reference | C++ FastBook | Rust reference | Rust FastBook |
|---|---|---|---|---|
| 1,000 | 242 / 259 | 12 / 28 | 574 / 240 | 64 / 14 |
| 10,000 | 3009 / 3200 | 37 / 12 | 2919 / 2969 | 139 / 44 |
| 100,000 | 48528 / 46204 | 16 / 40 | 20446 / 19900 | 30 / 24 |

## Reading the results

- **Throughput improved in both languages:** about 5x in C++ and about 2.4x to
  3.4x in Rust on this script.
- **Single-operation latency improved less than throughput.** Cancels and
  aggressive orders are faster at p50. Passive adds are not: FastBook is level
  with the C++ reference and slower than the Rust reference. The fenced timer
  hides overlap between operations, which may explain part of the gap with the
  throughput figures, but that was not tested.
- **The clearest change is in the worst case.** Reference cancel cost grows with
  queue length, while FastBook stays in the tens of nanoseconds at 100,000
  orders. The reference figures are close to linear in N.
- **The comparison between languages is weak.** Each language's two books were
  run on the same script, so compare within a language. The Rust reference beats
  the C++ reference here, most likely through different standard containers; this
  was not investigated.

## Reproducing

```sh
cmake -S cpp/fast -B build-fast -DCMAKE_BUILD_TYPE=Release
cmake --build build-fast -j
build-fast/fast_book_bench 2000000 20000 5

cd rust && cargo run --release --bin bench -- 2000000 20000 5
```

There is no Java FastBook or Java benchmark.
