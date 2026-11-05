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
  monotonic clock. The median cost of an empty timed section (about 20 ns here)
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
The figures are from one `bench/run_all.sh` session on an idle machine, with a
load average of 0.60 when it began. Java was not in this run, because `javac` was
not on the path.

Treat these as rough figures:

- There was no frequency locking and no core isolation. WSL2 runs under a
  hypervisor, so pinning a virtual CPU does not pin a physical one.
- Between the two runs, p50 cells moved by up to about 10%, and FastBook's p99
  cells by up to about 25%. Its p99.9 cells moved by up to a factor of two, and
  reverse-cancel FastBook cells by about the same. Small differences below mean
  nothing.
- The maximum columns, from about 50 microseconds to 3 ms, look like scheduler or
  hypervisor preemption rather than book behaviour. They do not rank the books.
  This was not confirmed by tracing.
- WSL and native Windows numbers are not comparable, and only WSL was measured.
  Nor do these numbers predict server hardware.
- A laptop benchmark of the book alone is not wire-to-wire latency.

## Results

Two complete runs, shown as run 1 / run 2. All latencies in ns.

### C++

| Book | Op | p50 | p99 | p99.9 |
|---|---|---|---|---|
| reference | add | 125 / 115 | 302 / 271 | 565 / 481 |
| reference | cancel | 294 / 279 | 593 / 522 | 1910 / 2167 |
| reference | marketable limit | 323 / 297 | 1042 / 892 | 3165 / 1980 |
| reference | market | 348 / 318 | 1228 / 1046 | 9629 / 2378 |
| FastBook | add | 32 / 35 | 265 / 216 | 377 / 407 |
| FastBook | cancel | 54 / 58 | 261 / 229 | 415 / 541 |
| FastBook | marketable limit | 87 / 92 | 381 / 300 | 638 / 1253 |
| FastBook | market | 86 / 83 | 361 / 302 | 578 / 622 |

Untimed throughput: reference 4.35 / 4.86 M ops/s, FastBook 32.64 / 36.47 M ops/s.

### Rust

| Book | Op | p50 | p99 | p99.9 |
|---|---|---|---|---|
| reference | add | 90 / 89 | 260 / 231 | 586 / 452 |
| reference | cancel | 170 / 167 | 393 / 329 | 942 / 754 |
| reference | marketable limit | 245 / 243 | 779 / 723 | 1931 / 1641 |
| reference | market | 235 / 232 | 954 / 855 | 2209 / 1623 |
| FastBook | add | 34 / 34 | 235 / 189 | 480 / 320 |
| FastBook | cancel | 55 / 55 | 244 / 210 | 704 / 413 |
| FastBook | marketable limit | 82 / 85 | 349 / 305 | 1178 / 778 |
| FastBook | market | 87 / 87 | 335 / 307 | 637 / 609 |

Untimed throughput: reference 7.01 / 7.44 M ops/s, FastBook 29.69 / 33.79 M ops/s.

### Reverse cancel, worst case

N orders rest at one price and are cancelled newest first, so the reference book
scans its whole queue each time. Mean ns per cancel, one pass per run.

| Orders | C++ reference | C++ FastBook | Rust reference | Rust FastBook |
|---|---|---|---|---|
| 1,000 | 300 / 290 | 11 / 16 | 218 / 219 | 19 / 16 |
| 10,000 | 2857 / 2719 | 27 / 12 | 1702 / 1712 | 37 / 30 |
| 100,000 | 43616 / 44351 | 12 / 9 | 18189 / 18535 | 20 / 17 |

### ITCH decode

One run per language, 5,000,000 synthetic frames, 7 passes. Mean ns per message.
The method is further down. The final book state and checksum matched between
the languages.

| Stage | C++ median | C++ best | Rust median | Rust best |
|---|---|---|---|---|
| decode | 13.6 | 13.5 | 19.0 | 18.3 |
| decode + FastBook | 24.6 | 23.9 | 39.9 | 35.1 |

## Reading the results

- **Throughput improved in both languages:** about 7.5x in C++ and 4.2x to 4.5x
  in Rust on this script.
- **Every operation is faster at p50.** Passive add takes 32 to 35 ns against 115
  to 125 ns in C++, and 34 ns against 89 to 90 ns in Rust. Cancel takes 54 to 58 ns
  against 279 to 294 ns, and 55 ns against 167 to 170 ns.
- **The tail gap is narrower.** At p99, add is 216 to 265 ns against 271 to 302 ns
  in C++. The fenced timer hides overlap between operations and the tail includes
  outside interruptions, which may explain part of this, but that was not tested.
- **The clearest change is in the worst case.** Reference cancel cost grows with
  queue length, while FastBook stays in the tens of nanoseconds or below at
  100,000 orders.
- **The comparison between languages is weak.** Each language's two books were
  run on the same script, so compare within a language. The Rust reference beats
  the C++ reference here, most likely through different standard containers; this
  was not investigated. The two FastBooks are close on the book tables, and C++
  is ahead on the ITCH stages.

## Reproducing

```sh
cmake -S cpp/fast -B build-fast -DCMAKE_BUILD_TYPE=Release
cmake --build build-fast -j
build-fast/fast_book_bench 2000000 20000 5

cd rust && cargo run --release --bin bench -- 2000000 20000 5
```

`bench/run_all.sh` runs everything on this page in one go, plus the ITCH and
Java benchmarks below, and writes the output to one file. Run it on an idle
machine, since other load changes these figures a lot.

## ITCH decode and the Java book: method

ITCH results are in the tables above. The Java benchmark was not in the idle run,
so no Java number is quoted.

**ITCH** (`cpp/fast/itch_bench.cpp`, `rust/src/bin/itch_bench.rs`). Both build
the same synthetic stream of framed ITCH 5.0 messages from an integer-only
seeded generator: 5,000,000 messages of types A F E C X D U P around 20,000 to
40,000 live orders, already in memory. Each pass walks the whole buffer and is
timed as one block, so a figure is mean ns per message with no per-message
timer. The table gives the median and the best of 7 passes. `decode` reads
every field; `decode + FastBook` also mirrors each message into a fresh book.
Each run prints the final book state and a field checksum, and the two languages
must print the same line. There is no socket, kernel or capture path in these
figures, and synthetic frames are not real flow.

**FastBook add.** `limit` now walks the id map once for the duplicate check and
the insert, when the order does not cross. The idle runs above include this change,
but no idle A/B run isolated it, so its own share of the add figure is not known.
Hardware counters are not available in
WSL2, so this came from reading the path, not from a counter profile.

**Java** (`java/bench.sh`). A plain `System.nanoTime` loop, not JMH: JMH would
add a dependency that the no-Maven build here does not have. It drives the
reference `LimitOrderBook` through a script of adds, cancels and market orders
after a warm-up, and reports mean ns per operation and heap bytes allocated per
operation. There is no Java FastBook and no GC-free Java path, so that allocation figure is the cost of
the reference book's objects. Without JMH, nothing guards against dead-code or
on-stack-replacement effects beyond a consumed checksum, so treat the result as
a rough figure.
