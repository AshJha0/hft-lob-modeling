#!/usr/bin/env bash
# Every latency benchmark in one go, for an idle Linux or WSL machine.  Method:
# docs/LATENCY.md.  Output goes to the terminal and to the file printed at the
# end.  Stop other work first: these runs pin to CPU 2 and assume it is free.
# Usage: bash bench/run_all.sh [output_file]
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
out="${1:-$root/bench/results-$(date +%Y%m%d-%H%M%S).txt}"
work="${TMPDIR:-/tmp}/lobmod_bench"
mkdir -p "$work"

run() {
    cd "$root"

    echo "== machine"
    date -u
    git rev-parse HEAD
    git status --short | head -20
    uname -a
    grep -m1 'model name' /proc/cpuinfo || true
    uptime
    g++ --version | head -1
    rustc --version
    java -version 2>&1 | head -1 || true

    echo "== build"
    cmake -S cpp/fast -B "$work/cpp" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-O3 >/dev/null
    cmake --build "$work/cpp" -j >/dev/null
    (cd rust && CARGO_TARGET_DIR="$work/rust" cargo build --release --bins 2>&1 | tail -1)

    # Two complete runs, the form docs/LATENCY.md reports.
    for pass in 1 2; do
        echo "== FastBook against the reference, C++, run $pass"
        "$work/cpp/fast_book_bench" 2000000 20000 5
        echo "== FastBook against the reference, Rust, run $pass"
        "$work/rust/release/bench" 2000000 20000 5
    done

    echo "== ITCH, C++"
    "$work/cpp/itch_bench" 5000000 20000 7 | tee "$work/itch_cpp.txt"
    echo "== ITCH, Rust"
    "$work/rust/release/itch_bench" 5000000 20000 7 | tee "$work/itch_rust.txt"
    if [ "$(grep '^state' "$work/itch_cpp.txt")" = "$(grep '^state' "$work/itch_rust.txt")" ]; then
        echo "ITCH state lines identical"
    else
        echo "ITCH state lines DIFFER" >&2
        exit 1
    fi

    if command -v javac >/dev/null; then
        echo "== Java"
        bash java/bench.sh 2000000 20000 7
    else
        echo "== Java skipped: no javac on PATH"
    fi
}

run 2>&1 | tee "$out"
echo "results in $out"
