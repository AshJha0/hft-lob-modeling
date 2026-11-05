#!/bin/sh
# Build both fuzz harnesses with clang and run each for $1 seconds (default 60).
set -e
cd "$(dirname "$0")/.."
secs="${1:-60}"
out=/tmp/lobmod_fuzz
mkdir -p "$out/corpus_book" "$out/corpus_csv"
flags="-std=c++17 -g -O1 -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all -Icpp/include -Icpp/fast"
clang++ $flags fuzz/fuzz_book.cpp cpp/src/book.cpp -o "$out/fuzz_book"
clang++ $flags fuzz/fuzz_csv.cpp cpp/src/csv.cpp cpp/src/book.cpp cpp/src/replay.cpp -o "$out/fuzz_csv"
"$out/fuzz_book" -max_total_time="$secs" -print_final_stats=1 "$out/corpus_book" 2>&1 | tail -8
"$out/fuzz_csv" -max_total_time="$secs" -print_final_stats=1 "$out/corpus_csv" 2>&1 | tail -8
