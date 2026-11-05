#!/bin/sh
# Build the fuzz harnesses with clang and run each for $1 seconds (default 60).
# Exits non-zero if any harness finds something; CI relies on that.
set -e
cd "$(dirname "$0")/.."
secs="${1:-60}"
out=/tmp/lobmod_fuzz
mkdir -p "$out/corpus_book" "$out/corpus_csv" "$out/corpus_itch"
flags="-std=c++17 -g -O1 -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all -Icpp/include -Icpp/fast"
clang++ $flags fuzz/fuzz_book.cpp cpp/src/book.cpp -o "$out/fuzz_book"
clang++ $flags fuzz/fuzz_csv.cpp cpp/src/csv.cpp cpp/src/book.cpp cpp/src/replay.cpp -o "$out/fuzz_csv"
clang++ $flags fuzz/fuzz_itch.cpp cpp/src/book.cpp -o "$out/fuzz_itch"
status=0
for h in book csv itch; do
    # No pipe into tail here: it would hide the harness's exit status.
    if "$out/fuzz_$h" -max_total_time="$secs" -print_final_stats=1 "$out/corpus_$h" \
        >"$out/$h.log" 2>&1; then
        tail -8 "$out/$h.log"
    else
        echo "fuzz_$h FAILED" >&2
        tail -60 "$out/$h.log" >&2
        status=1
    fi
done
exit "$status"
