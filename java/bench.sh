#!/usr/bin/env bash
# Build and run the Java book benchmark (no Maven/Gradle, no JMH).
# Usage: bash bench.sh [ops] [live_orders] [repeats]
set -euo pipefail
cd "$(dirname "$0")"

rm -rf out/bench
mkdir -p out/bench
javac -Xlint:all -Werror -d out/bench $(find src/main/java src/bench/java -name '*.java')
java -cp out/bench com.quant.lobmod.BookBench "$@"
