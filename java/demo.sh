#!/usr/bin/env bash
# Run the end-to-end demo (builds first if needed).
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d out/main ]; then
    bash build.sh
fi

java -cp out/main com.quant.lobmod.Demo ../data
