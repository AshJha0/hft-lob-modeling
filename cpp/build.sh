#!/bin/sh
# Build the lobmod C++ library, demo and tests (Release).
set -e
cd "$(dirname "$0")"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
