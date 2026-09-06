#!/usr/bin/env bash
# Build the lobmod Java library, demo and tests into out/ (no Maven/Gradle).
set -euo pipefail
cd "$(dirname "$0")"

JUNIT=/usr/share/java/junit4.jar
HAMCREST=/usr/share/java/hamcrest.jar

rm -rf out
mkdir -p out/main out/test

javac -Xlint:all -Werror -d out/main $(find src/main/java -name '*.java')
javac -Xlint:all -Werror -cp "out/main:$JUNIT:$HAMCREST" -d out/test \
    $(find src/test/java -name '*.java')

echo "build OK"
