#!/usr/bin/env bash
# Build the lobmod Java library, demo and tests into out/ (no Maven/Gradle).
set -euo pipefail
cd "$(dirname "$0")"

# JUNIT_CP overrides the JUnit + Hamcrest classpath.
JUNIT_CP="${JUNIT_CP:-/usr/share/java/junit4.jar:/usr/share/java/hamcrest.jar}"

rm -rf out
mkdir -p out/main out/test

javac -Xlint:all -Werror -d out/main $(find src/main/java -name '*.java')
javac -Xlint:all -Werror -cp "out/main:$JUNIT_CP" -d out/test \
    $(find src/test/java -name '*.java')

echo "build OK"
