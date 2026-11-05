#!/usr/bin/env bash
# Run every JUnit 4 *Test class found under out/test.
set -euo pipefail
cd "$(dirname "$0")"

# JUNIT_CP overrides the JUnit + Hamcrest classpath.
JUNIT_CP="${JUNIT_CP:-/usr/share/java/junit4.jar:/usr/share/java/hamcrest.jar}"

if [ ! -d out/test ]; then
    bash build.sh
fi

TESTS=$(cd out/test && find . -name '*Test.class' ! -name '*\$*' \
    | sed 's|^\./||; s|\.class$||; s|/|.|g' | sort)

java -cp "out/main:out/test:$JUNIT_CP" org.junit.runner.JUnitCore $TESTS
