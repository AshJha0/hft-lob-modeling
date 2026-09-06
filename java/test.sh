#!/usr/bin/env bash
# Run every JUnit 4 *Test class found under out/test.
set -euo pipefail
cd "$(dirname "$0")"

JUNIT=/usr/share/java/junit4.jar
HAMCREST=/usr/share/java/hamcrest.jar

if [ ! -d out/test ]; then
    bash build.sh
fi

TESTS=$(cd out/test && find . -name '*Test.class' ! -name '*\$*' \
    | sed 's|^\./||; s|\.class$||; s|/|.|g' | sort)

java -cp "out/main:out/test:$JUNIT:$HAMCREST" org.junit.runner.JUnitCore $TESTS
