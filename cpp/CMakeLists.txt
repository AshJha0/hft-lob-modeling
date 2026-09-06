cmake_minimum_required(VERSION 3.16)
project(lobmod_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra)

# The bundled datasets live one level above the cpp/ directory; tests and the
# demo both resolve them through this compile definition.
set(LOBMOD_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../data")

add_library(lobmod
    src/book.cpp
    src/csv.cpp
    src/replay.cpp
    src/zi.cpp
    src/neldermead.cpp
    src/hawkes.cpp
    src/gof.cpp
    src/features.cpp
    src/nn.cpp
)
target_include_directories(lobmod PUBLIC include)

add_executable(demo src/demo.cpp)
target_link_libraries(demo PRIVATE lobmod)
target_compile_definitions(demo PRIVATE LOBMOD_DATA_DIR="${LOBMOD_DATA_DIR}")

enable_testing()
find_package(GTest REQUIRED)

add_executable(lobmod_tests
    tests/test_book.cpp
    tests/test_hawkes.cpp
    tests/test_gof.cpp
    tests/test_features.cpp
    tests/test_nn.cpp
    tests/test_golden.cpp
)
target_link_libraries(lobmod_tests PRIVATE lobmod GTest::gtest GTest::gtest_main)
target_compile_definitions(lobmod_tests PRIVATE LOBMOD_DATA_DIR="${LOBMOD_DATA_DIR}")

include(GoogleTest)
add_test(NAME lobmod_tests COMMAND lobmod_tests)
