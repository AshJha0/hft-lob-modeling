# FastBook: header-only book, its tests and its benchmark.  Works as a
# subdirectory of cpp/ and also on its own (cmake -S cpp/fast), which needs no
# GoogleTest because these tests use a plain main().
cmake_minimum_required(VERSION 3.16)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(lobmod_fast CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release)
    endif()
    add_compile_options(-Wall -Wextra)
    option(LOBMOD_SANITIZE "Build with AddressSanitizer and UBSan" OFF)
    if(LOBMOD_SANITIZE)
        add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=all
                            -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address,undefined)
    endif()
    add_library(lobmod ../src/book.cpp)
    target_include_directories(lobmod PUBLIC ../include)
    enable_testing()
endif()

add_library(lobmod_fast INTERFACE)
target_include_directories(lobmod_fast INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lobmod_fast INTERFACE lobmod)

add_executable(fast_book_tests ../tests/test_fast_book.cpp)
target_link_libraries(fast_book_tests PRIVATE lobmod_fast)
add_test(NAME fast_book_tests COMMAND fast_book_tests)

add_executable(fast_book_bench bench.cpp)
target_link_libraries(fast_book_bench PRIVATE lobmod_fast)
