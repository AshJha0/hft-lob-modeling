// libFuzzer harness for the C++ CSV layer.  The contract is that bad input
// raises std::invalid_argument; anything else escaping, or a sanitizer report,
// is a finding.  The file reader takes a path, so whole-file inputs go through
// one temporary file per process.
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "lobmod/csv.hpp"
#include "lobmod/replay.hpp"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t len) {
    const std::string text(reinterpret_cast<const char*>(data), len);

    try {
        lobmod::parse_int64_strict(text, "fuzz");
    } catch (const std::invalid_argument&) {
    }
    try {
        lobmod::parse_double_strict(text, "fuzz");
    } catch (const std::invalid_argument&) {
    }
    lobmod::split_csv_fields(text);

    static const std::string path =
        "/tmp/lobmod_fuzz_csv_" + std::to_string(static_cast<long>(getpid())) + ".csv";
    if (std::FILE* f = std::fopen(path.c_str(), "wb")) {
        std::fwrite(text.data(), 1, text.size(), f);
        std::fclose(f);
        try {
            lobmod::replay(lobmod::read_events_csv(path));
        } catch (const std::invalid_argument&) {
        }
    }
    return 0;
}
