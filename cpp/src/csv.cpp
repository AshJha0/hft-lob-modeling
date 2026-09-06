#include "lobmod/csv.hpp"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace lobmod {

std::vector<std::string> read_csv_lines(const std::string& path,
                                        const std::string& expected_header) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::invalid_argument("cannot open CSV: " + path);
    }
    std::stringstream buf;
    buf << in.rdbuf();
    const std::string text = buf.str();
    std::vector<std::string> lines;
    std::string cur;
    for (char c : text) {
        if (c == '\n') {
            lines.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    lines.push_back(cur); // text after the last '\n' (possibly empty)
    for (std::string& ln : lines) {
        if (!ln.empty() && ln.back() == '\r') ln.pop_back();
    }
    while (!lines.empty() && lines.back().empty()) lines.pop_back();
    if (lines.empty() || lines.front() != expected_header) {
        throw std::invalid_argument("unexpected CSV header in " + path + ": '" +
                                    (lines.empty() ? std::string("<empty file>")
                                                   : lines.front()) +
                                    "'; expected '" + expected_header + "'");
    }
    std::vector<std::string> data(lines.begin() + 1, lines.end());
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (data[i].empty()) {
            throw std::invalid_argument("blank line at row " + std::to_string(i + 1) +
                                        " of " + path +
                                        " (only trailing blank lines are allowed)");
        }
    }
    return data;
}

std::vector<std::string> split_csv_fields(const std::string& line) {
    std::vector<std::string> out;
    std::string field;
    for (char c : line) {
        if (c == ',') {
            out.push_back(field);
            field.clear();
        } else {
            field.push_back(c);
        }
    }
    out.push_back(field);
    return out;
}

std::int64_t parse_int64_strict(const std::string& field,
                                const std::string& context) {
    std::size_t i = 0;
    if (!field.empty() && field[0] == '-') i = 1;
    if (i >= field.size()) {
        throw std::invalid_argument("malformed integer field '" + field + "' in " +
                                    context);
    }
    for (std::size_t j = i; j < field.size(); ++j) {
        if (field[j] < '0' || field[j] > '9') {
            throw std::invalid_argument("malformed integer field '" + field +
                                        "' in " + context);
        }
    }
    errno = 0;
    char* end = nullptr;
    const long long v = std::strtoll(field.c_str(), &end, 10);
    if (errno == ERANGE || end != field.c_str() + field.size()) {
        throw std::invalid_argument("integer field '" + field +
                                    "' out of 64-bit range in " + context);
    }
    return static_cast<std::int64_t>(v);
}

double parse_double_strict(const std::string& field, const std::string& context) {
    // accept only [+-]digits[.digits][e[+-]digits] / [+-].digits[...]
    std::size_t i = 0;
    const std::size_t n = field.size();
    if (i < n && (field[i] == '+' || field[i] == '-')) ++i;
    std::size_t digits = 0;
    while (i < n && field[i] >= '0' && field[i] <= '9') { ++i; ++digits; }
    if (i < n && field[i] == '.') {
        ++i;
        while (i < n && field[i] >= '0' && field[i] <= '9') { ++i; ++digits; }
    }
    bool ok = digits > 0;
    if (ok && i < n && (field[i] == 'e' || field[i] == 'E')) {
        ++i;
        if (i < n && (field[i] == '+' || field[i] == '-')) ++i;
        std::size_t exp_digits = 0;
        while (i < n && field[i] >= '0' && field[i] <= '9') { ++i; ++exp_digits; }
        ok = exp_digits > 0;
    }
    if (!ok || i != n) {
        throw std::invalid_argument("malformed time field '" + field + "' in " +
                                    context);
    }
    const double v = std::strtod(field.c_str(), nullptr);
    if (!std::isfinite(v)) {
        throw std::invalid_argument("non-finite time '" + field + "' in " + context);
    }
    return v;
}

} // namespace lobmod
