#ifndef LOBMOD_CSV_HPP
#define LOBMOD_CSV_HPP

/// \file csv.hpp
/// \brief Pinned CSV parsing primitives shared by the event-stream and
///        Hawkes-time readers (API_SPEC.md section 2.4).
///
/// Rules (identical in every language port):
/// * the header line must match exactly; a trailing '\r' is stripped from
///   every line first (CRLF files are accepted);
/// * trailing blank lines are ignored; a blank line before the last data
///   row is an error;
/// * integer fields match `-?[0-9]+` and must fit a signed 64-bit integer;
///   time fields are finite decimal floats;
/// * every violation throws std::invalid_argument (never std::out_of_range).

#include <cstdint>
#include <string>
#include <vector>

namespace lobmod {

/// Read a pinned-schema CSV: validate the header, strip '\r', drop trailing
/// blank lines and return the remaining data lines.
/// \throws std::invalid_argument on an unreadable file, bad header, or a
///         blank line before the last data row.
std::vector<std::string> read_csv_lines(const std::string& path,
                                        const std::string& expected_header);

/// Split one line on ',' keeping empty fields (a trailing comma yields an
/// extra empty field).
std::vector<std::string> split_csv_fields(const std::string& line);

/// Strict signed 64-bit integer parse (`-?[0-9]+`, in range).
/// \throws std::invalid_argument otherwise.
std::int64_t parse_int64_strict(const std::string& field,
                                const std::string& context);

/// Strict finite double parse (decimal or exponent notation only).
/// \throws std::invalid_argument otherwise.
double parse_double_strict(const std::string& field, const std::string& context);

} // namespace lobmod

#endif // LOBMOD_CSV_HPP
