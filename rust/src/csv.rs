//! Pinned CSV parsing primitives shared by the event-stream and Hawkes-time
//! readers (`API_SPEC.md` section 2.4).
//!
//! Rules (identical in every language port):
//! * the header line must match exactly; a trailing `\r` is stripped from
//!   every line first (CRLF files are accepted);
//! * trailing blank lines are ignored; a blank line before the last data row
//!   is an error;
//! * integer fields match `-?[0-9]+` and must fit a signed 64-bit integer;
//!   time fields are finite decimal floats;
//! * every violation is a [`LobError::Parse`] (file problems: [`LobError::Io`]).

use std::fs;

use crate::error::{LobError, Result};

/// Read a pinned-schema CSV: validate the header, strip `\r`, drop trailing
/// blank lines and return the remaining data lines.
///
/// # Errors
/// [`LobError::Io`] if the file cannot be read; [`LobError::Parse`] on a bad
/// header or a blank line before the last data row.
pub fn read_csv_lines(path: &str, expected_header: &str) -> Result<Vec<String>> {
    let text = fs::read_to_string(path).map_err(|e| LobError::Io(format!("{path}: {e}")))?;
    let mut lines: Vec<String> = text
        .split('\n')
        .map(|l| l.strip_suffix('\r').unwrap_or(l).to_string())
        .collect();
    while lines.last().is_some_and(|l| l.is_empty()) {
        lines.pop();
    }
    match lines.first() {
        Some(h) if h == expected_header => {}
        other => {
            return Err(LobError::Parse(format!(
                "unexpected CSV header in {path}: {:?}; expected {expected_header:?}",
                other.map(String::as_str).unwrap_or("<empty file>")
            )))
        }
    }
    let data: Vec<String> = lines.into_iter().skip(1).collect();
    if let Some(i) = data.iter().position(|l| l.is_empty()) {
        return Err(LobError::Parse(format!(
            "blank line at row {} of {path} (only trailing blank lines are allowed)",
            i + 1
        )));
    }
    Ok(data)
}

/// Strict signed 64-bit integer parse (`-?[0-9]+`, in range).
///
/// # Errors
/// [`LobError::Parse`] otherwise.
pub fn parse_int64_strict(field: &str, context: &str) -> Result<i64> {
    let digits = field.strip_prefix('-').unwrap_or(field);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(LobError::Parse(format!(
            "malformed integer field {field:?} in {context}"
        )));
    }
    field.parse::<i64>().map_err(|_| {
        LobError::Parse(format!(
            "integer field {field:?} out of 64-bit range in {context}"
        ))
    })
}

/// Strict finite double parse (`[+-]digits[.digits][e[+-]digits]` only).
///
/// # Errors
/// [`LobError::Parse`] otherwise.
pub fn parse_double_strict(field: &str, context: &str) -> Result<f64> {
    let b = field.as_bytes();
    let n = b.len();
    let mut i = 0;
    if i < n && (b[i] == b'+' || b[i] == b'-') {
        i += 1;
    }
    let mut digits = 0;
    while i < n && b[i].is_ascii_digit() {
        i += 1;
        digits += 1;
    }
    if i < n && b[i] == b'.' {
        i += 1;
        while i < n && b[i].is_ascii_digit() {
            i += 1;
            digits += 1;
        }
    }
    let mut ok = digits > 0;
    if ok && i < n && (b[i] == b'e' || b[i] == b'E') {
        i += 1;
        if i < n && (b[i] == b'+' || b[i] == b'-') {
            i += 1;
        }
        let mut exp_digits = 0;
        while i < n && b[i].is_ascii_digit() {
            i += 1;
            exp_digits += 1;
        }
        ok = exp_digits > 0;
    }
    if !ok || i != n {
        return Err(LobError::Parse(format!(
            "malformed time field {field:?} in {context}"
        )));
    }
    let v: f64 = field
        .parse()
        .map_err(|_| LobError::Parse(format!("malformed time field {field:?} in {context}")))?;
    if !v.is_finite() {
        return Err(LobError::Parse(format!(
            "non-finite time {field:?} in {context}"
        )));
    }
    Ok(v)
}
