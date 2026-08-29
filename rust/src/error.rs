//! Error type for the `lobmod` crate.
//!
//! Per the cross-language contract, invalid input is *reported* through
//! `Result<_, LobError>` — the library never panics on bad input.  Recoverable
//! model diagnostics (near-critical branching, non-convergence) are carried in
//! result structs, never returned as errors.

use std::fmt;

/// Error returned by all fallible `lobmod` operations (manual thiserror-style enum).
#[derive(Debug, Clone, PartialEq)]
pub enum LobError {
    /// Invalid argument or domain violation (bad price/size/parameter/...).
    InvalidInput(String),
    /// Cancel of an order id that is not live in the book
    /// (never rested, already cancelled, or fully filled).
    UnknownOrder(i64),
    /// I/O failure while reading a bundled data file.
    Io(String),
    /// Malformed CSV/JSON content.
    Parse(String),
}

impl fmt::Display for LobError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LobError::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            LobError::UnknownOrder(id) => write!(f, "cancel of unknown order_id {id}"),
            LobError::Io(msg) => write!(f, "i/o error: {msg}"),
            LobError::Parse(msg) => write!(f, "parse error: {msg}"),
        }
    }
}

impl std::error::Error for LobError {}

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, LobError>;
