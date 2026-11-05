//! Binary Nasdaq TotalView-ITCH 5.0 decoder and its [`FastBook`] mirror.
//!
//! The same decoder as `cpp/fast/itch_decode.hpp`.  [`decode`] reads one
//! message into a [`Message`] without allocating; `stock` and `attribution`
//! borrow from the input.  Integers on the wire are big-endian, and prices are
//! four-decimal fixed point, so one tick here is 0.0001 dollars.
//!
//! Types A F E C X D U P are decoded.  The other known types are checked for
//! length and give `Ok(None)`.  A type outside the table gives
//! [`DecodeError::UnknownType`]; the frame prefix still gives its length, so a
//! caller can step over it.  Nothing is keyed by stock locate: [`apply`]
//! mirrors one instrument, so the caller filters first.

use crate::book::Side;
use crate::fast_book::{FastBook, FastError};

/// Why a message was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// Fewer bytes than the type needs.
    Truncated,
    /// More bytes than the type has.
    BadLength,
    /// A side byte other than `B` or `S`.
    BadSide,
    /// Not in the length table.
    UnknownType,
}

/// Message length by type byte, frame prefix excluded; 0 if unknown.
#[must_use]
pub const fn message_length(kind: u8) -> usize {
    match kind {
        b'A' => 36,
        b'F' => 40,
        b'E' => 31,
        b'C' => 36,
        b'X' => 23,
        b'D' => 19,
        b'U' => 35,
        b'P' => 44,
        b'Q' => 40,
        b'R' => 39,
        b'S' => 12,
        b'H' => 25,
        b'Y' => 20,
        b'L' => 26,
        b'I' => 50,
        b'J' => 35,
        b'K' => 28,
        b'V' => 35,
        _ => 0,
    }
}

/// One decoded message.  Fields a type does not carry are zero or `None`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Message<'a> {
    /// Type byte.
    pub kind: u8,
    /// `b'B'` or `b'S'`: A F P.
    pub side: u8,
    /// `b'Y'` or `b'N'`: C.
    pub printable: u8,
    /// Stock locate.
    pub locate: u16,
    /// Tracking number.
    pub tracking: u16,
    /// A F P size; E C executed; X cancelled; U new size.
    pub shares: u32,
    /// A F P U limit price; C execution price.
    pub price: u32,
    /// Nanoseconds since midnight.
    pub timestamp_ns: u64,
    /// Order reference; for U, the original.
    pub reference: u64,
    /// U only.
    pub new_reference: u64,
    /// E C P.
    pub match_number: u64,
    /// Space padded: A F P.
    pub stock: Option<&'a [u8; 8]>,
    /// F only.
    pub attribution: Option<&'a [u8; 4]>,
}

#[inline(always)]
fn be16(p: &[u8], at: usize) -> u16 {
    u16::from_be_bytes([p[at], p[at + 1]])
}

#[inline(always)]
fn be32(p: &[u8], at: usize) -> u32 {
    u32::from_be_bytes([p[at], p[at + 1], p[at + 2], p[at + 3]])
}

#[inline(always)]
fn be64(p: &[u8], at: usize) -> u64 {
    (u64::from(be32(p, at)) << 32) | u64::from(be32(p, at + 4))
}

#[inline(always)]
fn order_fields<'a>(p: &'a [u8], m: &mut Message<'a>) -> Result<(), DecodeError> {
    m.reference = be64(p, 11);
    m.side = p[19];
    m.shares = be32(p, 20);
    m.stock = p[24..32].try_into().ok();
    m.price = be32(p, 32);
    if m.side == b'B' || m.side == b'S' {
        Ok(())
    } else {
        Err(DecodeError::BadSide)
    }
}

/// Decode exactly one message: `p` has no frame prefix.  `Ok(None)` is a known
/// type this decoder does not unpack.
///
/// # Errors
/// See [`DecodeError`].
#[inline]
pub fn decode(p: &[u8]) -> Result<Option<Message<'_>>, DecodeError> {
    let Some(&kind) = p.first() else {
        return Err(DecodeError::Truncated);
    };
    let want = message_length(kind);
    if want == 0 {
        return Err(DecodeError::UnknownType);
    }
    if p.len() < want {
        return Err(DecodeError::Truncated);
    }
    if p.len() > want {
        return Err(DecodeError::BadLength);
    }
    let mut m = Message {
        kind,
        locate: be16(p, 1),
        tracking: be16(p, 3),
        timestamp_ns: (u64::from(be16(p, 5)) << 32) | u64::from(be32(p, 7)),
        ..Message::default()
    };
    match kind {
        b'A' => order_fields(p, &mut m)?,
        b'F' => {
            order_fields(p, &mut m)?;
            m.attribution = p[36..40].try_into().ok();
        }
        b'P' => {
            order_fields(p, &mut m)?;
            m.match_number = be64(p, 36);
        }
        b'E' | b'C' => {
            m.reference = be64(p, 11);
            m.shares = be32(p, 19);
            m.match_number = be64(p, 23);
            if kind == b'C' {
                m.printable = p[31];
                m.price = be32(p, 32);
            }
        }
        b'X' => {
            m.reference = be64(p, 11);
            m.shares = be32(p, 19);
        }
        b'D' => m.reference = be64(p, 11),
        b'U' => {
            m.reference = be64(p, 11);
            m.new_reference = be64(p, 19);
            m.shares = be32(p, 27);
            m.price = be32(p, 31);
        }
        _ => return Ok(None),
    }
    Ok(Some(m))
}

/// Walks a buffer of frames: a two-byte big-endian length, then the message.
/// Stops at the end or on a partial last frame; [`Frames::rest`] shows which.
#[derive(Debug, Clone)]
pub struct Frames<'a> {
    buf: &'a [u8],
}

impl<'a> Frames<'a> {
    /// Start at the first frame of `buf`.
    #[must_use]
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf }
    }

    /// Bytes not yet consumed.
    #[must_use]
    pub fn rest(&self) -> &'a [u8] {
        self.buf
    }
}

impl<'a> Iterator for Frames<'a> {
    type Item = &'a [u8];

    #[inline]
    fn next(&mut self) -> Option<&'a [u8]> {
        if self.buf.len() < 2 {
            return None;
        }
        let body = usize::from(be16(self.buf, 0));
        if self.buf.len() - 2 < body {
            return None;
        }
        let (msg, rest) = self.buf[2..].split_at(body);
        self.buf = rest;
        Some(msg)
    }
}

/// Mirror one decoded message into a book that holds one instrument.  The feed
/// is the authority on what rests, so nothing here matches: adds go through
/// [`FastBook::rest`], and a P message changes nothing because it reports a
/// hidden order.  Reference numbers become ids by bit cast.
///
/// # Errors
/// The book's refusals, which on a consistent feed mean the band or pool is
/// too small or a message was missed.
#[inline]
pub fn apply(book: &mut FastBook, m: &Message<'_>) -> Result<(), FastError> {
    let id = m.reference as i64;
    match m.kind {
        b'A' | b'F' => {
            let side = if m.side == b'B' { Side::Bid } else { Side::Ask };
            book.rest(side, i64::from(m.price), i64::from(m.shares), id)
                .map(|_| ())
        }
        b'E' | b'C' | b'X' => book.reduce(id, i64::from(m.shares)).map(|_| ()),
        b'D' => book.cancel(id).map(|_| ()),
        b'U' => book
            .replace(
                id,
                m.new_reference as i64,
                i64::from(m.price),
                i64::from(m.shares),
            )
            .map(|_| ()),
        _ => Ok(()),
    }
}
