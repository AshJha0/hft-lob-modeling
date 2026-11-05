#ifndef LOBMOD_ITCH_DECODE_HPP
#define LOBMOD_ITCH_DECODE_HPP

/// \file itch_decode.hpp
/// \brief Binary Nasdaq TotalView-ITCH 5.0 decoder and its FastBook mirror.
///
/// decode() reads one message into a Message without allocating or copying
/// anything but scalars: `stock` and `attribution` point into the input, so
/// they live as long as the input buffer.  All integers on the wire are
/// big-endian.  Prices are four-decimal fixed point, so one tick here is
/// 0.0001 dollars.
///
/// Types A F E C X D U P are decoded.  The other known types are checked for
/// length, get their header filled in and return Decode::Ignored.  A type not
/// in the table returns Decode::UnknownType; the frame prefix still gives its
/// length, so a caller can step over it.
///
/// The same decoder is in rust/src/itch_decode.rs.  Nothing here is keyed by
/// stock locate: apply() mirrors one instrument, so the caller filters first.

#include <cstddef>
#include <cstdint>

#include "fast_book.hpp"

namespace lobmod {
namespace itch {

enum class Decode : std::uint8_t {
    Ok = 0,
    Ignored,     ///< a known type this decoder does not unpack
    Truncated,   ///< fewer bytes than the type needs
    BadLength,   ///< more bytes than the type has
    BadSide,     ///< a side byte other than 'B' or 'S'
    UnknownType, ///< not in the length table
};

/// Message length by type byte, frame prefix excluded; 0 if unknown.
constexpr std::size_t message_length(std::uint8_t type) noexcept {
    switch (type) {
    case 'A': return 36;
    case 'F': return 40;
    case 'E': return 31;
    case 'C': return 36;
    case 'X': return 23;
    case 'D': return 19;
    case 'U': return 35;
    case 'P': return 44;
    case 'Q': return 40;
    case 'R': return 39;
    case 'S': return 12;
    case 'H': return 25;
    case 'Y': return 20;
    case 'L': return 26;
    case 'I': return 50;
    case 'J': return 35;
    case 'K': return 28;
    case 'V': return 35;
    default: return 0;
    }
}

/// Fields a type does not carry are zero or null.
struct Message {
    std::uint8_t type = 0;
    std::uint8_t side = 0;      ///< 'B' or 'S': A F P
    std::uint8_t printable = 0; ///< 'Y' or 'N': C
    std::uint16_t locate = 0;
    std::uint16_t tracking = 0;
    std::uint32_t shares = 0; ///< A F P size; E C executed; X cancelled; U new size
    std::uint32_t price = 0;  ///< A F P U limit price; C execution price
    std::uint64_t timestamp_ns = 0; ///< since midnight
    std::uint64_t ref = 0;          ///< order reference; U: the original
    std::uint64_t new_ref = 0;      ///< U
    std::uint64_t match = 0;        ///< E C P
    const std::uint8_t* stock = nullptr;       ///< 8 bytes, space padded: A F P
    const std::uint8_t* attribution = nullptr; ///< 4 bytes: F
};

namespace detail {

inline std::uint16_t be16(const std::uint8_t* p) noexcept {
    return static_cast<std::uint16_t>((p[0] << 8) | p[1]);
}
inline std::uint32_t be32(const std::uint8_t* p) noexcept {
    return (std::uint32_t{p[0]} << 24) | (std::uint32_t{p[1]} << 16) |
           (std::uint32_t{p[2]} << 8) | std::uint32_t{p[3]};
}
inline std::uint64_t be48(const std::uint8_t* p) noexcept {
    return (std::uint64_t{be16(p)} << 32) | be32(p + 2);
}
inline std::uint64_t be64(const std::uint8_t* p) noexcept {
    return (std::uint64_t{be32(p)} << 32) | be32(p + 4);
}

} // namespace detail

/// Decode exactly one message: `n` is its length with no frame prefix.  On
/// anything but Ok or Ignored, `m` is left unspecified.
inline Decode decode(const std::uint8_t* p, std::size_t n, Message& m) noexcept {
    using namespace detail;
    if (n == 0) return Decode::Truncated;
    const std::size_t want = message_length(p[0]);
    if (want == 0) return Decode::UnknownType;
    if (n < want) return Decode::Truncated;
    if (n > want) return Decode::BadLength;
    m = Message{};
    m.type = p[0];
    m.locate = be16(p + 1);
    m.tracking = be16(p + 3);
    m.timestamp_ns = be48(p + 5);
    switch (m.type) {
    case 'F':
        m.attribution = p + 36;
        [[fallthrough]];
    case 'A':
        m.ref = be64(p + 11);
        m.side = p[19];
        m.shares = be32(p + 20);
        m.stock = p + 24;
        m.price = be32(p + 32);
        break;
    case 'P':
        m.ref = be64(p + 11);
        m.side = p[19];
        m.shares = be32(p + 20);
        m.stock = p + 24;
        m.price = be32(p + 32);
        m.match = be64(p + 36);
        break;
    case 'C':
        m.printable = p[31];
        m.price = be32(p + 32);
        [[fallthrough]];
    case 'E':
        m.ref = be64(p + 11);
        m.shares = be32(p + 19);
        m.match = be64(p + 23);
        return Decode::Ok;
    case 'X':
        m.ref = be64(p + 11);
        m.shares = be32(p + 19);
        return Decode::Ok;
    case 'D':
        m.ref = be64(p + 11);
        return Decode::Ok;
    case 'U':
        m.ref = be64(p + 11);
        m.new_ref = be64(p + 19);
        m.shares = be32(p + 27);
        m.price = be32(p + 31);
        return Decode::Ok;
    default:
        return Decode::Ignored;
    }
    return m.side == 'B' || m.side == 'S' ? Decode::Ok : Decode::BadSide;
}

/// Step over one frame: a two-byte big-endian length, then the message.
/// Returns false at the end of the buffer or on a partial last frame; `*at`
/// then stays on the frame that did not fit.
inline bool next_frame(const std::uint8_t* buf, std::size_t n, std::size_t* at,
                       const std::uint8_t** msg, std::size_t* len) noexcept {
    if (n - *at < 2) return false;
    const std::size_t body = detail::be16(buf + *at);
    if (n - *at - 2 < body) return false;
    *msg = buf + *at + 2;
    *len = body;
    *at += 2 + body;
    return true;
}

/// Mirror one decoded message into a book that holds one instrument.  The feed
/// is the authority on what rests, so nothing here matches: adds go through
/// FastBook::rest, and a P message changes nothing because it reports a hidden
/// order.  Reference numbers become ids by bit cast.  Non-Ok statuses are the
/// book's refusals, which on a consistent feed mean the band or pool is too
/// small or a message was missed.
inline fast::Status apply(fast::FastBook& book, const Message& m) noexcept {
    const auto id = static_cast<std::int64_t>(m.ref);
    switch (m.type) {
    case 'A':
    case 'F':
        return book
            .rest(m.side == 'B' ? fast::Side::Buy : fast::Side::Sell, m.price, m.shares, id)
            .status;
    case 'E':
    case 'C':
    case 'X':
        return book.reduce(id, m.shares).status;
    case 'D':
        return book.cancel(id).status;
    case 'U':
        return book.replace(id, static_cast<std::int64_t>(m.new_ref), m.price, m.shares).status;
    default:
        return fast::Status::Ok;
    }
}

} // namespace itch
} // namespace lobmod

#endif // LOBMOD_ITCH_DECODE_HPP
