// libFuzzer harness: raw bytes as ITCH frames, through the decoder and into a
// small FastBook.  Refusals are expected; a sanitizer report or a broken
// invariant is a finding.
#include <cstddef>
#include <cstdint>

#include "itch_decode.hpp"

using lobmod::fast::Config;
using lobmod::fast::FastBook;
using lobmod::fast::Side;
using namespace lobmod::itch;

namespace {

void require(bool ok) {
    if (!ok) __builtin_trap();
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t len) {
    FastBook book(Config{1, 512, 64}); // small pool: PoolFull is reachable
    std::size_t at = 0, n = 0;
    const std::uint8_t* msg = nullptr;
    while (next_frame(data, len, &at, &msg, &n)) {
        Message m;
        const Decode d = decode(msg, n, m);
        if (d == Decode::Ok || d == Decode::Ignored) require(n == message_length(msg[0]));
        if (d != Decode::Ok) continue;
        require(m.type == msg[0]);
        if (m.stock != nullptr) require(m.stock >= msg && m.stock + 8 <= msg + n);
        if (m.attribution != nullptr) {
            require(m.attribution >= msg && m.attribution + 4 <= msg + n);
        }
        // Raw values almost never land in the band or hit a live id; fold them.
        m.ref &= 0x7F;
        m.new_ref &= 0x7F;
        m.price &= 0x3FF;
        m.shares &= 0xFF;
        apply(book, m);
        for (const Side s : {Side::Buy, Side::Sell}) {
            require(book.total_size(s) >= 0 && book.n_orders(s) >= 0);
            require(book.n_orders(s) <= 64);
            require((book.n_orders(s) == 0) == (book.total_size(s) == 0));
        }
    }
    return 0;
}
