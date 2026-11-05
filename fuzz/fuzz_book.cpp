// libFuzzer harness: four bytes per operation, run through the reference book
// and FastBook together.  Any disagreement traps.
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "fast_book.hpp"

using lobmod::Fill;
using lobmod::LimitOrderBook;
using lobmod::fast::Config;
using lobmod::fast::FastBook;
using lobmod::fast::Side;
using lobmod::fast::Status;

namespace {

constexpr std::int64_t kLo = 100;
constexpr std::int64_t kLevels = 200;

void require(bool ok) {
    if (!ok) __builtin_trap();
}

bool same(const std::vector<Fill>& a, const std::vector<Fill>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].maker_id != b[i].maker_id || a[i].price != b[i].price ||
            a[i].size != b[i].size) {
            return false;
        }
    }
    return true;
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t len) {
    LimitOrderBook ref(1.0, false);
    FastBook fb(Config{kLo, kLevels, 64}); // small pool: PoolFull is reachable
    std::vector<Fill> got;
    auto sink = [&got](std::int64_t id, std::int64_t px, std::int64_t sz) {
        got.push_back(Fill{id, px, sz});
    };
    for (std::size_t i = 0; i + 4 <= len; i += 4) {
        const std::uint8_t kind = data[i] & 3;
        const bool buy = (data[i] & 4) != 0;
        const Side side = buy ? Side::Buy : Side::Sell;
        const char code = buy ? 'B' : 'S';
        const std::int64_t price = 60 + data[i + 1] + (data[i] >> 7) * 100; // 60..415
        const std::int64_t size = static_cast<std::int8_t>(data[i + 2]);
        const std::int64_t id = data[i + 3];
        got.clear();
        if (kind <= 1) {
            const auto res = fb.limit(side, price, size, id, sink);
            if (res.status == Status::OutOfBand || res.status == Status::PoolFull) {
                require(got.empty()); // band and pool are FastBook's own limits
            } else {
                try {
                    const auto want = ref.limit_ticks(code, price, size, id);
                    require(res.status == Status::Ok && res.value == want.second);
                    require(same(want.first, got));
                } catch (const std::invalid_argument&) {
                    require(res.status != Status::Ok);
                }
            }
        } else if (kind == 2) {
            const auto res = fb.cancel(id);
            try {
                const std::int64_t want = ref.cancel(id);
                require(res.status == Status::Ok && res.value == want);
            } catch (const std::invalid_argument&) {
                require(res.status == Status::UnknownId);
            }
        } else {
            const auto res = fb.market(side, size, sink);
            try {
                const auto want = ref.market(code, size);
                require(res.status == Status::Ok && res.value == want.second);
                require(same(want.first, got));
            } catch (const std::invalid_argument&) {
                require(res.status == Status::BadSize);
            }
        }
        require(ref.best_bid().value_or(0) == fb.best_bid());
        require(ref.best_ask().value_or(0) == fb.best_ask());
        require(ref.best_bid_size() == fb.best_bid_size());
        require(ref.best_ask_size() == fb.best_ask_size());
        require(ref.total_size('B') == fb.total_size(Side::Buy));
        require(ref.total_size('S') == fb.total_size(Side::Sell));
    }
    return 0;
}
