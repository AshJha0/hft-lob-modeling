#ifndef LOBMOD_ZI_HPP
#define LOBMOD_ZI_HPP

/// \file zi.hpp
/// \brief Zero-intelligence order-flow generator (pinned, seeded).
///
/// Smith-Farmer style flow: Poisson event-type rates around the best quotes
/// with geometric placement of passive limit orders.  Everything is pinned —
/// RNG (Lcg64), draw order, placement and size formulas — so that every
/// language regenerates the identical bundled 5000-event stream.
///
/// Seed phase (events 1..12): for i = 0..5 submit a buy limit at 999 - i
/// size 10 then a sell limit at 1001 + i size 10 (ids = seq numbers).
///
/// Stochastic phase: with nb, ns the live order counts per side, the rates
/// are [limit_buy 1, limit_sell 1, cancel_buy 0.04*nb, cancel_sell 0.04*ns,
/// market_buy 0.25, market_sell 0.25] in that exact order; one uniform picks
/// the type by cumulative rate.  Limits are aggressive (priced at the
/// opposite best) with probability 0.12 when that best exists, otherwise
/// placed geometric(0.35) + 1 ticks behind the opposite best (anchor 1000
/// when the opposite side is empty), clamped to >= 1; size 1 + randint(5).
/// Cancels pick uniformly from the side's ascending live ids.  Market orders
/// are an occasional sweep 40 + randint(40) with probability 0.02, else
/// 1 + randint(3).

#include <cstdint>
#include <vector>

#include "lobmod/replay.hpp"

namespace lobmod {

/// Generate a deterministic zero-intelligence event stream of n_events
/// events (seq numbered from 1) from the pinned Lcg64(seed).
/// \throws std::invalid_argument if n_events < 12 or the seed is negative.
std::vector<Event> generate_events(std::int64_t n_events, std::int64_t seed);

} // namespace lobmod

#endif // LOBMOD_ZI_HPP
