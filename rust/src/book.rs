//! Limit order book with price-time (FIFO) priority on a discrete tick grid.
//!
//! * Prices live on a grid of integer multiples of `tick_size`; internally a
//!   price is stored as an integer number of ticks.
//! * Each price level holds a FIFO queue of resting orders `(order_id, size)`;
//!   earlier arrivals at a level fill first (time priority), better prices
//!   fill first (price priority).
//! * Three event types:
//!   - **limit add**: rests at its price; if it crosses the opposite best it
//!     first executes as a marketable order against all crossing levels, and
//!     only the unfilled remainder (if any) rests.
//!   - **cancel by id**: removes an order entirely; unknown id is an error.
//!   - **market order**: walks the opposite side level by level, filling FIFO
//!     within each level; if the book runs out, the unfilled remainder is
//!     *reported*, never an error.
//! * Derived quantities: best bid/ask, mid, spread, microprice
//!   `(q_a * P_b + q_b * P_a) / (q_a + q_b)` with best-level sizes `q_b, q_a`
//!   (size-weighted mid tilted toward the thin side), and summed depth over
//!   the top `k` price levels per side.
//!
//! All quantities that need both sides of the book return NaN when a side is
//! empty (guarded, never an error).
//!
//! Tick conversion (pinned, identical in every port): `ratio = price / tick`,
//! must be below `2^52`, rounded half up; accepted iff
//! `|ratio - ticks| <= min(1e-9 * max(1, ratio), 0.01)` and `ticks >= 1`.  A
//! bad price is an error — never a negative price and never an unbounded
//! market order.  [`LimitOrderBook::limit_ticks`] takes the integer tick
//! count directly, up to `MAX_TICKS` (9e18; the replay path).
//!
//! Size capacity (pinned): a limit add is rejected before any mutation if the
//! side's running total plus the submitted size exceeds `MAX_TICKS`, so every
//! size sum fits an `i64`.
//!
//! Memory: `trades` grows for the life of the book; construct with
//! [`LimitOrderBook::with_options`] (`record_trades = false`) or call
//! [`LimitOrderBook::clear_history`] for long sessions.

use std::collections::{BTreeMap, HashMap, VecDeque};

use crate::error::{LobError, Result};

/// Largest accepted tick count / size (fits a signed 64-bit integer).
pub const MAX_TICKS: i64 = 9_000_000_000_000_000_000;
/// At or above this ratio doubles have no fractional bits (2^52).
const MAX_GRID_RATIO: f64 = 4_503_599_627_370_496.0;
/// Absolute cap on the grid residual, in ticks.
const GRID_TOL_CAP: f64 = 0.01;

/// Book side: bids (`B`) or asks (`S`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    /// Buy side ('B').
    Bid,
    /// Sell side ('S').
    Ask,
}

impl Side {
    /// The single-character code used by the event CSV schema.
    pub fn code(self) -> char {
        match self {
            Side::Bid => 'B',
            Side::Ask => 'S',
        }
    }

    /// Parse the CSV code; errors on anything but `"B"` / `"S"`.
    pub fn from_code(code: &str) -> Result<Side> {
        match code {
            "B" => Ok(Side::Bid),
            "S" => Ok(Side::Ask),
            other => Err(LobError::InvalidInput(format!(
                "side must be 'B' or 'S', got {other:?}"
            ))),
        }
    }
}

/// One execution against a resting order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fill {
    /// Id of the resting (passive) order that was hit.
    pub maker_id: i64,
    /// Execution price in *ticks* (the resting order's price).
    pub price: i64,
    /// Executed quantity (> 0).
    pub size: i64,
}

/// Price-time priority limit order book (see module docs).
#[derive(Debug, Clone)]
pub struct LimitOrderBook {
    tick_size: f64,
    record_trades: bool,
    /// price level (ticks) -> FIFO queue of (order_id, remaining size)
    bids: BTreeMap<i64, VecDeque<(i64, i64)>>,
    asks: BTreeMap<i64, VecDeque<(i64, i64)>>,
    /// order_id -> (side, price ticks)
    orders: HashMap<i64, (Side, i64)>,
    /// Running total resting size per side (never above `MAX_TICKS`).
    bid_total: i64,
    ask_total: i64,
    /// Executed trades as [`Fill`] records, in execution order.
    pub trades: Vec<Fill>,
}

impl LimitOrderBook {
    /// Create an empty book.
    ///
    /// # Errors
    /// [`LobError::InvalidInput`] unless `tick_size` is positive and finite.
    pub fn new(tick_size: f64) -> Result<Self> {
        Self::with_options(tick_size, true)
    }

    /// Create an empty book, optionally without recording fills in
    /// [`LimitOrderBook::trades`] (matching is unaffected).
    ///
    /// # Errors
    /// [`LobError::InvalidInput`] unless `tick_size` is positive and finite.
    pub fn with_options(tick_size: f64, record_trades: bool) -> Result<Self> {
        if !(tick_size.is_finite() && tick_size > 0.0) {
            return Err(LobError::InvalidInput(format!(
                "tick_size must be positive and finite, got {tick_size}"
            )));
        }
        Ok(Self {
            tick_size,
            record_trades,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            orders: HashMap::new(),
            bid_total: 0,
            ask_total: 0,
            trades: Vec::new(),
        })
    }

    /// The book's tick size.
    pub fn tick_size(&self) -> f64 {
        self.tick_size
    }

    /// Whether fills are being recorded in `trades`.
    pub fn record_trades(&self) -> bool {
        self.record_trades
    }

    /// Drop the accumulated `trades` (book unchanged).
    pub fn clear_history(&mut self) {
        self.trades.clear();
    }

    /// Validate that `price` sits on the tick grid and convert it to ticks.
    ///
    /// Pinned rule: ratio below `2^52`, round half up, residual within
    /// `min(1e-9 * max(1, ratio), 0.01)`, result at least 1.
    fn to_ticks(&self, price: f64) -> Result<i64> {
        if !price.is_finite() || price <= 0.0 {
            return Err(LobError::InvalidInput(format!(
                "price must be positive and finite, got {price}"
            )));
        }
        let ratio = price / self.tick_size;
        if ratio >= MAX_GRID_RATIO {
            return Err(LobError::InvalidInput(format!(
                "price {price} is {ratio:e} ticks, too large to validate on the grid \
                 (limit 2^52); use limit_ticks for counts up to MAX_TICKS"
            )));
        }
        let mut ticks = ratio.floor();
        if ratio - ticks >= 0.5 {
            ticks += 1.0;
        }
        if (ratio - ticks).abs() > (1e-9 * ratio.max(1.0)).min(GRID_TOL_CAP) {
            return Err(LobError::InvalidInput(format!(
                "price {price} is not a multiple of tick_size {}",
                self.tick_size
            )));
        }
        if ticks < 1.0 {
            return Err(LobError::InvalidInput(format!(
                "price {price} rounds to 0 ticks; prices must be >= 1 tick"
            )));
        }
        Ok(ticks as i64) // in [1, 2^52]: exact conversion
    }

    fn check_ticks(ticks: i64) -> Result<i64> {
        if ticks < 1 {
            return Err(LobError::InvalidInput(format!(
                "price_ticks must be >= 1, got {ticks}"
            )));
        }
        if ticks > MAX_TICKS {
            return Err(LobError::InvalidInput(format!(
                "price_ticks {ticks} exceeds MAX_TICKS"
            )));
        }
        Ok(ticks)
    }

    fn check_size(size: i64) -> Result<i64> {
        if size <= 0 || size > MAX_TICKS {
            return Err(LobError::InvalidInput(format!(
                "size must be a positive integer <= 9e18, got {size}"
            )));
        }
        Ok(size)
    }

    // ------------------------------------------------------------- inspection

    /// Highest bid price in ticks, or `None` if the bid side is empty.
    pub fn best_bid(&self) -> Option<i64> {
        self.bids.keys().next_back().copied()
    }

    /// Lowest ask price in ticks, or `None` if the ask side is empty.
    pub fn best_ask(&self) -> Option<i64> {
        self.asks.keys().next().copied()
    }

    /// Total resting size at the best bid level (0 if the side is empty).
    pub fn best_bid_size(&self) -> i64 {
        self.best_bid()
            .map(|p| self.bids[&p].iter().map(|o| o.1).sum())
            .unwrap_or(0)
    }

    /// Total resting size at the best ask level (0 if the side is empty).
    pub fn best_ask_size(&self) -> i64 {
        self.best_ask()
            .map(|p| self.asks[&p].iter().map(|o| o.1).sum())
            .unwrap_or(0)
    }

    /// Mid price in ticks; NaN if either side is empty.
    ///
    /// Pinned: the exact integer `bid + ask` (at most 1.8e19, so it fits a
    /// `u64`) rounded once to the nearest double, then halved.
    pub fn mid(&self) -> f64 {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => 0.5 * (b as u64 + a as u64) as f64,
            _ => f64::NAN,
        }
    }

    /// Bid-ask spread in ticks; NaN if either side is empty.
    pub fn spread(&self) -> f64 {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => (a - b) as f64,
            _ => f64::NAN,
        }
    }

    /// Size-weighted mid: `(q_a * P_b + q_b * P_a) / (q_a + q_b)`.
    ///
    /// The microprice leans toward the side with less depth — the direction
    /// the next mid move is more likely to take.  NaN when either side is
    /// empty.
    pub fn microprice(&self) -> f64 {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => {
                let qb = self.best_bid_size() as f64;
                let qa = self.best_ask_size() as f64;
                (qa * b as f64 + qb * a as f64) / (qa + qb)
            }
            _ => f64::NAN,
        }
    }

    /// Total resting size over the top `k` price levels on `side`.
    ///
    /// Fewer than `k` levels: sums what exists.
    ///
    /// # Errors
    /// [`LobError::InvalidInput`] if `k == 0`.
    pub fn depth(&self, side: Side, k: usize) -> Result<i64> {
        if k == 0 {
            return Err(LobError::InvalidInput(
                "k must be a positive integer, got 0".to_string(),
            ));
        }
        let level_sum = |q: &VecDeque<(i64, i64)>| q.iter().map(|o| o.1).sum::<i64>();
        Ok(match side {
            Side::Bid => self.bids.values().rev().take(k).map(level_sum).sum(),
            Side::Ask => self.asks.values().take(k).map(level_sum).sum(),
        })
    }

    /// Sorted ids of live resting orders on `side` (pinned ascending ordering
    /// used by the zero-intelligence generator's cancel selection).
    pub fn order_ids(&self, side: Side) -> Vec<i64> {
        let mut ids: Vec<i64> = self
            .orders
            .iter()
            .filter(|(_, (s, _))| *s == side)
            .map(|(&id, _)| id)
            .collect();
        ids.sort_unstable();
        ids
    }

    /// True if `order_id` is currently resting in the book.
    pub fn has_order(&self, order_id: i64) -> bool {
        self.orders.contains_key(&order_id)
    }

    /// Total resting size on `side` across all levels (O(1)).
    pub fn total_size(&self, side: Side) -> i64 {
        match side {
            Side::Bid => self.bid_total,
            Side::Ask => self.ask_total,
        }
    }

    /// Number of live resting orders on `side`.
    pub fn n_orders(&self, side: Side) -> usize {
        self.orders.values().filter(|(s, _)| *s == side).count()
    }

    // -------------------------------------------------------------- execution

    /// Execute up to `size` against the side opposite to `side`.
    ///
    /// `limit_ticks` bounds acceptable prices (`None` = market order, walks
    /// everything).  Consumes levels best-first and orders FIFO within a
    /// level.  Returns (fills, unfilled remainder).
    fn execute(&mut self, side: Side, size: i64, limit_ticks: Option<i64>) -> (Vec<Fill>, i64) {
        let mut fills: Vec<Fill> = Vec::new();
        let mut remaining = size;
        while remaining > 0 {
            let best = match side {
                Side::Bid => self.asks.keys().next().copied(),
                Side::Ask => self.bids.keys().next_back().copied(),
            };
            let Some(best) = best else { break };
            if let Some(lim) = limit_ticks {
                // a crossed limit never trades through its own limit price
                let through = match side {
                    Side::Bid => best > lim,
                    Side::Ask => best < lim,
                };
                if through {
                    break;
                }
            }
            let opp = match side {
                Side::Bid => &mut self.asks,
                Side::Ask => &mut self.bids,
            };
            let queue = opp.get_mut(&best).expect("best level exists");
            while remaining > 0 && !queue.is_empty() {
                let front = queue.front_mut().expect("non-empty queue");
                let take = remaining.min(front.1);
                front.1 -= take;
                remaining -= take;
                fills.push(Fill {
                    maker_id: front.0,
                    price: best,
                    size: take,
                });
                if front.1 == 0 {
                    let (id, _) = queue.pop_front().expect("front exists");
                    self.orders.remove(&id);
                }
            }
            if queue.is_empty() {
                opp.remove(&best);
            }
        }
        match side {
            Side::Bid => self.ask_total -= size - remaining,
            Side::Ask => self.bid_total -= size - remaining,
        }
        if self.record_trades {
            self.trades.extend(fills.iter().copied());
        }
        (fills, remaining)
    }

    /// Submit a limit order.
    ///
    /// A crossed limit (buy priced at/above best ask, or sell at/below best
    /// bid) converts to a marketable execution against every crossing level;
    /// the unfilled remainder rests at `price` (appended to that level's
    /// queue tail).  Executions happen at the resting orders' prices.
    ///
    /// Returns `(fills, rested_size)` — `rested_size` is 0 if fully executed.
    ///
    /// # Errors
    /// [`LobError::InvalidInput`] on bad price/size, an off-grid price, a
    /// price at or above `2^52` ticks, a duplicate live `order_id`, or a size
    /// that would take the side's total above `MAX_TICKS`.
    pub fn limit(
        &mut self,
        side: Side,
        price: f64,
        size: i64,
        order_id: i64,
    ) -> Result<(Vec<Fill>, i64)> {
        Self::check_size(size)?;
        let ticks = self.to_ticks(price)?;
        self.limit_ticks(side, ticks, size, order_id)
    }

    /// Submit a limit order priced directly in ticks (no float conversion);
    /// identical semantics to [`LimitOrderBook::limit`].  `price_ticks` must
    /// lie in `[1, MAX_TICKS]`.
    ///
    /// # Errors
    /// As [`LimitOrderBook::limit`].
    pub fn limit_ticks(
        &mut self,
        side: Side,
        price_ticks: i64,
        size: i64,
        order_id: i64,
    ) -> Result<(Vec<Fill>, i64)> {
        Self::check_size(size)?;
        let ticks = Self::check_ticks(price_ticks)?;
        if self.orders.contains_key(&order_id) {
            return Err(LobError::InvalidInput(format!(
                "order_id {order_id} is already live in the book"
            )));
        }
        if size > MAX_TICKS - self.total_size(side) {
            return Err(LobError::InvalidInput(format!(
                "size {size} would take side {} total resting size above MAX_TICKS",
                side.code()
            )));
        }
        let (fills, remaining) = self.execute(side, size, Some(ticks));
        if remaining > 0 {
            let levels = match side {
                Side::Bid => &mut self.bids,
                Side::Ask => &mut self.asks,
            };
            levels
                .entry(ticks)
                .or_default()
                .push_back((order_id, remaining));
            self.orders.insert(order_id, (side, ticks));
            match side {
                Side::Bid => self.bid_total += remaining,
                Side::Ask => self.ask_total += remaining,
            }
        }
        Ok((fills, remaining))
    }

    /// Cancel a resting order by id; returns the size removed.
    ///
    /// # Errors
    /// [`LobError::UnknownOrder`] if the id is not live (never resting,
    /// already cancelled, or fully filled).
    pub fn cancel(&mut self, order_id: i64) -> Result<i64> {
        let Some((side, ticks)) = self.orders.remove(&order_id) else {
            return Err(LobError::UnknownOrder(order_id));
        };
        let levels = match side {
            Side::Bid => &mut self.bids,
            Side::Ask => &mut self.asks,
        };
        let queue = levels.get_mut(&ticks).expect("level of a live order");
        let mut removed = 0;
        if let Some(pos) = queue.iter().position(|o| o.0 == order_id) {
            removed = queue[pos].1;
            queue.remove(pos);
        }
        if queue.is_empty() {
            levels.remove(&ticks);
        }
        match side {
            Side::Bid => self.bid_total -= removed,
            Side::Ask => self.ask_total -= removed,
        }
        Ok(removed)
    }

    /// Submit a market order that walks the opposite side of the book.
    ///
    /// A market order larger than the total opposite depth fills what is
    /// available; the unfilled remainder is returned (reported), which is not
    /// an error.
    ///
    /// Returns `(fills, unfilled)`.
    ///
    /// # Errors
    /// [`LobError::InvalidInput`] on non-positive `size`.
    pub fn market(&mut self, side: Side, size: i64) -> Result<(Vec<Fill>, i64)> {
        Self::check_size(size)?;
        Ok(self.execute(side, size, None))
    }
}
