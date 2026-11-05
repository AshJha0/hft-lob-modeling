//! Preallocated limit order book with the matching semantics of
//! [`crate::book::LimitOrderBook`], for a bounded price band.
//!
//! The same design as `cpp/fast/fast_book.hpp`: a flat ladder per side, a
//! hierarchical bitset to find the next live level, a node pool with an
//! intrusive doubly linked FIFO per level, cached totals, and an
//! open-addressing id map with backward-shift deletion.  All memory is taken
//! in [`FastBook::new`]; operations return a [`FastError`] rather than
//! allocating a message.  The code is safe Rust, so indexing stays
//! bounds-checked.
//!
//! Deliberate differences from the reference book:
//!
//! * a limit outside the band returns [`FastError::OutOfBand`] and changes
//!   nothing, even if it would have crossed; market orders are unaffected;
//! * with the pool full every limit returns [`FastError::PoolFull`] and changes
//!   nothing, including one that would have executed in full;
//! * prices are tick counts only, and there is no trade log: fills go to the
//!   caller's closure, called with `(maker_id, price_ticks, size)`.
//!
//! The id hash is not keyed, so ids from an untrusted source need remapping.

use crate::book::{Side, MAX_TICKS};

const NIL: u32 = u32::MAX;

/// Why an operation was refused.  A refused operation changes nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastError {
    /// Size at or below zero, or above `MAX_TICKS`.
    BadSize,
    /// Price below one tick or above `MAX_TICKS`.
    BadPrice,
    /// A valid price outside the configured band.
    OutOfBand,
    /// The id is already resting.
    DuplicateId,
    /// The side's total would pass `MAX_TICKS`.
    SideOverflow,
    /// No free node.
    PoolFull,
    /// Cancel of an id that is not resting.
    UnknownId,
}

/// Band and capacity, fixed at construction.
#[derive(Debug, Clone, Copy)]
pub struct FastConfig {
    /// Lowest price in the band (at least 1).
    pub min_price_ticks: i64,
    /// Ticks in the band, at most `2^24`.
    pub n_levels: i64,
    /// Live orders, at most `2^30`.
    pub max_orders: u32,
}

/// Up to four levels of 64-way summary words.
#[derive(Debug)]
struct LevelBits {
    words: Vec<u64>,
    off: [usize; 4],
    depth: usize,
}

impl LevelBits {
    fn new(n: usize) -> Self {
        let mut off = [0; 4];
        let (mut count, mut total, mut depth) = (n, 0, 0);
        loop {
            count = count.div_ceil(64);
            off[depth] = total;
            depth += 1;
            total += count;
            if count <= 1 {
                break;
            }
        }
        Self {
            words: vec![0; total],
            off,
            depth,
        }
    }

    fn set(&mut self, mut i: usize) {
        for h in 0..self.depth {
            let w = &mut self.words[self.off[h] + (i >> 6)];
            let was_empty = *w == 0;
            *w |= 1u64 << (i & 63);
            if !was_empty {
                return;
            }
            i >>= 6;
        }
    }

    fn clear(&mut self, mut i: usize) {
        for h in 0..self.depth {
            let w = &mut self.words[self.off[h] + (i >> 6)];
            *w &= !(1u64 << (i & 63));
            if *w != 0 {
                return;
            }
            i >>= 6;
        }
    }

    /// Highest set position strictly below `i`, or -1.
    fn prev(&self, mut i: usize) -> i64 {
        for h in 0..self.depth {
            let idx = i >> 6;
            let w = self.words[self.off[h] + idx] & ((1u64 << (i & 63)) - 1);
            if w != 0 {
                let mut pos = idx * 64 + 63 - w.leading_zeros() as usize;
                for g in (0..h).rev() {
                    pos = pos * 64 + 63 - self.words[self.off[g] + pos].leading_zeros() as usize;
                }
                return pos as i64;
            }
            i = idx;
        }
        -1
    }

    /// Lowest set position strictly above `i`, or -1.
    fn next(&self, mut i: usize) -> i64 {
        for h in 0..self.depth {
            let idx = i >> 6;
            let bit = i & 63;
            let mask = if bit == 63 { 0 } else { !0u64 << (bit + 1) };
            let w = self.words[self.off[h] + idx] & mask;
            if w != 0 {
                let mut pos = idx * 64 + w.trailing_zeros() as usize;
                for g in (0..h).rev() {
                    pos = pos * 64 + self.words[self.off[g] + pos].trailing_zeros() as usize;
                }
                return pos as i64;
            }
            i = idx;
        }
        -1
    }
}

/// id -> pool handle, stored plus one so that zero marks an empty slot and
/// every `i64` is usable as an id.
#[derive(Debug)]
struct IdMap {
    slots: Vec<(i64, u32)>,
    mask: usize,
    shift: u32,
}

impl IdMap {
    fn new(max_orders: u32) -> Self {
        let mut bits = 4u32;
        while (1u64 << bits) < 2 * u64::from(max_orders) {
            bits += 1;
        }
        Self {
            slots: vec![(0, 0); 1usize << bits],
            mask: (1usize << bits) - 1,
            shift: 64 - bits,
        }
    }

    #[inline]
    fn home(&self, id: i64) -> usize {
        let x = id as u64;
        ((x ^ (x >> 32)).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize
    }

    #[inline]
    fn find(&self, id: i64) -> Option<usize> {
        let mut i = self.home(id);
        while self.slots[i].1 != 0 {
            if self.slots[i].0 == id {
                return Some(i);
            }
            i = (i + 1) & self.mask;
        }
        None
    }

    /// The slot holding `id`, or the empty slot that ends its probe run.  One
    /// walk then serves both the duplicate check and the insert.
    #[inline]
    fn probe(&self, id: i64) -> usize {
        let mut i = self.home(id);
        while self.slots[i].1 != 0 && self.slots[i].0 != id {
            i = (i + 1) & self.mask;
        }
        i
    }

    #[inline]
    fn occupied(&self, at: usize) -> bool {
        self.slots[at].1 != 0
    }

    /// Backward-shift deletion: pull later entries of the probe run into the
    /// hole unless that would move one before its home slot.
    #[inline]
    fn erase_at(&mut self, mut hole: usize) {
        let mut j = hole;
        loop {
            j = (j + 1) & self.mask;
            if self.slots[j].1 == 0 {
                break;
            }
            let k = self.home(self.slots[j].0);
            if (j.wrapping_sub(k) & self.mask) >= (j.wrapping_sub(hole) & self.mask) {
                self.slots[hole] = self.slots[j];
                hole = j;
            }
        }
        self.slots[hole].1 = 0;
    }
}

#[derive(Debug, Clone, Copy)]
struct Node {
    id: i64,
    size: i64,
    prev: u32,
    next: u32,
    level: u32,
    side: Side,
}

#[derive(Debug, Clone, Copy)]
struct Level {
    total: i64,
    head: u32,
    tail: u32,
}

#[derive(Debug)]
struct SideState {
    levels: Vec<Level>,
    bits: LevelBits,
    /// Ladder index of the best level, or -1.
    best: i64,
    total: i64,
    orders: i64,
}

impl SideState {
    fn new(n: usize) -> Self {
        Self {
            levels: vec![
                Level {
                    total: 0,
                    head: NIL,
                    tail: NIL,
                };
                n
            ],
            bits: LevelBits::new(n),
            best: -1,
            total: 0,
            orders: 0,
        }
    }

    #[inline]
    fn level_emptied(&mut self, side: Side, level: u32) {
        self.bits.clear(level as usize);
        if self.best == i64::from(level) {
            self.best = match side {
                Side::Bid => self.bits.prev(level as usize),
                Side::Ask => self.bits.next(level as usize),
            };
        }
    }
}

#[inline]
fn slot(side: Side) -> usize {
    match side {
        Side::Bid => 0,
        Side::Ask => 1,
    }
}

/// See the module docs.
#[derive(Debug)]
pub struct FastBook {
    min_price: i64,
    n_levels: i64,
    sides: [SideState; 2],
    nodes: Vec<Node>,
    ids: IdMap,
    free: u32,
}

impl FastBook {
    /// Allocate a book.  Returns `None` on a bad band or pool size.
    pub fn new(cfg: FastConfig) -> Option<Self> {
        let band_ok = cfg.min_price_ticks >= 1
            && cfg.min_price_ticks <= MAX_TICKS
            && cfg.n_levels >= 1
            && cfg.n_levels <= 1 << 24
            && cfg.n_levels - 1 <= MAX_TICKS - cfg.min_price_ticks;
        if !band_ok || cfg.max_orders < 1 || cfg.max_orders > 1 << 30 {
            return None;
        }
        let n = cfg.n_levels as usize;
        let nodes = (0..cfg.max_orders)
            .map(|i| Node {
                id: 0,
                size: 0,
                prev: NIL,
                next: if i + 1 < cfg.max_orders { i + 1 } else { NIL },
                level: 0,
                side: Side::Bid,
            })
            .collect();
        Some(Self {
            min_price: cfg.min_price_ticks,
            n_levels: cfg.n_levels,
            sides: [SideState::new(n), SideState::new(n)],
            nodes,
            ids: IdMap::new(cfg.max_orders),
            free: 0,
        })
    }

    /// Submit a limit order.  Returns the rested size.
    ///
    /// # Errors
    /// See [`FastError`]; a refused order has no effect.
    #[inline]
    pub fn limit<F: FnMut(i64, i64, i64)>(
        &mut self,
        side: Side,
        price: i64,
        size: i64,
        id: i64,
        sink: &mut F,
    ) -> Result<i64, FastError> {
        if size <= 0 || size > MAX_TICKS {
            return Err(FastError::BadSize);
        }
        if !(1..=MAX_TICKS).contains(&price) {
            return Err(FastError::BadPrice);
        }
        if price < self.min_price || price - self.min_price >= self.n_levels {
            return Err(FastError::OutOfBand);
        }
        let at = self.ids.probe(id);
        if self.ids.occupied(at) {
            return Err(FastError::DuplicateId);
        }
        if size > MAX_TICKS - self.sides[slot(side)].total {
            return Err(FastError::SideOverflow);
        }
        if self.free == NIL {
            return Err(FastError::PoolFull);
        }
        let level = (price - self.min_price) as u32;
        if !self.crosses(side, price) {
            // No matching, so the map and the probed slot stay as they were.
            self.rest_order(side, level, size, id, at);
            return Ok(size);
        }
        let rest = self.execute(side, size, Some(price), sink);
        if rest > 0 {
            let at = self.ids.probe(id);
            self.rest_order(side, level, rest, id, at);
        }
        Ok(rest)
    }

    /// Add an order without matching, for mirroring a feed: the venue has
    /// already decided what rests, and a feed book can be locked or crossed
    /// before the open.  Returns the rested size.
    ///
    /// # Errors
    /// The same refusals as [`FastBook::limit`].
    #[inline]
    pub fn rest(&mut self, side: Side, price: i64, size: i64, id: i64) -> Result<i64, FastError> {
        if size <= 0 || size > MAX_TICKS {
            return Err(FastError::BadSize);
        }
        if !(1..=MAX_TICKS).contains(&price) {
            return Err(FastError::BadPrice);
        }
        if price < self.min_price || price - self.min_price >= self.n_levels {
            return Err(FastError::OutOfBand);
        }
        let at = self.ids.probe(id);
        if self.ids.occupied(at) {
            return Err(FastError::DuplicateId);
        }
        if size > MAX_TICKS - self.sides[slot(side)].total {
            return Err(FastError::SideOverflow);
        }
        if self.free == NIL {
            return Err(FastError::PoolFull);
        }
        self.rest_order(side, (price - self.min_price) as u32, size, id, at);
        Ok(size)
    }

    /// Move a resting order to a new id, price and size on the same side; it
    /// loses its queue place.  `new_id` may equal `id`.  Returns the rested
    /// size.
    ///
    /// # Errors
    /// The refusals of [`FastBook::rest`] and [`FastError::UnknownId`]; any
    /// refusal changes nothing.
    #[inline]
    pub fn replace(
        &mut self,
        id: i64,
        new_id: i64,
        price: i64,
        size: i64,
    ) -> Result<i64, FastError> {
        let at = self.ids.find(id).ok_or(FastError::UnknownId)?;
        if size <= 0 || size > MAX_TICKS {
            return Err(FastError::BadSize);
        }
        if !(1..=MAX_TICKS).contains(&price) {
            return Err(FastError::BadPrice);
        }
        if price < self.min_price || price - self.min_price >= self.n_levels {
            return Err(FastError::OutOfBand);
        }
        if new_id != id && self.ids.find(new_id).is_some() {
            return Err(FastError::DuplicateId);
        }
        let n = self.nodes[(self.ids.slots[at].1 - 1) as usize];
        if size > MAX_TICKS - (self.sides[slot(n.side)].total - n.size) {
            return Err(FastError::SideOverflow);
        }
        self.cancel(id)?;
        self.rest(n.side, price, size, new_id)
    }

    /// Take `by` off a resting order, keeping its queue place; reducing to
    /// zero removes it.  Returns the size left.
    ///
    /// # Errors
    /// [`FastError::UnknownId`], or [`FastError::BadSize`] if `by` is not in
    /// `1..=resting size`; the order is then unchanged.
    #[inline]
    pub fn reduce(&mut self, id: i64, by: i64) -> Result<i64, FastError> {
        let at = self.ids.find(id).ok_or(FastError::UnknownId)?;
        let h = (self.ids.slots[at].1 - 1) as usize;
        let n = self.nodes[h];
        if by <= 0 || by > n.size {
            return Err(FastError::BadSize);
        }
        if by == n.size {
            self.cancel(id)?;
            return Ok(0);
        }
        self.nodes[h].size -= by;
        let s = &mut self.sides[slot(n.side)];
        s.levels[n.level as usize].total -= by;
        s.total -= by;
        Ok(n.size - by)
    }

    /// Submit a market order.  Returns the unfilled size.
    ///
    /// # Errors
    /// [`FastError::BadSize`].
    #[inline]
    pub fn market<F: FnMut(i64, i64, i64)>(
        &mut self,
        side: Side,
        size: i64,
        sink: &mut F,
    ) -> Result<i64, FastError> {
        if size <= 0 || size > MAX_TICKS {
            return Err(FastError::BadSize);
        }
        Ok(self.execute(side, size, None, sink))
    }

    /// Cancel a resting order.  Returns the size removed.
    ///
    /// # Errors
    /// [`FastError::UnknownId`].
    #[inline]
    pub fn cancel(&mut self, id: i64) -> Result<i64, FastError> {
        let at = self.ids.find(id).ok_or(FastError::UnknownId)?;
        let h = self.ids.slots[at].1 - 1;
        self.ids.erase_at(at);
        let n = self.nodes[h as usize];
        if n.prev != NIL {
            self.nodes[n.prev as usize].next = n.next;
        }
        if n.next != NIL {
            self.nodes[n.next as usize].prev = n.prev;
        }
        let s = &mut self.sides[slot(n.side)];
        let lv = &mut s.levels[n.level as usize];
        if n.prev == NIL {
            lv.head = n.next;
        }
        if n.next == NIL {
            lv.tail = n.prev;
        }
        lv.total -= n.size;
        let emptied = lv.head == NIL;
        s.total -= n.size;
        s.orders -= 1;
        if emptied {
            s.level_emptied(n.side, n.level);
        }
        self.nodes[h as usize].next = self.free;
        self.free = h;
        Ok(n.size)
    }

    /// Highest bid in ticks, if any.
    pub fn best_bid(&self) -> Option<i64> {
        let b = self.sides[0].best;
        (b >= 0).then_some(self.min_price + b)
    }

    /// Lowest ask in ticks, if any.
    pub fn best_ask(&self) -> Option<i64> {
        let b = self.sides[1].best;
        (b >= 0).then_some(self.min_price + b)
    }

    fn best_size(&self, i: usize) -> i64 {
        let s = &self.sides[i];
        if s.best < 0 {
            0
        } else {
            s.levels[s.best as usize].total
        }
    }

    /// Size at the best bid, 0 if empty.
    pub fn best_bid_size(&self) -> i64 {
        self.best_size(0)
    }

    /// Size at the best ask, 0 if empty.
    pub fn best_ask_size(&self) -> i64 {
        self.best_size(1)
    }

    /// Mid in ticks, same pinned formula as the reference; NaN if a side is empty.
    pub fn mid(&self) -> f64 {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => 0.5 * (b as u64 + a as u64) as f64,
            _ => f64::NAN,
        }
    }

    /// Spread in ticks; NaN if a side is empty.
    pub fn spread(&self) -> f64 {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => (a - b) as f64,
            _ => f64::NAN,
        }
    }

    /// Size-weighted mid, same formula as the reference; NaN if a side is empty.
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

    /// Total size over the top `k` live levels.
    pub fn depth(&self, side: Side, k: usize) -> i64 {
        let s = &self.sides[slot(side)];
        let (mut total, mut at, mut taken) = (0, s.best, 0);
        while at >= 0 && taken < k {
            total += s.levels[at as usize].total;
            at = match side {
                Side::Bid => s.bits.prev(at as usize),
                Side::Ask => s.bits.next(at as usize),
            };
            taken += 1;
        }
        total
    }

    /// Total resting size on `side`.
    pub fn total_size(&self, side: Side) -> i64 {
        self.sides[slot(side)].total
    }

    /// Live orders on `side`.
    pub fn n_orders(&self, side: Side) -> i64 {
        self.sides[slot(side)].orders
    }

    /// True if `id` is resting.
    pub fn has_order(&self, id: i64) -> bool {
        self.ids.find(id).is_some()
    }

    #[inline]
    fn crosses(&self, side: Side, price: i64) -> bool {
        let (opp, buy) = match side {
            Side::Bid => (&self.sides[1], true),
            Side::Ask => (&self.sides[0], false),
        };
        if opp.best < 0 {
            return false;
        }
        let touch = self.min_price + opp.best;
        if buy {
            touch <= price
        } else {
            touch >= price
        }
    }

    #[inline]
    fn rest_order(&mut self, side: Side, level: u32, size: i64, id: i64, at: usize) {
        let h = self.free;
        self.free = self.nodes[h as usize].next;
        let s = &mut self.sides[slot(side)];
        let lv = &mut s.levels[level as usize];
        let tail = lv.tail;
        self.nodes[h as usize] = Node {
            id,
            size,
            prev: tail,
            next: NIL,
            level,
            side,
        };
        lv.tail = h;
        lv.total += size;
        if tail != NIL {
            self.nodes[tail as usize].next = h;
        } else {
            lv.head = h;
            s.bits.set(level as usize);
            let at = i64::from(level);
            let better = match side {
                Side::Bid => at > s.best,
                Side::Ask => at < s.best,
            };
            if s.best < 0 || better {
                s.best = at;
            }
        }
        s.total += size;
        s.orders += 1;
        self.ids.slots[at] = (id, h + 1);
    }

    #[inline]
    fn execute<F: FnMut(i64, i64, i64)>(
        &mut self,
        side: Side,
        size: i64,
        limit: Option<i64>,
        sink: &mut F,
    ) -> i64 {
        let opp_side = match side {
            Side::Bid => Side::Ask,
            Side::Ask => Side::Bid,
        };
        let opp = &mut self.sides[slot(opp_side)];
        let mut remaining = size;
        while remaining > 0 && opp.best >= 0 {
            let price = self.min_price + opp.best;
            if let Some(l) = limit {
                let beyond = match side {
                    Side::Bid => price > l,
                    Side::Ask => price < l,
                };
                if beyond {
                    break;
                }
            }
            let level = opp.best as u32;
            let lv = &mut opp.levels[level as usize];
            while remaining > 0 && lv.head != NIL {
                let h = lv.head;
                let n = &mut self.nodes[h as usize];
                let take = remaining.min(n.size);
                n.size -= take;
                remaining -= take;
                lv.total -= take;
                sink(n.id, price, take);
                if n.size == 0 {
                    let (id, next) = (n.id, n.next);
                    n.next = self.free;
                    self.free = h;
                    lv.head = next;
                    if next != NIL {
                        self.nodes[next as usize].prev = NIL;
                    } else {
                        lv.tail = NIL;
                    }
                    if let Some(at) = self.ids.find(id) {
                        self.ids.erase_at(at);
                    }
                    opp.orders -= 1;
                }
            }
            if lv.head == NIL {
                opp.level_emptied(opp_side, level);
            }
        }
        opp.total -= size - remaining;
        remaining
    }
}
