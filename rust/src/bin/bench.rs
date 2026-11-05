//! Reference book against FastBook on one deep-book script; the same script
//! and method as `cpp/fast/bench.cpp`, described in `docs/LATENCY.md`.
//!
//! Usage: `cargo run --release --bin bench -- [ops] [live_orders] [repeats]`

use std::time::Instant;

use lobmod::book::{LimitOrderBook, Side};
use lobmod::fast_book::{FastBook, FastConfig};

#[cfg(target_arch = "x86_64")]
mod timer {
    use std::arch::x86_64::{__rdtscp, _mm_lfence, _rdtsc};

    #[inline(always)]
    pub fn begin() -> u64 {
        // SAFETY: fences and TSC reads have no memory effects.
        unsafe {
            _mm_lfence();
            let t = _rdtsc();
            _mm_lfence();
            t
        }
    }

    #[inline(always)]
    pub fn end() -> u64 {
        let mut aux = 0u32;
        // SAFETY: aux is a valid u32 to write; the fence has no memory effects.
        unsafe {
            let t = __rdtscp(&mut aux);
            _mm_lfence();
            t
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
mod timer {
    use std::sync::OnceLock;
    use std::time::Instant;

    static START: OnceLock<Instant> = OnceLock::new();

    pub fn begin() -> u64 {
        START.get_or_init(Instant::now).elapsed().as_nanos() as u64
    }

    pub fn end() -> u64 {
        begin()
    }
}

#[cfg(target_os = "linux")]
fn pin() -> bool {
    extern "C" {
        fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
    }
    let mask = [1u64 << 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    // SAFETY: the mask is a live 128-byte buffer, the size of a cpu_set_t.
    unsafe { sched_setaffinity(0, std::mem::size_of_val(&mask), mask.as_ptr()) == 0 }
}

#[cfg(not(target_os = "linux"))]
fn pin() -> bool {
    false
}

fn ticks_per_ns() -> f64 {
    let c0 = Instant::now();
    let t0 = timer::begin();
    while c0.elapsed().as_secs_f64() < 0.2 {}
    let t1 = timer::end();
    (t1 - t0) as f64 / (c0.elapsed().as_secs_f64() * 1e9)
}

fn timer_overhead() -> u64 {
    let mut d: Vec<u64> = (0..1_000_000)
        .map(|_| {
            let a = timer::begin();
            timer::end() - a
        })
        .collect();
    d.sort_unstable();
    d[d.len() / 2]
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % (hi - lo + 1) as u64) as i64
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

const ADD: usize = 0;
const CANCEL: usize = 1;
const CROSS: usize = 2;
const MARKET: usize = 3;
const OP_NAMES: [&str; 4] = ["add", "cancel", "marketable limit", "market"];

#[derive(Clone, Copy)]
struct Op {
    price: i64,
    size: i64,
    id: i64,
    kind: usize,
    side: Side,
}

const BAND_LEVELS: i64 = 1 << 17;
const MID: i64 = BAND_LEVELS / 2;

fn config(pool: u32) -> FastConfig {
    FastConfig {
        min_price_ticks: 1,
        n_levels: BAND_LEVELS,
        max_orders: pool,
    }
}

fn distance(rng: &mut Rng) -> i64 {
    let mut d = 0;
    while d < 500 && rng.unit() < 0.97 {
        d += 1;
    }
    d
}

struct Script {
    rng: Rng,
    shadow: FastBook,
    ops: Vec<Op>,
    ids: Vec<i64>,
    next_id: i64,
    mid: i64,
}

impl Script {
    fn add(&mut self, side: Side) {
        let d = 1 + distance(&mut self.rng);
        let px = if side == Side::Bid {
            self.mid - d
        } else {
            self.mid + d
        };
        let crosses = match side {
            Side::Bid => self.shadow.best_ask().is_some_and(|a| px >= a),
            Side::Ask => self.shadow.best_bid().is_some_and(|b| px <= b),
        };
        if crosses {
            return;
        }
        self.next_id += 1;
        let op = Op {
            price: px,
            size: self.rng.range(1, 100),
            id: self.next_id,
            kind: ADD,
            side,
        };
        self.shadow
            .limit(side, op.price, op.size, op.id, &mut |_, _, _| {})
            .unwrap();
        self.ops.push(op);
        self.ids.push(op.id);
    }

    fn pick_side(&mut self) -> Side {
        if self.rng.unit() < 0.5 {
            Side::Bid
        } else {
            Side::Ask
        }
    }
}

/// Runs the script through a shadow FastBook so every cancel hits a live order.
fn make_script(n_ops: usize, live: usize, pool: u32) -> (Vec<Op>, usize) {
    let mut s = Script {
        rng: Rng(42),
        shadow: FastBook::new(config(pool)).unwrap(),
        ops: Vec::with_capacity(n_ops + 2 * live),
        ids: Vec::new(),
        next_id: 0,
        mid: MID,
    };
    while s.ids.len() < live {
        let side = s.pick_side();
        s.add(side);
    }
    let prefill = s.ops.len();
    while s.ops.len() < prefill + n_ops {
        if s.ops.len().is_multiple_of(1000) {
            s.mid += if s.rng.unit() < 0.5 { -1 } else { 1 };
        }
        let side = s.pick_side();
        let resting = (s.shadow.n_orders(Side::Bid) + s.shadow.n_orders(Side::Ask)) as usize;
        let u = s.rng.unit();
        let p_add = if resting > live { 0.40 } else { 0.55 };
        if u < p_add {
            s.add(side);
        } else if u < 0.95 {
            while !s.ids.is_empty() {
                let at = (s.rng.next() % s.ids.len() as u64) as usize;
                let id = s.ids.swap_remove(at);
                if s.shadow.has_order(id) {
                    s.shadow.cancel(id).unwrap();
                    s.ops.push(Op {
                        price: 0,
                        size: 0,
                        id,
                        kind: CANCEL,
                        side,
                    });
                    break;
                }
            }
        } else if u < 0.98 {
            let touch = match side {
                Side::Bid => s.shadow.best_ask(),
                Side::Ask => s.shadow.best_bid(),
            };
            let Some(touch) = touch else { continue };
            let reach = s.rng.range(0, 2);
            let size = s.rng.range(1, 300);
            s.next_id += 1;
            let op = Op {
                price: if side == Side::Bid {
                    touch + reach
                } else {
                    touch - reach
                },
                size,
                id: s.next_id,
                kind: CROSS,
                side,
            };
            let rest = s
                .shadow
                .limit(side, op.price, op.size, op.id, &mut |_, _, _| {})
                .unwrap();
            s.ops.push(op);
            if rest > 0 {
                s.ids.push(op.id);
            }
        } else {
            let op = Op {
                price: 0,
                size: s.rng.range(1, 300),
                id: 0,
                kind: MARKET,
                side,
            };
            s.shadow.market(side, op.size, &mut |_, _, _| {}).unwrap();
            s.ops.push(op);
        }
    }
    (s.ops, prefill)
}

#[inline(always)]
fn mix(id: i64, px: i64, sz: i64) -> u64 {
    (id ^ px.wrapping_mul(31).wrapping_add(sz)) as u64
}

trait Adapter {
    const NAME: &'static str;
    fn make(pool: u32) -> Self;
    fn run(&mut self, op: &Op);
    fn checksum(&self) -> u64;
}

struct Reference {
    book: LimitOrderBook,
    sum: u64,
}

impl Adapter for Reference {
    const NAME: &'static str = "reference";
    fn make(_pool: u32) -> Self {
        Self {
            book: LimitOrderBook::with_options(1.0, false).unwrap(),
            sum: 0,
        }
    }
    #[inline]
    fn run(&mut self, op: &Op) {
        let fills = match op.kind {
            CANCEL => {
                self.book.cancel(op.id).unwrap();
                return;
            }
            MARKET => self.book.market(op.side, op.size).unwrap().0,
            _ => {
                let r = self.book.limit_ticks(op.side, op.price, op.size, op.id);
                r.unwrap().0
            }
        };
        for f in fills {
            self.sum = self.sum.wrapping_add(mix(f.maker_id, f.price, f.size));
        }
    }
    fn checksum(&self) -> u64 {
        self.sum
    }
}

struct Fast {
    book: FastBook,
    sum: u64,
}

impl Adapter for Fast {
    const NAME: &'static str = "FastBook";
    fn make(pool: u32) -> Self {
        Self {
            book: FastBook::new(config(pool)).unwrap(),
            sum: 0,
        }
    }
    #[inline]
    fn run(&mut self, op: &Op) {
        let sum = &mut self.sum;
        let mut sink = |id, px, sz| *sum = sum.wrapping_add(mix(id, px, sz));
        let done = match op.kind {
            CANCEL => self.book.cancel(op.id),
            MARKET => self.book.market(op.side, op.size, &mut sink),
            _ => self
                .book
                .limit(op.side, op.price, op.size, op.id, &mut sink),
        };
        done.unwrap();
    }
    fn checksum(&self) -> u64 {
        self.sum
    }
}

fn summarise(d: &mut [u64], overhead: u64, tpn: f64) -> [f64; 4] {
    d.sort_unstable();
    let at = |q: f64| {
        let v = d[(q * (d.len() - 1) as f64) as usize];
        v.saturating_sub(overhead) as f64 / tpn
    };
    [at(0.5), at(0.99), at(0.999), at(1.0)]
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

struct Run<'a> {
    ops: &'a [Op],
    prefill: usize,
    repeats: usize,
    overhead: u64,
    tpn: f64,
    pool: u32,
}

fn measure<A: Adapter>(r: &Run) -> u64 {
    let mut counts = [0usize; 4];
    for op in &r.ops[r.prefill..] {
        counts[op.kind] += 1;
    }
    let mut runs: [Vec<[f64; 4]>; 4] = Default::default();
    let mut throughput = Vec::new();
    let mut checksum = 0;
    for _ in 0..r.repeats {
        let mut lat: Vec<Vec<u64>> = counts.iter().map(|&c| vec![0u64; c]).collect();
        let mut fill = [0usize; 4];
        let mut a = A::make(r.pool);
        for op in &r.ops[..r.prefill] {
            a.run(op);
        }
        for op in &r.ops[r.prefill..] {
            let t0 = timer::begin();
            a.run(op);
            let t1 = timer::end();
            lat[op.kind][fill[op.kind]] = t1 - t0;
            fill[op.kind] += 1;
        }
        checksum = a.checksum();
        drop(a);
        for t in 0..4 {
            runs[t].push(summarise(&mut lat[t], r.overhead, r.tpn));
        }
        let mut a = A::make(r.pool);
        for op in &r.ops[..r.prefill] {
            a.run(op);
        }
        let c0 = Instant::now();
        for op in &r.ops[r.prefill..] {
            a.run(op);
        }
        let s = c0.elapsed().as_secs_f64();
        throughput.push((r.ops.len() - r.prefill) as f64 / s);
        assert_eq!(a.checksum(), checksum, "checksum drift in {}", A::NAME);
    }
    for t in 0..4 {
        let col = |i: usize| median(runs[t].iter().map(|s| s[i]).collect());
        let max = runs[t].iter().map(|s| s[3]).fold(0.0, f64::max);
        println!(
            "| {} | {} | {} | {:.0} | {:.0} | {:.0} | {:.0} |",
            A::NAME,
            OP_NAMES[t],
            counts[t],
            col(0),
            col(1),
            col(2),
            max
        );
    }
    println!(
        "{} untimed throughput: {:.2} M ops/s (median of {})",
        A::NAME,
        median(throughput) / 1e6,
        r.repeats
    );
    checksum
}

fn reverse_cancel<A: Adapter>(n: i64, pool: u32) -> f64 {
    let mut a = A::make(pool);
    let op = |id, kind, price, size| Op {
        price,
        size,
        id,
        kind,
        side: Side::Bid,
    };
    for i in 1..=n {
        a.run(&op(i, ADD, MID, 10));
    }
    let c0 = Instant::now();
    for i in (1..=n).rev() {
        a.run(&op(i, CANCEL, 0, 0));
    }
    c0.elapsed().as_secs_f64() * 1e9 / n as f64
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg =
        |i: usize, default: usize| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(default);
    let (n_ops, live, repeats) = (arg(1, 2_000_000), arg(2, 20_000), arg(3, 5));
    let mut pool = 1u32 << 17;
    while (pool as usize) < 4 * live {
        pool *= 2;
    }
    let pinned = pin();
    let tpn = ticks_per_ns();
    let overhead = timer_overhead();
    println!(
        "Rust pinned={} ticks/ns={:.3} timer_overhead={:.1} ns ops={} live={} repeats={} pool={}",
        u8::from(pinned),
        tpn,
        overhead as f64 / tpn,
        n_ops,
        live,
        repeats,
        pool
    );
    let (ops, prefill) = make_script(n_ops, live, pool);
    let run = Run {
        ops: &ops,
        prefill,
        repeats,
        overhead,
        tpn,
        pool,
    };
    println!("| book | op | count | p50 ns | p99 ns | p99.9 ns | max ns |\n|---|---|---|---|---|---|---|");
    let a = measure::<Reference>(&run);
    let b = measure::<Fast>(&run);
    println!(
        "fills checksum {}",
        if a == b { "identical" } else { "DIFFERENT" }
    );
    println!("| orders in one level | reference ns/cancel | FastBook ns/cancel |\n|---|---|---|");
    for n in [1000, 10_000, 100_000] {
        let r = reverse_cancel::<Reference>(n, pool);
        let f = reverse_cancel::<Fast>(n, pool);
        println!("| {n} | {r:.0} | {f:.0} |");
    }
    std::process::exit(i32::from(a != b));
}
