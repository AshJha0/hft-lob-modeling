//! FastBook and the ITCH decoder must not touch the heap after construction.
//!
//! This file is its own test binary so that the counting allocator applies
//! here only.  The counter is per thread and armed around the measured code,
//! so the test harness's own allocations on other threads do not count.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use lobmod::book::Side;
use lobmod::fast_book::{FastBook, FastConfig};
use lobmod::itch_decode::{apply, decode, Frames};

thread_local! {
    static ARMED: Cell<bool> = const { Cell::new(false) };
    static CALLS: Cell<u64> = const { Cell::new(0) };
}

struct Counting;

fn note() {
    // try_with: the allocator can run while thread locals are being torn down.
    let _ = ARMED.try_with(|armed| {
        if armed.get() {
            let _ = CALLS.try_with(|c| c.set(c.get() + 1));
        }
    });
}

// SAFETY: every call forwards to System with the same arguments.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note();
        System.alloc(layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note();
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note();
        System.realloc(ptr, layout, new_size)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        note();
        System.dealloc(ptr, layout);
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

fn allocations_in(f: impl FnOnce()) -> u64 {
    CALLS.with(|c| c.set(0));
    ARMED.with(|a| a.set(true));
    f();
    ARMED.with(|a| a.set(false));
    CALLS.with(Cell::get)
}

fn book() -> FastBook {
    FastBook::new(FastConfig {
        min_price_ticks: 1,
        n_levels: 4096,
        max_orders: 1024,
    })
    .unwrap()
}

#[test]
fn the_counter_counts() {
    let n = allocations_in(|| drop(std::hint::black_box(vec![1u8; 64])));
    assert!(n >= 2, "alloc and dealloc should both register, saw {n}");
}

#[test]
fn fast_book_operations_do_not_allocate() {
    let mut b = book();
    let mut filled = 0i64;
    let mut sink = |_: i64, _: i64, size: i64| filled += size;
    let n = allocations_in(|| {
        let mut x = 88_172_645_463_325_252u64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for i in 0..200_000i64 {
            let side = if next() & 1 == 0 {
                Side::Bid
            } else {
                Side::Ask
            };
            let price = 1 + (next() % 4096) as i64;
            let id = (next() % 2048) as i64;
            let size = 1 + (next() % 50) as i64;
            // Refusals are part of the path under test, so results are ignored.
            match i % 8 {
                0..=2 => {
                    let _ = b.limit(side, price, size, id, &mut sink);
                }
                3 => {
                    let _ = b.rest(side, price, size, id);
                }
                4 | 5 => {
                    let _ = b.cancel(id);
                }
                6 => {
                    let _ = b.reduce(id, size);
                }
                _ => {
                    let _ = b.market(side, size, &mut sink);
                    let _ = b.replace(id, id + 1, price, size);
                }
            }
            std::hint::black_box((b.best_bid(), b.best_ask(), b.depth(side, 5), b.mid()));
        }
    });
    assert_eq!(n, 0);
    assert!(filled > 0, "the script should have matched something");
}

#[test]
fn itch_decode_and_mirror_do_not_allocate() {
    let mut buf = Vec::new();
    let frame = |buf: &mut Vec<u8>, kind: u8, tail: &[u8]| {
        let len = 11 + tail.len();
        buf.extend_from_slice(&(len as u16).to_be_bytes());
        buf.push(kind);
        buf.extend_from_slice(&[0; 10]);
        buf.extend_from_slice(tail);
    };
    let mut order = Vec::new();
    order.extend_from_slice(&7u64.to_be_bytes());
    order.push(b'B');
    order.extend_from_slice(&100u32.to_be_bytes());
    order.extend_from_slice(b"AAPL    ");
    order.extend_from_slice(&50u32.to_be_bytes());
    frame(&mut buf, b'A', &order);
    let mut x = Vec::new();
    x.extend_from_slice(&7u64.to_be_bytes());
    x.extend_from_slice(&10u32.to_be_bytes());
    frame(&mut buf, b'X', &x);
    frame(&mut buf, b'D', &7u64.to_be_bytes());

    let mut b = book();
    let n = allocations_in(|| {
        for _ in 0..10_000 {
            for f in Frames::new(&buf) {
                let m = decode(f).unwrap().unwrap();
                apply(&mut b, &m).unwrap();
            }
        }
    });
    assert_eq!(n, 0);
    assert_eq!(b.n_orders(Side::Bid), 0);
}
