# COOKBOOK — `lobmod` recipes

Task-oriented recipes. Recipes 1-13 cover all four languages. Recipes 14-19
are Python only because those layers are, and recipe 20 is C++ and Rust. The
snippets are extracted and executed as part of the review checklist, and
"Verifying the snippets" at the end says what was run and how. Paths assume the
bundled `../data/` files relative to the language directory.

Conventions, so the snippets stay short:

- **Python** — run from `python/`: `PYTHONPATH=src python3 snippet.py`. Each
  snippet is complete.
- **C++** — each block is the body of `int main()` with this prelude, built
  from `cpp/` after `bash build.sh` with
  `g++ -std=c++17 -I include snippet.cpp build/liblobmod.a`:

  ```cpp
  #include <lobmod/lobmod.hpp>   // umbrella header: every public header
  #include <cmath>
  #include <iostream>
  #include <stdexcept>
  #include <vector>
  int main() { /* snippet */ return 0; }
  ```

- **Rust** — each block is the body of
  `fn main() -> Result<(), lobmod::LobError> { /* snippet */ Ok(()) }` in a
  crate that depends on `lobmod = { path = "rust" }`, run from `rust/`.
  Fallible calls use `?`; nothing panics on bad input.
- **Java** — each block is the body of
  `public static void main(String[] args)` in a class that starts with
  `import com.quant.lobmod.*; import java.nio.file.Path; import java.util.List;`,
  compiled with `javac -cp out/main` and run from `java/` after `bash build.sh`.

All book prices returned by the library are in **ticks**; `Event.price` in the
CSV schema is a tick count; every integer (ticks, sizes, ids) is 64-bit.

---

## 1. How do I build a book by hand and read the top of book?

**Python**

```python
from lobmod import LimitOrderBook

book = LimitOrderBook(tick_size=1.0)
book.limit("B", 99.0, 5, order_id=1)    # bid 99 x 5
book.limit("B", 98.0, 8, order_id=2)
book.limit("S", 101.0, 3, order_id=3)   # ask 101 x 3
print(book.best_bid(), book.best_ask())  # 99 101  (in ticks)
print(book.mid(), book.spread())         # 100.0 2.0
print(book.microprice())                 # (3*99 + 5*101) / 8 = 100.25
print(book.depth("B", 2))                # 13
# prices already in ticks skip the float conversion entirely:
book.limit_ticks("S", 102, 4, order_id=4)
print(book.depth("S", 2))                # 7
```

**C++**

```cpp
lobmod::LimitOrderBook book(1.0);
book.limit('B', 99.0, 5, 1);
book.limit('B', 98.0, 8, 2);
book.limit('S', 101.0, 3, 3);
std::cout << *book.best_bid() << " " << *book.best_ask() << "\n";   // 99 101
std::cout << book.mid() << " " << book.spread() << " "
          << book.microprice() << " " << book.depth('B', 2) << "\n"; // 100 2 100.25 13
book.limit_ticks('S', 102, 4, 4);         // integer ticks, no float conversion
std::cout << book.depth('S', 2) << "\n";  // 7
```

**Rust**

```rust
use lobmod::{LimitOrderBook, Side};

let mut book = LimitOrderBook::new(1.0)?;
book.limit(Side::Bid, 99.0, 5, 1)?;
book.limit(Side::Bid, 98.0, 8, 2)?;
book.limit(Side::Ask, 101.0, 3, 3)?;
println!("{:?} {:?}", book.best_bid(), book.best_ask()); // Some(99) Some(101)
println!("{} {} {}", book.mid(), book.spread(), book.microprice());
println!("{}", book.depth(Side::Bid, 2)?);               // 13
book.limit_ticks(Side::Ask, 102, 4, 4)?;                 // integer ticks
println!("{}", book.depth(Side::Ask, 2)?);               // 7
```

**Java**

```java
LimitOrderBook book = new LimitOrderBook(1.0);
book.limit('B', 99.0, 5, 1);
book.limit('B', 98.0, 8, 2);
book.limit('S', 101.0, 3, 3);
System.out.println(book.bestBid().getAsLong() + " " + book.bestAsk().getAsLong()); // 99 101
System.out.printf("%f %f %f %d%n",
    book.mid(), book.spread(), book.microprice(), book.depth('B', 2));
book.limitTicks('S', 102, 4, 4);          // integer ticks, no float conversion
System.out.println(book.depth('S', 2));   // 7
```

---

## 2. How do I replay the bundled event stream and take snapshots?

`Event.price` is a tick count, so `replay` takes no tick size — only the
number of depth levels to record.

**Python**

```python
from lobmod import read_events_csv, replay

events = read_events_csv("../data/lob_events.csv")
states = replay(events, depth_k=3)
i = 1000 - 1                       # golden snapshot after 1-based event 1000
print(states.best_bid[i], states.best_ask[i])   # 999.0 1000.0
print(states.mid[i], states.microprice[i])      # 999.5 999.33...
print(states.bid_depth_k[i], states.ask_depth_k[i])  # 21.0 27.0
```

**C++**

```cpp
auto events = lobmod::read_events_csv("../data/lob_events.csv");
auto states = lobmod::replay(events, /*depth_k=*/3);
std::size_t i = 1000 - 1;
std::cout << states.best_bid[i] << " " << states.best_ask[i] << " "
          << states.mid[i] << "\n";                      // 999 1000 999.5
```

**Rust**

```rust
use lobmod::{read_events_csv, replay};

let events = read_events_csv("../data/lob_events.csv")?;
let states = replay(&events, 3)?;
let i = 1000 - 1;
println!("{} {} {}", states.best_bid[i], states.best_ask[i], states.mid[i]);
```

**Java**

```java
List<Event> events = Replay.readEventsCsv(Path.of("../data/lob_events.csv"));
BookStates states = Replay.replay(events, 3);
int i = 1000 - 1;
System.out.printf("%f %f %f%n",
    states.bestBid()[i], states.bestAsk()[i], states.mid()[i]);
```

---

## 3. How do I handle the classic edge cases (sweep, bad cancel, crossed limit, bad price)?

**Python**

```python
import math
from lobmod import LimitOrderBook

book = LimitOrderBook()
book.limit("S", 101.0, 5, 1)
book.limit("S", 102.0, 5, 2)

# market order bigger than the whole book: reported, never an error
fills, unfilled = book.market("B", 25)
print(len(fills), unfilled)          # 2 15  (10 filled, 15 reported unfilled)
print(math.isnan(book.mid()))        # True — ask side now empty, NaN guarded

# cancel of an unknown id: error
try:
    book.cancel(99)
except ValueError as exc:
    print(exc)                       # cancel of unknown order_id 99

# crossed limit executes first, remainder rests at its own price
book.limit("B", 100.0, 4, 3)
fills, rested = book.limit("S", 99.0, 10, 4)  # crosses the 100-bid
print(fills[0].price, rested)        # 100 6 — executed AT the bid, 6 rests at 99

# a price that rounds to 0 ticks, or exceeds 9e18 ticks, is an error —
# it never becomes a tick-0 order or an unbounded market order
for bad in (1e-12, 1e300):
    try:
        book.limit("B", bad, 1, 5)
    except ValueError as exc:
        print("rejected:", str(exc)[:40])
```

**C++** — same flow; every rejection throws `std::invalid_argument`:

```cpp
lobmod::LimitOrderBook book(1.0);
book.limit('S', 101.0, 5, 1);
book.limit('S', 102.0, 5, 2);
auto [fills, unfilled] = book.market('B', 25);     // unfilled reported, not thrown
std::cout << fills.size() << " " << unfilled << " " << std::isnan(book.mid()) << "\n"; // 2 15 1
try { book.cancel(99); } catch (const std::invalid_argument& e) { std::cout << e.what() << "\n"; }
book.limit('B', 100.0, 4, 3);
auto [f2, rested] = book.limit('S', 99.0, 10, 4);  // marketable limit
std::cout << f2[0].price << " " << rested << "\n"; // 100 6
for (double bad : {1e-12, 1e300}) {
    try { book.limit('B', bad, 1, 5); }
    catch (const std::invalid_argument& e) { std::cout << "rejected: " << e.what() << "\n"; }
}
```

**Rust** — errors come back as `Result`:

```rust
use lobmod::{LimitOrderBook, Side};

let mut book = LimitOrderBook::new(1.0)?;
book.limit(Side::Ask, 101.0, 5, 1)?;
book.limit(Side::Ask, 102.0, 5, 2)?;
let (fills, unfilled) = book.market(Side::Bid, 25)?; // Ok: unfilled reported
println!("{} {} {}", fills.len(), unfilled, book.mid().is_nan()); // 2 15 true
assert!(book.cancel(99).is_err());                   // unknown id -> LobError
book.limit(Side::Bid, 100.0, 4, 3)?;
let (f2, rested) = book.limit(Side::Ask, 99.0, 10, 4)?;
println!("{} {}", f2[0].price, rested);              // 100 6
assert!(book.limit(Side::Bid, 1e-12, 1, 5).is_err()); // rounds to 0 ticks
assert!(book.limit(Side::Bid, 1e300, 1, 5).is_err()); // above 9e18 ticks
```

**Java** — every rejection throws `IllegalArgumentException`:

```java
LimitOrderBook book = new LimitOrderBook(1.0);
book.limit('S', 101.0, 5, 1);
book.limit('S', 102.0, 5, 2);
LimitOrderBook.ExecResult r = book.market('B', 25);   // r.remainder() reported
System.out.println(r.fills().size() + " " + r.remainder() + " " + Double.isNaN(book.mid()));
try { book.cancel(99); } catch (IllegalArgumentException e) { System.out.println(e.getMessage()); }
book.limit('B', 100.0, 4, 3);
LimitOrderBook.ExecResult r2 = book.limit('S', 99.0, 10, 4);
System.out.println(r2.fills().get(0).price() + " " + r2.remainder()); // 100 6
for (double bad : new double[] {1e-12, 1e300}) {
    try { book.limit('B', bad, 1, 5); }
    catch (IllegalArgumentException e) { System.out.println("rejected: " + e.getMessage()); }
}
```

---

## 4. How do I generate a synthetic zero-intelligence stream and save it?

Identical seeds produce identical streams in every language (`data/lob_events.csv`
is `generate_events(5000, 20240613)`), and every port's writer reproduces the
pinned CSV schema.

**Python**

```python
from lobmod import generate_events, read_events_csv, replay, write_events_csv

events = generate_events(n_events=5000, seed=20240613)
write_events_csv("/tmp/my_stream.csv", events)
assert read_events_csv("/tmp/my_stream.csv") == events   # exact round trip
states = replay(events)
print(states.mid[-1])                # 1003.5 — matches the bundled stream
```

**C++**

```cpp
auto events = lobmod::generate_events(5000, 20240613);
lobmod::write_events_csv("/tmp/my_stream_cpp.csv", events);
auto back = lobmod::read_events_csv("/tmp/my_stream_cpp.csv");
std::cout << back.size() << " " << lobmod::replay(events).mid.back() << "\n"; // 5000 1003.5
```

**Rust**

```rust
let events = lobmod::generate_events(5000, 20240613)?;
lobmod::write_events_csv("/tmp/my_stream_rs.csv", &events)?;
let back = lobmod::read_events_csv("/tmp/my_stream_rs.csv")?;
assert_eq!(back, events);
println!("{}", lobmod::replay(&events, 3)?.mid[4999]); // 1003.5
```

**Java**

```java
List<Event> events = ZiGenerator.generateEvents(5000, 20240613L);
Replay.writeEventsCsv(Path.of("/tmp/my_stream_java.csv"), events);
List<Event> back = Replay.readEventsCsv(Path.of("/tmp/my_stream_java.csv"));
System.out.println(back.equals(events) + " " + Replay.replay(events).mid()[4999]); // true 1003.5
```

---

## 5. How do I evaluate a Hawkes log-likelihood at given parameters?

The `t` / `t,type` CSV readers apply the same strict rules as the event
reader (CRLF tolerated, only trailing blank lines, finite values).

**Python**

```python
from lobmod import read_times_csv, uni_loglik, uni_loglik_brute

t = read_times_csv("../data/hawkes_uni.csv")
ll = uni_loglik(t, T=1000.0, mu=0.4, alpha=0.6, beta=1.2)
print(ll)                                # -738.033935432713 (golden, 1e-8)
# brute-force O(N^2) cross-check (tests use 1e-10):
print(abs(ll - uni_loglik_brute(t, 1000.0, 0.4, 0.6, 1.2)) < 1e-10)
```

**C++**

```cpp
std::vector<double> t = lobmod::read_times_csv("../data/hawkes_uni.csv");
double ll = lobmod::uni_loglik(t, 1000.0, 0.4, 0.6, 1.2);
std::cout.precision(15);
std::cout << ll << "\n";                 // -738.033935432713
```

**Rust**

```rust
let t = lobmod::read_times_csv("../data/hawkes_uni.csv")?;
let ll = lobmod::uni_loglik(&t, 1000.0, 0.4, 0.6, 1.2)?;
println!("{ll:.12}");                    // -738.033935432713
```

**Java**

```java
double[] t = Hawkes.readTimesCsv(Path.of("../data/hawkes_uni.csv"));
double ll = Hawkes.uniLoglik(t, 1000.0, 0.4, 0.6, 1.2);
System.out.printf("%.12f%n", ll);        // -738.033935432713
```

---

## 6. How do I fit a univariate Hawkes model and read the branching ratio?

The fit warns (never throws) at branching ≥ 0.9 / ≥ 1. `converged` is the
Nelder-Mead tolerance flag: `False` means the iteration cap (`maxiter`,
default 5000) stopped the search and the parameters are the best vertex found.

**Python**

```python
from lobmod import fit_uni, read_times_csv

t = read_times_csv("../data/hawkes_uni.csv")
fit = fit_uni(t, T=1000.0)
print(fit.mu, fit.alpha, fit.beta)   # 0.5052 0.9795 1.6792 (golden, 1e-4)
print(fit.branching)                 # 0.5833235621941935 (golden, 1e-6)
print(fit.loglik, fit.converged, fit.n_iter)
capped = fit_uni(t, T=1000.0, maxiter=1)
print(capped.converged)              # False — honest: the cap stopped it
```

**C++**

```cpp
std::vector<double> t = lobmod::read_times_csv("../data/hawkes_uni.csv");
lobmod::HawkesUniFit fit = lobmod::fit_uni(t, 1000.0);
std::cout << fit.mu << " " << fit.alpha << " " << fit.beta
          << " branching=" << fit.branching << " converged=" << fit.converged
          << " warning='" << fit.warning << "'\n";
```

**Rust**

```rust
let t = lobmod::read_times_csv("../data/hawkes_uni.csv")?;
let fit = lobmod::fit_uni(&t, 1000.0)?;
println!("{} {} {} branching={} converged={} warning={:?}",
         fit.mu, fit.alpha, fit.beta, fit.branching, fit.converged, fit.warning);
let capped = lobmod::fit_uni_with(&t, 1000.0, 1)?;
println!("{}", capped.converged);        // false
```

**Java**

```java
double[] t = Hawkes.readTimesCsv(Path.of("../data/hawkes_uni.csv"));
Hawkes.UniFit fit = Hawkes.fitUni(t, 1000.0);
System.out.printf("%f %f %f branching=%f converged=%b warning=%s%n",
    fit.mu(), fit.alpha(), fit.beta(), fit.branching(), fit.converged(), fit.warning());
System.out.println(Hawkes.fitUni(t, 1000.0, 1).converged());   // false
```

---

## 7. How do I simulate a Hawkes process (Ogata thinning)?

Same seed ⇒ identical event times in all four languages (pinned two uniforms
per candidate). Simulation refuses non-stationary parameters (branching ≥ 1)
and errors if it ever exceeds 5·10⁷ candidate draws.

**Python**

```python
from lobmod import simulate_uni, simulate_biv, write_times_csv, write_marked_times_csv

t = simulate_uni(mu=0.5, alpha=1.0, beta=1.6, T=1000.0, seed=101)
print(t.size)                        # 1211 — regenerates data/hawkes_uni.csv
tb, kb = simulate_biv(mu=(0.3, 0.3), alpha=[[0.6, 0.3], [0.25, 0.5]],
                      beta=2.0, T=600.0, seed=33)
print(tb.size)                       # 611 — regenerates data/hawkes_biv.csv
write_times_csv("/tmp/uni.csv", t)             # header t, %.17g
write_marked_times_csv("/tmp/biv.csv", tb, kb) # header t,type
```

**C++**

```cpp
std::vector<double> t = lobmod::simulate_uni(0.5, 1.0, 1.6, 1000.0, 101);
auto [tb, kb] = lobmod::simulate_biv({0.3, 0.3}, {{{0.6, 0.3}, {0.25, 0.5}}},
                                     2.0, 600.0, 33);
std::cout << t.size() << " " << tb.size() << "\n";   // 1211 611
lobmod::write_times_csv("/tmp/uni_cpp.csv", t);
lobmod::write_marked_times_csv("/tmp/biv_cpp.csv", tb, kb);
```

**Rust**

```rust
let t = lobmod::simulate_uni(0.5, 1.0, 1.6, 1000.0, 101)?;
let (tb, kb) = lobmod::simulate_biv(&[0.3, 0.3], &[[0.6, 0.3], [0.25, 0.5]],
                                    2.0, 600.0, 33)?;
println!("{} {}", t.len(), tb.len());                // 1211 611
lobmod::write_times_csv("/tmp/uni_rs.csv", &t)?;
lobmod::write_marked_times_csv("/tmp/biv_rs.csv", &tb, &kb)?;
```

**Java**

```java
double[] t = Hawkes.simulateUni(0.5, 1.0, 1.6, 1000.0, 101L);
Hawkes.BivSample s = Hawkes.simulateBiv(new double[]{0.3, 0.3},
    new double[][]{{0.6, 0.3}, {0.25, 0.5}}, 2.0, 600.0, 33L);
System.out.println(t.length + " " + s.times().length);   // 1211 611
Hawkes.writeTimesCsv(Path.of("/tmp/uni_java.csv"), t);
Hawkes.writeMarkedTimesCsv(Path.of("/tmp/biv_java.csv"), s.times(), s.types());
```

---

## 8. How do I check goodness of fit (time-rescaling + KS)?

**Python**

```python
from lobmod import fit_uni, ks_test_exp, read_times_csv, uni_residuals

t = read_times_csv("../data/hawkes_uni.csv")
fit = fit_uni(t, T=1000.0)
tau = uni_residuals(t, 1000.0, fit.mu, fit.alpha, fit.beta)
print(tau.mean(), (tau > 0).all())   # ~1.0 (Exp(1) mean), True
d, p = ks_test_exp(tau)
print(d, p)                          # 0.017205 0.863177 (golden, 1e-6): no rejection (weak test, fitted params)
```

**C++**

```cpp
std::vector<double> t = lobmod::read_times_csv("../data/hawkes_uni.csv");
lobmod::HawkesUniFit fit = lobmod::fit_uni(t, 1000.0);
auto tau = lobmod::uni_residuals(t, 1000.0, fit.mu, fit.alpha, fit.beta);
auto [d, p] = lobmod::ks_test_exp(tau);
std::cout << d << " " << p << "\n";  // 0.0172 0.8632
```

**Rust**

```rust
let t = lobmod::read_times_csv("../data/hawkes_uni.csv")?;
let fit = lobmod::fit_uni(&t, 1000.0)?;
let tau = lobmod::uni_residuals(&t, 1000.0, fit.mu, fit.alpha, fit.beta)?;
let (d, p) = lobmod::ks_test_exp(&tau)?;
println!("{d:.6} {p:.6}");           // 0.017205 0.863177
```

**Java**

```java
double[] t = Hawkes.readTimesCsv(Path.of("../data/hawkes_uni.csv"));
Hawkes.UniFit fit = Hawkes.fitUni(t, 1000.0);
double[] tau = Hawkes.uniResiduals(t, 1000.0, fit.mu(), fit.alpha(), fit.beta());
Gof.KsResult ks = Gof.ksTestExp(tau);
System.out.printf("%.6f %.6f%n", ks.stat(), ks.pValue());   // 0.017205 0.863177
```

---

## 9. How do I fit the bivariate buy/sell model and interpret it?

**Python**

```python
from lobmod import fit_biv, read_marked_times_csv, spectral_radius_2x2

t, k = read_marked_times_csv("../data/hawkes_biv.csv")   # k: 0 = buy, 1 = sell
fit = fit_biv(t, k, T=600.0)
print(fit.alpha)          # [[0.727 0.303] [0.295 0.585]] (golden, 1e-3)
print(fit.spectral_radius)  # 0.45678479107028125 (golden, 1e-6): stationary
print(spectral_radius_2x2(fit.branching) == fit.spectral_radius)   # True
# alpha[0,1] answers: how strongly does a SELL excite future BUYS?
# self > cross here because the simulation was set up that way; not evidence about markets.
```

**C++**

```cpp
auto [t, k] = lobmod::read_marked_times_csv("../data/hawkes_biv.csv");
lobmod::HawkesBivFit fit = lobmod::fit_biv(t, k, 600.0);
double rad = lobmod::spectral_radius_2x2(fit.branching);
std::cout << fit.alpha[0][0] << " " << fit.alpha[0][1] << " "
          << fit.alpha[1][0] << " " << fit.alpha[1][1] << " radius=" << rad << "\n";
```

**Rust**

```rust
let (t, k) = lobmod::read_marked_times_csv("../data/hawkes_biv.csv")?;
let fit = lobmod::fit_biv(&t, &k, 600.0)?;
let rad = lobmod::spectral_radius_2x2(&fit.branching)?;
println!("{:?} radius={rad}", fit.alpha);
```

**Java**

```java
Hawkes.BivSample data = Hawkes.readMarkedTimesCsv(Path.of("../data/hawkes_biv.csv"));
Hawkes.BivFit fit = Hawkes.fitBiv(data.times(), data.types(), 600.0);
double rad = Hawkes.spectralRadius2x2(fit.branching());
System.out.printf("%f %f %f %f radius=%f%n", fit.alpha()[0][0], fit.alpha()[0][1],
    fit.alpha()[1][0], fit.alpha()[1][1], rad);
```

---

## 10. How do I compute OFI and build the feature/label set?

`ofi_at(states, i, w)` errors iff `i < w` (`e_0` is undefined); `i == w` is
the first valid index. Windows are counted in events, not clock time.

**Python**

```python
from lobmod import (read_events_csv, replay, load_feature_config,
                    compute_features, ofi_at, FEATURE_NAMES)

events = read_events_csv("../data/lob_events.csv")
cfg = load_feature_config("../data/features_config.json")
states = replay(events, depth_k=cfg.depth_k)
print(ofi_at(states, event_index=2199, window=100))  # -46.0 (golden, 1e-10)
fs = compute_features(states, cfg)
print(len(fs), FEATURE_NAMES)   # 4348 ['ofi','queue_imbalance','spread','depth_ratio','ret']
print(fs.X.shape, fs.y[:10])    # labels: 0 down, 1 flat, 2 up
```

**C++**

```cpp
auto events = lobmod::read_events_csv("../data/lob_events.csv");
auto cfg = lobmod::load_feature_config("../data/features_config.json");
auto states = lobmod::replay(events, cfg.depth_k);
double ofi = lobmod::ofi_at(states, 2199, 100);
lobmod::FeatureSet fs = lobmod::compute_features(states, cfg);
std::cout << ofi << " " << fs.size() << " " << fs.X[0].size() << "\n"; // -46 4348 5
```

**Rust**

```rust
let events = lobmod::read_events_csv("../data/lob_events.csv")?;
let cfg = lobmod::load_feature_config("../data/features_config.json")?;
let states = lobmod::replay(&events, cfg.depth_k)?;
let ofi = lobmod::ofi_at(&states, 2199, 100)?;
let fs = lobmod::compute_features(&states, &cfg)?;
println!("{ofi} {} {}", fs.len(), fs.x.cols());          // -46 4348 5
```

**Java**

```java
List<Event> events = Replay.readEventsCsv(Path.of("../data/lob_events.csv"));
FeatureConfig cfg = FeatureConfig.load(Path.of("../data/features_config.json"));
BookStates states = Replay.replay(events, cfg.depthK());
double ofi = Features.ofiAt(states, 2199, 100);
FeatureSet fs = Features.computeFeatures(states, cfg);
System.out.println(ofi + " " + fs.size() + " " + fs.x()[0].length);   // -46.0 4348 5
```

---

## 11. How do I train the MLP and baselines without leaking?

Split chronologically first, fit the standardizer on the **train split only**,
then transform both splits. Row-selection helpers (`x_rows`, `y_rows`, …)
take the split's index vector / range.

**Python**

```python
import numpy as np
from lobmod import (MLP, Standardizer, accuracy, chronological_split,
                    compute_features, confusion_matrix, load_feature_config,
                    majority_class, read_events_csv, replay)

cfg = load_feature_config("../data/features_config.json")
fs = compute_features(replay(read_events_csv("../data/lob_events.csv"), cfg.depth_k), cfg)
tr, te = chronological_split(len(fs), cfg.train_frac)      # first 70% train
std = Standardizer().fit(fs.X[tr])                         # train moments only
x_tr, x_te = std.transform(fs.X[tr]), std.transform(fs.X[te])

net = MLP(5, cfg.hidden, 3)                                # (16, 8) hidden
hist = net.fit(x_tr, fs.y[tr], cfg.learning_rate, cfg.epochs)
print(hist[0], hist[-1])          # 1.0806390310 -> 0.6760014200 (goldens, 1e-8)

logistic = MLP(5, (), 3)                                   # 0 hidden layers
logistic.fit(x_tr, fs.y[tr], cfg.learning_rate, cfg.epochs)
maj = majority_class(fs.y[tr])
print(accuracy(fs.y[te], net.predict(x_te)))               # 0.7042145594
print(accuracy(fs.y[te], logistic.predict(x_te)))          # 0.6996168582
print(accuracy(fs.y[te], np.full(te.size, maj)))           # 0.6582375479
print(confusion_matrix(fs.y[te], net.predict(x_te), 3))
```

**C++**

```cpp
auto cfg = lobmod::load_feature_config("../data/features_config.json");
auto fs = lobmod::compute_features(
    lobmod::replay(lobmod::read_events_csv("../data/lob_events.csv"), cfg.depth_k), cfg);
auto [tr, te] = lobmod::chronological_split(fs.size(), cfg.train_frac);
lobmod::Standardizer sc;
sc.fit(fs.x_rows(tr));                                     // train moments only
lobmod::Matrix x_tr = sc.transform(fs.x_rows(tr)), x_te = sc.transform(fs.x_rows(te));
lobmod::MLP net(5, cfg.hidden, 3);
auto hist = net.fit(x_tr, fs.y_rows(tr), cfg.learning_rate, cfg.epochs);
lobmod::MLP logistic(5, {}, 3);
logistic.fit(x_tr, fs.y_rows(tr), cfg.learning_rate, cfg.epochs);
std::vector<int> y_te = fs.y_rows(te);
std::vector<int> maj(y_te.size(), lobmod::majority_class(fs.y_rows(tr)));
std::cout.precision(10);
std::cout << hist.front() << " " << hist.back() << "\n";   // 1.080639031 0.67600142
std::cout << lobmod::accuracy(y_te, net.predict(x_te)) << " "
          << lobmod::accuracy(y_te, logistic.predict(x_te)) << " "
          << lobmod::accuracy(y_te, maj) << "\n";          // 0.7042 0.6996 0.6582
```

**Rust**

```rust
use lobmod::{Mlp, Standardizer};

let cfg = lobmod::load_feature_config("../data/features_config.json")?;
let events = lobmod::read_events_csv("../data/lob_events.csv")?;
let fs = lobmod::compute_features(&lobmod::replay(&events, cfg.depth_k)?, &cfg)?;
let (tr, te) = lobmod::chronological_split(fs.len(), cfg.train_frac)?;
let mut std = Standardizer::new();
std.fit(&fs.x_rows(&tr)?)?;                                 // train moments only
let (x_tr, x_te) = (std.transform(&fs.x_rows(&tr)?)?, std.transform(&fs.x_rows(&te)?)?);
let (y_tr, y_te) = (fs.y_rows(&tr)?, fs.y_rows(&te)?);
let mut net = Mlp::new(5, &cfg.hidden, 3)?;
let hist = net.fit(&x_tr, &y_tr, cfg.learning_rate, cfg.epochs)?;
let mut logistic = Mlp::new(5, &[], 3)?;
logistic.fit(&x_tr, &y_tr, cfg.learning_rate, cfg.epochs)?;
let maj = vec![lobmod::majority_class(&y_tr)?; y_te.len()];
println!("{:.10} {:.10}", hist[0], hist[hist.len() - 1]); // 1.0806390310 0.6760014200
println!("{:.4} {:.4} {:.4}",
         lobmod::accuracy(&y_te, &net.predict(&x_te)?)?,
         lobmod::accuracy(&y_te, &logistic.predict(&x_te)?)?,
         lobmod::accuracy(&y_te, &maj)?);                  // 0.7042 0.6996 0.6582
```

**Java**

```java
FeatureConfig cfg = FeatureConfig.load(Path.of("../data/features_config.json"));
List<Event> events = Replay.readEventsCsv(Path.of("../data/lob_events.csv"));
FeatureSet fs = Features.computeFeatures(Replay.replay(events, cfg.depthK()), cfg);
Features.Split split = Features.chronologicalSplit(fs.size(), cfg.trainFrac());
Standardizer std = new Standardizer().fit(fs.xRows(split.train()));   // train moments only
double[][] xTr = std.transform(fs.xRows(split.train()));
double[][] xTe = std.transform(fs.xRows(split.test()));
int[] yTr = fs.yRows(split.train());
int[] yTe = fs.yRows(split.test());
Mlp net = new Mlp(5, cfg.hidden(), 3);
double[] hist = net.fit(xTr, yTr, cfg.learningRate(), cfg.epochs());
Mlp logistic = new Mlp(5, new int[0], 3);
logistic.fit(xTr, yTr, cfg.learningRate(), cfg.epochs());
int[] maj = new int[yTe.length];
java.util.Arrays.fill(maj, Metrics.majorityClass(yTr));
System.out.printf("%.10f %.10f%n", hist[0], hist[hist.length - 1]); // 1.0806390310 0.6760014200
System.out.printf("%.4f %.4f %.4f%n", Metrics.accuracy(yTe, net.predict(xTe)),
    Metrics.accuracy(yTe, logistic.predict(xTe)), Metrics.accuracy(yTe, maj));
```

---

## 12. How do I compute signal PnL after transaction costs?

Position per test row is `y_pred − 1 ∈ {−1, 0, +1}` (any other label is an
error); net PnL subtracts half the spread per non-flat signal, charged on
entry only. This is a signal diagnostic (overlapping positions, mid-to-mid,
no fees or latency), not a tradable strategy.

**Python**

```python
from lobmod import (MLP, Standardizer, chronological_split, compute_features,
                    load_feature_config, read_events_csv, replay, signal_pnl)

cfg = load_feature_config("../data/features_config.json")
fs = compute_features(replay(read_events_csv("../data/lob_events.csv"), cfg.depth_k), cfg)
tr, te = chronological_split(len(fs), cfg.train_frac)
std = Standardizer().fit(fs.X[tr])
net = MLP(5, cfg.hidden, 3)
net.fit(std.transform(fs.X[tr]), fs.y[tr], cfg.learning_rate, cfg.epochs)
pred = net.predict(std.transform(fs.X[te]))
gross, net_pnl, n_trades = signal_pnl(pred, fs.fwd_ret[te], fs.spread[te])
print(gross, net_pnl, n_trades)      # 45.0 -139.5 177 (golden, 1e-8)
# positive gross, negative net: the edge is smaller than the half-spread.
```

**C++**

```cpp
auto cfg = lobmod::load_feature_config("../data/features_config.json");
auto fs = lobmod::compute_features(
    lobmod::replay(lobmod::read_events_csv("../data/lob_events.csv"), cfg.depth_k), cfg);
auto [tr, te] = lobmod::chronological_split(fs.size(), cfg.train_frac);
lobmod::Standardizer sc;
sc.fit(fs.x_rows(tr));
lobmod::MLP net(5, cfg.hidden, 3);
net.fit(sc.transform(fs.x_rows(tr)), fs.y_rows(tr), cfg.learning_rate, cfg.epochs);
std::vector<int> pred = net.predict(sc.transform(fs.x_rows(te)));
auto [gross, net_pnl, n_trades] =
    lobmod::signal_pnl(pred, fs.fwd_ret_rows(te), fs.spread_rows(te));
std::cout << gross << " " << net_pnl << " " << n_trades << "\n";   // 45 -139.5 177
```

**Rust**

```rust
use lobmod::{Mlp, Standardizer};

let cfg = lobmod::load_feature_config("../data/features_config.json")?;
let events = lobmod::read_events_csv("../data/lob_events.csv")?;
let fs = lobmod::compute_features(&lobmod::replay(&events, cfg.depth_k)?, &cfg)?;
let (tr, te) = lobmod::chronological_split(fs.len(), cfg.train_frac)?;
let mut std = Standardizer::new();
std.fit(&fs.x_rows(&tr)?)?;
let mut net = Mlp::new(5, &cfg.hidden, 3)?;
net.fit(&std.transform(&fs.x_rows(&tr)?)?, &fs.y_rows(&tr)?, cfg.learning_rate, cfg.epochs)?;
let pred = net.predict(&std.transform(&fs.x_rows(&te)?)?)?;
let (gross, net_pnl, n_trades) =
    lobmod::signal_pnl(&pred, &fs.fwd_ret_rows(&te)?, &fs.spread_rows(&te)?)?;
println!("{gross} {net_pnl} {n_trades}");                // 45 -139.5 177
```

**Java**

```java
FeatureConfig cfg = FeatureConfig.load(Path.of("../data/features_config.json"));
List<Event> events = Replay.readEventsCsv(Path.of("../data/lob_events.csv"));
FeatureSet fs = Features.computeFeatures(Replay.replay(events, cfg.depthK()), cfg);
Features.Split split = Features.chronologicalSplit(fs.size(), cfg.trainFrac());
Standardizer std = new Standardizer().fit(fs.xRows(split.train()));
Mlp net = new Mlp(5, cfg.hidden(), 3);
net.fit(std.transform(fs.xRows(split.train())), fs.yRows(split.train()),
    cfg.learningRate(), cfg.epochs());
int[] pred = net.predict(std.transform(fs.xRows(split.test())));
Metrics.Pnl pnl = Metrics.signalPnl(pred, fs.fwdRetRows(split.test()), fs.spreadRows(split.test()));
System.out.printf("gross=%f net=%f trades=%d%n", pnl.gross(), pnl.net(), pnl.nTrades()); // 45 -139.5 177
```

---

## 13. How do I reduce, replace and execute orders, and replay a v2 event file?

`reduce` takes size off and keeps the order's queue place. `replace` is a new
id at the tail of the new level, so priority is lost. `execute` fills a named
order at its own price. The file `data/lob_events_v2.csv` is a 17-event fixture
with nanosecond timestamps. `read_events_v2_csv` also reads a v1 file, with
`ts_ns = 0`.

**Python**

```python
from lobmod import LimitOrderBook
from lobmod.replay import EventV2, apply_event_v2, read_events_v2_csv, replay_v2

book = LimitOrderBook(tick_size=1.0)
book.limit_ticks("B", 100, 10, 1)
book.limit_ticks("B", 100, 5, 2)
print(book.reduce(1, 4))                  # 6: size left, id 1 keeps its place
print(book.order_ids("B"))                # [1, 2]
print(book.execute(1, 2))                 # Fill(maker_id=1, price=100, size=2)
print(book.replace_ticks(1, 3, 100, 4))   # ([], 4): id 3, back of the queue
print(book.order_ids("B"))                # [2, 3]
apply_event_v2(book, EventV2(1, 1_000, "R", "B", 0, 1, 2))   # reduce id 2 by 1
print(book.best_bid_size())               # 8

events = read_events_v2_csv("../data/lob_events_v2.csv")
states = replay_v2(events)
print(len(events), events[0].ts_ns)       # 17 34200000001500
print(states.best_bid[-1], states.best_ask[-1])   # 999.0 1000.0
```

**C++**

```cpp
lobmod::LimitOrderBook book(1.0);
book.limit_ticks('B', 100, 10, 1);
book.limit_ticks('B', 100, 5, 2);
std::cout << book.reduce(1, 4) << "\n";                  // 6
lobmod::Fill f = book.execute(1, 2);
std::cout << f.price << " " << f.size << "\n";           // 100 2
auto [fills, rested] = book.replace_ticks(1, 3, 100, 4);
std::cout << fills.size() << " " << rested << "\n";      // 0 4
std::cout << book.order_ids('B')[0] << "\n";             // 2
lobmod::apply_event_v2(book, lobmod::EventV2{1, 1000, 'R', 'B', 0, 1, 2, 0});
std::cout << book.best_bid_size() << "\n";               // 8

auto events = lobmod::read_events_v2_csv("../data/lob_events_v2.csv");
auto states = lobmod::replay_v2(events);
std::cout << events.size() << " " << states.best_bid.back() << " "
          << states.best_ask.back() << "\n";             // 17 999 1000
```

**Rust**

```rust
use lobmod::replay::{apply_event_v2, read_events_v2_csv, replay_v2, EventV2};
use lobmod::{LimitOrderBook, Side};

let mut book = LimitOrderBook::new(1.0)?;
book.limit_ticks(Side::Bid, 100, 10, 1)?;
book.limit_ticks(Side::Bid, 100, 5, 2)?;
println!("{}", book.reduce(1, 4)?);                      // 6
let f = book.execute(1, 2)?;
println!("{} {}", f.price, f.size);                      // 100 2
let (fills, rested) = book.replace_ticks(1, 3, 100, 4)?;
println!("{} {}", fills.len(), rested);                  // 0 4
println!("{:?}", book.order_ids(Side::Bid));             // [2, 3]
let ev = EventV2 { seq: 1, ts_ns: 1000, etype: 'R', side: Side::Bid, price: 0,
                   size: 1, order_id: 2, new_order_id: 0 };
apply_event_v2(&mut book, &ev)?;
println!("{}", book.best_bid_size());                    // 8

let events = read_events_v2_csv("../data/lob_events_v2.csv")?;
let states = replay_v2(&events, 3)?;
println!("{} {} {}", events.len(), states.best_bid.last().unwrap(),
         states.best_ask.last().unwrap());               // 17 999 1000
```

**Java**

```java
LimitOrderBook book = new LimitOrderBook(1.0);
book.limitTicks('B', 100, 10, 1);
book.limitTicks('B', 100, 5, 2);
System.out.println(book.reduce(1, 4));                   // 6
Fill f = book.execute(1, 2);
System.out.println(f.price() + " " + f.size());          // 100 2
LimitOrderBook.ExecResult r = book.replaceTicks(1, 3, 100, 4);
System.out.println(r.fills().size() + " " + r.remainder()); // 0 4
System.out.println(book.orderIds('B')[0]);               // 2
Replay.applyEventV2(book, new EventV2(1, 1000, 'R', 'B', 0, 1, 2, 0));
System.out.println(book.bestBidSize());                  // 8

List<EventV2> events = Replay.readEventsV2Csv(Path.of("../data/lob_events_v2.csv"));
BookStates st = Replay.replayV2(events, 3);
System.out.println(events.size() + " " + st.bestBid()[events.size() - 1]); // 17 999.0
```

## 14. How do I run the ITCH 5.0 parser on a small byte fixture?

The parser is Python only and has been tested on hand-built fixtures. This
recipe builds one in memory: a directory message, two adds, a partial cancel,
an execution and a replace. Prices have four implied decimals, so 1,000,000
is 100.0000 and the raw value is the tick count. Events that refer to a resting
order by id carry price 0.

**Python**

```python
import io
import struct

from lobmod.feeds import BookIntegrityChecker, FeedEvent, Itch50Parser

def head(kind, ts_ns, locate=1):
    return kind + struct.pack(">HH", locate, 0) + ts_ns.to_bytes(6, "big")

def frame(msg):
    return struct.pack(">H", len(msg)) + msg

stock = b"TEST    "
messages = [
    head(b"R", 1) + stock + b"Q" + bytes(19),                                 # directory
    head(b"A", 2) + struct.pack(">QcI", 1, b"B", 100) + stock + struct.pack(">I", 1_000_000),
    head(b"A", 3) + struct.pack(">QcI", 2, b"S", 50) + stock + struct.pack(">I", 1_000_100),
    head(b"X", 4) + struct.pack(">QI", 1, 30),                                # cancel 30
    head(b"E", 5) + struct.pack(">QIQ", 1, 20, 77),                           # execute 20
    head(b"U", 6) + struct.pack(">QQII", 2, 3, 40, 1_000_200),                # replace 2 -> 3
]
stream = io.BytesIO(b"".join(map(frame, messages)))

parser = Itch50Parser(symbols=["TEST"])       # strict: any anomaly raises ItchError
checker = BookIntegrityChecker()
for item in parser.parse(stream):
    if isinstance(item, FeedEvent):
        checker.apply(item.event)
        print(item.event.etype, item.event.side, item.event.price, item.event.size)
# L B 1000000 100 / L S 1000100 50 / R B 0 30 / E B 0 20 / U S 1000200 40

print(checker.book.best_bid(), checker.book.best_bid_size())   # 1000000 50
print(checker.book.best_ask(), checker.book.best_ask_size())   # 1000200 40
print(checker.verify())                                        # (1, 1) orders per side
print(dict(parser.stats))                                      # {'messages': 6}
```

## 15. How do I get standard errors and a valid goodness-of-fit p-value?

`lobmod.research` is Python only and outside the golden contract. The table
KS p-value ignores estimation. The parametric bootstrap refits on every
simulated path, so its p-value is the one to quote. `n_boot=19` keeps the
recipe quick; `research/run_all.py` uses 49, and a real check wants more.

**Python**

```python
from lobmod import fit_uni, read_times_csv, uni_residuals
from lobmod.research import bootstrap_ks_uni, inference_uni, ljung_box

t = read_times_csv("../data/hawkes_uni.csv")
T = 1000.0
fit = fit_uni(t, T)
inf = inference_uni(t, T, fit)
for name, est, se in zip(inf.names, inf.estimate, inf.se):
    print(f"{name} {est:.3f} se {se:.3f}")   # mu 0.505 se 0.037, alpha 0.979 se 0.089, beta 1.679 se 0.161
print(f"{inf.branching:.3f} se {inf.branching_se:.3f}")   # 0.583 se 0.033

gof = bootstrap_ks_uni(t, T, n_boot=19, seed=1, fit=fit)
print(f"{gof.statistic:.4f} {gof.p_naive:.3f} {gof.p_value:.3f}")   # 0.0172 0.863 0.450

lb = ljung_box(uni_residuals(t, T, fit.mu, fit.alpha, fit.beta), lags=10)
print(f"{lb.statistic:.2f} {lb.p_value:.3f}")   # 6.78 0.746
```

## 16. How do I run purged walk-forward folds and an executable backtest?

`walk_forward_splits` removes training rows whose label window reaches the test
block, then an embargo. `run_backtest` pays the spread both ways, with a delay
in events and a fee per side, so it is harsher than `signal_pnl`.

**Python**

```python
from lobmod import MLP, Standardizer, compute_features, load_feature_config, read_events_csv, replay
from lobmod.research import BacktestConfig, run_backtest, walk_forward_splits

cfg = load_feature_config("../data/features_config.json")
states = replay(read_events_csv("../data/lob_events.csv"), cfg.depth_k)
fs = compute_features(states, cfg)

for fold in walk_forward_splits(fs.event_index, n_folds=3, horizon=cfg.horizon, embargo=50):
    scaler = Standardizer().fit(fs.X[fold.train])
    model = MLP(5, (), 3)                              # no hidden layer: logistic
    model.fit(scaler.transform(fs.X[fold.train]), fs.y[fold.train], cfg.learning_rate, 200)
    pred = model.predict(scaler.transform(fs.X[fold.test]))
    s = run_backtest(states, fs.event_index[fold.test], pred - 1,
                     BacktestConfig(hold=cfg.horizon, delay=1, fee_per_side=0.1)).stats
    print(fold.train.size, fold.test.size, int(s["n_trades"]), round(s["net"], 1))
# 2983 435 8 -9.6
# 3418 435 6 -10.2
# 3853 435 17 -26.4
```

Three folds of one synthetic stream show the mechanics and nothing more. The
multi-seed figures are in `research/results.md`.

## 17. How do I fit multi-exponential, power-law and seasonal Hawkes models?

`simulate_cluster` draws a path with any kernel shape, because you hand it the
parent-to-child lags. Here the truth has two timescales and branching 0.6.

**Python**

```python
import numpy as np

from lobmod.research.kernels import fit_power_law, fit_sum_exp, simulate_cluster
from lobmod.research.seasonality import select_baseline

rng = np.random.default_rng(7)
T = 1000.0

def two_timescales(k):
    fast = rng.random(k) < 0.5
    return np.where(fast, rng.exponential(1 / 20.0, k), rng.exponential(1 / 0.5, k))

t = simulate_cluster(lambda x: np.full_like(x, 1.0), 1.0, 0.6, two_timescales, T, rng)
for fit in (fit_sum_exp(t, T, K=1), fit_sum_exp(t, T, K=2), fit_power_law(t, T)):
    print(f"{fit.name}: n {fit.branching:.3f}, BIC {fit.bic:.1f}")
# exp K=1: n 0.383, BIC -1120.1      (branching badly understated)
# exp K=2: n 0.629, BIC -1206.7      (lowest BIC, near the true 0.6)
# power-law mix: n 0.641, BIC -1184.2

def u_shape(x):
    return 1.0 + 12.0 * (x / T - 0.5) ** 2      # busy open and close, between 1 and 4

u = simulate_cluster(u_shape, 4.0, 0.5, lambda k: rng.exponential(1 / 3.0, k), T, rng)
print(round(fit_sum_exp(u, T, K=1).branching, 3))   # 0.696: seasonality read as excitation
best, baseline, table = select_baseline(u, T)
print(baseline.J, round(best.branching, 3))         # 2 0.563 (true n is 0.5)
```

One path per case: use `research/results_models.md` for the 60-path figures.

## 18. How do I label rows by what a round trip would actually pay?

The pinned labels say which way the mid moved. `cost_aware_labels` says whether
crossing the spread both ways, after fees and a delay, would have paid.

**Python**

```python
import numpy as np

from lobmod import compute_features, load_feature_config, read_events_csv, replay
from lobmod.research.labels import cost_aware_labels

cfg = load_feature_config("../data/features_config.json")
states = replay(read_events_csv("../data/lob_events.csv"), cfg.depth_k)
fs = compute_features(states, cfg)

print(round(float(np.mean(fs.y != 1)), 3))      # 0.326: pinned labels, not flat
for H in (10, 200):
    lab = cost_aware_labels(states, fs.event_index, horizon=H, fee_per_side=0.1, delay=1)
    print(H, round(float(np.mean(lab.y[lab.valid] != 1)), 3))   # 10 0.021, then 200 0.401
```

## 19. How do I run the execution simulator and read the attribution?

Every flow parameter is assumed, and the agents get a free fair-value feed
unless `sees_signal=False`. Read the attribution, not the P&L level, and use
many seeds as `research/run_sim.py` does. One short run is shown here.

**Python**

```python
from lobmod.sim import AgentSpec, AvellanedaStoikov, FlowParams, NaiveTouch, Simulator
from lobmod.sim import attribute, realised_spread

agents = [
    AgentSpec("naive", NaiveTouch(cap=5), md_latency=0.01, oe_latency=0.01),
    AgentSpec("as", AvellanedaStoikov(cap=5), md_latency=0.01, oe_latency=0.01),
]
res = Simulator(FlowParams(), agents, T=120.0, seed=1).run()
print(res.n_events)                                  # 8962
a = attribute(res, "naive", horizon=1.0)
print(a.spread, a.adverse, a.inventory, a.total, a.lots)   # 104.0 -28.0 -2.0 74.0 126
b = attribute(res, "as", horizon=1.0)
print(b.spread, b.adverse, b.inventory, b.total, b.lots)   # 48.5 15.0 -39.5 24.0 30
print(round(realised_spread(res, "naive", [1.0])[1.0], 3)) # 0.603 ticks per lot
```

With zero fees the three terms add up to `total`, which equals the booked P&L.
Positive P&L here is not an edge: see `research/results_sim.md`.

## 20. How do I use FastBook?

C++ and Rust only, integer ticks only. You fix the price band and the pool up
front, fills go to a callback, and refusals are return values. Build C++ with
`#include "fast_book.hpp"` added to the prelude and `-I fast` added to the
command.

**C++**

```cpp
using namespace lobmod::fast;
Config cfg;
cfg.min_price_ticks = 900;
cfg.n_levels = 200;          // band 900..1099
cfg.max_orders = 1024;
FastBook book(cfg);
auto sink = [](std::int64_t maker, std::int64_t px, std::int64_t qty) {
    std::cout << maker << " " << px << " " << qty << "\n";
};
book.limit(Side::Buy, 1000, 5, 1, sink);
book.limit(Side::Sell, 1002, 3, 2, sink);
OpResult r = book.limit(Side::Buy, 1002, 4, 3, sink);    // prints 2 1002 3
std::cout << r.value << " " << book.best_bid() << "\n";  // 1 1002
std::cout << (book.limit(Side::Buy, 5000, 1, 4, sink).status == Status::OutOfBand)
          << "\n";                                       // 1
std::cout << book.cancel(1).value << "\n";               // 5
```

**Rust**

```rust
use lobmod::fast_book::{FastBook, FastConfig, FastError};
use lobmod::Side;

let cfg = FastConfig { min_price_ticks: 900, n_levels: 200, max_orders: 1024 };
let mut book = FastBook::new(cfg).expect("bad band or pool size");
let mut sink = |maker: i64, px: i64, qty: i64| println!("{maker} {px} {qty}");
book.limit(Side::Bid, 1000, 5, 1, &mut sink).unwrap();
book.limit(Side::Ask, 1002, 3, 2, &mut sink).unwrap();
let rested = book.limit(Side::Bid, 1002, 4, 3, &mut sink).unwrap(); // prints 2 1002 3
println!("{} {:?}", rested, book.best_bid());            // 1 Some(1002)
let refused = book.limit(Side::Bid, 5000, 1, 4, &mut sink);
println!("{}", refused == Err(FastError::OutOfBand));    // true
println!("{}", book.cancel(1).unwrap());                 // 5
```

There is no FastBook in Python or Java. Timings are in `docs/LATENCY.md`.

---

## Verifying the snippets

Every fenced block was extracted, wrapped as the conventions describe and run on
2026-11-05 from a clean export of the code, with these tools: Python 3.12 and
numpy 2.4 on Windows, cargo release profile on Windows, JDK 27 on Windows, and
g++ 15.2 under WSL (library compiled directly from `cpp/src`, not through
`build.sh`). All 60 blocks compiled and ran without error, and the printed
values of recipes 13-20 match their comments. Comments in recipes 1-12 were
checked when they were written and were not compared line by line in this pass.
Recipe 17 uses numpy's generator, so its values can move with the numpy version.
Re-run the same check after any API change: a snippet that stops compiling is a
documentation bug.
