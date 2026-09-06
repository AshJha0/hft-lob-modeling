# COOKBOOK — `lobmod` recipes

Task-oriented recipes in all four languages. **Every snippet below compiles
and runs against the real library** — they are extracted and executed as part
of the review checklist (see "Verifying the snippets" at the end). Paths
assume the bundled `../data/` files relative to the language directory.

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
print(d, p)                          # 0.017205 0.863177 (golden, 1e-6): PASS
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
# self terms > cross terms => flow herds by side (order splitting).
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

## Verifying the snippets

Every fenced block above was extracted, wrapped exactly as described in the
conventions, compiled against the built library and executed from the
language directory (Python 3.11, g++ 13 `-std=c++17`, cargo 1.95 release
profile, OpenJDK 21), and produced the values shown in the comments. Re-run
the same check after any API change: any snippet that stops compiling is a
documentation bug.
