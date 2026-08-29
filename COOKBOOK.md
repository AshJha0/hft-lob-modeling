# COOKBOOK — `lobmod` recipes

Task-oriented recipes in all four languages. Python snippets are runnable as-is
from the `python/` directory (`cd python && PYTHONPATH=src python3 <snippet>.py`);
paths assume the bundled `../data/` files. C++ uses namespace `lobmod` (headers
under `include/lobmod/`), Rust the `lobmod` crate (fallible APIs return
`Result<_, LobError>`), Java the `com.quant.lobmod` package — all naming per
`API_SPEC.md`.

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
```

**C++**

```cpp
#include <lobmod/book.hpp>

lobmod::LimitOrderBook book(1.0);
book.limit('B', 99.0, 5, 1);
book.limit('B', 98.0, 8, 2);
book.limit('S', 101.0, 3, 3);
std::cout << book.mid() << " " << book.spread() << " "
          << book.microprice() << " " << book.depth('B', 2) << "\n";
```

**Rust**

```rust
use lobmod::{LimitOrderBook, Side};

let mut book = LimitOrderBook::new(1.0)?;
book.limit(Side::Buy, 99.0, 5, 1)?;
book.limit(Side::Buy, 98.0, 8, 2)?;
book.limit(Side::Sell, 101.0, 3, 3)?;
println!("{} {} {}", book.mid(), book.spread(), book.microprice());
println!("{}", book.depth(Side::Buy, 2)?);
```

**Java**

```java
import com.quant.lobmod.LimitOrderBook;

LimitOrderBook book = new LimitOrderBook(1.0);
book.limit('B', 99.0, 5, 1);
book.limit('B', 98.0, 8, 2);
book.limit('S', 101.0, 3, 3);
System.out.printf("%f %f %f %d%n",
    book.mid(), book.spread(), book.microprice(), book.depth('B', 2));
```

---

## 2. How do I replay the bundled event stream and take snapshots?

**Python**

```python
from lobmod import read_events_csv, replay

events = read_events_csv("../data/lob_events.csv")
states = replay(events, depth_k=3, tick_size=1.0)
i = 1000 - 1                       # golden snapshot after 1-based event 1000
print(states.best_bid[i], states.best_ask[i])   # 999.0 1000.0
print(states.mid[i], states.microprice[i])      # 999.5 ...
print(states.bid_depth_k[i], states.ask_depth_k[i])  # 21.0 27.0
```

**C++**

```cpp
#include <lobmod/replay.hpp>

auto events = lobmod::read_events_csv("../data/lob_events.csv");
auto states = lobmod::replay(events, /*depth_k=*/3, /*tick_size=*/1.0);
std::size_t i = 1000 - 1;
std::cout << states.best_bid[i] << " " << states.best_ask[i] << " "
          << states.mid[i] << "\n";
```

**Rust**

```rust
use lobmod::{read_events_csv, replay};

let events = read_events_csv("../data/lob_events.csv")?;
let states = replay(&events, 3, 1.0)?;
let i = 1000 - 1;
println!("{} {} {}", states.best_bid[i], states.best_ask[i], states.mid[i]);
```

**Java**

```java
import com.quant.lobmod.Replay;
import com.quant.lobmod.BookStates;

var events = Replay.readEventsCsv("../data/lob_events.csv");
BookStates states = Replay.replay(events, 3, 1.0);
int i = 1000 - 1;
System.out.printf("%f %f %f%n",
    states.bestBid[i], states.bestAsk[i], states.mid[i]);
```

---

## 3. How do I handle the classic edge cases (sweep, bad cancel, crossed limit)?

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
```

**C++** — same flow; the bad cancel throws `std::invalid_argument`:

```cpp
lobmod::LimitOrderBook book(1.0);
book.limit('S', 101.0, 5, 1);
auto [fills, unfilled] = book.market('B', 25);   // unfilled reported
try { book.cancel(99); } catch (const std::invalid_argument& e) { /* ... */ }
auto [f2, rested] = book.limit('S', 99.0, 10, 4); // marketable limit
```

**Rust** — errors come back as `Result`:

```rust
let mut book = LimitOrderBook::new(1.0)?;
book.limit(Side::Sell, 101.0, 5, 1)?;
let (fills, unfilled) = book.market(Side::Buy, 25)?; // Ok: unfilled reported
assert!(book.mid().is_nan());
assert!(book.cancel(99).is_err());                   // unknown id -> LobError
```

**Java** — the bad cancel throws `IllegalArgumentException`:

```java
LimitOrderBook book = new LimitOrderBook(1.0);
book.limit('S', 101.0, 5, 1);
LimitOrderBook.MarketResult r = book.market('B', 25); // r.unfilled reported
try { book.cancel(99); } catch (IllegalArgumentException e) { /* expected */ }
```

---

## 4. How do I generate a synthetic zero-intelligence stream?

Identical seeds produce identical streams in every language (`data/lob_events.csv`
is `generate_events(5000, 20240613)`).

**Python**

```python
from lobmod import generate_events, replay, write_events_csv

events = generate_events(n_events=5000, seed=20240613)
write_events_csv("/tmp/my_stream.csv", events)
states = replay(events)
print(states.mid[-1])                # 1003.5 — matches the bundled stream
```

**C++**

```cpp
auto events = lobmod::generate_events(5000, 20240613ULL);
lobmod::write_events_csv("my_stream.csv", events);
```

**Rust**

```rust
let events = lobmod::generate_events(5000, 20240613)?;
lobmod::write_events_csv("my_stream.csv", &events)?;
```

**Java**

```java
var events = com.quant.lobmod.ZiGenerator.generateEvents(5000, 20240613L);
com.quant.lobmod.Replay.writeEventsCsv("my_stream.csv", events);
```

---

## 5. How do I evaluate a Hawkes log-likelihood at given parameters?

**Python**

```python
import numpy as np
from lobmod import uni_loglik, uni_loglik_brute

t = np.loadtxt("../data/hawkes_uni.csv", skiprows=1)
ll = uni_loglik(t, T=1000.0, mu=0.4, alpha=0.6, beta=1.2)
print(ll)                                # -738.033935432713 (golden, 1e-8)
# brute-force O(N^2) cross-check (tests use 1e-10):
print(abs(ll - uni_loglik_brute(t, 1000.0, 0.4, 0.6, 1.2)) < 1e-10)
```

**C++**

```cpp
#include <lobmod/hawkes.hpp>
std::vector<double> t = lobmod::read_times_csv("../data/hawkes_uni.csv");
double ll = lobmod::uni_loglik(t, 1000.0, 0.4, 0.6, 1.2);
```

**Rust**

```rust
let t = lobmod::read_times_csv("../data/hawkes_uni.csv")?;
let ll = lobmod::uni_loglik(&t, 1000.0, 0.4, 0.6, 1.2)?;
```

**Java**

```java
double[] t = com.quant.lobmod.Hawkes.readTimesCsv("../data/hawkes_uni.csv");
double ll = com.quant.lobmod.Hawkes.uniLoglik(t, 1000.0, 0.4, 0.6, 1.2);
```

---

## 6. How do I fit a univariate Hawkes model and read the branching ratio?

The fit warns (never throws) at branching ≥ 0.9 / ≥ 1.

**Python**

```python
import numpy as np
from lobmod import fit_uni

t = np.loadtxt("../data/hawkes_uni.csv", skiprows=1)
fit = fit_uni(t, T=1000.0)
print(fit.mu, fit.alpha, fit.beta)   # 0.5052 0.9795 1.6792 (golden, 1e-4)
print(fit.branching)                 # 0.5833235621941935 (golden, 1e-6)
print(fit.loglik, fit.converged, fit.n_iter)
```

**C++**

```cpp
lobmod::HawkesUniFit fit = lobmod::fit_uni(t, 1000.0);
std::cout << fit.mu << " " << fit.alpha << " " << fit.beta
          << " branching=" << fit.branching << "\n";
```

**Rust**

```rust
let fit = lobmod::fit_uni(&t, 1000.0)?;
println!("{} {} {} branching={}", fit.mu, fit.alpha, fit.beta, fit.branching);
```

**Java**

```java
Hawkes.UniFit fit = Hawkes.fitUni(t, 1000.0);
System.out.printf("%f %f %f branching=%f%n", fit.mu, fit.alpha, fit.beta, fit.branching);
```

---

## 7. How do I simulate a Hawkes process (Ogata thinning)?

Same seed ⇒ identical event times in all four languages (pinned two uniforms
per candidate). Simulation refuses non-stationary parameters (branching ≥ 1).

**Python**

```python
from lobmod import simulate_uni, simulate_biv

t = simulate_uni(mu=0.5, alpha=1.0, beta=1.6, T=1000.0, seed=101)
print(t.size)                        # 1211 — regenerates data/hawkes_uni.csv
tb, kb = simulate_biv(mu=(0.3, 0.3), alpha=[[0.6, 0.3], [0.25, 0.5]],
                      beta=2.0, T=600.0, seed=33)
print(tb.size)                       # 611 — regenerates data/hawkes_biv.csv
```

**C++**

```cpp
std::vector<double> t = lobmod::simulate_uni(0.5, 1.0, 1.6, 1000.0, 101ULL);
auto [tb, kb] = lobmod::simulate_biv({0.3, 0.3}, {{0.6, 0.3}, {0.25, 0.5}},
                                     2.0, 600.0, 33ULL);
```

**Rust**

```rust
let t = lobmod::simulate_uni(0.5, 1.0, 1.6, 1000.0, 101)?;
let (tb, kb) = lobmod::simulate_biv([0.3, 0.3], [[0.6, 0.3], [0.25, 0.5]],
                                    2.0, 600.0, 33)?;
```

**Java**

```java
double[] t = Hawkes.simulateUni(0.5, 1.0, 1.6, 1000.0, 101L);
Hawkes.BivSample s = Hawkes.simulateBiv(new double[]{0.3, 0.3},
    new double[][]{{0.6, 0.3}, {0.25, 0.5}}, 2.0, 600.0, 33L);
```

---

## 8. How do I check goodness of fit (time-rescaling + KS)?

**Python**

```python
import numpy as np
from lobmod import fit_uni, uni_residuals, ks_test_exp

t = np.loadtxt("../data/hawkes_uni.csv", skiprows=1)
fit = fit_uni(t, T=1000.0)
tau = uni_residuals(t, 1000.0, fit.mu, fit.alpha, fit.beta)
print(tau.mean())                    # ~1.0 (Exp(1) mean) and all tau > 0
d, p = ks_test_exp(tau)
print(d, p)                          # 0.017205 0.863177 (golden, 1e-6): PASS
```

**C++**

```cpp
auto tau = lobmod::uni_residuals(t, 1000.0, fit.mu, fit.alpha, fit.beta);
auto [d, p] = lobmod::ks_test_exp(tau);
```

**Rust**

```rust
let tau = lobmod::uni_residuals(&t, 1000.0, fit.mu, fit.alpha, fit.beta)?;
let (d, p) = lobmod::ks_test_exp(&tau)?;
```

**Java**

```java
double[] tau = Hawkes.uniResiduals(t, 1000.0, fit.mu, fit.alpha, fit.beta);
Gof.KsResult ks = Gof.ksTestExp(tau);   // ks.stat, ks.pValue
```

---

## 9. How do I fit the bivariate buy/sell model and interpret it?

**Python**

```python
import numpy as np
from lobmod import fit_biv, spectral_radius_2x2

raw = np.loadtxt("../data/hawkes_biv.csv", skiprows=1, delimiter=",")
t, k = raw[:, 0], raw[:, 1].astype(int)     # k: 0 = buy, 1 = sell
fit = fit_biv(t, k, T=600.0)
print(fit.alpha)          # [[0.727 0.303] [0.295 0.585]] (golden, 1e-3)
print(fit.spectral_radius)  # 0.45678479107028125 (golden, 1e-6): stationary
# alpha[0,1] answers: how strongly does a SELL excite future BUYS?
# self terms > cross terms => flow herds by side (order splitting).
```

**C++**

```cpp
lobmod::HawkesBivFit fit = lobmod::fit_biv(t, k, 600.0);
double rad = lobmod::spectral_radius_2x2(fit.branching);
```

**Rust**

```rust
let fit = lobmod::fit_biv(&t, &k, 600.0)?;
let rad = lobmod::spectral_radius_2x2(&fit.branching)?;
```

**Java**

```java
Hawkes.BivFit fit = Hawkes.fitBiv(t, k, 600.0);
double rad = Hawkes.spectralRadius2x2(fit.branching);
```

---

## 10. How do I compute OFI and build the feature/label set?

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
auto cfg = lobmod::load_feature_config("../data/features_config.json");
auto states = lobmod::replay(events, cfg.depth_k, 1.0);
double ofi = lobmod::ofi_at(states, 2199, 100);
lobmod::FeatureSet fs = lobmod::compute_features(states, cfg);
```

**Rust**

```rust
let cfg = lobmod::load_feature_config("../data/features_config.json")?;
let states = lobmod::replay(&events, cfg.depth_k, 1.0)?;
let ofi = lobmod::ofi_at(&states, 2199, 100)?;
let fs = lobmod::compute_features(&states, &cfg)?;
```

**Java**

```java
FeatureConfig cfg = Features.loadConfig("../data/features_config.json");
BookStates states = Replay.replay(events, cfg.depthK, 1.0);
double ofi = Features.ofiAt(states, 2199, 100);
FeatureSet fs = Features.computeFeatures(states, cfg);
```

---

## 11. How do I train the MLP and baselines without leaking?

Split chronologically first, fit the standardizer on the **train split only**,
then transform both splits.

**Python**

```python
import numpy as np
from lobmod import (MLP, Standardizer, accuracy, chronological_split,
                    confusion_matrix, majority_class)
# ... fs, cfg from recipe 10 ...
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
auto [tr, te] = lobmod::chronological_split(fs.size(), cfg.train_frac);
lobmod::Standardizer std_;
std_.fit(fs.x(tr));
lobmod::Mlp net(5, {16, 8}, 3);
auto hist = net.fit(std_.transform(fs.x(tr)), fs.y(tr),
                    cfg.learning_rate, cfg.epochs);
double acc = lobmod::accuracy(fs.y(te), net.predict(std_.transform(fs.x(te))));
```

**Rust**

```rust
let (tr, te) = lobmod::chronological_split(fs.len(), cfg.train_frac)?;
let std = lobmod::Standardizer::fit(&fs.x_rows(&tr))?;
let mut net = lobmod::Mlp::new(5, &[16, 8], 3)?;
let hist = net.fit(&std.transform(&fs.x_rows(&tr))?, &fs.y_rows(&tr),
                   cfg.learning_rate, cfg.epochs)?;
let acc = lobmod::accuracy(&fs.y_rows(&te),
                           &net.predict(&std.transform(&fs.x_rows(&te))?))?;
```

**Java**

```java
int[][] split = Features.chronologicalSplit(fs.size(), cfg.trainFrac);
Standardizer std = new Standardizer().fit(fs.x(split[0]));
Mlp net = new Mlp(5, new int[]{16, 8}, 3);
double[] hist = net.fit(std.transform(fs.x(split[0])), fs.y(split[0]),
                        cfg.learningRate, cfg.epochs);
double acc = Metrics.accuracy(fs.y(split[1]),
                              net.predict(std.transform(fs.x(split[1]))));
```

---

## 12. How do I compute signal PnL after transaction costs?

Position per test row is `y_pred − 1 ∈ {−1, 0, +1}`; net PnL subtracts half
the spread per non-flat signal.

**Python**

```python
from lobmod import signal_pnl
# ... net, fs, te from recipe 11 ...
pred = net.predict(x_te)
gross, net_pnl, n_trades = signal_pnl(pred, fs.fwd_ret[te], fs.spread[te])
print(gross, net_pnl, n_trades)      # 45.0 -139.5 177 (golden, 1e-8)
# positive gross, negative net: the edge is smaller than the half-spread.
```

**C++**

```cpp
auto [gross, net_pnl, n_trades] =
    lobmod::signal_pnl(pred, fs.fwd_ret(te), fs.spread(te));
```

**Rust**

```rust
let (gross, net_pnl, n_trades) =
    lobmod::signal_pnl(&pred, &fs.fwd_ret_rows(&te), &fs.spread_rows(&te))?;
```

**Java**

```java
Metrics.PnlResult pnl = Metrics.signalPnl(pred, fs.fwdRet(split[1]), fs.spread(split[1]));
System.out.printf("gross=%f net=%f trades=%d%n", pnl.gross, pnl.net, pnl.nTrades);
```
