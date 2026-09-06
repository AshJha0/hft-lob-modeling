# `lobmod` — API Specification (cross-language contract)

Binding contract for the Python (`lobmod`), C++ (`lobmod::`), Rust (crate
`lobmod`) and Java (`com.quant.lobmod`) implementations. Every formula,
constant, iteration order and tolerance below is **pinned**: implementations
must reproduce `data/golden/golden.json` bit-for-tolerance from the bundled
CSVs. All floating point is IEEE-754 double.

Error behavior (everywhere a rule below says "error"): Python raises
`ValueError` with a message; C++ throws `std::invalid_argument` (or
`std::domain_error` for numeric-domain violations); Rust returns
`Result<_, LobError>` (manual thiserror-style enum, no panic on bad input);
Java throws `IllegalArgumentException`. Recoverable model diagnostics
(near-critical branching, non-convergence) are **reported, never thrown**.
Every integer quantity in this contract (ticks, sizes, order ids, sequence
numbers, seeds) is a **signed 64-bit integer** in every port; nothing is
silently narrowed, wrapped or masked — out-of-range values are errors.
Python additionally accepts any `numbers.Integral` (e.g. `numpy.int64`) but
never `bool`; non-integral floats are never truncated to integers.

Units: book prices are tick counts (`tick_size` converts currency to ticks
and is otherwise never used); sizes are unit-less integers; Hawkes times and
`T` share one arbitrary time unit; PnL is in ticks per unit size.

## 1. Pinned RNG — `Lcg64`

The only source of randomness in the whole project (ZI generator, Ogata
thinning). Never use a language's built-in RNG.

```
state_0   = seed, an integer in [0, 2^63 - 1] (error otherwise; never masked)
state_n+1 = (6364136223846793005 * state_n + 1442695040888963407) mod 2^64
next_u64():        advance once, return new state
uniform():         (next_u64() >> 11) * 2^-53            in [0, 1)
exponential(rate): -log1p(-uniform()) / rate             (error: rate <= 0)
geometric(p):      floor( log1p(-uniform()) / log1p(-p) ) in {0,1,2,...}
                   (error: p not in (0,1))
randint(n):        min(floor(uniform() * n), n-1)        (error: n <= 0)
```

Draw order matters; consume uniforms exactly as the algorithms below state.

## 2. Limit order book

### 2.1 Model

* Prices are positive integer multiples of `tick_size` (> 0, finite; error
  otherwise). Internally a price is its integer tick count; all book prices
  in this contract are **in ticks**. A submitted currency price is converted
  by the pinned rule

  ```
  ratio = price / tick_size                      (price <= 0 or non-finite: error)
  ticks = floor(ratio); if ratio - ticks >= 0.5: ticks += 1    (round half up)
  error if |ratio - ticks| > 1e-9 * max(1, ratio)              (off-grid)
  error if ticks < 1                                           (rounds to 0 ticks)
  error if ticks > MAX_TICKS = 9e18                            (64-bit range)
  ```

  An out-of-range price is therefore always an error and can never be
  reinterpreted as a negative price or as an unbounded (market) order.
  Every port also exposes `limit_ticks(side, price_ticks, size, order_id)`
  taking the integer tick count directly (same validation, no float
  conversion); it is the path used by replay.
* Two sides `B` (bids) and `S` (asks). Each price level holds a FIFO queue
  of orders `(order_id, remaining_size)`. Sizes are positive integers (error
  otherwise; also error above `MAX_TICKS`). `order_id` must not collide with
  a live resting order (error).
* Best bid = max bid level, best ask = min ask level.
* Optional history: a book records every fill (`trades`) and, in Python and
  C++, every event (`log`) for its lifetime; both grow without bound. Each
  port offers a constructor flag `record_trades` (default true) to disable
  recording and a `clear_history()` call to drop what was recorded. The
  matching semantics do not depend on the flag.

### 2.2 Events and matching (price-time priority)

* **`limit(side, price, size, order_id) -> (fills, rested_size)`**
  If the order crosses (`B` with `price >= best_ask`, `S` with
  `price <= best_bid`) it first executes as a taker against the opposite
  side: walk levels best-first while the level price is within the limit
  price; within a level fill FIFO from the queue front, partially reducing
  the front order if needed; fully-filled resting orders leave the book.
  Executions happen **at the resting order's price**. Any unfilled remainder
  rests at `price` (appended to that level's queue tail). A crossed limit
  never trades through its own limit price.
* **`cancel(order_id) -> removed_size`** removes the whole resting order.
  Unknown id (never existed, already cancelled, or fully filled): **error**.
* **`market(side, size) -> (fills, unfilled)`** walks the opposite side with
  no price bound. If `size` exceeds total opposite depth, fill everything
  available and **report** the remainder in `unfilled` — not an error.

Each fill records `(maker_id, price_ticks, fill_size)` in execution order.

### 2.3 Derived quantities (`NaN` guarded — never an error)

With best bid/ask prices `Pb, Pa` and *total level sizes at best* `qb, qa`:

```
mid        = (Pb + Pa) / 2                     NaN if either side empty
spread     = Pa - Pb                           NaN if either side empty
microprice = (qa * Pb + qb * Pa) / (qa + qb)   NaN if either side empty
depth(side, k) = sum of level sizes over the best k levels of side
                 (fewer levels: sum what exists; error: k <= 0, bad side)
```

### 2.4 Event stream schema (`data/lob_events.csv`)

Header exactly `seq,type,side,price,size,order_id`. `seq` is 1-based.
`type` in `{L, C, M}`; `side` in `{B, S}`. For `L`: price (**ticks**), size,
new order id (= seq in the bundled stream). For `C`: `price = 0`,
`size = 0`, id of the order to cancel (side echoes the resting side, not
used by replay). For `M`: `price = 0`, `order_id = 0`.

Pinned CSV parsing rules (all three bundled schemas — this one, `t`, and
`t,type` — and every port):

* the header line must match exactly; a trailing `\r` (CRLF files) is
  stripped from every line before any check;
* trailing blank lines are ignored; a blank line before the last data row
  is an error;
* a row has exactly the schema's field count, split on `,` with no
  surrounding whitespace (a trailing comma is an extra, empty field: error);
* integer fields match `-?[0-9]+` and fit a signed 64-bit integer; time
  fields are decimal floats that must be finite; `type` must be one of
  `L`/`C`/`M` and `side` one of `B`/`S` (validated at parse time);
* anything else is an error of the language's invalid-input family (C++:
  `std::invalid_argument` even for out-of-range integers).

Every port provides `read_events_csv` / `write_events_csv` (this schema),
`read_times_csv` / `write_times_csv` (header `t`) and
`read_marked_times_csv` / `write_marked_times_csv` (header `t,type`, types
validated to 0/1). Writers emit `\n` line endings; time writers must produce
text that parses back to the identical double (Python/C++ use `%.17g`).

**Replay** = apply events in order to a fresh book with `tick_size = 1`
through `limit_ticks` (event prices are tick counts — replay takes **no**
tick-size parameter), recording **after each event**: best bid/ask price
(NaN if none), best-level sizes (0 if none), depth over top `k = 3` levels
per side (`depth_k` positive integer, error otherwise), mid, microprice,
spread. Replay books do not record history.

### 2.5 Zero-intelligence generator (produces `lob_events.csv`, seed 20240613)

RNG = `Lcg64(seed)`. Maintain an internal book; apply each emitted event
immediately.

*Seed phase* — events 1..12: for `i = 0..5`, emit
`L B (999 - i) 10 id=seq` then `L S (1001 + i) 10 id=seq`.

*Stochastic phase* — while fewer than `n_events` (= 5000) events, with
`nb, ns` = live resting-order counts per side:

```
rates = [ limit_buy 1.0, limit_sell 1.0,
          cancel_buy 0.04*nb, cancel_sell 0.04*ns,
          market_buy 0.25, market_sell 0.25 ]        (this exact order)
x = uniform() * sum(rates); pick the first type whose cumulative rate > x
```

* limit (buy shown; sell mirrors): `u2 = uniform()`. If `u2 < 0.12` **and**
  best ask exists: `price = best_ask` (aggressive/marketable). Else
  `k = geometric(0.35)`; anchor = best ask if it exists else 1000;
  `price = anchor - (1 + k)`; then `price = max(price, 1)`.
  `size = 1 + randint(5)`; `order_id = seq`.
* cancel: `ids` = live order ids on the chosen side **sorted ascending**;
  cancel `ids[randint(len(ids))]`. (Rate is 0 when the side is empty, so
  this branch is never picked with no orders.)
* market: `u3 = uniform()`; `size = 40 + randint(40)` if `u3 < 0.02`
  (occasional sweep) else `1 + randint(3)`.

`n_events < 12`: error.

## 3. Hawkes processes (exponential kernel)

Times `t_i` and the horizon `T` are in one arbitrary time unit (the bundled
data uses seconds); `mu` and `beta` are per unit time, `alpha` is per event.

### 3.1 Univariate

`lambda(t) = mu + alpha * sum_{t_i < t} exp(-beta (t - t_i))`, params
`mu > 0`, `alpha >= 0`, `beta > 0` (error otherwise). Event-time input
validation (all likelihood/residual functions): non-empty, finite, strictly
increasing, `t_1 > 0`, `t_N <= T`, `T > 0` finite; error otherwise.

Log-likelihood on `[0, T]` — pinned O(N) recursion:

```
R_1 = 0;  R_i = exp(-beta (t_i - t_{i-1})) * (1 + R_{i-1})
ll = sum_i log(mu + alpha R_i) - mu T
     + (alpha/beta) * sum_i expm1(-beta (T - t_i))
```

Use `expm1` for the tail terms. A brute-force O(N^2) evaluator (direct
double sum) must exist for tests and must agree with the recursion to 1e-10.

### 3.2 Bivariate (pinned parameterization)

Types `m, n` in `{0 = buy, 1 = sell}`; **full 2x2 alpha, scalar shared
beta**: `lambda_m(t) = mu_m + sum_n alpha_mn * sum_{t_i^n < t}
exp(-beta (t - t_i^n))`. Validation: `mu_m > 0`, `alpha_mn >= 0`,
`beta > 0`, types in {0,1} aligned with times, else error.

Likelihood over the merged stream (events `(t_k, m_k)` in time order),
keeping decayed sums `S_0, S_1` (both start 0):

```
per event k:  w = exp(-beta (t_k - t_{k-1}))  (t_0 = 0)
              S_0 *= w; S_1 *= w
              ll += log( mu_{m_k} + alpha_{m_k,0} S_0 + alpha_{m_k,1} S_1 )
              S_{m_k} += 1
finally:      ll -= (mu_0 + mu_1) T
              ll += sum_n ((alpha_0n + alpha_1n)/beta)
                         * sum_{i: type n} expm1(-beta (T - t_i^n))
```

### 3.3 Simulation — Ogata thinning (pinned; RNG = `Lcg64(seed)`)

Univariate (error if `alpha/beta >= 1`):

```
t = 0; S = 0
loop: lam_bar = mu + alpha S              # intensity just after last step
      w = exponential(lam_bar); t += w   # 1st uniform
      if t > T: stop
      S *= exp(-beta w); lam = mu + alpha S
      if uniform() * lam_bar <= lam:     # 2nd uniform
          record t; S += 1
```

Bivariate (error if branching spectral radius >= 1): identical with
`S = (S_0, S_1)`, `lam_bar = lam_0 + lam_1` evaluated before the step, both
sums decayed by `exp(-beta w)`, and acceptance: `d = uniform() * lam_bar`;
accept iff `d <= lam_0 + lam_1`, with type `0` iff `d <= lam_0` else `1`;
then `S_type += 1`. Exactly two uniforms per loop iteration in both cases.

Bound: both simulators perform at most `MAX_CANDIDATES = 50,000,000` loop
iterations (candidate draws) and raise an error if that bound is reached
(the loop terminates almost surely but has no finite worst case).

### 3.4 MLE — pinned Nelder-Mead setup

Nelder-Mead (section 5) minimizes the negative log-likelihood over
**log-transformed** parameters (positivity by construction). Objectives are
pure (total functions that never raise): any overflow/non-finite evaluation
returns the penalty `1e10`, and so does `exp` underflow of a parameter that
must be strictly positive (`mu`, `mu_0`, `mu_1`, `beta` = 0 exactly);
`alpha = 0` by underflow is a valid (Poisson) evaluation.

* Univariate: `x = (log mu, log alpha, log beta)`, start
  `x0 = (log(0.5 N / T), log 0.5, 0)`. One NM run.
* Bivariate: `x = (log mu0, log mu1, log a00, log a01, log a10, log a11,
  log beta)`, start `mu_m = 0.5 N_m / T`, all `a_mn = 0.25`, `beta = 1`.
  **Two chained NM runs** (second starts from the first's optimum); report
  the second run's result, summed iteration counts. Error if either type is
  absent.

Report: fitted params, maximized loglik, branching `alpha/beta` (uni) or
matrix `alpha_mn/beta` plus its spectral radius (biv), convergence flag,
iterations. Stationarity diagnostics are **warnings, not errors**: warn
"near-critical" when branching (radius) `>= 0.9`, "non-stationary" when
`>= 1`; the fit still returns.

2x2 spectral radius — pinned closed form: `tr = m00 + m11`,
`det = m00 m11 - m01 m10`, `disc = tr^2 - 4 det`; if `disc >= 0`:
`max(|tr + sqrt(disc)|, |tr - sqrt(disc)|) / 2`, else `sqrt(det)`.

### 3.5 Time-rescaling residuals

Univariate: with `S_1 = 0`, `S_i = exp(-beta (t_i - t_{i-1})) (S_{i-1}+1)`:
`Lambda(t_i) = mu t_i + (alpha/beta) ((i-1) - S_i)`;
residuals `tau_i = Lambda(t_i) - Lambda(t_{i-1})` (`Lambda(t_0) = 0`).
Under the true model `tau_i ~ iid Exp(1)`; `Lambda` is nondecreasing.

Bivariate: walk the merged stream keeping decayed sums `S_n` and prior
event counts `C_n`; at each type-`m` event (before adding it),
`Lambda_m(t_k) = mu_m t_k + sum_n (alpha_mn/beta)(C_n - S_n(t_k))`.
Residuals are per-type first differences; return `[tau_buy, tau_sell]`.

## 4. Goodness of fit — KS vs Exp(1)

Sample validation: non-empty, finite, all `>= 0`; error otherwise (tiny N
is allowed — asymptotic p is indicative only, documented, not an error).

```
sort x ascending; F_i = -expm1(-x_(i))       # 1 - exp(-x), accurately
D = max_i max( i/n - F_i , F_i - (i-1)/n )
lambda = (sqrt(n) + 0.12 + 0.11/sqrt(n)) * D  # Stephens correction
if lambda >= 0.1:
    Q = 2 * sum_{j=1..100} (-1)^{j-1} exp(-2 j^2 lambda^2)
else:                                          # dual form, NR eq. 14.3.20
    Q = 1 - (sqrt(2 pi) / lambda) * sum_{j=1..100} exp(-(2j-1)^2 pi^2 / (8 lambda^2))
p = clip(Q, 0, 1)
p = 1 exactly when D = 0. ks_pvalue errors on D outside [0,1] or n <= 0.
```

Both series are pinned at exactly 100 terms. The alternating series alone is
wrong for small `lambda` (100 terms give 0.867 instead of 1.000 at
`lambda = 0.01`); the dual form is exact there and its terms underflow to 0,
so `p = 1` for e.g. `D = 1e-4, n = 100`. Both forms agree to machine
precision at the switch point.

## 5. Nelder-Mead (native in every language; scipy-compatible configuration)

Minimize `f: R^n -> double` from `x0` (error: empty/non-finite `x0`,
non-positive tolerances). Pinned:

* Coefficients: reflection 1, expansion 2, contraction 0.5, shrink 0.5.
* Initial simplex: vertex 0 = `x0`; vertex `i` multiplies coordinate `i-1`
  by 1.05 if nonzero, else sets it to 0.00025.
* Iteration (Lagarias et al. 1998): sort vertices ascending by `f`
  (**stable** — ties keep prior order, key `(f_i, index)`); centroid of all
  but worst; reflect `xr = c + (c - x_worst)`, `fr = f(xr)`.
  - `fr < f_best`: expand `xe = c + 2 (c - x_worst)`; take `xe` iff
    `fe < fr`, else `xr`.
  - `f_best <= fr < f_secondworst`: take `xr`.
  - else if `fr < f_worst`: outside contraction `xc = c + 0.5 (xr - c)`;
    accept iff `fc <= fr`, else shrink.
  - else: inside contraction `xc = c - 0.5 (c - x_worst)`; accept iff
    `fc < f_worst`, else shrink.
  - shrink: `x_i = x_best + 0.5 (x_i - x_best)` for all non-best vertices,
    re-evaluating `f`.
* Termination *before* an iteration: converged when max abs coordinate
  spread to the best vertex `< xatol` AND max abs `f` spread to the best
  `< fatol`; cap at `maxiter` iterations. Defaults (used by all Hawkes
  fits): `xatol = 1e-8`, `fatol = 1e-10`, `maxiter = 5000`.
* Result: best vertex, its `f`, iterations, `converged` (tolerance stop,
  not the cap). Fully deterministic.

## 6. Features and labels (`data/features_config.json`)

Config (pinned values in parentheses): `ofi_window` (50), `ret_window`
(20), `horizon` H (10), `epsilon` (0.25), `depth_k` (3), `hidden`
([16, 8]), `learning_rate` (0.1), `epochs` (800), `train_frac` (0.7).
Validation: positive integers / positive lr / `epsilon >= 0` /
`train_frac` in (0,1), error otherwise.

Inputs are the per-event replay states (section 2.4), 0-based index `i`.

**OFI** (Cont-Kukanov-Stoikov). Per-event contribution for `n >= 1`
(NaN if any of the four best prices involved is NaN; `e_0 = NaN`):

```
e_n =   (Pb_n >= Pb_{n-1} ? qb_n     : 0)
      - (Pb_n <= Pb_{n-1} ? qb_{n-1} : 0)
      - (Pa_n <= Pa_{n-1} ? qa_n     : 0)
      + (Pa_n >= Pa_{n-1} ? qa_{n-1} : 0)
OFI_i(w) = sum_{n=i-w+1..i} e_n        (error iff i < w: e_0 is undefined)
```

`ofi_at(states, i, w)` errors iff `i < w` (also for `w <= 0` or `i` out of
range); `i = w` is the first valid index and sums `e_1 .. e_w`. Window,
lookback and horizon are counted in events, not clock time.

**Feature vector** at event `i`, pinned column order:

```
[ OFI_i(ofi_window),
  QI_i  = (qb_i - qa_i) / (qb_i + qa_i),
  spread_i,
  depth_ratio_i = bid_depth_k_i / ask_depth_k_i,
  ret_i = mid_i - mid_{i-ret_window} ]
```

**Label** (classes pinned `0 = down, 1 = flat, 2 = up`):
`d = mid_{i+H} - mid_i`; `y = 2` if `d > epsilon`, `0` if `d < -epsilon`,
else `1`.

A row is emitted iff `i >= max(ofi_window, ret_window)`,
`i + H <= n_events - 1`, every `e_n` in the OFI window is finite,
`mid_i`, `mid_{i-ret_window}`, `mid_{i+H}`, `spread_i` are finite,
`qb_i + qa_i > 0` and `ask_depth_k_i > 0`. Rows keep replay order. Also
carried per row (for PnL): `fwd_ret = d` and `spread_i`. Zero emitted rows:
error.

**Split**: chronological, first `floor(train_frac * n)` rows train, rest
test (error if either side empty).

## 7. Neural network (from scratch — pinned, deterministic, NO RNG)

Architecture for the task: input 5 -> dense 16 ReLU -> dense 8 ReLU ->
dense 3 -> softmax. The logistic-regression baseline is the same machine
with zero hidden layers (input -> dense 3 -> softmax). Weights `W_l` have
shape `(n_in, n_out)`; activations are row vectors: `Z = A W + b`.

* **Init** (layer index `l` = 0,1,2 top to bottom; also `l = 0` for the
  logistic model): `W_l[i][j] = 0.5 * sqrt(2 / n_in) *
  sin(l * 10000 + i * n_out + j + 1)` (argument in radians, plain double
  `sin`), `b_l = 0`.
* **Standardization**: fit on the training split only; per column
  `mean = sum/n`, `std = sqrt(sum((x-mean)^2)/n)` (population, divisor n);
  if `std < 1e-12` use 1. Transform train AND test with these moments.
* **Forward**: hidden `A = max(Z, 0)`; output probabilities = softmax with
  the max of each row subtracted before `exp`. Loss = mean cross-entropy
  `-(1/n) sum_i log(max(P[i, y_i], 1e-300))`.
* **Backward** (exact equations): `G_L = (P - onehot(y)) / n`;
  `dW_l = A_{l-1}^T G_l`; `db_l = column sums of G_l`;
  `G_{l-1} = (G_l W_l^T) .* [Z_{l-1} > 0]`.
* **Training**: full-batch gradient descent — per epoch exactly one update
  `W -= lr dW`, `b -= lr db` computed on the entire training set, then
  record the loss on the entire training set (the recorded history entry
  for that epoch). No shuffling, no minibatches, no momentum,
  no regularization. Pinned `lr = 0.1`, `epochs = 800` for both MLP and
  logistic baseline.
* **Prediction**: argmax probability, ties to the lowest class index.
* Validation errors: non-finite/empty X, labels outside `[0, n_classes)`,
  `lr <= 0`, `epochs <= 0`, wrong feature count, sizes `<= 0`.

**Baselines/metrics**: majority baseline predicts the most frequent
training label (ties: smallest label). `accuracy` = matching fraction.
Confusion matrix: rows = true, cols = predicted. **Signal PnL** on the test
rows: every `y_pred` must be in `{0, 1, 2}` (error otherwise); position
`pos = y_pred - 1` in {-1, 0, +1}; `gross = sum pos * fwd_ret`;
`net = gross - sum |pos| * spread_i / 2` (half-spread cost per non-flat
signal, charged on entry only — the exit is marked at mid); also report the
trade count. PnL is in ticks per unit size. It is a signal diagnostic, not a
tradable strategy: positions from consecutive rows overlap (one signal per
event, each held `H` events); no fees, latency or fill probability.

Degenerate all-one-class labels: training must still run (loss finite and
decreasing); the majority baseline then wins — not an error.

## 8. Bundled data (regenerated by `data/generate_data.py`)

* `lob_events.csv` — ZI stream, `n = 5000`, seed `20240613` (section 2.5).
* `hawkes_uni.csv` — header `t`; `simulate_uni(mu=0.5, alpha=1.0,
  beta=1.6, T=1000, seed=101)`; 1211 events; `%.17g` formatting.
* `hawkes_biv.csv` — header `t,type`; `simulate_biv(mu=(0.3, 0.3),
  alpha=[[0.6, 0.3], [0.25, 0.5]], beta=2.0, T=600, seed=33)`; 611 events.
* `dgp_params.json` — the true parameters/seeds above plus the pinned
  loglik evaluation points.
* `features_config.json` — section 6 values.
* Generation validates the loglik recursions against brute force (1e-10)
  and the backprop gradient against central finite differences (rel. 1e-6)
  before writing goldens.

## 9. Golden cases (`data/golden/golden.json`) — 18 cases

Schema: `{"cases": [{"name", "inputs": {flat scalars}, "expect":
{flat scalars}, "tol"}]}`; compare `|computed - expected| <= tol`.
Every language's test suite must assert all of:

| # | name | expect keys | tol |
|---|------|-------------|-----|
| 1-3 | `book_snapshot_event_{1000,3000,5000}` | best_bid, best_ask, bid_size, ask_size, bid_depth_k, ask_depth_k, mid (after that 1-based event; depth k=3) | 1e-12 |
| 4 | `microprice_event_2500` | microprice | 1e-12 |
| 5 | `ofi_event_2200_w100` | ofi (window 100 ending at event 2200) | 1e-10 |
| 6 | `hawkes_uni_loglik_pinned` | loglik at mu=0.4, alpha=0.6, beta=1.2, T=1000 | 1e-8 |
| 7 | `hawkes_uni_fit` | mu, alpha, beta (section 3.4 setup) | 1e-4 |
| 8 | `hawkes_uni_branching` | branching | 1e-6 |
| 9 | `hawkes_uni_ks_fitted` | ks_stat, p_value (residuals at the fitted params) | 1e-6 |
| 10 | `hawkes_biv_loglik_pinned` | loglik at mu=(0.25,0.25), a=[[0.5,0.2],[0.2,0.5]], beta=1.8, T=600 | 1e-8 |
| 11 | `hawkes_biv_fit_alpha` | a00, a01, a10, a11 | 1e-3 |
| 12 | `hawkes_biv_spectral_radius` | spectral_radius (fitted branching matrix) | 1e-6 |
| 13 | `nn_loss_epoch1` | loss (training loss after epoch 1's update) | 1e-8 |
| 14 | `nn_loss_final` | loss (after epoch 800) | 1e-8 |
| 15 | `nn_holdout_accuracy` | accuracy (MLP, test split) | 1e-10 |
| 16 | `logistic_accuracy` | accuracy | 1e-10 |
| 17 | `majority_accuracy` | accuracy | 1e-10 |
| 18 | `signal_pnl_after_costs` | gross, net, n_trades (MLP signal, test split) | 1e-8 |

Cases 13-18 use the full pipeline: replay `lob_events.csv` -> features
(section 6 config) -> chronological 70/30 split -> standardize on train ->
train MLP (16, 8) and logistic with lr 0.1 for 800 full-batch epochs.

## 10. Required non-golden tests (every language)

Book invariants over the full replay (`best_bid < best_ask` whenever both
exist; size conservation: limit size added = rested + cancelled + maker
fills + marketable-limit taker fills; `depth(side, k) <= depth(side, k+1)`
and `depth(side, 1) = best size` at every event); FIFO and price priority
via targeted sequences; market order walking multiple levels; market order
bigger than the book (partial fill reported) and on an empty book; crossed
limit executing then resting and stopping at its own price; price
improvement (aggressive limit executes at the resting price); cancel of
unknown id errors; reuse of a cancelled id succeeds; empty-side NaN guards;
tick validation including prices that round to 0 ticks (error, book
unchanged) and large tick counts (`tick_size = 1e-8`, price 60000 rests at
exactly 6,000,000,000,000 ticks; price 1e300 errors); CSV robustness (CRLF
and trailing blank line accepted; trailing comma, non-integer field,
interior blank line, reordered header rejected); bundled streams regenerate
bit-for-bit from their seeds.
Hawkes: recursion vs brute force (1e-10, uni and biv); simulation
determinism per seed; refit of the bundled samples close to the DGP
(loose); near-critical DGP (mu=0.2, alpha=1.9, beta=2.0, T=150, seed=999)
fits with a warning; Poisson limit (`simulate_uni(1, 0, 1, 2000, 5)` refits
with branching < 0.05, converged, loglik >= Poisson MLE loglik - 1e-6, and
`uni_loglik` with alpha = 0 is independent of beta); single-event paths
(`t = [0.7], T = 1` closed form, one positive residual, KS works, `fit_uni`
returns; `t = [T]` exactly); objectives are pure under underflow (the
bivariate objective at `x = (-800, -800, 0, 0, 0, 0, -800)` returns 1e10,
`fit_biv` with `maxiter = 1` reports `converged = false`); residuals strictly
positive (compensator monotone), mean near 1; KS at true params fails to
reject (p > 0.05), wrong params reject.
KS: tiny samples (n = 1, 3) work; p in [0,1] monotone in D; small-lambda
accuracy (`ks_pvalue(1e-4, 100)` within 1e-9 of 1). NN: backprop vs central
finite differences (rel. 1e-6) on a small net, on a no-hidden-layer
logistic head and on a 2-class net; loss decreases; softmax rows sum to 1;
retraining is bit-identical; degenerate one-class labels train and lose to
majority; `signal_pnl` rejects labels outside {0, 1, 2}; validation errors
surface per the error-behavior preamble.
