# LEARN — High-Frequency Limit Order Book Modeling

This document teaches the theory behind `lobmod`, from the mechanics of a limit
order book to Hawkes processes and a from-scratch neural network, ending with
the two doses of realism every microstructure student needs: transaction costs
and what real HFT infrastructure actually looks like. Section 10 then covers the
layers added around the pinned core. Notation follows
`API_SPEC.md`; all worked numbers reproduce entries of `data/golden/golden.json`.

---

## 1. The limit order book

### 1.1 What a book is

Modern electronic markets are *double auctions running continuously*. Anyone can
post a **limit order**: a commitment to buy up to size `q` at price no worse
than `p` (a bid) or sell at no less than `p` (an ask). Orders that do not
execute immediately *rest* in the **limit order book** (LOB), organized as:

- a **bid side**: price levels below the market, best (highest) first;
- an **ask side**: price levels above the market, best (lowest) first;
- within each price level, a **FIFO queue** of resting orders.

Prices live on a discrete grid of integer multiples of the **tick size**. In
`lobmod` a price is internally a signed 64-bit integer tick count (a currency
price is converted once, at the API boundary, by rounding half up and
requiring `1 <= ticks <= 9e18` within a `1e-9` relative grid tolerance (capped at 0.01 tick) —
anything else is an error, never a silent rescale or a tick-0 order); the
event stream carries tick counts directly, and the bundled stream uses
`tick_size = 1`.

### 1.2 Price-time priority and matching

When a marketable order arrives, the matching engine allocates fills by
**price priority** (better-priced resting orders fill first) then **time
priority** (within a level, the earliest arrival fills first). Three event
types drive everything:

- **Limit add** `limit(side, price, size, id)`. If the order *crosses* — a buy
  priced at or above the best ask, or a sell at or below the best bid — it
  first executes as a taker: walk opposite levels best-first while the level
  price is within the limit, filling FIFO inside each level, partially reducing
  the front order if needed. Executions happen **at the resting order's
  price** (the passive side sets the price; the aggressor gets "price
  improvement" if it crossed by more than one level would need). Any unfilled
  remainder rests at the limit price, appended to that level's queue tail. A
  limit never trades through its own price.
- **Cancel** `cancel(id)`: removes a resting order entirely. Cancelling an
  unknown id (never existed, already cancelled, or fully filled) is an error —
  real venues reject such messages too.
- **Market order** `market(side, size)`: walks the opposite side with no price
  bound. If it exceeds the entire opposite depth, it fills everything and the
  remainder is *reported*, not an error — exactly what an exchange does when a
  sweep exhausts the book.

### 1.3 Derived quantities: mid, spread, microprice

With best bid/ask prices $P_b, P_a$ and *total sizes at those levels*
$q_b, q_a$:

$$
\text{mid} = \frac{P_b + P_a}{2}, \qquad
\text{spread} = P_a - P_b, \qquad
\text{microprice} = \frac{q_a P_b + q_b P_a}{q_a + q_b}.
$$

The microprice is a size-weighted mid that **leans toward the thin side**: if
the ask queue is small ($q_a \ll q_b$), the ask is likely to be consumed first
and the price to tick up, so the microprice sits nearer $P_a$. It is a more
useful one-line summary of the touch than the mid, and its weights match the
one-step "which queue depletes first" probabilities under a symmetric-race
approximation.

A naming warning. What this project calls `microprice` is the size-weighted
mid above. It is not the micro-price of Stoikov (2018), which is defined as the
limit of expected future mids given the current imbalance and spread and has
to be estimated from data. Stoikov introduces that quantity partly because the
size-weighted mid is noisy and is not a martingale.

All three are NaN-guarded when a side is empty (no best quote → no mid), never
an error: an empty side is a legitimate market state, e.g. after a large sweep.

**Worked example (golden `microprice_event_2500`, tol 1e-12).** After event
2500 of the bundled replay: $P_b = 986$, $P_a = 987$, $q_b = 15$, $q_a = 14$.

$$
\text{microprice} = \frac{14 \cdot 986 + 15 \cdot 987}{29}
= \frac{28609}{29} = 986.5172413793103\ldots
$$

The bid queue is slightly heavier, so the microprice sits above the mid
(986.5): buying pressure at the touch tilts the expected next move up.

**Worked example (golden `book_snapshot_event_1000`).** After event 1000:
best bid 999 (size 2), best ask 1000 (size 4), depth over the top 3 levels
21 (bid) / 27 (ask), mid 999.5. Every language must reproduce these integers
exactly — book replay is pure integer bookkeeping, so tolerance 1e-12 is really
"exact".

### 1.4 Market conventions: equity vs FX

- **Equities** trade on lit central limit order books with a regulated tick
  (US: $0.01 above $1; Europe: MiFID II tick tables keyed to price and
  liquidity band). Queue priority at large-tick stocks is enormously valuable —
  the spread is nearly always one tick, so you earn the spread by *being early
  in the queue*, which is why this project models FIFO queues explicitly.
- **FX** is an over-the-counter dealer market. The closest analogues to a CLOB
  are the primary venues (EBS, Refinitiv Matching) with their own conventions:
  prices in *pips* (a 10th of a pip as tick on modern feeds), sizes in millions
  of base currency, and **last-look**: a liquidity provider may reject your
  aggression after the fact, something a matching engine never does. Much FX
  microstructure modeling therefore treats quotes as indicative, not firm.
- Both markets share the mechanics modeled here (price-time priority within a
  venue), but equity data comes with consolidated tapes and clean event feeds
  (ITCH-style add/cancel/execute messages — the `L/C/M` schema of
  `data/lob_events.csv` covers ITCH's add (`A`), full delete (`D`) and the
  aggressor side of executions; schema v2 adds partial cancels (`X`), replace
  with priority loss (`U`) and executions against a named resting order
  (`E`/`C`), see section 10.1; hidden or iceberg orders and auctions are still
  not modeled), while FX depth data is fragmented and often quote-based.

---

## 2. Stylized facts of order flow

Real order flow is emphatically *not* Poisson. Robust empirical regularities:

1. **Clustering / burstiness.** Trades and quote changes arrive in clusters:
   the distribution of inter-event times is far heavier-tailed than exponential.
   The measure to use is dispersion: the coefficient of variation of the gaps
   is 1 for a Poisson process and 1.63 in the bundled (simulated) Hawkes sample.
   The smallest gap tells you little. With 1210 gaps averaging 0.822, a Poisson
   process produces a gap as small as the observed 0.0005 about half the time.
2. **Self-excitation.** An event raises the probability of further events:
   trades trigger cancellations, quote updates, and more trades (order
   splitting, stop-loss cascades, algos reacting to each other). This is the
   defining feature Hawkes processes capture.
3. **Cross-excitation by side.** Buy market orders excite more buy orders
   (herding, parent-order splitting) and, more weakly, sell orders (market
   makers rebalancing, mean-reversion traders). The bundled bivariate sample was
   simulated with self terms above cross terms, and the fit recovers that
   ordering: self (0.727, 0.585), cross (0.303, 0.295). That checks the
   estimator. It is not evidence about markets.
4. **Long memory of order-flow signs.** The autocorrelation of trade signs
   decays like a power law over thousands of trades (metaorder splitting) —
   one reason a single exponential kernel is an approximation (see §3.4).
5. **Mean-reverting micro-prices with predictable micro-moves.** Book
   imbalance predicts the direction of the next mid move (§4), yet prices are
   nearly unpredictable at longer horizons — the predictability lives inside
   the spread (§7).
6. **Intraday seasonality.** Activity is U-shaped over the day (open/close
   heavy). A constant baseline $\mu$ is thus only locally reasonable; serious
   fits use a time-varying $\mu(t)$ or restrict to intraday windows.

The zero-intelligence generator (`lobmod.zi`) deliberately produces only *some*
of these facts: it creates spreads and depletion events from fixed event-type
probabilities, but no self-excitation — that is what the Hawkes layer adds on
top. It is inspired by Smith, Farmer, Gillemot & Krishnamurthy (2003) and differs
from that model: prices are placed at a geometric distance off the opposite
best, and there is no event clock.

---

## 3. Point processes: from Poisson to Hawkes

### 3.1 Conditional intensity

A temporal point process is a random set of event times $0 < t_1 < t_2 < \dots$
Its **conditional intensity** is the instantaneous arrival rate given history
$\mathcal{H}_t$:

$$
\lambda(t) = \lim_{\Delta \downarrow 0}
\frac{\Pr(\text{event in } [t, t+\Delta) \mid \mathcal{H}_t)}{\Delta}.
$$

A homogeneous Poisson process has $\lambda(t) = \mu$ constant: memoryless,
exponential inter-arrivals, no clustering. Order flow needs history dependence.

### 3.2 The Hawkes process

Hawkes (1971): each past event adds a decaying bump to the intensity,

$$
\lambda(t) = \mu + \sum_{t_i < t} \phi(t - t_i),
\qquad \phi(s) = \alpha e^{-\beta s} \ (s > 0),
$$

with background rate $\mu > 0$, jump size $\alpha \ge 0$, decay $\beta > 0$.

**Branching interpretation.** A Hawkes process is exactly a Poisson cluster
process: immigrant events arrive at rate $\mu$; each event (immigrant or not)
independently spawns children at rate $\phi(t - t_{\text{parent}})$. The
expected number of children per event is the **branching ratio**

$$
n = \int_0^\infty \phi(s)\,ds = \frac{\alpha}{\beta}.
$$

- $n < 1$: subcritical, stationary; a fraction $n$ of all events are
  endogenous "echoes" of other events.
- $n = 1$: critical; every cascade dies out with probability 1, but its
  expected size is infinite and no stationary version exists with $\mu > 0$.
- $n > 1$: supercritical; cascades survive with positive probability and the
  process explodes.
- The stationary mean rate is $\bar\lambda = \mu / (1 - n)$ — the immigrant
  rate amplified by the cascade multiplier $1/(1-n)$.

**Worked example (goldens `hawkes_uni_fit`, `hawkes_uni_branching`).** The
bundled `hawkes_uni.csv` was simulated with $(\mu, \alpha, \beta) = (0.5, 1.0,
1.6)$, so true $n = 0.625$. The MLE recovers $(0.5052, 0.9795, 1.6792)$ and

$$
\hat n = \frac{0.9794946}{1.6791617} = 0.5833236 \quad (\text{tol } 10^{-6}),
$$

i.e. an estimated 58% of events are endogenous offspring. Sanity check on the
rate: $\hat\mu/(1-\hat n) = 0.5052/0.4167 \approx 1.212$ events per unit time,
matching the observed $1211/1000$.

Near-critical estimates ($n \to 1$) are common on real trade data (Hardiman-
Bercot-Bouchaud report $n \approx 0.9$–1 for E-mini futures). `lobmod` warns —
never errors — at $n \ge 0.9$ (near-critical) and $n \ge 1$ (non-stationary),
because a diagnostic should not destroy the fit that produced it.

### 3.3 Bivariate mutual excitation

For buy ($m=0$) and sell ($m=1$) market orders, the pinned parameterization is
a full $2\times 2$ excitation matrix with one shared decay:

$$
\lambda_m(t) = \mu_m + \sum_{n \in \{0,1\}} \alpha_{mn}
\sum_{t_i^n < t} e^{-\beta (t - t_i^n)}.
$$

$\alpha_{01}$ reads "how much a *sell* excites future *buys*". The branching
matrix is $N = \alpha/\beta$; stationarity requires its **spectral radius**
$\rho(N) < 1$. For a $2\times 2$ matrix the closed form is pinned: with trace
`tr` and determinant `det`,

$$
\rho = \begin{cases}
\max\left(\left|\tfrac{tr + \sqrt{tr^2 - 4\,det}}{2}\right|,
\left|\tfrac{tr - \sqrt{tr^2 - 4\,det}}{2}\right|\right) & tr^2 \ge 4\,det \\[4pt]
\sqrt{det} & \text{otherwise (complex pair).}
\end{cases}
$$

Golden `hawkes_biv_spectral_radius`: $\rho = 0.4567848$ on the fitted matrix —
comfortably stationary, self-excitation dominant (see the demo's
interpretation table).

Sharing a scalar $\beta$ across the four kernels is a pragmatic pin: it keeps
the likelihood recursion to two decayed sums, halves the parameter count, and
avoids the ridge of near-equivalent $(\alpha_{mn}, \beta_{mn})$ combinations
that makes per-pair decays hard to identify on moderate samples.

### 3.4 Why exponential kernels?

Three reasons, one honest caveat:

1. **Markov structure ⇒ O(N) everything.** With $\phi(s) = \alpha e^{-\beta s}$
   the excitation state is a *single number* $S(t) = \sum_{t_i<t}
   e^{-\beta(t-t_i)}$ that decays multiplicatively between events and jumps by 1
   at each event. Likelihood, simulation and residuals all become linear-time
   recursions. A power-law kernel has no finite-dimensional state: every
   evaluation is O(N), the likelihood O(N²).
2. **Exact simulation bound.** Between events the intensity only decays, so the
   intensity *just after the last step* is a valid upper bound for Ogata
   thinning (§3.6) — no ad hoc dominating rate needed.
3. **Closed-form compensator.** $\int \phi = \alpha/\beta$ gives the branching
   ratio and the likelihood's integral term analytically, with `expm1`
   available for the numerically delicate tails.

*Caveat:* empirical market kernels are closer to power laws
($\phi(s) \sim s^{-1.15}$ over four decades of timescales). The standard fix
retains tractability: a **sum of exponentials** $\sum_k \alpha_k e^{-\beta_k s}$
with $\beta_k$ on a geometric grid approximates a power law while keeping one
state variable per term. The single-exponential model here is the honest first
rung of that ladder.

### 3.5 Maximum likelihood

The log-likelihood of any point process observed on $[0, T]$ is

$$
\ell = \sum_i \log \lambda(t_i) - \int_0^T \lambda(s)\,ds .
$$

(Sketch: the density of "events exactly at $t_1,\dots,t_N$ and nowhere else"
is the product of intensities at events times the survival factor
$e^{-\int \lambda}$.) For the exponential Hawkes model define
$R_i = \sum_{j<i} e^{-\beta(t_i - t_j)}$. Splitting the sum at $t_{i-1}$
gives Ozaki's recursion — the term $j = i-1$ contributes 1, everything older
just decays by the common factor:

$$
R_1 = 0, \qquad R_i = e^{-\beta (t_i - t_{i-1})} (1 + R_{i-1}),
$$

and the pinned O(N) likelihood

$$
\ell = \sum_i \log(\mu + \alpha R_i) - \mu T
+ \frac{\alpha}{\beta} \sum_i \left(e^{-\beta (T - t_i)} - 1\right),
$$

where the tail terms use `expm1(-beta*(T - t_i))` — for events far from $T$
the exponential underflows to 0 and `expm1` returns exactly −1 with no
cancellation.

**Worked toy example** (check by hand): $t = (1, 2, 4)$, $T = 5$, $\mu = 0.5$,
$\alpha = 0.8$, $\beta = 1$. Then $R_1 = 0$, $R_2 = e^{-1} = 0.36788$,
$R_3 = e^{-2}(1 + e^{-1}) = 0.18512$, and

$$
\ell = \log 0.5 + \log(0.5 + 0.8 \cdot 0.36788) + \log(0.5 + 0.8 \cdot 0.18512)
- 2.5 + 0.8\,[ (e^{-4}\!-\!1) + (e^{-3}\!-\!1) + (e^{-1}\!-\!1) ]
= -5.9083646,
$$

which `uni_loglik` and the brute-force O(N²) evaluator both return to 1e-10 —
the same cross-check `generate_data.py` runs before writing goldens. Golden
`hawkes_uni_loglik_pinned` fixes $\ell = -738.033935432713$ at
$(\mu,\alpha,\beta) = (0.4, 0.6, 1.2)$ on the bundled 1211 events.

**Optimization.** The likelihood is maximized with the project's native
Nelder-Mead (scipy-compatible configuration, pinned to the last coefficient
and tie-break)
over **log-transformed parameters** $x = (\log\mu, \log\alpha, \log\beta)$:
positivity by construction, no constraint handling, and better-conditioned
steps since the likelihood is closer to quadratic in logs. Any non-finite or
overflowing evaluation returns the penalty $10^{10}$, keeping the objective
pure and total. The bivariate fit chains **two** NM runs (the second polishes
the first's optimum) — in 7 dimensions a single simplex can collapse
prematurely; the restart is a cheap, fully deterministic guard.

### 3.6 Simulation: Ogata thinning

To simulate an inhomogeneous process with intensity $\lambda(t) \le \bar\lambda$,
draw candidates from a Poisson process of rate $\bar\lambda$ and accept each
with probability $\lambda(t)/\bar\lambda$. For exponential Hawkes the bound is
free: after the state decays only downward until the next event, so with the
decayed sum $S$,

```
loop: lam_bar = mu + alpha*S          # valid ceiling until next arrival
      w  = Exp(lam_bar); t += w       # uniform #1
      if t > T stop
      S *= exp(-beta*w); lam = mu + alpha*S
      accept iff u * lam_bar <= lam   # uniform #2
      on accept: record t, S += 1
```

Exactly two uniforms per loop iteration (pinned — this is what makes the
simulated CSVs reproducible across languages). The bivariate version thins the
*total* intensity and attributes the accepted event to type 0 iff
$u\,\bar\lambda \le \lambda_0$, else type 1.

### 3.7 Goodness of fit: the time-rescaling theorem

If $\Lambda(t) = \int_0^t \lambda(s)\,ds$ is the true compensator, the theorem
says the rescaled times $\Lambda(t_i)$ form a **unit-rate Poisson process**;
equivalently, the increments

$$
\tau_i = \Lambda(t_i) - \Lambda(t_{i-1}) \; \stackrel{iid}{\sim} \; \text{Exp}(1).
$$

Intuition: $\Lambda$ stretches busy periods and compresses quiet ones until the
clock ticks at unit speed. For the exponential Hawkes model the compensator at
event times is again recursive: with $R_i$ from the likelihood recursion,

$$
\Lambda(t_i) = \mu t_i + \frac{\alpha}{\beta}\left[(i-1) - R_i\right].
$$

Model checking then reduces to a distribution test of $\{\tau_i\}$ against
Exp(1), done natively with a Kolmogorov-Smirnov test:

$$
D_n = \max_i \max\!\left(\tfrac{i}{n} - F(x_{(i)}),\; F(x_{(i)}) - \tfrac{i-1}{n}\right),
\qquad F(x) = 1 - e^{-x},
$$

computed as `-expm1(-x)` for accuracy near 0, with the Stephens-corrected
asymptotic p-value (pinned 100-term Kolmogorov series)

$$
\lambda^* = \left(\sqrt{n} + 0.12 + \tfrac{0.11}{\sqrt{n}}\right) D_n,
\qquad p = 2\sum_{j=1}^{100} (-1)^{j-1} e^{-2 j^2 \lambda^{*2}} \in [0,1].
$$

That series is the branch for $\lambda^* \ge 0.1$. It converges slowly near
zero, so below 0.1 the code switches to the dual (Jacobi theta) form of the same
function.

Golden `hawkes_uni_ks_fitted`: $D = 0.0172047$, $p = 0.8631775$ on the fitted
residuals — no rejection, and the residual mean is 0.99963 ≈ 1 as theory
demands. The demo's deliberately wrong parameters give $p \approx 2\cdot10^{-14}$,
but those are parameters nobody fitted, so that number says little about power.

Two caveats. With tiny $n$ the asymptotic p-value is only indicative. The larger
one: the p-value assumes parameters fixed in advance, and testing residuals at
*fitted* parameters makes the test far too lenient, not slightly conservative.
The fit absorbs most of the discrepancy. In the v1.2.0 review's simulation, 0 of 400 runs rejected at
5% under the true model, where a calibrated test would reject about 20, and wrong
kernels were rejected only 3–5% of the time. Read "no rejection" here as very
weak evidence. A parametric bootstrap that refits on each simulated path gives
a calibrated test; see `lobmod.research` and `research/results.md`.

---

## 4. Order-flow imbalance and short-horizon predictability

Cont, Kukanov & Stoikov (2014) showed that a single variable — **order-flow
imbalance** — explains mid-price changes remarkably well over short windows.
The idea: every book event at the best quotes shifts the supply/demand balance.
The pinned per-event contribution between consecutive states $n-1 \to n$ is

$$
e_n = \mathbb{1}\{P_b^n \ge P_b^{n-1}\}\, q_b^n
- \mathbb{1}\{P_b^n \le P_b^{n-1}\}\, q_b^{n-1}
- \mathbb{1}\{P_a^n \le P_a^{n-1}\}\, q_a^n
+ \mathbb{1}\{P_a^n \ge P_a^{n-1}\}\, q_a^{n-1},
$$

and $\text{OFI}_i(w) = \sum_{n=i-w+1}^{i} e_n$. Reading the indicators: a bid
price *improvement* or size *increase* is buying pressure ($+$); a bid retreat
or depletion is selling pressure ($-$); the ask side mirrors. When the price is
unchanged the two indicator terms on that side combine into the *size change*
$q^n - q^{n-1}$. OFI aggregates limit adds, cancels and market orders into one
signed flow number — Cont et al.'s point is that these all matter, not just
trades. Golden `ofi_event_2200_w100`: OFI over the 100 events ending at event
2200 is exactly $-46.0$ (net selling pressure in that window).

The full pinned feature vector at event $i$:

| # | feature | formula | intuition |
|---|---------|---------|-----------|
| 1 | `ofi` | $\sum e_n$ over trailing 50 events | net signed order flow |
| 2 | `queue_imbalance` | $(q_b - q_a)/(q_b + q_a)$ | which best queue is heavier, in $[-1,1]$ |
| 3 | `spread` | $P_a - P_b$ | cost of immediacy / information state |
| 4 | `depth_ratio` | bid depth$_3$ / ask depth$_3$ | imbalance beyond the touch |
| 5 | `ret` | $\text{mid}_i - \text{mid}_{i-20}$ | short-horizon momentum/reversion |

The **label** is the mid move over the next $H = 10$ events with a dead zone:
$d = \text{mid}_{i+H} - \text{mid}_i$; class 2 (up) if $d > \varepsilon$, 0
(down) if $d < -\varepsilon$, else 1 (flat), with $\varepsilon = 0.25$ ticks.
The dead zone keeps half-tick bounces of the mid (spread breathing) out of the
directional classes — without it, the classes are dominated by microstructure
noise that no feature can or should predict.

Why is any of this predictable at all? Because queues deplete mechanically: if
the ask queue is nearly empty and flow is buying, the ask *will* tick up with
high probability. This is real, robust, and well documented. Whether it is
*monetizable* is a different question — §7.

---

## 5. Machine learning on LOB data: doing it honestly

The pipeline is deliberately old-fashioned, because the discipline matters more
than the model:

1. **Chronological split, always.** First $\lfloor 0.7 n\rfloor$ rows train,
   the rest test — never a random shuffle. Microstructure features are strongly
   autocorrelated; a random split puts rows from the same burst on both sides
   and inflates test accuracy. With overlapping labels (each label looks $H$
   events ahead), shuffling is outright leakage: a test row's label window can
   overlap a training row's. (Fancier versions add a *purge/embargo gap* of at
   least $H$ events between train and test; with $H=10$ against a 1305-row
   test set the effect here is negligible, but know the technique.)
2. **Standardize on train only.** Means and (population, divisor-$n$) standard
   deviations are fitted on the training split and *applied* to both splits.
   Fitting moments on all data leaks test information into training — a subtle,
   classic mistake.
3. **Labels only from information available later; features only from
   information available now.** Every feature at row $i$ uses states
   $\le i$; the label uses state $i + H$. Rows where any input is undefined
   (NaN mid from an empty side, zero denominators, incomplete OFI window) are
   dropped rather than imputed.
4. **Baselines are non-negotiable.** The MLP is compared against multinomial
   logistic regression (identical training machinery, zero hidden layers) and
   the majority class. Goldens: MLP 0.7042145594, logistic 0.6996168582,
   majority 0.6582375479 on the 1305-row holdout. The gaps are the story: most
   of the accuracy is the flat-class base rate, and the MLP's ~0.5 point lead over
   logistic on this one split does not hold up: across 60 streams MLP minus
   logistic is −0.53 points [−0.95, −0.12], and an earlier 100-seed run gave −0.59
   (SE 0.16). What the models learn is generator
   mechanics, since spread alone gives 0.693 of the 0.704. Compare models over
   many seeds before reading anything into a gap this small.
5. **Degenerate cases must not crash.** All-one-class labels still train (loss
   finite, decreasing) and the majority baseline wins; that is a reported
   outcome, not an exception.

---

## 6. A neural network from first principles

`lobmod.nn` implements exactly the mathematics below — no frameworks, no
autograd, and (deliberately) no randomness anywhere.

### 6.1 Forward pass

Row-vector convention, $Z = A W + b$ with $W_l$ of shape
$(n_{\text{in}}, n_{\text{out}})$. For the task: input 5 → dense 16, ReLU →
dense 8, ReLU → dense 3 → softmax.

$$
A_0 = X,\qquad Z_l = A_{l-1} W_l + b_l,\qquad A_l = \max(Z_l, 0),
$$

$$
P_{ic} = \frac{e^{Z_{ic} - m_i}}{\sum_{c'} e^{Z_{ic'} - m_i}},
\quad m_i = \max_c Z_{ic}
\qquad\text{(max-shift: subtracting } m_i \text{ cancels in the ratio}
$$
$$
\text{but keeps every } e^{\cdot} \le 1 \text{ — no overflow, ever).}
$$

Loss = mean cross-entropy, with the picked probability floored at $10^{-300}$
so a confidently wrong prediction yields a huge but finite loss instead of
$-\infty$:

$$
\mathcal{L} = -\frac{1}{n} \sum_i \log \max(P_{i, y_i}, 10^{-300}).
$$

### 6.2 Backward pass — the equations as implemented

The softmax + cross-entropy head has the famous clean gradient. Derivation for
one row: $\mathcal{L}_i = -\log P_{i y_i}$ with
$P_{ic} = e^{Z_{ic}}/\sum e^{Z_{ic'}}$ gives
$\partial \mathcal{L}_i / \partial Z_{ic} = P_{ic} - \mathbb{1}\{c = y_i\}$.
Averaging over the batch:

$$
G_L = \frac{P - Y}{n} \qquad (Y = \text{one-hot labels}).
$$

Then plain chain rule, layer by layer downward:

$$
\nabla_{W_l} = A_{l-1}^{\top} G_l, \qquad
\nabla_{b_l} = \mathbf{1}^{\top} G_l \ \text{(column sums)}, \qquad
G_{l-1} = \left(G_l W_l^{\top}\right) \odot \mathbb{1}\{Z_{l-1} > 0\},
$$

the last factor being the ReLU gate (derivative 1 where the pre-activation was
positive, 0 elsewhere; the measure-zero point $Z=0$ is assigned 0 — pinned).
These four lines *are* `loss_and_grads` in every language; the test suite
verifies them against central finite differences to relative 1e-6.

### 6.3 Determinism instead of randomness

Cross-language golden reproduction of a *trained network* to 1e-8 requires zero
randomness:

- **Pinned init**: $W_l[i,j] = 0.5\sqrt{2/n_{\text{in}}}\,
  \sin(10000\,l + i\,n_{\text{out}} + j + 1)$, $b_l = 0$. The sine of distinct
  integers gives well-spread values in $[-1,1]$ (deterministic pseudo-noise);
  $\sqrt{2/n_{\text{in}}}$ comes from He scaling, but this init is smaller
  than He: the 0.5 factor and $E[\sin^2] \approx 1/2$ put the weight variance
  about 8x below $2/n_{\text{in}}$, so ReLU activations shrink layer by layer
  at the start of training rather than holding steady.
- **Full-batch gradient descent**: exactly one update per epoch on the whole
  training set, `lr = 0.1`, 800 epochs, no shuffling, minibatches, momentum or
  regularization. The recorded history entry per epoch is the training loss
  *after* that epoch's update.
- **Pinned standardization** (§5) and argmax **ties to the lowest class index**.

Goldens: loss 1.0806390310 after epoch 1 (just below the untrained
$\log 3 \approx 1.0986$ — one step already helps), 0.6760014200 after epoch
800. Any deviation at the 1e-8 level flags a divergence in matrix-multiply
order, `exp`/`sin` libm behavior, or update arithmetic in a port.

This setup trades sample efficiency for verifiability — the correct trade in a
reference implementation. Production training would use shuffled minibatches,
Adam, early stopping and multiple seeds; none of that is portable to
bit-tolerance across four languages.

---

## 7. Why transaction costs kill most microstructure signals

The pinned evaluation trades the MLP signal on every test row: position
$\text{pos} = y_{\text{pred}} - 1 \in \{-1, 0, +1\}$,

$$
\text{gross} = \sum_i \text{pos}_i \cdot (\text{mid}_{i+H} - \text{mid}_i),
\qquad
\text{net} = \text{gross} - \sum_i |\text{pos}_i| \cdot \frac{\text{spread}_i}{2}.
$$

This is a scoring rule, not a strategy. Each non-flat signal is charged one
half-spread on entry and marked mid-to-mid, with no exit cost. Signals overlap,
since each holds for $H$ events: the 177 signals form only 53 distinct runs.
Golden `signal_pnl_after_costs`: **gross +45.0 ticks, net −139.5 ticks over 177
signals**. An executable taker on the bundled stream, which crosses the spread
both ways, makes 35 round trips for −49 ticks (see `research/results.md`). The
arithmetic of the scoring rule:

- Average gross edge per trade: $45/177 \approx 0.25$ ticks.
- Average half-spread paid: $(45.0 - (-139.5))/177 = 184.5/177 \approx 1.04$
  ticks per trade. Note this is far above half the *global* mean spread
  (1.23/2 ≈ 0.6 ticks): the mean spread on the rows where the classifier
  actually fires is 2.08 ticks. Spread is one of the five features, and the
  model fires disproportionately when the spread is wide — precisely the
  states where crossing is most expensive.
- Edge < cost, therefore net < 0. The 70% accuracy is mostly the flat class, not
  directional skill: of the 177 signals actually traded, 87 were right, 63 wrong
  and 27 landed flat. Even the right ones are worth a fraction of the spread
  you pay to act on them.

And this accounting is optimistic: a taker also pays exchange fees, the far side
of the spread on exit, the impact of its own order, and latency (by the time you
act, the state that generated the signal has partly resolved). Adverse selection
is a separate cost that falls on passive fills: a resting order tends to fill
just as the price moves through it, so a strategy that tries to save the spread
by posting pays this instead. This is why documented book-imbalance
predictability coexists peacefully with efficient markets: **the signal lives
inside the bid-ask bounce**. The firms that do monetize such signals either
(a) capture the spread instead of paying it — posting limit orders and using
the signal to decide *when to lean or pull* — which converts the problem into
queue-position management, or (b) use the signal as a micro-timing overlay on
trades they had to do anyway (execution algos), where the cost is already sunk.

---

## 8. How real HFT systems differ from this project

This project is a *model laboratory*. A production HFT stack differs on every
axis:

- **Latency budget.** Tick-to-trade for competitive firms is measured in
  hundreds of *nanoseconds* (FPGA feed handlers, kernel-bypass NICs,
  pre-computed responses), not the milliseconds a Python replay implies.
  Decision logic that cannot be evaluated in a bounded, tiny time is moved
  off the hot path or into hardware.
- **Colocation and determinism of the race.** Servers sit in the exchange's
  data center with equalized cable lengths; microwave/shortwave links carry
  cross-venue signals faster than fiber. Speed decides who wins each race to
  hit a stale quote or join a queue first — a dimension absent from the pinned
  core, where events are processed one at a time. `lobmod.sim` adds an assumed
  fixed delay for *your* agent (section 10.4), with no jitter and no real venue.
- **Queue position as the asset.** With one-tick spreads, expected earnings of
  a passive order depend sharply on where it stands in the FIFO queue (early
  orders fill on benign flow; late orders fill mostly when the level is about
  to break — adverse selection *within* the level). Real systems track their
  own queue position estimate per order and value it explicitly. The pinned
  book has faithful FIFO queues and no "our" order. `lobmod.sim` places agents'
  orders in that book and logs the lots ahead at entry, against synthetic flow.
- **Feeds and state reconstruction.** Real systems build the book from
  ITCH-style market-data feeds (OUCH is the order-entry protocol, not a feed), handle A/B feed arbitration, gaps,
  retransmits, and venue-specific auction/halt states; messages can arrive out
  of order across venues. The bundled stream is clean, complete and sequenced by
  construction. `lobmod.feeds` reads recorded files, not live feeds.
- **Risk and controls.** Kill switches, fat-finger and self-trade prevention,
  position/notional limits checked in-line, exchange throttles and
  message-rate budgets, and (post-Knight-Capital) regulatory-grade deployment
  discipline.
- **Fragmentation.** One instrument trades on many venues plus dark pools;
  smart order routing, fee/rebate structures (maker-taker), and the national
  best bid/offer are a whole layer this single-book model abstracts away.

None of this invalidates the models — Hawkes intensities and OFI regressions
are estimated *offline* on captured data in real firms too — but it locates
them: research layer, not the trading loop.

### 8.1 How a desk would use this

A concrete, honest workflow for a market-microstructure research desk:

1. **Reconstruct and audit.** Convert a captured feed into v2 events with
   `lobmod.feeds` (COOKBOOK recipe 24), which maps executions, partial cancels
   and replaces to `E`, `R` and `U`. Use prices in *integer ticks* (`price_ticks = round(price /
   tick)`; the library refuses off-grid prices, prices that round to 0 ticks,
   and tick counts beyond 9e18, so a wrong `tick_size` fails loudly instead of
   silently rescaling). `replay` then rebuilds top-of-book state after every
   message; the size-conservation invariant in the tests is the check that
   the capture was complete (every unit of limit size is rested, cancelled or
   filled — a gap in the feed breaks it).
2. **Calibrate event clustering.** Extract market-order timestamps per side
   into the `t,type` schema and fit the bivariate Hawkes model
   (`fit_biv`). Read the branching matrix: the diagonal is order splitting /
   herding within a side, the off-diagonal the buy-sell feedback; the
   spectral radius says how close the flow is to criticality (0.9+ is a
   regime warning, not an error). Validate with the time-rescaling KS test
   before trusting any of the numbers — a rejected fit is a mis-specified
   kernel (try a different decay, or a sum of exponentials), not a smaller
   p-value to report.
3. **Screen signals, not strategies.** Build OFI / imbalance features on the
   replayed states and train the deterministic classifier with a
   *chronological* split. Use `signal_pnl` as a screening diagnostic only:
   it is mid-to-mid, overlapping and charges half a spread on entry. If a
   signal does not survive *that*, it will not survive fees, adverse
   selection and latency; if it does, the next step is an event-driven
   test with queue position and fill probability. `lobmod.sim` shows the shape
   of one, but calibrating it to real data failed (section 10.13), so a desk
   would need its own.
4. **Reproduce across languages.** The same goldens are asserted by the
   Python, C++, Rust and Java ports, so a research result computed in Python
   can be re-run in the production language and compared to 1e-8, which is
   the practical reason the whole pipeline is pinned.

What the library leaves to the desk: clock-time feature windows (the pinned
pipeline counts *events*), calendars and sessions, auctions and halts as book
states, and any model calibrated to real data. Timestamps, reduce, replace,
execute, fees, latency and "our" order in the queue now exist in the layers
described in section 10.

---

## 9. Common pitfalls and numerical issues

1. **`exp` vs `expm1` in likelihood tails.** For events far from $T$,
   $e^{-\beta(T-t_i)}$ underflows; computing $e^x - 1$ directly loses all
   precision for small $|x|$ too. The pinned formulas use `expm1` in every
   compensator tail. Same for the Exp(1) CDF: `-expm1(-x)`, not `1 - exp(-x)`.
2. **Log-likelihood at $\lambda \to 0$.** With $\mu$ unconstrained an optimizer
   will probe $\mu \le 0$ where $\log\lambda$ is undefined. The log-transform
   of parameters removes the boundary entirely; the $10^{10}$ penalty catches
   overflow at the other end. Never let an objective raise.
3. **Softmax overflow.** Logits of ±800 appear easily late in training;
   without the max-shift, `exp` overflows to inf and probabilities become NaN.
   With it, the largest exponent is exactly 0.
4. **Cross-entropy of a confident wrong answer.** $P_{i,y_i}$ can underflow to
   exactly 0.0; $\log 0 = -\infty$ poisons the mean. Hence the $10^{-300}$
   floor — large loss, finite gradient path.
5. **Hawkes identifiability.** $\alpha$ and $\beta$ are correlated in the
   likelihood (a taller, faster-decaying kernel mimics a shorter, slower one);
   with few events the surface has a long curved valley. Symptoms: slow
   Nelder-Mead convergence, fitted $\beta$ far from truth while $\alpha/\beta$
   is stable. The branching ratio is the robust quantity — report it.
6. **Near-critical fits.** As $\hat n \to 1$ the process is bursty at all
   scales, effective sample size collapses and $\hat\mu \to 0$ compensates.
   Treat $\hat n \ge 0.9$ as a model-adequacy flag (the pinned warning), not a
   trading insight.
7. **Time-ordering bugs.** Every likelihood/residual routine validates strictly
   increasing positive times with $t_N \le T$. Duplicate timestamps (common in
   real millisecond data) break the theory silently — deduplicate or jitter
   *before* fitting, and know which you did.
8. **Floating-point mid vs integer ticks.** Book state arithmetic here is
   integer (exact); only derived quantities are doubles. The moment you store
   prices as raw doubles (e.g. 986.01), tick-grid validation and equality
   comparisons need the 1e-9 relative-tolerance guard the spec pins — never
   `==` on float prices.
9. **Leakage via standardization or shuffling** (§5). The two most common ways
   published microstructure backtests flatter themselves.
10. **KS on estimated parameters and tiny samples.** Both make the stated
    p-value wrong. With fitted parameters it is far too large (0/400 rejections
    in the v1.2.0 review at 5% under the true model, §3.7), so use the bootstrap in `lobmod.research`.
    With small $n$ the asymptotic formula is inaccurate; such input is legal
    and the caveat is on the caller.

---

## 10. Beyond the pinned core

Sections 1 to 9 describe the pinned, four-language core. This section covers the
layers around it. Bundled data is simulated, and the IEX results in
`research/results_iex.md` are the only real-data numbers published. Other numbers
come from `research/results_models.md`, `research/results_sim.md` and
`docs/LATENCY.md`.

A note on sources in sections 10.7 to 10.11: Crossref confirmed that each cited
paper exists with the details given in the README references. That confirms the
citation, not my reading of it. Where a formula is derived here you can check it
yourself; other statements about what a paper says are from memory and are marked
so.

### 10.1 L3 events

**The idea.** A level-3 feed reports each order's life, not only the top of the
book. Taking size off an order is different from replacing it: one keeps its
place in the queue and the other goes to the back.

**Formally.** Schema v2 adds `ts_ns` and three types to `L`, `C`, `M`. For a
resting order with size $q$:

- `R` (reduce by $d$): $q \leftarrow q - d$, queue place kept.
- `E` (execute $d$): a fill at the order's own price, queue place kept for any remainder.
- `U` (replace): the old id leaves, and a new id arrives at the tail of the new
  level. A crossing replace executes first, as a limit would.

The exact edge cases are pinned in `API_SPEC.md`.

**Worked example.** Ids 1 (size 10) and 2 (size 5) rest at 100, id 1 first.
Reduce id 1 by 4 and it stays first with size 6. Execute 2 against it: size 4,
still first. Replace it with id 3 at the same price and size: the queue is now
2, 3. The price and size did not change, yet the order lost its place. COOKBOOK
recipe 13 runs this.

### 10.2 Richer Hawkes models, and why one exponential misleads

**The idea.** One exponential has a single memory length. If excitation has a
fast part and a slow part, the fit settles on a compromise, and it can call slow
excitation "baseline". Slow changes in the baseline can do the opposite and
look like excitation.

**Formally.** With a mixture of exponential densities,

$$
\lambda(t) = \mu(t) + \sum_k a_k \sum_{t_j < t} b_k e^{-b_k (t - t_j)},
\qquad n = \sum_k a_k,
$$

where $a_k$ is the expected number of children through component $k$ and $n$ is
the branching ratio. The O(N) recursion of section 3.5 runs once per component.
A power-law (Lomax) tail $(a/c)(1 + t/c)^{-(a+1)}$ is a Gamma mixture of
exponentials, so `fit_power_law` discretises it on 16 fixed rates with three
kernel parameters $(n, a, c)$. Approximating a power law with exponentials on a
geometric grid follows Bochud & Challet (2007) and Hardiman et al. (2013). A
seasonal baseline makes $\mu(t)$ piecewise constant or a linear spline, fitted
jointly with the kernel.

**Worked example** (simulated paths, T = 1000, 60 paths per case):

| Truth | Fitted model | Mean $\hat n$ | Note |
|---|---|---|---|
| two timescales, $n = 0.6$ | one exponential | 0.310 | KS rejects 3% of paths, Ljung-Box 82% |
| two timescales, $n = 0.6$ | two exponentials | 0.596 | BIC beats one exponential on 98% |
| power law, $n = 0.6$ | one exponential | 0.465 | |
| power law, $n = 0.6$ | power-law mix | 0.606 | |
| U-shaped $\mu(t)$, $n = 0.5$ | flat $\mu$, one exponential | 0.862 | |
| U-shaped $\mu(t)$, $n = 0.5$ | spline, J by BIC | 0.518 | |

So the error goes both ways, and a clean KS result does not catch it: in the first
row KS rejects almost nothing while residual dependence is plain. These fits
also knew the right family was among the candidates. A real kernel need not be.
Non-stationarity inflating the branching ratio is discussed on real data by
Filimonov & Sornette (2015); this repository has not tested that.

### 10.3 Cost-aware labels

**The idea.** "The mid went up" does not mean a taker made money, since a taker
buys at the ask and sells at the bid. Label a row by what the round trip pays.

**Formally.** Decide at $i$, enter at $e = i + \text{delay}$, exit at $x = e + H$,
with fee $c$ per side:

$$
\text{long}_i = \text{bid}_x - \text{ask}_e - 2c, \qquad
\text{short}_i = \text{bid}_e - \text{ask}_x - 2c.
$$

The label is UP if $\text{long}_i > \text{margin}$, DOWN if
$\text{short}_i > \text{margin}$, and FLAT otherwise. Rows with an empty side are
marked invalid.

**Worked example.** Ask 101 at entry, bid 103 at exit, fee 0.1: long is
$103 - 101 - 0.2 = 1.8$, so UP. If the bid at exit is 101, the mid may well have
risen, but long is $-0.2$, so FLAT. Over 30 ZI streams with no fee or delay, the
share of events where a long round trip pays is 0.011 at $H = 10$ and 0.203 at
$H = 200$. The mean of the better round trip, which needs perfect foresight, is
−0.868 ticks at $H = 10$. Models trained on these labels mostly learn not to
trade at short horizons.

### 10.4 The execution simulator

**The idea.** A backtest replays a tape that cannot react to you. `lobmod.sim`
puts agents' orders in a real book, behind a delay, among flow that responds to
the book, so that queue position and stale quotes have a cost.

**Formally.** Bivariate Hawkes market orders arrive with limit and cancel rates
that depend on touch size and imbalance. This is in the spirit of queue-reactive
models, with assumed rates. A hidden fair value jumps as a compound Poisson
process, and informed takers hit quotes left stale by a jump. An agent sees data
`md_latency` late, and its orders arrive `oe_latency` late. P&L for a fill with
signed quantity $q$, price $p$, pre-trade mid $m_0$, mid $m_h$ one horizon later
and final mid $m_T$ splits exactly:

$$
q(m_T - p) = \underbrace{q(m_0 - p)}_{\text{spread}}
+ \underbrace{q(m_h - m_0)}_{\text{adverse selection}}
+ \underbrace{q(m_T - m_h)}_{\text{inventory}}.
$$

Realised spread per lot at horizon $h$ is the mean of $\text{sign}\cdot(m_{t+h} - p)$.
Three strategies are included: a touch-joining quoter, an Avellaneda-Stoikov
quoter with a rolling horizon, and a taker that crosses stale quotes.

**Worked example.** Naive maker, 10 ms one way, 40 seeds, 600 s, zero fees:
spread 363.6, adverse −116.3, inventory 22.6, total 269.8 ticks per run (rounded
terms). About a third of what it earned at the touch went to the next second's move.

**Two caveats that change the reading.**

- *The free fair-value feed.* Agents get a delayed view of the hidden fair value
  at no cost, and that feed is what creates the cancel race. With it, the naive
  maker gains 73.2 [52.0, 94.5] ticks going from 100 ms to 2 ms. Without it the
  same difference is −10.8 [−32.1, 10.5]: the latency effect is not
  distinguishable from zero. So the result is about the value of a fast signal
  in this model. It is not a measurement of what latency is worth.
- *Maker levels are not an edge.* Every parameter was set by judgement, and the
  background book is not competitive, so nothing drives maker profit to zero
  except the optional fast rival. Positive P&L reflects the assumed share of
  uninformed flow. Only signs and orderings inside the model are worth reading.

An older zero-intelligence replay showed P&L rising with a looser inventory cap,
362 ticks at cap 20. A control was run to test whether mean reversion in the ZI mid
explains that. It was inconclusive ([results_control.md](research/results_control.md)):
mean reversion is only partly supported, and the cause is not settled.

### 10.5 How FastBook differs

**The idea.** The reference book favours clarity. Its cancel scans a queue, and
its containers allocate. FastBook keeps the same matching rules and pays for
speed by fixing its limits in advance. It is fast for three plain reasons. Nothing
is searched, because an id leads straight to a node and a price straight to a
ladder entry. Nothing is allocated. And a level's FIFO is linked through the
nodes, so removing an order touches only its two neighbours.

**Formally.** It has a flat ladder with one entry per tick in a fixed band, a
hierarchical bitset to find the next live level, a node pool with an intrusive
doubly linked FIFO per level (O(1) unlink), and an open-addressing id map. All
memory is taken at construction. A limit outside the band returns an error even
if it would have crossed, a full pool refuses every limit, prices are ticks
only, and fills go to a caller's callback. `rest`, `reduce` and `replace` exist so
that the ITCH decoder can mirror a feed, where the venue has already matched.
There is no `execute` that reports a fill, and no Java or Python version. The
layout is the FastBook diagram in `docs/ARCHITECTURE.md`.

**Worked example.** From `docs/LATENCY.md`, on an idle laptop under WSL2, book
only: reverse cancel of 100,000 orders at one price costs a mean 43,616 / 44,351
ns per cancel for the C++ reference book (two runs) and 12 / 9 ns for FastBook.
Throughput on the mixed script is about 7.5x in C++ and 4.2x to 4.5x in Rust. At
p50 a passive add takes 32 to 35 ns against 115 to 125 ns for the C++ reference
and 34 ns against 89 to 90 ns for the Rust one. None of these figures is
wire-to-wire.

Measure on a quiet machine. An add is cheap in the reference book when the level
exists, since it appends to a queue and updates a map, so its gain is smaller than
cancel's: 2.6x to 3.9x at p50 against 3x to 5.4x. Timings taken while other work
was running hid that gain, and they are not quoted here.

### 10.6 One order, from feed message to P&L

**The idea.** Follow a single order through every layer: bytes, event, book, fill,
accounting.

**Formally.** ITCH message `A` becomes event `L`, `E` becomes event `E` plus a trade
record, and `D` becomes `C`. The execution message carries no price, since the
order it names already has one, so the book supplies the price. The attribution is
the three-term split of section 10.4.

**Worked example.** These messages are built by hand. Prices have four implied
decimals, so one unit is 0.0001 dollars. We follow order 1, a bid for 100 shares at
100.00. Suppose it is ours.

```python
import io
import struct

from lobmod.feeds import BookIntegrityChecker, FeedEvent, Itch50Parser

def head(kind, ts_ns):
    return kind + struct.pack(">HH", 1, 0) + ts_ns.to_bytes(6, "big")

def frame(msg):
    return struct.pack(">H", len(msg)) + msg

stock = b"TEST    "

def add(ts, oid, side, size, px):
    return head(b"A", ts) + struct.pack(">QcI", oid, side, size) + stock + struct.pack(">I", px)

messages = [
    head(b"R", 1) + stock + b"Q" + bytes(19),         # directory: locate 1 is TEST
    add(2, 1, b"B", 100, 1_000_000),                  # our bid, 100 at 100.00
    add(3, 2, b"S", 100, 1_000_200),                  # an offer at 100.02
    head(b"E", 4) + struct.pack(">QIQ", 1, 30, 77),   # a seller takes 30 of ours
    add(5, 3, b"S", 50, 1_000_100),                   # a better offer arrives
    head(b"D", 6) + struct.pack(">Q", 3),             # and is pulled
]
parser = Itch50Parser(symbols=["TEST"])
checker = BookIntegrityChecker()
mids = []
for item in parser.parse(io.BytesIO(b"".join(map(frame, messages)))):
    if isinstance(item, FeedEvent):
        checker.apply(item.event)
        mids.append(checker.book.mid())
print(mids[2:])                                              # [1000100.0, 1000050.0, 1000100.0]
print(checker.book.best_bid(), checker.book.best_bid_size())   # 1000000 70

q, p = 30, 1_000_000
m0, mh, mT = 1_000_100, 1_000_050, 1_000_100
print(q * (m0 - p), q * (mh - m0), q * (mT - mh), q * (mT - p))   # 3000 -1500 1500 3000
```

The order rests with 100, and the execution leaves 70 in the same queue place. We
bought 30 at 100.00 when the mid was 100.01, so spread capture is 3,000 units, or
$0.30. The mid then fell half a cent, which costs 1,500 units of adverse selection.
It came back by the end, for +1,500 of inventory. Total 3,000, which equals
$q(m_T - p)$, as it must. Here the next mid change stands in for the one-horizon
move, and we still hold 30 shares marked at a mid we might not be able to trade at.

### 10.7 Market making: inventory and adverse selection

**The idea.** A market maker earns the spread when buyers and sellers alternate.
Two things take it back. Inventory: fills do not balance, and a position held is a
bet. Adverse selection: the people who hit your quote may know something, so fills
tend to come just before the price moves against you.

**Formally.** Avellaneda and Stoikov shade quotes around a reservation price. With
mid or reference $s$, inventory $q$, risk aversion $\gamma$, variance rate
$\sigma^2$, horizon $\tau$ and fill-rate decay $\kappa$, the form coded in
`sim.AvellanedaStoikov` is

$$
r = s - q\gamma\sigma^2\tau, \qquad
\delta = \frac{\gamma\sigma^2\tau}{2} + \frac{1}{\gamma}\ln\left(1 + \frac{\gamma}{\kappa}\right),
$$

quoting $r \pm \delta$ rounded outward to ticks. The paper has a terminal date.
Here $\tau$ is a fixed rolling horizon, which is a simplification. Inventory
models of dealer quotes go back at least to Ho and Stoll (1981).

**Worked example.** With the defaults $\gamma = 0.1$, $\sigma^2 = 0.7$, $\tau = 5$,
$\kappa = 0.5$, we get $\gamma\sigma^2\tau = 0.35$ and $\delta = 0.175 + 1.823 =
1.998$. With $s = 10000$ and $q = 0$ the code quotes 9998 and 10002. At $q = 2$,
$r = 9999.3$, and it quotes 9997 and 10002: a lower bid, so it buys less, while the
ask rounds to the same tick. In `lobmod.sim` the two risks show up by name:
`inventory` and `adverse` in the attribution, and informed takers and
`p_informed` as their source. Recipe 21 turns the informed flow off, and the
adverse term changes sign.

### 10.8 Kyle: how much a trade should move the price

**The idea.** A dealer cannot tell an informed order from noise, so every order
moves the price a little. The less noise there is, the more each order says.

**Formally.** One period. Value $v \sim N(\mu, \sigma_v^2)$, noise $u \sim N(0,
\sigma_u^2)$, and an insider sends $x$. The dealer sees $y = x + u$ and sets
$p = \mu + \lambda y$. The insider maximises $x(v - \mu - \lambda x)$, so
$x = \beta(v - \mu)$ with $\beta = 1/(2\lambda)$. Dealer pricing needs
$\lambda = \operatorname{cov}(v, y)/\operatorname{var}(y) =
\beta\sigma_v^2/(\beta^2\sigma_v^2 + \sigma_u^2)$. Solving the two gives
$\lambda = \sigma_v/(2\sigma_u)$.

**Worked example.** $\sigma_v = 2$ dollars and $\sigma_u = 100$ shares give
$\lambda = 0.01$: a net 50 shares of buying moves the price 50 cents. Double the
noise and the move halves. This repository has no Kyle model. The closest thing is
regressing mid changes on OFI, as in section 4.

### 10.9 Glosten-Milgrom: why spreads exist with no costs

**The idea.** A dealer quotes so that it breaks even against whoever turns up. Since
a buy is more likely when the news is good, the ask must sit above the current
expected value. The spread is the price of being picked off.

**Formally.** Value is $H$ or $L$ with equal chance. A share $\pi$ of traders knows
which; the rest buy or sell at random. Then $P(H \mid \text{buy}) = (1 + \pi)/2$, so
the ask is $E[v \mid \text{buy}] = L + (H - L)(1 + \pi)/2$, the bid mirrors it, and
the spread is $\pi(H - L)$.

**Worked example.** $H = 101$, $L = 99$, $\pi = 0.2$: ask 100.2, bid 99.8, spread
0.4. After a buy, the dealer's belief in $H$ rises to 0.6, and quotes move up. So
quotes learn from trades. `lobmod.sim` has informed takers but no competitive
dealer, so its spreads are not set by this break-even rule.

### 10.10 Square-root impact

**The idea.** A large order moves the price less than in proportion to its size:
four times the size costs about twice the impact.

**Formally.** From memory of the empirical literature, for example Tóth et al.
(2011): $I(Q) \approx Y \sigma \sqrt{Q/V}$, with $\sigma$ daily volatility, $V$ daily
volume and $Y$ a constant of order one. It is an empirical regularity for
metaorders, which are parent orders split over time, and it is not derived here.

**Worked example.** $\sigma = 2\%$ and $Q/V = 1\%$ give $I \approx Y \times 0.2\%$,
or about 20 basis points if $Y = 1$. At $Q/V = 4\%$ it is about 40. Nothing here
implements or tests this: `lobmod.sim` has no metaorders, and its impact comes only
from walking the book.

### 10.11 Lee-Ready: who was the aggressor?

**The idea.** Many trade files do not say which side initiated a trade, so you
infer it from where the trade printed.

**Formally.** The quote rule says a trade above the mid is a buy, and below it a
sell. A trade at the mid falls to the tick test: a buy if the price rose from the
last different price, a sell if it fell. Lee and Ready (1991) also dealt with the
delay between quotes and trades. Timestamp problems in fast markets are the
subject of Holden and Jacobsen (2014), which I cite from memory
(https://doi.org/10.1111/jofi.12127; the record is on Crossref).

**Worked example.** Quotes 100.00 and 100.02, so the mid is 100.01. A trade at 100.02
is a buy. A trade at 100.01 after a last different price of 100.00 is a buy by the
tick test. This repository mostly does not need the rule: an ITCH or DEEP+ execution
names a resting order, so the aggressor is the other side. `research/iex_replay.py`
uses only the first half of it, for fills with no side, and drops a fill at the mid.

### 10.12 How to read a backtest honestly

**The idea.** A backtest is a claim, and most claims fail for dull reasons. Each of
these mistakes was made, or nearly made, in this repository.

1. **Overlapping trades.** `signal_pnl` opens a trade on each signal with no
   position limit, so holds overlap, and 177 trades are not 177 independent
   observations. Count effective bets, and use `run_backtest`, which pays the
   spread both ways.
2. **One seed.** On the bundled split the MLP beats logistic. Across 60 streams
   MLP minus logistic is −0.53 points [−0.95, −0.12]. Ask how the result varies across
   seeds, before you ask why it won.
3. **Weak goodness of fit with fitted parameters.** KS on fitted parameters
   rejected 0 of 400 true models and only 3 to 5% of wrong kernels in the v1.2.0
   review. "Not rejected"
   was close to a statement about the test. Use a test that refits, like the
   bootstrap.
4. **A horizon no taker can win.** At $H = 10$ a long round trip pays on 1.1% of
   events, and even perfect foresight averages −0.868 ticks. Accuracy at that
   horizon is unrelated to money. Work out the best possible P&L before you model.
5. **Calibration that failed.** The criteria were written first, and the attempt
   failed (section 10.13). Had the targets been chosen afterwards, it might have
   looked fine. Write the pass mark down first, and report the failure.
6. **A control with no verdict.** The +362 control's gate could not be met at any
   setting, so it gave no verdict under its rule. The honest report is
   "inconclusive", even though a story was available.

Before believing a result, ask: what would have made this fail, and was that
written down before the run?

### 10.13 What was run on real data

Data provided for free by IEX. By accessing or using IEX Historical Data, you agree to the IEX Historical Data Terms of Use. Terms: <https://www.iex.io/products/market-data-connectivity/hist-terms>.

One IEX DEEP+ file, 12 December 2025, was replayed for AAPL, MSFT and INTC with no gaps or anomalies reported, every displayed execution at the touch and an empty book at the end of the day. Those are the only real-data numbers published, in [research/results_iex.md](research/results_iex.md). They describe IEX's own thin displayed book, which is not the NBBO, on one day.

The parser and book were replayed on public ITCH 5.0 sample files for ten symbols with no parser or book anomalies and every execution at the touch. That is internal consistency, not comparison with a venue snapshot. Raw data is not redistributed.

Calibration of `lobmod.sim` to real data was attempted against criteria fixed in advance ([research/calibration_criteria.md](research/calibration_criteria.md)) and FAILED: it did not meet the criteria in sample, parameters did not carry across the day, and the held-out day was not run. The simulator remains uncalibrated and its P&L levels mean nothing.

Hawkes on real trades, qualitative only: a seasonal two-timescale model is preferred over a single exponential, but the fast rate hits its bound and branching is not resolved, so no value is quoted. Sub-millisecond burst structure cannot be attributed to a mechanism from this feed. The "Branching against
seasonality" diagram in `docs/ARCHITECTURE.md` shows why the two explanations are
hard to separate.

---

## 11. Knowledge check

**Q1. Why can the size-weighted mid (this project's `microprice`) say more
about the next move than the mid?** Because it weights the quotes by *opposite* queue sizes:
$(q_a P_b + q_b P_a)/(q_a+q_b)$. If the ask queue is thin it depletes first, so
the price is more likely to tick up, and the microprice already leans that way.
The mid throws queue information away. It is still not Stoikov's (2018)
micro-price, which is the limit of expected future mids and is built to be a
martingale; the size-weighted mid is not one.

**Q2. What is the branching ratio of a Hawkes process and why do practitioners
care?** $n = \int \phi = \alpha/\beta$: expected offspring per event. It splits
activity into exogenous ($1-n$ fraction) and endogenous ($n$) components,
determines stationarity ($n<1$), and amplifies the base rate to
$\mu/(1-n)$. Some published estimates near 1 are read as high endogeneity, with
most events echoes of others. Those estimates depend heavily on the baseline and
kernel chosen, and any link to episodes such as flash crashes is a hypothesis.

**Q3. Derive the Hawkes likelihood recursion.** Split
$R_i = \sum_{j<i} e^{-\beta(t_i-t_j)}$ at $j = i-1$: the $j=i-1$ term is
$e^{-\beta(t_i-t_{i-1})}$; every older term equals its value at $t_{i-1}$
multiplied by the same factor. Hence
$R_i = e^{-\beta(t_i-t_{i-1})}(1 + R_{i-1})$, $R_1 = 0$, and the sum of
$\log(\mu + \alpha R_i)$ plus the closed-form compensator gives $\ell$ in O(N).
Only works because the exponential kernel is memoryless in the decayed sum.

**Q4. How does Ogata thinning work, and why is the exponential kernel
convenient there?** Simulate candidates from a homogeneous ceiling rate
$\bar\lambda$, accept with probability $\lambda(t)/\bar\lambda$. With
exponential decay the intensity between events only falls, so the current
intensity *is* a valid ceiling — refreshed after every candidate, no external
bound needed. Rejected candidates still advance time (their memorylessness
makes the ceiling-rate process Poisson piecewise).

**Q5. How do you test whether a fitted point-process model is adequate?**
Time-rescaling: transform event times through the fitted compensator
$\Lambda$; increments must be iid Exp(1). Check with a KS test (plus a mean
≈ 1 sanity check and monotone $\Lambda$). Rejection means misspecification —
wrong kernel shape, missing marks, or nonconstant baseline. No rejection means
little when the same data estimated the parameters: in the v1.2.0 review that test rejected
0/400 true models and only 3–5% of wrong kernels, so refit inside a parametric
bootstrap (`lobmod.research`).

**Q6. Define OFI and explain why it beats trade imbalance.** OFI sums signed
best-quote supply/demand changes: bid improvements/additions minus bid
retreats/depletions, minus the mirrored ask terms. Trade imbalance sees only
market orders; OFI also counts limit-order additions and cancellations, which
move quotes just as surely. Cont et al. found mid-price changes approximately
linear in OFI with $R^2$ far above trade-count regressions.

**Q7. Your LOB classifier scores 70% out-of-sample. Why might it still lose
money?** (a) The base rate: majority class alone scores 66% — the model's
apparent edge is a few points, and of the 177 signals traded only 87 were right
(63 wrong, 27 flat). (b) The predicted moves are sub-spread: ~0.25
ticks gross per trade vs ~1.04 ticks half-spread cost on the rows it trades
(the signal fires when spreads are wide) — net negative (golden: +45.0 gross,
−139.5 net over 177 trades). (c) Exit costs, impact and latency make it worse
for a taker, and posting instead swaps the spread for adverse selection on
passive fills. Accuracy is not PnL.

**Q8. Why chronological rather than random train/test splits here?** Serial
dependence and overlapping labels. Random splits interleave train and test
rows from the same excitation burst, and with $H$-step-ahead labels a test
label window can overlap training inputs — direct leakage. Chronological
splits (ideally with an embargo ≥ $H$) measure what matters: performance on a
genuinely later period.

**Q9. Why does this project initialize network weights with a sine formula
instead of random draws?** Cross-language golden tests require the *trained*
network to match to 1e-8, so training must be bit-deterministic in four
languages: no RNG, no shuffling, full-batch updates. $\sin$ of distinct
integers provides fixed, well-spread pseudo-noise. The scale borrows He's
$\sqrt{2/n_{in}}$, but with the 0.5 factor the variance is about 8x below He. The lesson generalizes: determinism is
a testing tool; production uses random init over multiple seeds.

**Q10. A market order arrives for more size than the entire opposite side
holds. What should the matching engine do?** Fill everything available,
level by level FIFO, at each resting order's price, and report the unfilled
remainder — not an error. The book may be left one-sided; downstream
consumers must NaN-guard mid/spread/microprice. (This exact path is a pinned
edge case and test in every language.)

---

## 12. Topics not covered

This lab stops well short of the field. These are the next things to read, and
they are not implemented here, except where an entry says otherwise:

- **Almgren-Chriss (2001)**: optimal execution against impact and risk.
- **Propagators**: how impact decays over time. Square-root impact has a short
  section at 10.10.
- **Queue-reactive models** (Huang, Lehalle & Rosenbaum, 2015): rates that depend
  on book state. The simulator's flow is in that spirit, with assumed rates.
- **Hasbrouck (1991, 1995)**: VAR models of trades and quotes, and information
  shares.
- **The Epps effect**: correlations that shrink at short sampling intervals.
- **Marked Hawkes processes**: excitation that depends on order size.

Kyle, Glosten-Milgrom, Avellaneda-Stoikov and Lee-Ready now have short sections
(10.7 to 10.11), but the first three are not derived in full.

---

## 13. Further reading

- Hawkes (1971), *Spectra of some self-exciting and mutually exciting point
  processes*, Biometrika — the original.
- Ogata (1981), *On Lewis' simulation method for point processes*, IEEE IT —
  thinning. Ozaki (1979), Ann. Inst. Statist. Math. — the likelihood recursion.
- Bacry, Mastromatteo & Muzy (2015), *Hawkes processes in finance*, Market
  Microstructure and Liquidity — the survey to read after this document.
- Hardiman, Bercot & Bouchaud (2013), *Critical reflexivity in financial
  markets*, Eur. Phys. J. B — near-critical branching in real data.
- Bochud & Challet (2007), *Optimal approximations of power laws with
  exponentials*, Quantitative Finance — the exponential-mixture device.
- Filimonov & Sornette (2015), *Apparent criticality and calibration issues in
  the Hawkes self-excited point process model*, Quantitative Finance.
- Avellaneda & Stoikov (2008), *High-frequency trading in a limit order book*,
  Quantitative Finance — the quoting formula behind `sim.AvellanedaStoikov`.
- Brown et al. (2002), *The time-rescaling theorem and its application*,
  Neural Computation — the GOF methodology, beautifully explained.
- Cont, Kukanov & Stoikov (2014), *The price impact of order book events*,
  J. Financial Econometrics — OFI.
- Cont, Stoikov & Talreja (2010), *A stochastic model for order book
  dynamics*, Operations Research — queueing view of the LOB.
- Smith, Farmer, Gillemot & Krishnamurthy (2003), *Statistical theory of the
  continuous double auction*, Quantitative Finance — zero-intelligence flow.
- Gould et al. (2013), *Limit order books*, Quantitative Finance — broad LOB
  survey. Bouchaud, Bonart, Donier & Gould (2018), *Trades, Quotes and
  Prices*, CUP — the modern textbook.
- Zhang, Zohren & Roberts (2019), *DeepLOB*, IEEE TSP — deep learning on LOBs
  done at scale; contrast its CNN/LSTM with our deliberately tiny MLP.
- Kyle (1985), *Continuous auctions and insider trading*, Econometrica; Glosten &
  Milgrom (1985), J. Financial Economics; Lee & Ready (1991), J. Finance; Tóth et
  al. (2011), Physical Review X. Full details are in the README references.
- Harris (2003), *Trading and Exchanges* — market mechanics and conventions.
- Stephens (1970), J. Royal Stat. Soc. B — the KS small-sample correction;
  Press et al., *Numerical Recipes*, §14.3 — the pinned p-value form.
- Lagarias, Reeds, Wright & Wright (1998), SIAM J. Optim. — the Nelder-Mead
  variant pinned here.
