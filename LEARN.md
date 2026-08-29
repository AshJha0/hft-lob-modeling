# LEARN — High-Frequency Limit Order Book Modeling

This document teaches the theory behind `lobmod`, from the mechanics of a limit
order book to Hawkes processes and a from-scratch neural network, ending with
the two doses of realism every microstructure student needs: transaction costs
and what real HFT infrastructure actually looks like. Notation follows
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
`lobmod` a price is internally an integer tick count; the bundled stream uses
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
and the price to tick up, so the microprice sits nearer $P_a$. It is a better
short-horizon fair-value estimate than the mid, and its weights are exactly the
one-step "which queue depletes first" probabilities under a symmetric-race
approximation.

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
  (ITCH-style add/cancel/execute messages — exactly the `L/C/M` schema of
  `data/lob_events.csv`), while FX depth data is fragmented and often quote-based.

---

## 2. Stylized facts of order flow

Real order flow is emphatically *not* Poisson. Robust empirical regularities:

1. **Clustering / burstiness.** Trades and quote changes arrive in clusters:
   the distribution of inter-event times is far heavier-tailed than exponential.
   In the bundled Hawkes sample the minimum gap is 0.0005 vs a mean of 0.822 —
   a ratio no homogeneous Poisson process would plausibly produce over 1211 events.
2. **Self-excitation.** An event raises the probability of further events:
   trades trigger cancellations, quote updates, and more trades (order
   splitting, stop-loss cascades, algos reacting to each other). This is the
   defining feature Hawkes processes capture.
3. **Cross-excitation by side.** Buy market orders excite more buy orders
   (herding, parent-order splitting) and, more weakly, sell orders (market
   makers rebalancing, mean-reversion traders). The bivariate fit on the bundled
   data shows exactly this: self terms (0.727, 0.585) dominate cross terms
   (0.303, 0.295).
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
of these facts: it creates realistic book shapes, spreads and depletion events
from Poisson-type rates (Smith-Farmer style), but no self-excitation — that is
what the Hawkes layer adds on top.

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
- $n \ge 1$: supercritical; cascades don't die out, the process explodes.
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
Nelder-Mead (P02 configuration, pinned to the last coefficient and tie-break)
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
event times is again recursive: with $S_i$ from the likelihood recursion,

$$
\Lambda(t_i) = \mu t_i + \frac{\alpha}{\beta}\left[(i-1) - S_i\right].
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

Golden `hawkes_uni_ks_fitted`: $D = 0.0172047$, $p = 0.8631775$ on the fitted
residuals — no rejection, and the residual mean is 0.99963 ≈ 1 as theory
demands. The demo's deliberately wrong parameters give $p \approx 2\cdot10^{-14}$:
the test has real power. Two caveats are documented, not hidden: with tiny $n$
the asymptotic p-value is only indicative, and testing residuals at *fitted*
(not true) parameters makes the test slightly conservative — the estimation
step uses up some of the discrepancy.

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
   of the accuracy is the flat-class base rate; the nonlinearity adds ~0.5% over
   linear. That ordering — small, honest increments over strong baselines — is
   what real microstructure results look like.
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
  $\sqrt{2/n_{\text{in}}}$ is He scaling, which keeps ReLU activation variance
  roughly constant across layers.
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

The half-spread per non-flat signal is the *floor* cost of crossing the spread
on entry, marked mid-to-mid. Golden `signal_pnl_after_costs`: **gross +45.0
ticks, net −139.5 ticks over 177 trades**. The arithmetic is brutal and general:

- Average gross edge per trade: $45/177 \approx 0.25$ ticks.
- Average half-spread paid: mean spread ≈ 1.23 ticks ⇒ ≈ 0.6 ticks per trade.
- Edge < cost, therefore net < 0. Predicting direction with 70% accuracy is
  *not enough* when most correct predictions are worth a fraction of the
  spread you pay to act on them.

And this floor is optimistic: real trading adds exchange fees, adverse
selection (your marketable orders fill exactly when the queue is about to move
against you — the fills you get are worse than the average row), impact of your
own order, and latency (by the time you act, the state that generated the
signal has partly resolved). This is why documented book-imbalance
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
  hit a stale quote or join a queue first — a dimension entirely absent here,
  where events are processed one at a time with no notion of *your* latency.
- **Queue position as the asset.** With one-tick spreads, expected earnings of
  a passive order depend sharply on where it stands in the FIFO queue (early
  orders fill on benign flow; late orders fill mostly when the level is about
  to break — adverse selection *within* the level). Real systems track their
  own queue position estimate per order and value it explicitly. Our simulator
  has faithful FIFO queues but no concept of "our" order competing with
  others' reactions.
- **Feeds and state reconstruction.** Real systems build the book from
  ITCH/OUCH-style message feeds, handle A/B feed arbitration, gaps,
  retransmits, and venue-specific auction/halt states; messages can arrive out
  of order across venues. Here the event stream is clean, complete and
  sequenced by construction.
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
    p-value approximate (conservative for fitted params; asymptotic formula
    inaccurate for small $n$). The implementation documents rather than hides
    this — small-$n$ input is legal, the caveat is on the caller.

---

## 10. Interview-style Q&A

**Q1. Why is the microprice a better short-horizon predictor than the mid?**
Because it weights the quotes by *opposite* queue sizes:
$(q_a P_b + q_b P_a)/(q_a+q_b)$. If the ask queue is thin it depletes first, so
the price is more likely to tick up, and the microprice already leans that way.
The mid throws queue information away; empirically the microprice is closer to
a martingale at event scale.

**Q2. What is the branching ratio of a Hawkes process and why do practitioners
care?** $n = \int \phi = \alpha/\beta$: expected offspring per event. It splits
activity into exogenous ($1-n$ fraction) and endogenous ($n$) components,
determines stationarity ($n<1$), and amplifies the base rate to
$\mu/(1-n)$. Market estimates near 1 imply most events are echoes of other
events — high endogeneity, fragility to feedback loops (flash crashes).

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
wrong kernel shape, missing marks, or nonconstant baseline. Remember the test
is conservative when the same data estimated the parameters.

**Q6. Define OFI and explain why it beats trade imbalance.** OFI sums signed
best-quote supply/demand changes: bid improvements/additions minus bid
retreats/depletions, minus the mirrored ask terms. Trade imbalance sees only
market orders; OFI also counts limit-order additions and cancellations, which
move quotes just as surely. Cont et al. found mid-price changes approximately
linear in OFI with $R^2$ far above trade-count regressions.

**Q7. Your LOB classifier scores 70% out-of-sample. Why might it still lose
money?** (a) The base rate: majority class alone scores 66% — the model's
genuine edge is a few points. (b) The predicted moves are sub-spread: ~0.25
ticks gross per trade vs ~0.6 ticks half-spread cost — net negative (golden:
+45.0 gross, −139.5 net). (c) Adverse selection and latency degrade realized
fills below backtest assumptions. Accuracy is not PnL.

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
integers provides fixed, well-spread pseudo-noise; the $\sqrt{2/n_{in}}$
factor is standard He scaling for ReLU. The lesson generalizes: determinism is
a testing tool; production uses random init over multiple seeds.

**Q10. A market order arrives for more size than the entire opposite side
holds. What should the matching engine do?** Fill everything available,
level by level FIFO, at each resting order's price, and report the unfilled
remainder — not an error. The book may be left one-sided; downstream
consumers must NaN-guard mid/spread/microprice. (This exact path is a pinned
edge case and test in every language.)

---

## 11. Further reading

- Hawkes (1971), *Spectra of some self-exciting and mutually exciting point
  processes*, Biometrika — the original.
- Ogata (1981), *On Lewis' simulation method for point processes*, IEEE IT —
  thinning. Ozaki (1979), Ann. Inst. Statist. Math. — the likelihood recursion.
- Bacry, Mastromatteo & Muzy (2015), *Hawkes processes in finance*, Market
  Microstructure and Liquidity — the survey to read after this document.
- Hardiman, Bercot & Bouchaud (2013), *Critical reflexivity in financial
  markets*, Eur. Phys. J. B — near-critical branching in real data.
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
- Harris (2003), *Trading and Exchanges* — market mechanics and conventions.
- Stephens (1970), J. Royal Stat. Soc. B — the KS small-sample correction;
  Press et al., *Numerical Recipes*, §14.3 — the pinned p-value form.
- Lagarias, Reeds, Wright & Wright (1998), SIAM J. Optim. — the Nelder-Mead
  variant pinned here.
