# Simulator calibration: pass criteria fixed before fitting

Written before any fit was run and before the ten-symbol extract was read. The only real-data numbers known at the time were the integrity results and a few summary statistics for three symbols from the first replay. If a criterion changes after results are seen, the change goes in the log at the end with its reason, and results are reported against both versions.

This file holds method only. Data from the licensed sample file, and any number derived from it, stay in the local data directory.

## What is being tested

Whether `lobmod.sim`, fitted to one symbol's summary statistics, reproduces that symbol's market statistics. A pass would mean the simulator matches these statistics on this data. It would not make simulated strategy P&L a forecast of real P&L. Failure is an acceptable result and is reported as such.

## Data and windows

- Regular hours only. The first and last 30 minutes are dropped, leaving 10:00 to 15:30.
- Fit window: 10:00 to 12:45. Within-day check window: 12:45 to 15:30.
- Ten symbols, chosen before any result for the seven new names was seen, to spread liquidity and price: AAPL, MSFT, INTC, NVDA, TSLA, SPY, F, CSX, ZION, JBLU.
- One tick is one cent. One simulated lot is the symbol's mean displayed add size divided by 2.5, the simulator's mean limit size.
- Trade statistics use aggressive orders, meaning fills merged by `aggregate_signs`. Non-displayed executions are left out because the feed does not give their side.

## Three evaluations

1. In sample: simulated statistics against the fit window.
2. Within-day check: the same fitted parameters against the check window. Same day, same stock: a weak test of stability, and not a substitute for item 3.
3. Held-out day: the same fitted parameters against another trading day. **This needs a second file, which is not available, so this evaluation is NOT DONE.** No claim may be made that depends on it.

## Statistics and tolerances

Simulated values come from four paths of 3600 seconds, with seeds not used in the fit, averaged over paths.

| id | statistic | definition | pass if |
|---|---|---|---|
| C1 | spread distribution | time-weighted share of spread at 1, 2, 3, 4, 5+ ticks | total variation distance <= 0.15 |
| C2a | event mix | shares of adds, cancels, trades | total variation distance <= 0.10 |
| C2b | event rate | events per second | ratio within [0.75, 1.33] |
| C3 | queue size | time-weighted touch size in lots | median ratio within [0.7, 1.43] and 90th percentile ratio within [0.5, 2.0] |
| C4 | trade-sign memory | autocorrelation of aggressive-order signs at lags 1, 10, 100 | absolute gap <= 0.10, 0.05, 0.03 respectively |
| C5 | OFI R2 | R2 of 10-second mid change on 10-second order-flow imbalance | absolute gap <= 0.15 |
| C6 | volatility | sd of 10-second mid change | ratio within [0.7, 1.43] |

A symbol passes an evaluation only if every row passes. The simulator passes an evaluation only if at least 8 of 10 symbols pass. Per-row pass counts are always reported, so a partial match stays visible.

## Fitting rules

- Stage 1 sets parameters directly from moments. Stage 2 runs Nelder-Mead on a few parameters against a distance built from the same statistics, with common random numbers.
- Fit budget, the same for every symbol: 60 distance evaluations, 2 paths of 600 seconds each. A symbol whose simulated rate makes this unaffordable may scale event rates down, and that is reported.
- The fit uses the fit window only. The check window is read only after fitting finishes.
- The simulator code is not changed to help a statistic pass.

## Change log

### 2026-11-06: model v2, written before any v2 fit

The v1 result stands: it failed, and that record is not changed. The rule that simulator code is not changed to help a statistic pass applied to the v1 evaluation. v2 is a new model, judged as a new evaluation. Every tolerance and the 8 of 10 pass mark are unchanged.

Why the model changed. A look at the v1 failures, using the fit day only, found four things. The fall in activity after midday is real, not an extract fault, and much of it is one burst late in the morning, so a fixed clock-time curve would memorise that day. The Hawkes rate sat on its upper bound for every symbol. Lag-1 sign memory was too weak and no fitted parameter could move it. Prices moved mainly by exogenous jumps, so order flow explained too little of them.

What v2 adds, all off by default: a two-state activity level with a mean multiplier of one, a Markov chain for aggressive-order side, and a permanent impact of aggressive orders on fair value. Impact joins the refined parameters. Fit budget and paths are unchanged.

Windows change. v2 is fitted on the whole of 10:00 to 15:30 of the first day and judged on that same window, so the afternoon is no longer unseen. Morning and afternoon rows are still reported, for comparison with v1, but they are in-sample diagnostics. The only out-of-sample evaluation is the held-out day, the second file, now available: the same parameters against its 10:00 to 15:30 window, run once. It is reported whatever the outcome, and no change follows it.

Known limit. Order timing below one millisecond is not modelled; no criterion row depends on it, and the rate bound remains.
