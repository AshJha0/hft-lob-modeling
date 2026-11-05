Data provided for free by IEX. By accessing or using IEX Historical Data, you agree to the IEX Historical Data Terms of Use.

Terms: <https://www.iex.io/products/market-data-connectivity/hist-terms>

# IEX replay: one day, three symbols

Written by hand from the JSON summary that `research/iex_replay.py` produced. These are the only real-data numbers published in this repository. No Nasdaq-derived number appears here or anywhere else in it, and nothing here should be read against one.

**Limits.** One trading day, 12 December 2025, and three symbols. IEX is one venue with a small share of volume, so none of this describes the consolidated market. A clean run shows that the file and our code agree with each other. It does not show that our book matches the venue's own. No statistic has a confidence interval, and the run was made once.

## What was run

```sh
python research/iex_replay.py 20251212_IEXTP1_DPLS1.0.pcap.gz OUT.json AAPL MSFT INTC
```

| item | value |
|---|---|
| file | `20251212_IEXTP1_DPLS1.0.pcap.gz` (DEEP+, order by order; pcapng inside, despite the name) |
| packets, all DEEP+ segments | 335,524,535 |
| messages, all symbols | 477,645,531 |
| messages of other symbols, skipped | 473,347,352 |
| retail indicator messages, skipped | 2,145,348 |
| heartbeats | 3,631 |
| streams | 1 |
| regular hours | taken from the system events in the file |

## Integrity

The parser stores a counter only once it has been incremented, so a missing key means zero. Most rows below rest on that: the keys are absent, not reported as 0.

| check | result |
|---|---|
| `seq_gap`, `gap_messages` | absent, so no gaps |
| `duplicate_segment`, `duplicate_messages` | absent |
| `unknown_type`, `unknown_order`, `ts_regression` and the other anomaly counters | absent |
| `late_start_messages` | present, 0 |
| `file_error` | null |
| per-symbol `fatal` | absent for all three |
| trade breaks | 0 for all three |
| book `verify` at end of day | ok for all three, with 0 orders resting on either side |

Every displayed execution hit an order at the touch of its side, and the `behind_touch`, `missing` and `price_differs_from_booked` counters are absent:

| symbol | book events | displayed executions | at the touch |
|---|---|---|---|
| AAPL | 863,008 | 5,242 | 5,242 |
| MSFT | 139,925 | 2,023 | 2,023 |
| INTC | 1,123,846 | 13,077 | 13,077 |

## Published statistics, regular hours

Spread is measured on IEX's own displayed book at each feed event with both sides present, so it is weighted by events and not by time. OFI R² is from the Cont, Kukanov and Stoikov regression of mid change on order flow imbalance in 10-second buckets. A bucket needs a mid at both ends, which is why MSFT has fewer of them.

| symbol | median spread (cents) | p99 spread (cents) | OFI buckets | OFI R² | OFI correlation |
|---|---|---|---|---|---|
| AAPL | 7 | 496 | 2,340 | 0.018 | 0.135 |
| MSFT | 41 | 425 | 2,278 | 0.023 | 0.152 |
| INTC | 1 | 3 | 2,340 | 0.250 | 0.500 |

Autocorrelation of trade signs, after merging fills with one timestamp and sign into one parent trade:

| symbol | parent trades | lag 1 | lag 10 | lag 100 | lag 1000 |
|---|---|---|---|---|---|
| AAPL | 9,779 | 0.174 | 0.053 | 0.016 | -0.008 |
| MSFT | 10,567 | 0.551 | 0.179 | 0.037 | -0.031 |
| INTC | 8,496 | 0.296 | 0.040 | 0.010 | -0.004 |

Signs decay with lag in all three. With about 10,000 trades, values near 0.01 cannot be told from zero, so this day does not show the slow power-law tail that the literature reports. Fills with no side are signed against the displayed mid, and a fill at that mid is dropped: 2,337 for AAPL, 331 for MSFT and 6,792 for INTC. On a wide book that signing is crude, so the MSFT figures in particular should be read loosely.

## Why these numbers look the way they do

IEX's own displayed book is thin, and it is not the NBBO. The best price displayed on IEX can sit well behind the best price in the market. So the 7 and 41 cent medians for AAPL and MSFT are a fact about one venue's displayed orders, not the cost of trading these stocks. No NBBO data was used here, so the size of that gap was not measured.

The low OFI R² for AAPL and MSFT most likely comes from the same place. If few orders are displayed, the mid of that book moves in jumps as isolated orders arrive and leave, and it is a poor stand-in for the market's price. This was not tested. INTC is the exception, with a 1 cent median spread and an R² of 0.25, so "wide and low" is true of two symbols out of three.

DEEP+ shows displayed orders only. Many IEX fills are not against displayed orders, and they arrive as trade messages with no order behind them. The book cannot see that liquidity:

| symbol | fills against displayed orders | fills between non-displayed orders | displayed share of fills | displayed share of volume |
|---|---|---|---|---|
| AAPL | 5,242 | 7,944 | 40% | 36% |
| MSFT | 2,016 | 10,331 | 16% | 8% |
| INTC | 13,058 | 7,709 | 63% | 54% |

## Outside check

The last eligible trades were AAPL 278.315, MSFT 478.53 and INTC 37.83, each within about four seconds of the end of the regular session. Third-party closes for that day were about 278.28 for AAPL and 478.17 for MSFT. Those are aggregator figures: one other source gave 277.84 for AAPL, and no figure was found for INTC.

This is a plausibility check only. An official close comes from the listing market and an IEX last trade does not, so they should be near each other without being equal. It says the prices decoded from the file are in the right place, and nothing more.

## What this does not show

- That our book equals IEX's. No venue snapshot was compared.
- Anything about other days, other symbols, or the DEEP level-book path, which this file does not use.
- Anything about the consolidated market, or about a tradable edge.
- A calibration of `lobmod.sim`. That attempt failed, see [calibration_criteria.md](calibration_criteria.md) and the README.
