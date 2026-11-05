"""Replay one IEX HIST capture and test the book against what the feed implies.

Usage: python research/iex_replay.py FILE.pcap.gz OUT.json [SYMBOL ...] [--progress LOG]

A progress line goes to stderr, and to LOG if given, about every 60 seconds.

Mirrors research/real_replay.py.  DEEP+ files give an order book, DEEP files a
level book; the message types decide.  Run once on one real DEEP+ file, with no
gaps or anomalies reported and an empty verified book at the end of the day: see
research/results_iex.md.  The level-book path has not met a real file.

Go/no-go checks, per symbol:
  * transport: sequence gaps, offset mismatches, duplicates;
  * framing and order-reference anomalies from the parser;
  * book integrity: never crossed, totals agree with the queues;
  * DEEP+ only: every displayed execution hits an order at the touch of its side,
    and how often the fill price differs from the booked price.

Published facts, regular hours only (from the system events R and M, so no time
zone is assumed; a file without them counts every message as regular hours):
  * Cont-Kukanov-Stoikov: mid change against OFI in 10-second buckets;
  * long memory in trade signs, per fill and aggregated.

The book is sampled once per feed event, since the specifications make messages
with one timestamp (DEEP+) or one Event Flags run (DEEP) atomic.  Aggregated signs
merge fills of one timestamp and sign into one parent trade.  A DEEP+ sign comes
from the resting side.  'T' and all DEEP fills have no side and are signed against
the prevailing mid, a fill at the mid being dropped.

IEX is one venue with a small share of volume, so none of this describes the
consolidated market.

The input file is never copied, and OUT holds summary statistics only.
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'python' / 'src'))
from lobmod.feeds.iex import (BookIntegrityChecker, FeedEvent, IexError, IexParser, IexStatus, IexTrade,
                              LevelBook, LevelUpdate, TradeBreak, iter_file)
from lobmod.feeds.itch50 import ItchError

BUCKET_NS = 10 * 10 ** 9
ATTRIBUTION = ('Data provided for free by IEX. By accessing or using IEX Historical Data, '
               'you agree to the IEX Historical Data Terms of Use.')
TERMS_URL = 'https://www.iex.io/products/market-data-connectivity/hist-terms'


class Symbol:
    def __init__(self):
        self.check = BookIntegrityChecker(strict=False)
        self.levels = LevelBook()
        self.fatal = None
        self.prev = None
        self.pending_ts = None
        self.ofi = Counter()
        self.mid_end = {}
        self.spreads = []
        self.touch = Counter()
        self.signs = []  # (ts, sign)
        self.unsigned = 0
        self.volume = 0
        self.breaks = 0
        self.n_samples = 0
        self.fills = Counter()  # by source and session: count and volume
        self.trade_ids = set()
        self.last_trade = None

    def bbo(self):
        if self.levels.counts['updates']:
            return self.levels.bbo()
        book = self.check.book
        return book.best_bid(), book.best_bid_size(), book.best_ask(), book.best_ask_size()

    def sample(self, ts_ns, rth):
        """Record the top of book as a settled state at ``ts_ns``."""
        bid, qb, ask, qa = self.bbo()
        if bid is None or ask is None or bid >= ask or not rth:
            self.prev = None
            return
        p = self.prev
        bucket = ts_ns // BUCKET_NS
        if p is not None:
            pb, pqb, pa, pqa = p
            self.ofi[bucket] += (qb if bid >= pb else 0) - (pqb if bid <= pb else 0) \
                - (qa if ask <= pa else 0) + (pqa if ask >= pa else 0)
        self.mid_end[bucket] = 0.5 * (bid + ask)
        self.n_samples += 1
        if self.n_samples % 50 == 0:
            self.spreads.append(ask - bid)
        self.prev = (bid, qb, ask, qa)

    def order_event(self, ev, rth):
        if self.pending_ts is not None and ev.ts_ns != self.pending_ts:
            self.sample(self.pending_ts, rth)
        self.pending_ts = ev.ts_ns
        if ev.etype == 'E':
            book = self.check.book
            loc = book._orders.get(ev.order_id)
            if loc is None:
                self.touch['missing'] += 1
            else:
                best = book.best_bid() if loc[0] == 'B' else book.best_ask()
                self.touch['at_touch' if loc[1] == best else 'behind_touch'] += 1
        self.check.apply(ev)

    def level_update(self, up, rth):
        if self.levels.apply(up):
            self.sample(up.ts_ns, rth)

    def trade(self, tr, rth):
        self.volume += tr.size
        session = 'rth' if rth else 'outside'
        self.fills[f'{tr.source}_{session}_n'] += 1
        self.fills[f'{tr.source}_{session}_volume'] += tr.size
        self.fills['notional_ticks'] += tr.size * tr.price
        if tr.trade_id in self.trade_ids:
            self.fills['repeated_trade_id'] += 1
        self.trade_ids.add(tr.trade_id)
        if tr.last_sale_eligible:
            self.last_trade = (tr.ts_ns, tr.price, tr.size)
        else:
            self.fills['not_last_sale_eligible'] += 1
        if tr.source == 'L' and tr.price != tr.booked_price:
            self.touch['price_differs_from_booked'] += 1
        if not rth:
            return
        if tr.side:
            sign = -1 if tr.side == 'B' else 1  # resting bid hit = seller
        else:
            bid, _, ask, _ = self.bbo()
            mid = None if bid is None or ask is None else 0.5 * (bid + ask)
            sign = 0 if mid is None or tr.price == mid else (1 if tr.price > mid else -1)
        if sign:
            self.signs.append((tr.ts_ns, sign))
        else:
            self.unsigned += 1


def acf(x, lags):
    x = np.asarray(x, float) - np.mean(x)
    v = np.mean(x * x)
    if v == 0:
        return {}
    return {str(k): round(float(np.mean(x[:-k] * x[k:]) / v), 4) for k in lags if k < len(x)}


def aggregate(signs):
    """Merge consecutive fills of one timestamp and one sign."""
    out = []
    last = None
    for item in signs:
        if item != last:
            out.append(item[1])
        last = item
    return out


def summarise(s):
    if s.pending_ts is not None:
        s.sample(s.pending_ts, True)
    out = {'volume': s.volume, 'n_fills_signed_rth': len(s.signs), 'unsigned_rth': s.unsigned,
           'trade_breaks': s.breaks, 'fills': dict(s.fills), 'last_eligible_trade': s.last_trade}
    if s.fatal:
        out['fatal'] = s.fatal
    if s.levels.counts['updates']:
        out['book'] = 'level'
        out['levels'] = dict(s.levels.counts)
        out['resting_levels'] = (len(s.levels.bids), len(s.levels.asks))
    else:
        out['book'] = 'order'
        out['events'] = dict(s.check.counts)
        out['touch'] = dict(s.touch)
        try:
            out['resting_bid_ask'] = s.check.verify()
            out['verify'] = 'ok'
        except ItchError as exc:
            out['verify'] = str(exc)
    if s.spreads:
        sp = np.array(s.spreads) / 100.0  # 0.0001 ticks to cents
        out['spread_cents'] = {'median': float(np.median(sp)), 'p99': float(np.percentile(sp, 99))}
    keys = sorted(s.mid_end)
    pairs = [(s.ofi[k], s.mid_end[k] - s.mid_end[j]) for j, k in zip(keys, keys[1:]) if k == j + 1]
    if len(pairs) > 30:
        o, d = np.array(pairs, float).T
        if o.std() > 0 and d.std() > 0:
            c = np.corrcoef(o, d)[0, 1]
            out['cks'] = {'buckets': len(o), 'r2': round(float(c * c), 4), 'corr': round(float(c), 4)}
    if len(s.signs) > 2000:
        out['sign_acf_fills'] = acf([x[1] for x in s.signs], (1, 10, 100, 1000))
        agg = aggregate(s.signs)
        out['sign_acf_aggregated'] = {'n': len(agg), 'acf': acf(agg, (1, 10, 100, 1000))}
    return out


def main(path, out, symbols, max_items=None, progress_log=None):
    t0 = time.time()
    n = 0

    def progress(capture, feed):
        line = (f"{time.strftime('%H:%M:%S')} +{time.time() - t0:.0f}s packets={capture['packets']} "
                f"messages={feed['messages']} items={n} gaps={feed['seq_gap']} "
                f"gap_messages={feed['gap_messages']} unknown_type={feed['unknown_type']} "
                f"unknown_order={feed['unknown_order']}")
        print(line, file=sys.stderr, flush=True)
        if progress_log:
            with open(progress_log, 'a') as fh:
                fh.write(line + '\n')

    parser = IexParser(symbols or None, strict=False)
    books = {}
    status = Counter()
    last_status = {}
    system_events = []
    rth = True  # until a system event says otherwise, see the module docstring
    saw_system = False
    file_error = None
    try:
        for item in iter_file(path, parser=parser, progress=progress):
            n += 1
            if isinstance(item, IexStatus):
                status[f'{item.msg_type}:{item.code}'] += 1
                if item.msg_type != 'S':
                    last_status[f'{item.symbol}:{item.msg_type}'] = (item.ts_ns, item.code, item.detail)
                else:
                    system_events.append((item.ts_ns, item.code))
                if item.msg_type == 'S' and item.code in 'OSRME':
                    saw_system = True
                    rth = item.code == 'R'
            elif isinstance(item, FeedEvent):
                s = books.setdefault(item.symbol, Symbol())
                if s.fatal is None:
                    try:
                        s.order_event(item.event, rth)
                    except ItchError as exc:
                        s.fatal = f'{exc} at ts {item.event.ts_ns}'
            elif isinstance(item, LevelUpdate):
                books.setdefault(item.symbol, Symbol()).level_update(item, rth)
            elif isinstance(item, IexTrade):
                books.setdefault(item.symbol, Symbol()).trade(item, rth)
            elif isinstance(item, TradeBreak):
                books.setdefault(item.symbol, Symbol()).breaks += 1
            if max_items and n >= max_items:
                break
    except IexError as exc:
        file_error = str(exc)
    res = {'attribution': ATTRIBUTION, 'terms': TERMS_URL, 'source_file': Path(path).name,
           'items': n, 'parser': dict(parser.stats), 'message_types': parser.type_counts(),
           'status_messages': dict(status), 'system_events': system_events, 'last_status': last_status,
           'regular_hours_from_system_events': saw_system, 'file_error': file_error,
           'seconds': round(time.time() - t0),
           'symbols': {k: summarise(v) for k, v in sorted(books.items())}}
    text = json.dumps(res, indent=1, default=str)
    Path(out).write_text(text)
    print(text)
    print(f'\n{ATTRIBUTION}\n{TERMS_URL}')


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description='Replay one IEX HIST capture.')
    cli.add_argument('path')
    cli.add_argument('out')
    cli.add_argument('symbols', nargs='*')
    cli.add_argument('--progress', help='also append the progress lines to this file')
    args = cli.parse_args()
    main(args.path, args.out, args.symbols, progress_log=args.progress)
