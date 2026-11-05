"""One pass over an ITCH 5.0 file: integrity checks plus reusable per-symbol extracts.

Usage: python research/real_replay.py FILE.gz OUT_DIR [--max-seconds N] [--max-frames N] SYMBOL ...

Writes OUT_DIR/replay_summary_extract.json and OUT_DIR/extract/SYMBOL.npz.  Both hold
data derived from the input file: keep them out of the repository.

Book checks per symbol: framing and order-reference anomalies from the parser,
crossed states, sizes that go non-positive, level totals against live orders at
the close, and whether each visible execution hits the touch of its side.

Arrays per symbol, regular hours only (``code = 2 * class + side``, class 0 add,
1 cancel or reduce, 2 execute, side 0 bid and 1 ask; a replace is a cancel then an add):

  ev_ts, ev_code, ev_dist, ev_size   every event; dist = ticks behind own touch, negative inside the spread
  tr_ts, tr_sign, tr_size, tr_price  visible executions, sign +1 for a buying aggressor
  hid_ts, hid_size                   non-displayed executions, side not used
  grid                               bid, ask, sizes, order counts every 100 ms, so its statistics are time weighted
  ofi, mid_end, bucket               10-second order-flow imbalance and last mid
  counts, occupancy                  events by state, seconds in each state

The state is (spread in ticks capped at 4, touch imbalance in 5 bins).  The book
here is a lean copy of the price-level totals: a lazy heap keeps the best price
cheap on deep books, where a max over a dict is not.
"""
import argparse
import gzip
import heapq
import json
import sys
import time
from array import array
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'python' / 'src'))
from lobmod.feeds.itch50 import FeedEvent, Itch50Parser, TradeReport

OPEN_NS = int(9.5 * 3600e9)
CLOSE_NS = 16 * 3600 * 10 ** 9
BUCKET_NS = 10 * 10 ** 9
GRID_NS = 10 ** 8
TICK = 100          # price units of 0.0001 per cent
N_SPREAD, N_IMB, N_CLASS, N_LOC = 4, 5, 3, 3   # LOC: inside spread, at touch, behind


def aggregate_signs(ts, signs):
    """Merge adjacent fills with the same nanosecond and sign into one aggressive order.

    One sweep prints one execution per resting order hit, all with its timestamp, so
    fill-level signs repeat mechanically and overstate trade-sign memory.  Fills
    from a different order at that same nanosecond cannot be told apart and merge
    too; that is the limit of what the feed shows.  Returns ``(ts, signs)`` arrays.
    """
    t = np.asarray(ts, dtype=np.int64)
    s = np.asarray(signs, dtype=np.int8)
    if t.shape != s.shape or t.ndim != 1:
        raise ValueError('ts and signs must be 1-D and equal in length')
    if t.size == 0:
        return t, s
    keep = np.ones(t.size, dtype=bool)
    keep[1:] = (t[1:] != t[:-1]) | (s[1:] != s[:-1])
    return t[keep], s[keep]


def acf(x, lags):
    x = np.asarray(x, float) - np.mean(x)
    v = np.mean(x * x)
    return {str(k): round(float(np.mean(x[:-k] * x[k:]) / v), 4) for k in lags if k < len(x)}


class LeanBook:
    """Price-level totals by order id.  Prices in feed units; quantities in shares."""

    def __init__(self):
        self.orders = {}                 # id -> [side 0/1, price, size]
        self.levels = ({}, {})           # price -> [size, n orders]
        self.heaps = ([], [])            # bid prices negated
        self.bad_size = 0

    def add(self, oid, side, price, size):
        self.orders[oid] = [side, price, size]
        lv = self.levels[side].get(price)
        if lv is None:
            self.levels[side][price] = [size, 1]
            heapq.heappush(self.heaps[side], price if side else -price)
        else:
            lv[0] += size
            lv[1] += 1

    def reduce(self, oid, size):
        """Take ``size`` off an order; ``None`` removes all of it."""
        o = self.orders[oid]
        left = 0 if size is None else o[2] - size
        if left < 0:
            self.bad_size += 1
            left = 0
        lv = self.levels[o[0]][o[1]]
        lv[0] -= o[2] - left
        o[2] = left
        if left == 0:
            del self.orders[oid]
            lv[1] -= 1
            if lv[1] == 0:
                del self.levels[o[0]][o[1]]

    def best(self, side):
        h, lv = self.heaps[side], self.levels[side]
        while h:
            p = h[0] if side else -h[0]
            if p in lv:
                return p
            heapq.heappop(h)
        return None

    def verify(self):
        for side in (0, 1):
            totals = Counter()
            for s, price, size in self.orders.values():
                if s == side:
                    totals[price] += size
            if {p: v[0] for p, v in self.levels[side].items()} != dict(totals):
                return f'level totals disagree on side {side}'
        return 'ok'


class Extract:
    def __init__(self):
        self.book = LeanBook()
        self.counts_by_type = Counter()
        self.touch = Counter()
        self.crossed = 0
        self.missing = 0
        self.volume = 0
        self.ev_ts, self.ev_code = array('q'), array('b')
        self.ev_dist, self.ev_size = array('i'), array('i')
        self.tr_ts, self.tr_sign = array('q'), array('b')
        self.tr_size, self.tr_price = array('i'), array('i')
        self.hid_ts, self.hid_size = array('q'), array('i')
        self.grid = array('q')           # bid, ask, qb, qa, nb, na per row
        self.next_grid = OPEN_NS
        self.ofi = Counter()
        self.mid_end = {}
        self.counts = np.zeros((N_SPREAD, N_IMB, N_CLASS, N_LOC), dtype=np.int64)
        self.occupancy = np.zeros((N_SPREAD, N_IMB))
        self.top = None                  # bid, ask, qb, qa, nb, na
        self.t_state = None
        self.state = None

    def _record(self, ts, cls, side, price, size):
        top = self.top
        if top is None:
            return
        own = top[side]
        dist = (price - own) // TICK if side else (own - price) // TICK
        self.ev_ts.append(ts)
        self.ev_code.append(2 * cls + side)
        self.ev_dist.append(max(-30000, min(30000, dist)))
        self.ev_size.append(size)
        if self.state is not None:
            loc = 0 if dist < 0 else (1 if dist == 0 else 2)
            self.counts[self.state[0], self.state[1], cls, loc] += 1

    def _refresh(self, ts):
        """Publish the new top of book, ageing the state it replaces."""
        book = self.book
        bid, ask = book.best(0), book.best(1)
        if self.state is not None:
            self.occupancy[self.state] += (ts - self.t_state) / 1e9
        if not OPEN_NS <= ts < CLOSE_NS or bid is None or ask is None:
            self.top = self.state = None
            return
        if bid >= ask:
            self.crossed += 1
        lb, la = book.levels[0][bid], book.levels[1][ask]
        new = (bid, ask, lb[0], la[0], lb[1], la[1])
        p = self.top
        bucket = ts // BUCKET_NS
        if p is not None:
            self.ofi[bucket] += (new[2] if bid >= p[0] else 0) - (p[2] if bid <= p[0] else 0) \
                - (new[3] if ask <= p[1] else 0) + (p[3] if ask >= p[1] else 0)
        self.mid_end[bucket] = 0.5 * (bid + ask)
        self.top = new
        total = new[2] + new[3]
        imb = min(int(N_IMB * new[2] / total), N_IMB - 1) if total else 2
        spread = max(1, min((ask - bid) // TICK, N_SPREAD)) - 1
        self.state = (spread, imb)
        self.t_state = ts

    def _fill_grid(self, ts):
        while self.next_grid <= ts and self.next_grid < CLOSE_NS:
            self.grid.extend(self.top if self.top is not None else (0, 0, 0, 0, 0, 0))
            self.next_grid += GRID_NS

    def apply(self, ev):
        ts, et, oid = ev.ts_ns, ev.etype, ev.order_id
        self.counts_by_type[et] += 1
        rth = OPEN_NS <= ts < CLOSE_NS
        if rth:
            self._fill_grid(ts)
        book = self.book
        if et == 'L':
            side = 0 if ev.side == 'B' else 1
            if rth:
                self._record(ts, 0, side, ev.price, ev.size)
            book.add(oid, side, ev.price, ev.size)
        else:
            o = book.orders.get(oid)
            if o is None:
                self.missing += 1
                return
            side, price, left = o
            if et == 'E':
                self.touch['at_touch' if price == book.best(side) else 'behind_touch'] += 1
                if rth:
                    self._record(ts, 2, side, price, ev.size)
                book.reduce(oid, ev.size)
            elif et == 'R':
                if rth:
                    self._record(ts, 1, side, price, ev.size)
                book.reduce(oid, ev.size)
            else:                         # 'C' or 'U'
                if rth:
                    self._record(ts, 1, side, price, left)
                book.reduce(oid, None)
                if et == 'U':
                    if rth:
                        self._record(ts, 0, side, ev.price, ev.size)
                    book.add(ev.new_order_id, side, ev.price, ev.size)
        self._refresh(ts)

    def trade(self, tr):
        if tr.printable:
            self.volume += tr.size
        if not OPEN_NS <= tr.ts_ns < CLOSE_NS:
            return
        if tr.source in ('E', 'C'):
            self.tr_ts.append(tr.ts_ns)
            self.tr_sign.append(-1 if tr.side == 'B' else 1)   # resting bid hit = seller
            self.tr_size.append(tr.size)
            self.tr_price.append(tr.price)
        elif tr.source == 'P':
            self.hid_ts.append(tr.ts_ns)
            self.hid_size.append(tr.size)

    def arrays(self):
        keys = sorted(self.mid_end)
        return {
            'ev_ts': np.frombuffer(self.ev_ts, np.int64), 'ev_code': np.frombuffer(self.ev_code, np.int8),
            'ev_dist': np.frombuffer(self.ev_dist, np.int32).astype(np.int16),
            'ev_size': np.frombuffer(self.ev_size, np.int32),
            'tr_ts': np.frombuffer(self.tr_ts, np.int64), 'tr_sign': np.frombuffer(self.tr_sign, np.int8),
            'tr_size': np.frombuffer(self.tr_size, np.int32), 'tr_price': np.frombuffer(self.tr_price, np.int32),
            'hid_ts': np.frombuffer(self.hid_ts, np.int64), 'hid_size': np.frombuffer(self.hid_size, np.int32),
            'grid': np.frombuffer(self.grid, np.int64).reshape(-1, 6).astype(np.int32),
            'bucket': np.array(keys, np.int64), 'ofi': np.array([self.ofi[k] for k in keys], np.int64),
            'mid_end': np.array([self.mid_end[k] for k in keys]),
            'counts': self.counts, 'occupancy': self.occupancy,
        }

    def summary(self):
        a = self.arrays()
        ts, sg = aggregate_signs(a['tr_ts'], a['tr_sign'])
        out = {'events': dict(self.counts_by_type), 'touch': dict(self.touch), 'volume': self.volume,
               'crossed_states': self.crossed, 'missing_orders': self.missing,
               'bad_sizes': self.book.bad_size, 'verify': self.book.verify(),
               'fills_rth': int(a['tr_ts'].size), 'orders_rth': int(ts.size), 'events_rth': int(a['ev_ts'].size)}
        if sg.size > 2000:
            out['sign_acf_orders'] = acf(sg, (1, 10, 100, 1000))
            out['sign_acf_fills'] = acf(a['tr_sign'], (1, 10, 100, 1000))
        k = a['bucket']
        adj = np.nonzero(k[1:] == k[:-1] + 1)[0]
        if adj.size > 30:
            c = np.corrcoef(a['ofi'][adj + 1], a['mid_end'][adj + 1] - a['mid_end'][adj])[0, 1]
            out['cks'] = {'buckets': int(adj.size), 'r2': round(float(c * c), 4)}
        return out


def main(path, out_dir, symbols, max_seconds=0, max_frames=0):
    t0 = time.time()
    out = Path(out_dir)
    (out / 'extract').mkdir(parents=True, exist_ok=True)
    parser = Itch50Parser(symbols, strict=False)
    books = {s: Extract() for s in symbols}
    names = {s.encode().ljust(8) for s in symbols}
    wanted = bytearray(65536)
    frames = chunks = 0
    tail = b''
    stopped = None

    def handle(msg):
        for item in parser.parse_message(msg):
            if isinstance(item, FeedEvent):
                books[item.symbol].apply(item.event)
            elif isinstance(item, TradeReport):
                books[item.symbol].trade(item)

    with gzip.open(path, 'rb') as fh:
        while stopped is None:
            chunk = fh.read(1 << 24)
            if not chunk:
                break
            buf = tail + chunk
            i, end = 0, len(buf)
            while i + 2 <= end:
                j = i + 2 + ((buf[i] << 8) | buf[i + 1])
                if j > end:
                    break
                if wanted[(buf[i + 3] << 8) | buf[i + 4]]:
                    handle(buf[i + 2:j])
                else:
                    t = buf[i + 2]
                    if t == 82:          # R: stock directory, learn locates
                        if buf[i + 13:i + 21] in names:
                            wanted[(buf[i + 3] << 8) | buf[i + 4]] = 1
                        handle(buf[i + 2:j])
                    elif t == 83:        # S: system event, locate 0
                        handle(buf[i + 2:j])
                frames += 1
                i = j
            tail = buf[i:]
            chunks += 1
            if chunks % 100 == 0:
                n = sum(b.counts_by_type.total() for b in books.values())
                print(f'{time.time() - t0:.0f}s frames {frames} events {n}', flush=True)
            if max_seconds and time.time() - t0 > max_seconds:
                stopped = 'max_seconds'
            if max_frames and frames >= max_frames:
                stopped = 'max_frames'
    res = {'frames': frames, 'parser': dict(parser.stats), 'stopped': stopped, 'tail_bytes': len(tail),
           'seconds': round(time.time() - t0), 'symbols': {}}
    for s, b in books.items():
        b._fill_grid(CLOSE_NS)
        res['symbols'][s] = b.summary()
        np.savez(out / 'extract' / f'{s}.npz', **b.arrays())
    (out / 'replay_summary_extract.json').write_text(json.dumps(res, indent=1, default=str))
    print('done', res['seconds'], flush=True)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('file')
    ap.add_argument('out_dir')
    ap.add_argument('symbols', nargs='+')
    ap.add_argument('--max-seconds', type=float, default=5300)
    ap.add_argument('--max-frames', type=int, default=0)
    a = ap.parse_args()
    main(a.file, a.out_dir, a.symbols, a.max_seconds, a.max_frames)
