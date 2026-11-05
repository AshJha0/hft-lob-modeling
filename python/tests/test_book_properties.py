"""Property tests: the book against a deliberately naive model.

The model keeps one flat list of resting orders and finds each match by
sorting it, so it shares no data structures with the book. Random operation
sequences must leave both with the same fills, quotes and resting sizes.
"""
import numpy as np
import pytest

from lobmod.book import LimitOrderBook


class NaiveBook:
    def __init__(self):
        self.orders = []  # [arrival, side, price, id, size]
        self.clock = 0

    def _match(self, side, size, limit):
        fills = []
        opp = [o for o in self.orders if o[1] != side]
        opp.sort(key=lambda o: (o[2] if side == "B" else -o[2], o[0]))
        for o in opp:
            if size == 0:
                break
            if limit is not None and (o[2] > limit if side == "B" else o[2] < limit):
                break
            take = min(size, o[4])
            o[4] -= take
            size -= take
            fills.append((o[3], o[2], take))
        self.orders = [o for o in self.orders if o[4] > 0]
        return fills, size

    def limit(self, side, price, size, oid):
        fills, rest = self._match(side, size, price)
        if rest:
            self.clock += 1
            self.orders.append([self.clock, side, price, oid, rest])
        return fills, rest

    def cancel(self, oid):
        hit = next(o for o in self.orders if o[3] == oid)
        self.orders.remove(hit)
        return hit[4]

    def best(self, side):
        prices = [o[2] for o in self.orders if o[1] == side]
        if not prices:
            return None
        return max(prices) if side == "B" else min(prices)

    def total(self, side):
        return sum(o[4] for o in self.orders if o[1] == side)


def as_tuples(fills):
    return [(f.maker_id, f.price, f.size) for f in fills]


@pytest.mark.parametrize("seed", range(40))
def test_book_matches_naive_model(seed):
    rng = np.random.default_rng(seed)
    book, model = LimitOrderBook(1.0), NaiveBook()
    next_id = 1
    for _ in range(400):
        op = rng.random()
        side = "B" if rng.random() < 0.5 else "S"
        live = [o[3] for o in model.orders]
        if op < 0.55:
            price, size = int(rng.integers(90, 111)), int(rng.integers(1, 20))
            got = book.limit_ticks(side, price, size, next_id)
            want = model.limit(side, price, size, next_id)
            next_id += 1
            assert (as_tuples(got[0]), got[1]) == want
        elif op < 0.8 and live:
            oid = int(rng.choice(live))
            assert book.cancel(oid) == model.cancel(oid)
        else:
            size = int(rng.integers(1, 40))
            got = book.market(side, size)
            want = model._match(side, size, None)
            assert (as_tuples(got[0]), got[1]) == want
        bid, ask = book.best_bid(), book.best_ask()
        assert bid == model.best("B") and ask == model.best("S")
        assert book.total_size("B") == model.total("B")
        assert book.total_size("S") == model.total("S")
        if bid is not None and ask is not None:
            assert bid < ask  # never left crossed
