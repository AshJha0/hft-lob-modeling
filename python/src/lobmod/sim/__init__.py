"""Event-time-plus-clock execution simulator (Python only, outside the golden contract).

The package joins the repo's parts into one synthetic market: Hawkes market
orders, queue-reactive limit and cancel flow, a hidden fair value with informed
takers, and agents whose orders sit in a real :class:`~lobmod.book.LimitOrderBook`
behind market-data and order-entry delays.

Every parameter is assumed, none is calibrated to data.  Results describe this
model, not a market, and nothing here is evidence of a tradable edge.
"""

from .accounting import Attribution, assess_edge, attribute, fill_prob_by_queue, realised_spread, summarise
from .engine import AgentSpec, Fees, FlowParams, SimResult, Simulator
from .legacy import legacy_maker, variance_ratio
from .strategies import AvellanedaStoikov, NaiveTouch, Quote, SignalTaker

__all__ = [
    "AgentSpec", "Attribution", "AvellanedaStoikov", "Fees", "FlowParams", "NaiveTouch", "Quote",
    "SignalTaker", "SimResult", "Simulator", "assess_edge", "attribute", "fill_prob_by_queue",
    "legacy_maker", "realised_spread", "summarise", "variance_ratio",
]
