"""lobmod — high-frequency limit-order-book modeling.

Components:
    * :mod:`lobmod.book` — price-time priority limit order book.
    * :mod:`lobmod.replay` — event schema, CSV i/o, deterministic replay.
    * :mod:`lobmod.zi` — pinned zero-intelligence flow generator.
    * :mod:`lobmod.hawkes` — Hawkes processes: likelihood, MLE, thinning.
    * :mod:`lobmod.gof` — KS goodness of fit for Exp(1) residuals.
    * :mod:`lobmod.features` — order-flow features and direction labels.
    * :mod:`lobmod.nn` — deterministic MLP / logistic baselines (no RNG).
    * :mod:`lobmod.neldermead` — pinned native Nelder-Mead.
    * :mod:`lobmod.rng` — pinned 64-bit LCG.
"""

from .book import Fill, LimitOrderBook
from .features import (
    FEATURE_NAMES,
    LABEL_DOWN,
    LABEL_FLAT,
    LABEL_UP,
    FeatureConfig,
    FeatureSet,
    chronological_split,
    compute_features,
    load_feature_config,
    ofi_at,
    ofi_series,
)
from .gof import ks_pvalue, ks_statistic_exp, ks_test_exp
from .hawkes import (
    HawkesBivFit,
    HawkesUniFit,
    biv_loglik,
    biv_loglik_brute,
    biv_residuals,
    fit_biv,
    fit_uni,
    simulate_biv,
    simulate_uni,
    spectral_radius_2x2,
    uni_loglik,
    uni_loglik_brute,
    uni_residuals,
)
from .neldermead import NMResult, nelder_mead
from .nn import (
    MLP,
    Standardizer,
    accuracy,
    confusion_matrix,
    majority_class,
    pinned_init,
    signal_pnl,
    softmax,
)
from .replay import BookStates, Event, apply_event, read_events_csv, replay, write_events_csv
from .rng import Lcg64
from .zi import generate_events

__all__ = [
    "Fill", "LimitOrderBook",
    "FEATURE_NAMES", "LABEL_DOWN", "LABEL_FLAT", "LABEL_UP",
    "FeatureConfig", "FeatureSet", "chronological_split", "compute_features",
    "load_feature_config", "ofi_at", "ofi_series",
    "ks_pvalue", "ks_statistic_exp", "ks_test_exp",
    "HawkesBivFit", "HawkesUniFit", "biv_loglik", "biv_loglik_brute",
    "biv_residuals", "fit_biv", "fit_uni", "simulate_biv", "simulate_uni",
    "spectral_radius_2x2", "uni_loglik", "uni_loglik_brute", "uni_residuals",
    "NMResult", "nelder_mead",
    "MLP", "Standardizer", "accuracy", "confusion_matrix", "majority_class",
    "pinned_init", "signal_pnl", "softmax",
    "BookStates", "Event", "apply_event", "read_events_csv", "replay",
    "write_events_csv",
    "Lcg64", "generate_events",
]

__version__ = "1.0.0"
