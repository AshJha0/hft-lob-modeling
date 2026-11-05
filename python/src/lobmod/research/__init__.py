"""Research tools: inference, honest evaluation and execution simulators.

These modules answer "how sure are we, and could it be traded?" for the pinned
models in :mod:`lobmod`.  They are **Python-only analysis tools, outside the
four-language golden contract**: nothing here is mirrored in C++, Rust or Java,
and nothing here has golden values.

RNG exception: the package rule "no numpy RNG" covers pinned paths only.  Code
here may use ``numpy.random.default_rng`` with an explicit seed, because its
outputs are never compared across languages.  Pinned generators
(:func:`lobmod.zi.generate_events`, :func:`lobmod.hawkes.simulate_uni`) are
still used wherever a stream or path is simulated.

Requires numpy only.  Import as ``lobmod.research``; the top-level ``lobmod``
namespace does not re-export it.

Everything here has only been exercised on the synthetic zero-intelligence
stream and simulated Hawkes paths.  It says nothing about real markets.
"""

from .backtest import (
    BacktestConfig,
    BacktestResult,
    RoundTrip,
    foresight_ceiling,
    run_backtest,
)
from .evaluation import (
    BootstrapCI,
    Split,
    block_bootstrap_ci,
    evaluate_stream,
    log_loss,
    macro_f1,
    matthews_corrcoef,
    multi_seed_compare,
    per_class_precision,
    walk_forward_splits,
)
from .inference import (
    BootstrapGof,
    HawkesInference,
    LjungBox,
    LrTest,
    bootstrap_ks_uni,
    chi2_sf,
    inference_biv,
    inference_uni,
    ljung_box,
    lr_test_no_cross,
    lr_test_poisson,
    numeric_hessian,
    residual_acf,
)
from .maker import MakerConfig, MakerFill, MakerResult, run_maker

__all__ = [
    "BacktestConfig", "BacktestResult", "RoundTrip", "foresight_ceiling",
    "run_backtest",
    "BootstrapCI", "Split", "block_bootstrap_ci", "evaluate_stream", "log_loss",
    "macro_f1", "matthews_corrcoef", "multi_seed_compare",
    "per_class_precision", "walk_forward_splits",
    "BootstrapGof", "HawkesInference", "LjungBox", "LrTest", "bootstrap_ks_uni",
    "chi2_sf", "inference_biv", "inference_uni", "ljung_box",
    "lr_test_no_cross", "lr_test_poisson", "numeric_hessian", "residual_acf",
    "MakerConfig", "MakerFill", "MakerResult", "run_maker",
]
