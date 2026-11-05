//! `lobmod` — high-frequency limit-order-book modeling.
//!
//! Rust implementation of the cross-language `lobmod` contract (`API_SPEC.md`):
//!
//! * [`book`] — price-time priority limit order book on a discrete tick grid.
//! * [`replay`] — event schema, CSV input, deterministic replay.
//! * [`zi`] — pinned zero-intelligence flow generator.
//! * [`hawkes`] — Hawkes processes: likelihood, MLE, Ogata thinning.
//! * [`gof`] — KS goodness of fit for Exp(1) residuals.
//! * [`features`] — order-flow features and direction labels.
//! * [`nn`] — deterministic MLP / logistic baselines (no RNG).
//! * [`neldermead`] — pinned native Nelder-Mead.
//! * [`rng`] — pinned 64-bit LCG (the only source of randomness).
//!
//! Every formula, constant and iteration order is pinned by the contract so
//! that this crate reproduces `data/golden/golden.json` from the bundled
//! CSVs; all floating point is IEEE-754 double.  Fallible operations return
//! [`Result<T, LobError>`](error::LobError) — the crate never panics on bad
//! input.

pub mod book;
pub mod csv;
pub mod error;
pub mod fast_book;
pub mod features;
pub mod gof;
pub mod hawkes;
pub mod matrix;
pub mod neldermead;
pub mod nn;
pub mod replay;
pub mod rng;
pub mod zi;

pub use book::{Fill, LimitOrderBook, Side, MAX_TICKS};
pub use error::{LobError, Result};
pub use features::{
    chronological_split, compute_features, load_feature_config, ofi_at, ofi_series, FeatureConfig,
    FeatureSet, FEATURE_NAMES, LABEL_DOWN, LABEL_FLAT, LABEL_UP,
};
pub use gof::{kolmogorov_q, ks_pvalue, ks_statistic_exp, ks_test_exp};
pub use hawkes::{
    biv_fit_objective, biv_loglik, biv_loglik_brute, biv_residuals, fit_biv, fit_biv_with, fit_uni,
    fit_uni_with, read_marked_times_csv, read_times_csv, simulate_biv, simulate_uni,
    spectral_radius_2x2, uni_fit_objective, uni_loglik, uni_loglik_brute, uni_residuals,
    write_marked_times_csv, write_times_csv, HawkesBivFit, HawkesUniFit, MAX_CANDIDATES,
};
pub use matrix::Matrix;
pub use neldermead::{nelder_mead, nelder_mead_default, NmResult};
pub use nn::{
    accuracy, confusion_matrix, majority_class, pinned_init, signal_pnl, softmax, Mlp, Standardizer,
};
pub use replay::{
    apply_event, read_events_csv, replay, write_events_csv, BookStates, Event, EventType,
};
pub use rng::{Lcg64, MAX_SEED};
pub use zi::generate_events;
