//! Order-flow features and direction labels from replayed book states.
//!
//! Order-flow imbalance (OFI)
//! --------------------------
//! Cont, Kukanov & Stoikov (2014) style.  For consecutive book states
//! `n-1 -> n` with best bid price/size `(Pb, qb)` and best ask `(Pa, qa)`,
//! the per-event order-flow contribution is (pinned)
//!
//! ```text
//! e_n =   1{Pb_n >= Pb_{n-1}} * qb_n  -  1{Pb_n <= Pb_{n-1}} * qb_{n-1}
//!       - 1{Pa_n <= Pa_{n-1}} * qa_n  +  1{Pa_n >= Pa_{n-1}} * qa_{n-1}
//! ```
//!
//! (bid improvements / additions push up, ask improvements / additions push
//! down; equal prices contribute the size *change* on that side).  The OFI
//! over a trailing window of `w` events ending at event `i` is
//! `OFI_i = sum_{n = i-w+1}^{i} e_n`.
//!
//! Feature vector (pinned column order) at event `i`:
//!
//! ```text
//! [ OFI_i,
//!   QI_i          = (qb_i - qa_i) / (qb_i + qa_i)   (queue imbalance),
//!   spread_i      = Pa_i - Pb_i                     (ticks),
//!   depth_ratio_i = bid_depth_k_i / ask_depth_k_i,
//!   ret_i         = mid_i - mid_{i - w_ret}         (short-horizon return) ]
//! ```
//!
//! Label: mid-price direction over the next `H` events with an epsilon dead
//! zone (pinned classes `0 = down, 1 = flat, 2 = up`):
//! `d = mid_{i+H} - mid_i`; `y = 2` if `d > epsilon`, `0` if `d < -epsilon`,
//! else `1`.
//!
//! Rows are emitted only where every input is defined: `i >= max(w_ofi,
//! w_ret)`, `i + H < n_events`, every `e_n` in the OFI window finite, the
//! mids/spread the row touches finite, and all denominators positive.

use std::fs;

use serde::Deserialize;

use crate::error::{LobError, Result};
use crate::matrix::Matrix;
use crate::replay::BookStates;

/// Pinned feature column order.
pub const FEATURE_NAMES: [&str; 5] = ["ofi", "queue_imbalance", "spread", "depth_ratio", "ret"];

/// Pinned label class: mid moves down more than epsilon.
pub const LABEL_DOWN: usize = 0;
/// Pinned label class: mid stays within the epsilon dead zone.
pub const LABEL_FLAT: usize = 1;
/// Pinned label class: mid moves up more than epsilon.
pub const LABEL_UP: usize = 2;

/// Pinned feature / label / NN hyperparameters (`data/features_config.json`).
#[derive(Debug, Clone, Deserialize)]
pub struct FeatureConfig {
    /// Trailing OFI window length in events.
    pub ofi_window: usize,
    /// Lookback for the short-horizon return feature.
    pub ret_window: usize,
    /// Label horizon H in events.
    pub horizon: usize,
    /// Dead-zone half-width for the flat class (ticks).
    pub epsilon: f64,
    /// Levels per side in the depth-ratio feature.
    pub depth_k: usize,
    /// MLP hidden layer sizes.
    pub hidden: Vec<usize>,
    /// Full-batch gradient-descent step.
    pub learning_rate: f64,
    /// Training epochs (full-batch steps).
    pub epochs: usize,
    /// Chronological train fraction.
    pub train_frac: f64,
}

impl FeatureConfig {
    /// Validate every field per the contract.
    ///
    /// # Errors
    /// [`LobError::InvalidInput`] on any out-of-range value.
    pub fn validate(&self) -> Result<()> {
        for (name, v) in [
            ("ofi_window", self.ofi_window),
            ("ret_window", self.ret_window),
            ("horizon", self.horizon),
            ("depth_k", self.depth_k),
            ("epochs", self.epochs),
        ] {
            if v == 0 {
                return Err(LobError::InvalidInput(format!(
                    "{name} must be a positive integer, got 0"
                )));
            }
        }
        if !(self.epsilon.is_finite() && self.epsilon >= 0.0) {
            return Err(LobError::InvalidInput(format!(
                "epsilon must be >= 0, got {}",
                self.epsilon
            )));
        }
        if !(self.learning_rate.is_finite() && self.learning_rate > 0.0) {
            return Err(LobError::InvalidInput(format!(
                "learning_rate must be positive, got {}",
                self.learning_rate
            )));
        }
        if !(self.train_frac > 0.0 && self.train_frac < 1.0) {
            return Err(LobError::InvalidInput(format!(
                "train_frac must be in (0, 1), got {}",
                self.train_frac
            )));
        }
        if self.hidden.iter().any(|&h| h == 0) {
            return Err(LobError::InvalidInput(format!(
                "hidden sizes must be positive integers, got {:?}",
                self.hidden
            )));
        }
        Ok(())
    }
}

/// Load and validate a [`FeatureConfig`] from its pinned JSON file.
///
/// # Errors
/// [`LobError::Io`] / [`LobError::Parse`] on file/JSON problems,
/// [`LobError::InvalidInput`] on out-of-range values.
pub fn load_feature_config(path: &str) -> Result<FeatureConfig> {
    let text = fs::read_to_string(path).map_err(|e| LobError::Io(format!("{path}: {e}")))?;
    let cfg: FeatureConfig =
        serde_json::from_str(&text).map_err(|e| LobError::Parse(format!("{path}: {e}")))?;
    cfg.validate()?;
    Ok(cfg)
}

/// Feature matrix with aligned labels and trading metadata.
#[derive(Debug, Clone)]
pub struct FeatureSet {
    /// `(n, 5)` feature matrix in [`FEATURE_NAMES`] order.
    pub x: Matrix,
    /// Integer labels in `{0, 1, 2}`.
    pub y: Vec<usize>,
    /// 0-based index into the [`BookStates`] arrays per row.
    pub event_index: Vec<usize>,
    /// Realized `mid_{i+H} - mid_i` (for PnL evaluation).
    pub fwd_ret: Vec<f64>,
    /// Spread at decision time (for cost modelling).
    pub spread: Vec<f64>,
}

impl FeatureSet {
    /// Number of feature rows.
    pub fn len(&self) -> usize {
        self.y.len()
    }

    /// True when no rows were emitted (never constructed — kept for API
    /// completeness).
    pub fn is_empty(&self) -> bool {
        self.y.is_empty()
    }
}

/// Per-event order-flow contributions `e_n` (module docs formula).
///
/// `e_0 = NaN` (no predecessor); any event whose own or previous state has an
/// empty side yields NaN.
pub fn ofi_series(states: &BookStates) -> Vec<f64> {
    let n = states.len();
    let mut e = vec![f64::NAN; n];
    for i in 1..n {
        let (pb, pb_prev) = (states.best_bid[i], states.best_bid[i - 1]);
        let (pa, pa_prev) = (states.best_ask[i], states.best_ask[i - 1]);
        if !(pb.is_finite() && pb_prev.is_finite() && pa.is_finite() && pa_prev.is_finite()) {
            continue; // stays NaN
        }
        let mut contrib = 0.0;
        if pb >= pb_prev {
            contrib += states.bid_size[i];
        }
        if pb <= pb_prev {
            contrib -= states.bid_size[i - 1];
        }
        if pa <= pa_prev {
            contrib -= states.ask_size[i];
        }
        if pa >= pa_prev {
            contrib += states.ask_size[i - 1];
        }
        e[i] = contrib;
    }
    e
}

/// OFI over the `window` events ending at `event_index` (0-based).
///
/// # Errors
/// [`LobError::InvalidInput`] if the window does not fit
/// (`event_index < window`) or arguments are invalid.
pub fn ofi_at(states: &BookStates, event_index: usize, window: usize) -> Result<f64> {
    if window == 0 {
        return Err(LobError::InvalidInput(
            "window must be a positive integer, got 0".to_string(),
        ));
    }
    let n = states.len();
    if event_index >= n {
        return Err(LobError::InvalidInput(format!(
            "event_index {event_index} out of range [0, {n})"
        )));
    }
    if event_index < window {
        return Err(LobError::InvalidInput(format!(
            "window {window} does not fit before event {event_index}"
        )));
    }
    let e = ofi_series(states);
    let mut sum = 0.0;
    for &v in &e[event_index - window + 1..=event_index] {
        sum += v;
    }
    Ok(sum)
}

/// Build the feature matrix and labels from replayed book states.
///
/// See the module docs for the pinned formulas; rows with any undefined
/// input are skipped and rows keep replay order.
///
/// # Errors
/// [`LobError::InvalidInput`] if the config is invalid or zero rows could be
/// emitted.
pub fn compute_features(states: &BookStates, cfg: &FeatureConfig) -> Result<FeatureSet> {
    cfg.validate()?;
    let n = states.len();
    let e = ofi_series(states);
    let start = cfg.ofi_window.max(cfg.ret_window);
    let stop = n.saturating_sub(cfg.horizon);
    let mut x_data: Vec<f64> = Vec::new();
    let mut y: Vec<usize> = Vec::new();
    let mut event_index: Vec<usize> = Vec::new();
    let mut fwd_ret: Vec<f64> = Vec::new();
    let mut spread_out: Vec<f64> = Vec::new();
    for i in start..stop {
        let window = &e[i - cfg.ofi_window + 1..=i];
        if window.iter().any(|v| !v.is_finite()) {
            continue;
        }
        let (qb, qa) = (states.bid_size[i], states.ask_size[i]);
        let mid_now = states.mid[i];
        let mid_past = states.mid[i - cfg.ret_window];
        let mid_fwd = states.mid[i + cfg.horizon];
        let spread = states.spread[i];
        let (bd, ad) = (states.bid_depth_k[i], states.ask_depth_k[i]);
        if !(mid_now.is_finite() && mid_past.is_finite() && mid_fwd.is_finite() && spread.is_finite())
        {
            continue;
        }
        if qb + qa <= 0.0 || ad <= 0.0 {
            continue;
        }
        let mut ofi = 0.0;
        for &v in window {
            ofi += v;
        }
        let qi = (qb - qa) / (qb + qa);
        let depth_ratio = bd / ad;
        let ret = mid_now - mid_past;
        let d = mid_fwd - mid_now;
        let label = if d > cfg.epsilon {
            LABEL_UP
        } else if d < -cfg.epsilon {
            LABEL_DOWN
        } else {
            LABEL_FLAT
        };
        x_data.extend_from_slice(&[ofi, qi, spread, depth_ratio, ret]);
        y.push(label);
        event_index.push(i);
        fwd_ret.push(d);
        spread_out.push(spread);
    }
    if y.is_empty() {
        return Err(LobError::InvalidInput(
            "no valid feature rows could be built from the states".to_string(),
        ));
    }
    let rows = y.len();
    Ok(FeatureSet {
        x: Matrix::from_vec(rows, 5, x_data),
        y,
        event_index,
        fwd_ret,
        spread: spread_out,
    })
}

/// Pinned chronological split: the first `floor(train_frac * n)` rows train,
/// the rest test.  Returns `(train_rows, test_rows)` as index ranges.
///
/// # Errors
/// [`LobError::InvalidInput`] if `n <= 1`, `train_frac` is outside `(0, 1)`,
/// or the split would leave either side empty.
pub fn chronological_split(
    n: usize,
    train_frac: f64,
) -> Result<(std::ops::Range<usize>, std::ops::Range<usize>)> {
    if n <= 1 {
        return Err(LobError::InvalidInput(format!(
            "n must be an integer > 1, got {n}"
        )));
    }
    if !(train_frac > 0.0 && train_frac < 1.0) {
        return Err(LobError::InvalidInput(format!(
            "train_frac must be in (0, 1), got {train_frac}"
        )));
    }
    let n_train = (train_frac * n as f64).floor() as usize;
    if n_train == 0 || n_train == n {
        return Err(LobError::InvalidInput(format!(
            "split leaves an empty side: n={n}, train_frac={train_frac}"
        )));
    }
    Ok((0..n_train, n_train..n))
}
