//! Feed-forward neural network from scratch — fully deterministic, no RNG.
//!
//! Architecture (pinned for the prediction task): input -> 16 ReLU -> 8 ReLU
//! -> 3-way softmax.  The same type with empty `hidden` *is* the multinomial
//! logistic-regression baseline (input -> softmax).
//!
//! Determinism
//! -----------
//! Cross-language golden reproducibility requires that training involve no
//! randomness at all:
//!
//! * **Pinned init** (no RNG): for layer `l` (0-based) with shape
//!   `(n_in, n_out)`,
//!   `W_l[i][j] = 0.5 * sqrt(2 / n_in) * sin(l * 10000 + i * n_out + j + 1)`,
//!   `b_l = 0`.  The `sin` of distinct integers gives well-spread values in
//!   [-1, 1]; the `sqrt(2/n_in)` factor is He-style scaling for ReLU units.
//! * **Full-batch gradient descent**: one gradient step per epoch on the
//!   entire training set (no shuffling, no minibatches, no momentum, no
//!   regularization), pinned learning rate and epoch count.
//! * **Pinned standardization**: per-feature `(x - mean) / std` with
//!   training-set population moments (`std` divisor n); `std < 1e-12` is
//!   replaced by 1.
//!
//! Forward / backward equations
//! ----------------------------
//! ```text
//! A_0 = X
//! Z_l = A_{l-1} W_l + b_l ;  A_l = relu(Z_l)      for hidden layers
//! Z_L = A_{L-1} W_L + b_L ;  P = softmax(Z_L)     (row-wise, max-shifted)
//! L(theta) = -(1/n) * sum_i log P[i, y_i]         (mean cross-entropy)
//!
//! G_L   = (P - Y) / n                             (Y one-hot)
//! dW_l  = A_{l-1}^T G_l ;   db_l = sum_rows(G_l)
//! G_{l-1} = (G_l W_l^T) .* 1{Z_{l-1} > 0}         (ReLU gate)
//! ```
//!
//! Update: `W_l -= lr * dW_l`, `b_l -= lr * db_l`.

use crate::error::{LobError, Result};
use crate::matrix::Matrix;

const STD_FLOOR: f64 = 1e-12;
const LOG_FLOOR: f64 = 1e-300;

/// Deterministic sin-based weight matrix (module docs formula).
///
/// # Errors
/// [`LobError::InvalidInput`] if `n_in` or `n_out` is zero.
pub fn pinned_init(n_in: usize, n_out: usize, layer_index: usize) -> Result<Matrix> {
    if n_in == 0 || n_out == 0 {
        return Err(LobError::InvalidInput(
            "n_in, n_out must be positive".to_string(),
        ));
    }
    let scale = 0.5 * (2.0 / n_in as f64).sqrt();
    let mut w = Matrix::zeros(n_in, n_out);
    for i in 0..n_in {
        for j in 0..n_out {
            let arg = (layer_index * 10000 + i * n_out + j + 1) as f64;
            w.set(i, j, scale * arg.sin());
        }
    }
    Ok(w)
}

fn check_matrix(x: &Matrix) -> Result<()> {
    if x.rows() == 0 || x.cols() == 0 {
        return Err(LobError::InvalidInput(
            "X must be a non-empty 2-D array".to_string(),
        ));
    }
    if !x.all_finite() {
        return Err(LobError::InvalidInput("X must be finite".to_string()));
    }
    Ok(())
}

fn check_labels(y: &[usize], n_rows: usize, n_classes: usize) -> Result<()> {
    if y.len() != n_rows {
        return Err(LobError::InvalidInput(format!(
            "y must have length {n_rows}, got {}",
            y.len()
        )));
    }
    if y.iter().any(|&v| v >= n_classes) {
        return Err(LobError::InvalidInput(format!(
            "labels must be in [0, {n_classes})"
        )));
    }
    Ok(())
}

/// Pinned per-feature standardization fitted on the training split only.
#[derive(Debug, Clone, Default)]
pub struct Standardizer {
    mean: Vec<f64>,
    std: Vec<f64>,
}

impl Standardizer {
    /// Fresh, unfitted standardizer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute population mean/std per column; std below 1e-12 becomes 1.
    ///
    /// # Errors
    /// [`LobError::InvalidInput`] on empty or non-finite input.
    pub fn fit(&mut self, x: &Matrix) -> Result<&mut Self> {
        check_matrix(x)?;
        let (n, d) = (x.rows(), x.cols());
        let mut mean = vec![0.0; d];
        for r in 0..n {
            let row = x.row(r);
            for (m, &v) in mean.iter_mut().zip(row) {
                *m += v;
            }
        }
        for m in mean.iter_mut() {
            *m /= n as f64;
        }
        let mut var = vec![0.0; d];
        for r in 0..n {
            let row = x.row(r);
            for c in 0..d {
                let diff = row[c] - mean[c];
                var[c] += diff * diff;
            }
        }
        let std: Vec<f64> = var
            .iter()
            .map(|v| {
                let s = (v / n as f64).sqrt(); // population (divisor n) — pinned
                if s < STD_FLOOR {
                    1.0
                } else {
                    s
                }
            })
            .collect();
        self.mean = mean;
        self.std = std;
        Ok(self)
    }

    /// Apply `(x - mean) / std`; requires [`Standardizer::fit`] first.
    ///
    /// # Errors
    /// [`LobError::InvalidInput`] if unfitted, or on a column-count mismatch.
    pub fn transform(&self, x: &Matrix) -> Result<Matrix> {
        if self.mean.is_empty() {
            return Err(LobError::InvalidInput(
                "Standardizer::transform called before fit".to_string(),
            ));
        }
        check_matrix(x)?;
        if x.cols() != self.mean.len() {
            return Err(LobError::InvalidInput(format!(
                "expected {} columns, got {}",
                self.mean.len(),
                x.cols()
            )));
        }
        let mut out = x.clone();
        for r in 0..out.rows() {
            let row = out.row_mut(r);
            for c in 0..row.len() {
                row[c] = (row[c] - self.mean[c]) / self.std[c];
            }
        }
        Ok(out)
    }
}

/// Row-wise softmax with the pinned max-shift for overflow safety.
pub fn softmax(z: &Matrix) -> Matrix {
    let mut p = z.clone();
    for r in 0..p.rows() {
        let row = p.row_mut(r);
        let mut mx = f64::NEG_INFINITY;
        for &v in row.iter() {
            if v > mx {
                mx = v;
            }
        }
        let mut sum = 0.0;
        for v in row.iter_mut() {
            *v = (*v - mx).exp();
            sum += *v;
        }
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
    p
}

/// `Z = A W + b` with explicit sequential accumulation (row vectors,
/// `W` of shape `(n_in, n_out)`).
fn dense(a: &Matrix, w: &Matrix, b: &[f64]) -> Matrix {
    let (n, d, m) = (a.rows(), a.cols(), w.cols());
    let mut z = Matrix::zeros(n, m);
    for r in 0..n {
        let arow = a.row(r);
        let zrow = z.row_mut(r);
        for (k, &av) in arow.iter().enumerate().take(d) {
            let wrow = w.row(k);
            for j in 0..m {
                zrow[j] += av * wrow[j];
            }
        }
        for j in 0..m {
            zrow[j] += b[j];
        }
    }
    z
}

/// Deterministic MLP classifier (see module docs).
#[derive(Debug, Clone)]
pub struct Mlp {
    sizes: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Vec<f64>>,
}

impl Mlp {
    /// Build a network `n_inputs -> hidden... -> n_classes` with the pinned
    /// deterministic init.  Empty `hidden` gives the logistic-regression
    /// baseline.
    ///
    /// # Errors
    /// [`LobError::InvalidInput`] on zero sizes or `n_classes < 2`.
    pub fn new(n_inputs: usize, hidden: &[usize], n_classes: usize) -> Result<Self> {
        if n_inputs == 0 || n_classes < 2 {
            return Err(LobError::InvalidInput(
                "n_inputs must be positive and n_classes >= 2".to_string(),
            ));
        }
        if hidden.iter().any(|&h| h == 0) {
            return Err(LobError::InvalidInput(format!(
                "hidden sizes must be positive integers, got {hidden:?}"
            )));
        }
        let mut sizes = vec![n_inputs];
        sizes.extend_from_slice(hidden);
        sizes.push(n_classes);
        let mut weights = Vec::with_capacity(sizes.len() - 1);
        let mut biases = Vec::with_capacity(sizes.len() - 1);
        for l in 0..sizes.len() - 1 {
            weights.push(pinned_init(sizes[l], sizes[l + 1], l)?);
            biases.push(vec![0.0; sizes[l + 1]]);
        }
        Ok(Self {
            sizes,
            weights,
            biases,
        })
    }

    /// Layer sizes including input and output.
    pub fn sizes(&self) -> &[usize] {
        &self.sizes
    }

    /// The weight matrices (for tests / inspection).
    pub fn weights(&self) -> &[Matrix] {
        &self.weights
    }

    /// The bias vectors (for tests / inspection).
    pub fn biases(&self) -> &[Vec<f64>] {
        &self.biases
    }

    /// Mutable weight matrices — for parameter perturbation such as
    /// finite-difference gradient checks; training itself must go through
    /// [`Mlp::fit`].
    pub fn weights_mut(&mut self) -> &mut [Matrix] {
        &mut self.weights
    }

    /// Mutable bias vectors (see [`Mlp::weights_mut`]).
    pub fn biases_mut(&mut self) -> &mut [Vec<f64>] {
        &mut self.biases
    }

    /// Forward pass: pre-activations `Z_l`, activations `A_l` (A_0 = X),
    /// output probabilities.
    fn forward(&self, x: &Matrix) -> (Vec<Matrix>, Vec<Matrix>, Matrix) {
        let n_layers = self.weights.len();
        let mut zs: Vec<Matrix> = Vec::with_capacity(n_layers);
        let mut activations: Vec<Matrix> = Vec::with_capacity(n_layers);
        activations.push(x.clone());
        for l in 0..n_layers {
            let z = dense(&activations[l], &self.weights[l], &self.biases[l]);
            if l < n_layers - 1 {
                let mut a = z.clone();
                for v in a.data_mut() {
                    if *v < 0.0 {
                        *v = 0.0; // ReLU
                    }
                }
                activations.push(a);
            }
            zs.push(z);
        }
        let probs = softmax(&zs[n_layers - 1]);
        (zs, activations, probs)
    }

    fn check_input(&self, x: &Matrix) -> Result<()> {
        check_matrix(x)?;
        if x.cols() != self.sizes[0] {
            return Err(LobError::InvalidInput(format!(
                "expected {} features, got {}",
                self.sizes[0],
                x.cols()
            )));
        }
        Ok(())
    }

    /// Class probabilities, shape `(n, n_classes)`; rows sum to 1.
    ///
    /// # Errors
    /// [`LobError::InvalidInput`] on invalid input or feature-count mismatch.
    pub fn predict_proba(&self, x: &Matrix) -> Result<Matrix> {
        self.check_input(x)?;
        Ok(self.forward(x).2)
    }

    /// Argmax class per row (ties resolve to the lowest class index).
    ///
    /// # Errors
    /// Same as [`Mlp::predict_proba`].
    pub fn predict(&self, x: &Matrix) -> Result<Vec<usize>> {
        let p = self.predict_proba(x)?;
        Ok((0..p.rows())
            .map(|r| {
                let row = p.row(r);
                let mut best = 0usize;
                for (c, &v) in row.iter().enumerate() {
                    if v > row[best] {
                        best = c;
                    }
                }
                best
            })
            .collect())
    }

    /// Mean cross-entropy `-(1/n) sum log P[i, y_i]` (probability floored at
    /// 1e-300 before the log so a confident wrong answer cannot produce
    /// `-inf`).
    ///
    /// # Errors
    /// [`LobError::InvalidInput`] on invalid input or labels.
    pub fn loss(&self, x: &Matrix, y: &[usize]) -> Result<f64> {
        self.check_input(x)?;
        check_labels(y, x.rows(), self.sizes[self.sizes.len() - 1])?;
        let probs = self.forward(x).2;
        let mut total = 0.0;
        for (r, &yi) in y.iter().enumerate() {
            total += probs.get(r, yi).max(LOG_FLOOR).ln();
        }
        Ok(-total / x.rows() as f64)
    }

    /// Loss plus gradients `dL/dW_l`, `dL/db_l` via backprop (exactly the
    /// pinned equations in the module docs).
    ///
    /// # Errors
    /// [`LobError::InvalidInput`] on invalid input or labels.
    #[allow(clippy::type_complexity)]
    pub fn loss_and_grads(
        &self,
        x: &Matrix,
        y: &[usize],
    ) -> Result<(f64, Vec<Matrix>, Vec<Vec<f64>>)> {
        self.check_input(x)?;
        let n_classes = self.sizes[self.sizes.len() - 1];
        check_labels(y, x.rows(), n_classes)?;
        let n = x.rows();
        let (zs, activations, probs) = self.forward(x);
        let mut total = 0.0;
        for (r, &yi) in y.iter().enumerate() {
            total += probs.get(r, yi).max(LOG_FLOOR).ln();
        }
        let loss = -total / n as f64;

        // G_L = (P - onehot(y)) / n
        let mut g = probs;
        for (r, &yi) in y.iter().enumerate() {
            let row = g.row_mut(r);
            row[yi] -= 1.0;
        }
        for v in g.data_mut() {
            *v /= n as f64;
        }

        let n_layers = self.weights.len();
        let mut grads_w: Vec<Matrix> = Vec::with_capacity(n_layers);
        let mut grads_b: Vec<Vec<f64>> = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            grads_w.push(Matrix::zeros(self.sizes[l], self.sizes[l + 1]));
            grads_b.push(vec![0.0; self.sizes[l + 1]]);
        }
        for l in (0..n_layers).rev() {
            let (d, m) = (self.sizes[l], self.sizes[l + 1]);
            // dW_l = A_{l-1}^T G_l  (accumulate over sample rows in order)
            let a = &activations[l];
            let dw = &mut grads_w[l];
            for r in 0..n {
                let arow = a.row(r);
                let grow = g.row(r);
                for k in 0..d {
                    let av = arow[k];
                    let dwrow = dw.row_mut(k);
                    for j in 0..m {
                        dwrow[j] += av * grow[j];
                    }
                }
            }
            // db_l = column sums of G_l
            let db = &mut grads_b[l];
            for r in 0..n {
                let grow = g.row(r);
                for j in 0..m {
                    db[j] += grow[j];
                }
            }
            if l > 0 {
                // G_{l-1} = (G_l W_l^T) .* [Z_{l-1} > 0]
                let w = &self.weights[l];
                let z_prev = &zs[l - 1];
                let mut g_prev = Matrix::zeros(n, d);
                for r in 0..n {
                    let grow = g.row(r);
                    let zrow = z_prev.row(r);
                    let out = g_prev.row_mut(r);
                    for (k, ov) in out.iter_mut().enumerate() {
                        if zrow[k] > 0.0 {
                            let wrow = w.row(k);
                            let mut acc = 0.0;
                            for j in 0..m {
                                acc += grow[j] * wrow[j];
                            }
                            *ov = acc;
                        }
                    }
                }
                g = g_prev;
            }
        }
        Ok((loss, grads_w, grads_b))
    }

    /// Full-batch gradient descent (pinned schedule).
    ///
    /// One gradient step per epoch on the whole training set; the returned
    /// history holds the training loss *after* each epoch's update
    /// (`history[0]` is the golden "loss after epoch 1").
    ///
    /// # Errors
    /// [`LobError::InvalidInput`] on invalid inputs or hyperparameters
    /// (`learning_rate <= 0`, `epochs == 0`).
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &[usize],
        learning_rate: f64,
        epochs: usize,
    ) -> Result<Vec<f64>> {
        if !(learning_rate.is_finite() && learning_rate > 0.0) {
            return Err(LobError::InvalidInput(format!(
                "learning_rate must be positive, got {learning_rate}"
            )));
        }
        if epochs == 0 {
            return Err(LobError::InvalidInput(
                "epochs must be a positive integer, got 0".to_string(),
            ));
        }
        self.check_input(x)?;
        check_labels(y, x.rows(), self.sizes[self.sizes.len() - 1])?;
        let mut history = Vec::with_capacity(epochs);
        for _ in 0..epochs {
            let (_, grads_w, grads_b) = self.loss_and_grads(x, y)?;
            for l in 0..self.weights.len() {
                let w = self.weights[l].data_mut();
                for (wv, gv) in w.iter_mut().zip(grads_w[l].data()) {
                    *wv -= learning_rate * gv;
                }
                for (bv, gv) in self.biases[l].iter_mut().zip(&grads_b[l]) {
                    *bv -= learning_rate * gv;
                }
            }
            history.push(self.loss(x, y)?);
        }
        Ok(history)
    }
}

// -------------------------------------------------------------------- metrics

/// Fraction of matching labels.
///
/// # Errors
/// [`LobError::InvalidInput`] on empty or mismatched arrays.
pub fn accuracy(y_true: &[usize], y_pred: &[usize]) -> Result<f64> {
    if y_true.is_empty() || y_true.len() != y_pred.len() {
        return Err(LobError::InvalidInput(
            "y_true and y_pred must be equal-length non-empty arrays".to_string(),
        ));
    }
    let matches = y_true
        .iter()
        .zip(y_pred)
        .filter(|(a, b)| a == b)
        .count();
    Ok(matches as f64 / y_true.len() as f64)
}

/// `(n_classes x n_classes)` counts, rows = true class, cols = predicted.
///
/// # Errors
/// [`LobError::InvalidInput`] on mismatched lengths or out-of-range labels.
pub fn confusion_matrix(
    y_true: &[usize],
    y_pred: &[usize],
    n_classes: usize,
) -> Result<Vec<Vec<usize>>> {
    check_labels(y_true, y_true.len(), n_classes)?;
    check_labels(y_pred, y_true.len(), n_classes)?;
    let mut cm = vec![vec![0usize; n_classes]; n_classes];
    for (&t, &p) in y_true.iter().zip(y_pred) {
        cm[t][p] += 1;
    }
    Ok(cm)
}

/// Most frequent label; ties resolve to the smallest label (pinned).
///
/// # Errors
/// [`LobError::InvalidInput`] on an empty array.
pub fn majority_class(y: &[usize]) -> Result<usize> {
    if y.is_empty() {
        return Err(LobError::InvalidInput(
            "y must be a non-empty 1-D array".to_string(),
        ));
    }
    let max_label = *y.iter().max().expect("non-empty");
    let mut counts = vec![0usize; max_label + 1];
    for &v in y {
        counts[v] += 1;
    }
    let mut best = 0usize;
    for (label, &c) in counts.iter().enumerate() {
        if c > counts[best] {
            best = label; // strict > keeps the smallest label on ties
        }
    }
    Ok(best)
}

/// PnL of trading the predicted direction with half-spread costs.
///
/// Position per row: +1 for class 2 (up), -1 for class 0 (down), 0 for flat.
/// Gross PnL adds `pos * fwd_ret` (mid-to-mid, ticks per unit size); net PnL
/// subtracts a half-spread cost `spread / 2` per non-flat signal, charged on
/// entry only (the exit is marked at mid).  Returns `(gross, net, n_trades)`.
///
/// This is a signal diagnostic, not a tradable strategy: positions of
/// consecutive rows overlap (one signal per event, each held `H` events);
/// no fees, latency or fill probability.
///
/// # Errors
/// [`LobError::InvalidInput`] on empty, mismatched, or non-finite inputs, or
/// a label outside `{0, 1, 2}`.
pub fn signal_pnl(y_pred: &[usize], fwd_ret: &[f64], spread: &[f64]) -> Result<(f64, f64, usize)> {
    if y_pred.is_empty() {
        return Err(LobError::InvalidInput(
            "empty prediction array".to_string(),
        ));
    }
    if y_pred.len() != fwd_ret.len() || y_pred.len() != spread.len() {
        return Err(LobError::InvalidInput(
            "y_pred, fwd_ret, spread must be equal-length 1-D arrays".to_string(),
        ));
    }
    if y_pred.iter().any(|&v| v > 2) {
        return Err(LobError::InvalidInput(
            "y_pred labels must be in {0, 1, 2}".to_string(),
        ));
    }
    if fwd_ret.iter().chain(spread).any(|v| !v.is_finite()) {
        return Err(LobError::InvalidInput(
            "fwd_ret and spread must be finite".to_string(),
        ));
    }
    let mut gross = 0.0;
    let mut cost = 0.0;
    let mut n_trades = 0usize;
    for i in 0..y_pred.len() {
        let pos = y_pred[i] as f64 - 1.0; // {0,1,2} -> {-1,0,+1}
        gross += pos * fwd_ret[i];
        cost += pos.abs() * spread[i] * 0.5;
        if y_pred[i] != 1 {
            n_trades += 1;
        }
    }
    Ok((gross, gross - cost, n_trades))
}
