//! Deterministic NN tests: pinned init, gradient check against central finite
//! differences, loss decrease, softmax normalization, bit-identical retrain,
//! degenerate labels, Nelder-Mead behaviour, and validation errors.

use lobmod::{
    accuracy, confusion_matrix, majority_class, nelder_mead, nelder_mead_default, pinned_init,
    signal_pnl, softmax, Matrix, Mlp, Standardizer,
};

/// Small deterministic dataset with all three classes present.
fn toy_data() -> (Matrix, Vec<usize>) {
    let mut x = Matrix::zeros(9, 3);
    for r in 0..9 {
        for c in 0..3 {
            // spread of positive/negative values, no randomness
            x.set(r, c, ((r * 3 + c) as f64 * 0.7).sin());
        }
    }
    let y = vec![0, 1, 2, 0, 1, 2, 2, 1, 0];
    (x, y)
}

#[test]
fn pinned_init_formula() {
    let w = pinned_init(3, 4, 1).unwrap();
    let scale = 0.5 * (2.0_f64 / 3.0).sqrt();
    // W_l[i][j] = scale * sin(l*10000 + i*n_out + j + 1)
    assert_eq!(w.get(0, 0), scale * (10001.0_f64).sin());
    assert_eq!(w.get(2, 3), scale * (10012.0_f64).sin());
    assert!(pinned_init(0, 4, 0).is_err());
}

#[test]
fn gradient_check_vs_central_differences() {
    let (x, y) = toy_data();
    let net = Mlp::new(3, &[4], 3).unwrap();
    let (_, grads_w, grads_b) = net.loss_and_grads(&x, &y).unwrap();
    let h = 1e-6;
    for l in 0..2 {
        // weights
        for i in 0..net.weights()[l].rows() {
            for j in 0..net.weights()[l].cols() {
                let mut plus = net.clone();
                let w = plus.weights_mut()[l].get(i, j);
                plus.weights_mut()[l].set(i, j, w + h);
                let mut minus = net.clone();
                minus.weights_mut()[l].set(i, j, w - h);
                let fd = (plus.loss(&x, &y).unwrap() - minus.loss(&x, &y).unwrap()) / (2.0 * h);
                let bp = grads_w[l].get(i, j);
                let denom = fd.abs().max(bp.abs()).max(1e-8);
                assert!(
                    (fd - bp).abs() / denom <= 1e-6,
                    "dW[{l}][{i}][{j}]: fd {fd} vs bp {bp}"
                );
            }
        }
        // biases
        for j in 0..net.biases()[l].len() {
            let mut plus = net.clone();
            let b = plus.biases_mut()[l][j];
            plus.biases_mut()[l][j] = b + h;
            let mut minus = net.clone();
            minus.biases_mut()[l][j] = b - h;
            let fd = (plus.loss(&x, &y).unwrap() - minus.loss(&x, &y).unwrap()) / (2.0 * h);
            let bp = grads_b[l][j];
            let denom = fd.abs().max(bp.abs()).max(1e-8);
            assert!(
                (fd - bp).abs() / denom <= 1e-6,
                "db[{l}][{j}]: fd {fd} vs bp {bp}"
            );
        }
    }
}

#[test]
fn loss_decreases_over_epochs() {
    let (x, y) = toy_data();
    let mut net = Mlp::new(3, &[4], 3).unwrap();
    let hist = net.fit(&x, &y, 0.1, 300).unwrap();
    assert_eq!(hist.len(), 300);
    assert!(hist.iter().all(|v| v.is_finite()));
    assert!(hist[299] < hist[0], "loss should decrease: {} -> {}", hist[0], hist[299]);
    // and substantially so on this separable toy problem
    assert!(hist[299] < 0.9 * hist[0]);
}

#[test]
fn softmax_rows_sum_to_one() {
    let (x, y) = toy_data();
    let mut net = Mlp::new(3, &[4], 3).unwrap();
    net.fit(&x, &y, 0.1, 50).unwrap();
    let p = net.predict_proba(&x).unwrap();
    for r in 0..p.rows() {
        let sum: f64 = p.row(r).iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "row {r} sums to {sum}");
        assert!(p.row(r).iter().all(|&v| v >= 0.0));
    }
    // softmax is shift-invariant and handles large logits (max-shift pinned)
    let z = Matrix::from_vec(1, 3, vec![1000.0, 1001.0, 1002.0]);
    let s = softmax(&z);
    let total: f64 = s.row(0).iter().sum();
    assert!((total - 1.0).abs() < 1e-12);
}

#[test]
fn retraining_is_bit_identical() {
    let (x, y) = toy_data();
    let mut a = Mlp::new(3, &[4, 3], 3).unwrap();
    let ha = a.fit(&x, &y, 0.1, 120).unwrap();
    let mut b = Mlp::new(3, &[4, 3], 3).unwrap();
    let hb = b.fit(&x, &y, 0.1, 120).unwrap();
    assert_eq!(ha, hb, "loss history must be bit-identical");
    assert_eq!(a.predict(&x).unwrap(), b.predict(&x).unwrap());
    for l in 0..a.weights().len() {
        assert_eq!(a.weights()[l].data(), b.weights()[l].data());
    }
}

#[test]
fn degenerate_one_class_labels_train_and_lose_to_majority() {
    let (x, _) = toy_data();
    let y = vec![1usize; x.rows()];
    let mut net = Mlp::new(3, &[4], 3).unwrap();
    let hist = net.fit(&x, &y, 0.1, 100).unwrap();
    assert!(hist.iter().all(|v| v.is_finite()));
    assert!(hist[99] < hist[0], "training must still run and improve");
    let maj = majority_class(&y).unwrap();
    assert_eq!(maj, 1);
    let pred_mj = vec![maj; y.len()];
    let acc_mj = accuracy(&y, &pred_mj).unwrap();
    let acc_nn = accuracy(&y, &net.predict(&x).unwrap()).unwrap();
    assert_eq!(acc_mj, 1.0);
    assert!(acc_mj >= acc_nn, "the majority baseline wins — not an error");
}

#[test]
fn standardizer_pinned_moments() {
    let x = Matrix::from_vec(4, 2, vec![1.0, 10.0, 2.0, 10.0, 3.0, 10.0, 4.0, 10.0]);
    let mut s = Standardizer::new();
    s.fit(&x).unwrap();
    let t = s.transform(&x).unwrap();
    // column 0: mean 2.5, population std sqrt(1.25)
    let std0 = (1.25_f64).sqrt();
    assert!((t.get(0, 0) - (1.0 - 2.5) / std0).abs() < 1e-15);
    // column 1 is constant: std < 1e-12 becomes 1, so values center to 0
    assert_eq!(t.get(0, 1), 0.0);
    // transform requires fit, and column counts must match
    assert!(Standardizer::new().transform(&x).is_err());
    let wrong = Matrix::zeros(2, 3);
    assert!(s.transform(&wrong).is_err());
}

#[test]
fn prediction_ties_resolve_to_lowest_class() {
    // an untrained single-layer net with symmetric input can produce exact
    // probability ties only in contrived cases; instead verify the argmax
    // rule directly on identical logits via a zero-input row.
    let net = Mlp::new(2, &[], 3).unwrap();
    let x = Matrix::zeros(1, 2); // logits = biases = 0 -> uniform probs
    assert_eq!(net.predict(&x).unwrap(), vec![0]);
}

#[test]
fn metrics_and_pnl() {
    let y_true = vec![0, 1, 2, 2];
    let y_pred = vec![0, 1, 1, 2];
    assert_eq!(accuracy(&y_true, &y_pred).unwrap(), 0.75);
    let cm = confusion_matrix(&y_true, &y_pred, 3).unwrap();
    assert_eq!(cm[2][1], 1);
    assert_eq!(cm[0][0], 1);
    assert_eq!(cm.iter().flatten().sum::<usize>(), 4);
    // ties in majority_class resolve to the smallest label
    assert_eq!(majority_class(&[2, 0, 2, 0]).unwrap(), 0);
    // pnl: positions {-1,0,+1}; costs charged on non-flat signals only
    let (gross, net, n) = signal_pnl(&[2, 1, 0], &[1.0, 5.0, -2.0], &[1.0, 1.0, 2.0]).unwrap();
    assert_eq!(gross, 1.0 * 1.0 + 0.0 + (-1.0) * (-2.0));
    assert_eq!(net, gross - (0.5 + 0.0 + 1.0));
    assert_eq!(n, 2);
}

#[test]
fn nn_validation_errors() {
    let (x, y) = toy_data();
    assert!(Mlp::new(0, &[4], 3).is_err());
    assert!(Mlp::new(3, &[0], 3).is_err());
    assert!(Mlp::new(3, &[4], 1).is_err());
    let mut net = Mlp::new(3, &[4], 3).unwrap();
    assert!(net.fit(&x, &y, 0.0, 10).is_err()); // lr <= 0
    assert!(net.fit(&x, &y, f64::NAN, 10).is_err());
    assert!(net.fit(&x, &y, 0.1, 0).is_err()); // epochs == 0
    let bad_y = vec![3usize; x.rows()]; // label out of range
    assert!(net.fit(&x, &bad_y, 0.1, 10).is_err());
    let short_y = vec![0usize; 2]; // length mismatch
    assert!(net.loss(&x, &short_y).is_err());
    let wrong_cols = Matrix::zeros(4, 2); // feature-count mismatch
    assert!(net.predict(&wrong_cols).is_err());
    let mut bad_x = x.clone();
    bad_x.set(0, 0, f64::NAN); // non-finite input
    assert!(net.predict(&bad_x).is_err());
    assert!(accuracy(&[], &[]).is_err());
    assert!(signal_pnl(&[], &[], &[]).is_err());
    assert!(signal_pnl(&[1], &[f64::NAN], &[1.0]).is_err());
    assert!(majority_class(&[]).is_err());
}

// ------------------------------------------------------------- Nelder-Mead

#[test]
fn nelder_mead_minimizes_quadratic() {
    let res = nelder_mead_default(
        |x| (x[0] - 2.0).powi(2) + (x[1] + 3.0).powi(2),
        &[0.0, 0.0],
    )
    .unwrap();
    assert!(res.converged);
    assert!((res.x[0] - 2.0).abs() < 1e-6 && (res.x[1] + 3.0).abs() < 1e-6);
    assert!(res.fval < 1e-12);
}

#[test]
fn nelder_mead_minimizes_rosenbrock() {
    let rosen = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
    let res = nelder_mead_default(rosen, &[-1.2, 1.0]).unwrap();
    assert!(res.converged);
    assert!((res.x[0] - 1.0).abs() < 1e-4 && (res.x[1] - 1.0).abs() < 1e-4);
}

#[test]
fn nelder_mead_iteration_cap_and_errors() {
    // a cap of 3 iterations cannot converge on Rosenbrock
    let rosen = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
    let res = nelder_mead(rosen, &[-1.2, 1.0], 1e-8, 1e-10, 3).unwrap();
    assert!(!res.converged);
    assert_eq!(res.n_iter, 3);
    assert!(nelder_mead(|_| 0.0, &[], 1e-8, 1e-10, 100).is_err());
    assert!(nelder_mead(|_| 0.0, &[f64::NAN], 1e-8, 1e-10, 100).is_err());
    assert!(nelder_mead(|x| x[0], &[1.0], 0.0, 1e-10, 100).is_err());
    assert!(nelder_mead(|x| x[0], &[1.0], 1e-8, 1e-10, 0).is_err());
}
