//! Feature pipeline tests: pinned OFI formula on hand-built states, label
//! epsilon dead zone, row-emission guards, chronological split, and config
//! validation.

mod common;

use lobmod::{
    chronological_split, compute_features, load_feature_config, ofi_at, ofi_series, BookStates,
    FeatureConfig, LABEL_DOWN, LABEL_FLAT, LABEL_UP,
};

/// Hand-built book states with a constant two-sided book except where noted.
fn constant_states(n: usize) -> BookStates {
    BookStates {
        best_bid: vec![99.0; n],
        best_ask: vec![101.0; n],
        bid_size: vec![5.0; n],
        ask_size: vec![5.0; n],
        bid_depth_k: vec![15.0; n],
        ask_depth_k: vec![15.0; n],
        mid: vec![100.0; n],
        microprice: vec![100.0; n],
        spread: vec![2.0; n],
    }
}

#[test]
fn ofi_series_pinned_formula() {
    let mut st = constant_states(4);
    // event 1: bid size grows 5 -> 8 at the same prices
    st.bid_size[1] = 8.0;
    // event 2: bid price improves 99 -> 100 with size 4 (old level had 8)
    st.best_bid[2] = 100.0;
    st.bid_size[2] = 4.0;
    // event 3: ask price improves (drops) 101 -> 100.5-like: use 100.0 is
    // crossed, so improve to 100.9 is off-grid; instead ask worsens 101 -> 102
    st.best_ask[3] = 102.0;
    st.ask_size[3] = 6.0;
    st.best_bid[3] = 100.0;
    st.bid_size[3] = 4.0;
    let e = ofi_series(&st);
    assert!(e[0].is_nan(), "e_0 has no predecessor");
    // event 1: equal prices both sides -> size changes: (8 - 5) - (5 - 5) = 3
    assert_eq!(e[1], 3.0);
    // event 2: bid up -> +qb_n = 4; ask equal -> -qa_n + qa_{n-1} = 0
    assert_eq!(e[2], 4.0);
    // event 3: bid equal -> qb_n - qb_{n-1} = 0; ask up (worse) -> +qa_{n-1} = 5
    assert_eq!(e[3], 5.0);
}

#[test]
fn ofi_nan_when_a_side_is_empty() {
    let mut st = constant_states(3);
    st.best_ask[1] = f64::NAN; // ask side empty at event 1
    let e = ofi_series(&st);
    assert!(e[1].is_nan());
    assert!(e[2].is_nan(), "previous state has an empty side");
}

#[test]
fn ofi_at_window_and_errors() {
    let st = constant_states(10);
    // constant book: every e_n = 0, so any fitting window sums to 0
    assert_eq!(ofi_at(&st, 5, 3).unwrap(), 0.0);
    assert!(ofi_at(&st, 5, 0).is_err()); // bad window
    assert!(ofi_at(&st, 20, 3).is_err()); // index out of range
    assert!(ofi_at(&st, 2, 3).is_err()); // window does not fit
}

fn tiny_config() -> FeatureConfig {
    FeatureConfig {
        ofi_window: 2,
        ret_window: 2,
        horizon: 1,
        epsilon: 0.25,
        depth_k: 3,
        hidden: vec![4],
        learning_rate: 0.1,
        epochs: 10,
        train_frac: 0.5,
    }
}

#[test]
fn labels_use_epsilon_dead_zone() {
    let mut st = constant_states(8);
    // mids: rows are emitted for i in [2, 7); label from mid_{i+1} - mid_i
    st.mid = vec![100.0, 100.0, 100.0, 100.5, 100.5, 100.2, 99.0, 99.0];
    let fs = compute_features(&st, &tiny_config()).unwrap();
    assert_eq!(fs.event_index, vec![2, 3, 4, 5, 6]);
    // d = +0.5 (> eps) -> up; 0 -> flat; -0.3 -> down; -1.2 -> down; 0 -> flat
    assert_eq!(
        fs.y,
        vec![LABEL_UP, LABEL_FLAT, LABEL_DOWN, LABEL_DOWN, LABEL_FLAT]
    );
    assert_eq!(fs.fwd_ret[0], 0.5);
    assert_eq!(fs.spread[0], 2.0);
    // feature columns: ret_i = mid_i - mid_{i-2}
    assert_eq!(fs.x.get(3, 4), 100.2 - 100.5);
    // queue imbalance of a balanced book is 0, depth ratio 1
    assert_eq!(fs.x.get(0, 1), 0.0);
    assert_eq!(fs.x.get(0, 3), 1.0);
}

#[test]
fn rows_with_undefined_inputs_are_skipped() {
    let mut st = constant_states(10);
    st.best_ask[4] = f64::NAN; // empty ask side at event 4
    st.mid[4] = f64::NAN;
    st.spread[4] = f64::NAN;
    st.ask_size[4] = 0.0;
    let fs = compute_features(&st, &tiny_config()).unwrap();
    // events 4, 5 (NaN in OFI window), and the row whose ret/mid touches 4
    // are dropped; remaining rows still keep replay order
    assert!(!fs.event_index.contains(&4));
    assert!(!fs.event_index.contains(&5));
    assert!(fs.event_index.windows(2).all(|w| w[0] < w[1]));
}

#[test]
fn all_undefined_rows_is_an_error() {
    let mut st = constant_states(6);
    for v in st.best_ask.iter_mut() {
        *v = f64::NAN; // ask side never exists
    }
    for v in st.mid.iter_mut() {
        *v = f64::NAN;
    }
    assert!(compute_features(&st, &tiny_config()).is_err());
}

#[test]
fn chronological_split_pinned() {
    let (tr, te) = chronological_split(10, 0.7).unwrap();
    assert_eq!((tr.start, tr.end), (0, 7));
    assert_eq!((te.start, te.end), (7, 10));
    // floor semantics
    let (tr, te) = chronological_split(9, 0.7).unwrap();
    assert_eq!(tr.end, 6);
    assert_eq!(te.len(), 3);
    assert!(chronological_split(1, 0.7).is_err());
    assert!(chronological_split(10, 0.0).is_err());
    assert!(chronological_split(10, 1.0).is_err());
    assert!(chronological_split(10, 0.01).is_err()); // empty train side
}

#[test]
fn feature_config_validation() {
    let good = load_feature_config(&common::data_path("features_config.json")).unwrap();
    assert_eq!(good.ofi_window, 50);
    assert_eq!(good.hidden, vec![16, 8]);
    let mut bad = tiny_config();
    bad.ofi_window = 0;
    assert!(bad.validate().is_err());
    let mut bad = tiny_config();
    bad.epsilon = -0.1;
    assert!(bad.validate().is_err());
    let mut bad = tiny_config();
    bad.learning_rate = 0.0;
    assert!(bad.validate().is_err());
    let mut bad = tiny_config();
    bad.train_frac = 1.5;
    assert!(bad.validate().is_err());
    let mut bad = tiny_config();
    bad.hidden = vec![16, 0];
    assert!(bad.validate().is_err());
    assert!(load_feature_config("/nonexistent.json").is_err());
}

// ---------------------------------------------------------------- review additions

#[test]
fn ofi_at_window_boundary_is_pinned() {
    // PT-3: error iff event_index < window; event_index == window is valid
    let st = constant_states(10);
    assert!(ofi_at(&st, 3, 4).is_err());
    let v = ofi_at(&st, 4, 4).unwrap(); // sums e_1..e_4: finite, zero
    assert!(v.is_finite());
    assert_eq!(v, 0.0);
    assert!(ofi_at(&st, 9, 10).is_err());
    assert_eq!(ofi_at(&st, 9, 9).unwrap(), 0.0);
}
