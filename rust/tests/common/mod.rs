//! Shared helpers for the integration test binaries.
//!
//! Each test binary compiles this module independently and uses a different
//! subset of it, so the module-level `dead_code` allowance is expected.
#![allow(dead_code)]

use std::collections::HashMap;
use std::fs;

use serde_json::Value;

/// Absolute path of a file inside the shared `data/` directory
/// (`../data` relative to the language dir, per the port brief).
pub fn data_path(name: &str) -> String {
    format!("{}/../data/{name}", env!("CARGO_MANIFEST_DIR"))
}

/// Parse `data/golden/golden.json` into a name -> case map.
pub fn load_golden() -> HashMap<String, Value> {
    let text = fs::read_to_string(data_path("golden/golden.json")).expect("golden.json readable");
    let root: Value = serde_json::from_str(&text).expect("golden.json is valid JSON");
    root["cases"]
        .as_array()
        .expect("cases array")
        .iter()
        .map(|c| (c["name"].as_str().expect("case name").to_string(), c.clone()))
        .collect()
}

/// Assert one golden case: every `expect` key must match `computed` within
/// the case's tolerance.
pub fn check_case(case: &Value, computed: &[(&str, f64)]) {
    let tol = case["tol"].as_f64().expect("tol");
    let name = case["name"].as_str().unwrap_or("?");
    let expect = case["expect"].as_object().expect("expect map");
    let computed: HashMap<&str, f64> = computed.iter().copied().collect();
    assert_eq!(
        expect.len(),
        computed.len(),
        "case {name}: expect-key count mismatch"
    );
    for (key, expected) in expect {
        let expected = expected.as_f64().expect("numeric expected value");
        let got = *computed
            .get(key.as_str())
            .unwrap_or_else(|| panic!("case {name}: missing computed key {key}"));
        assert!(
            (got - expected).abs() <= tol,
            "case {name}, key {key}: computed {got} vs expected {expected} (tol {tol})"
        );
    }
}

/// Read a one-column (`t`) or two-column (`t,type`) Hawkes CSV.
pub fn read_hawkes_csv(name: &str) -> (Vec<f64>, Vec<usize>) {
    let path = data_path(name);
    let text = fs::read_to_string(&path).expect("hawkes csv readable");
    let mut times = Vec::new();
    let mut types = Vec::new();
    for line in text.lines().skip(1) {
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split(',');
        times.push(parts.next().unwrap().parse::<f64>().expect("time"));
        if let Some(k) = parts.next() {
            types.push(k.parse::<usize>().expect("type"));
        }
    }
    (times, types)
}
