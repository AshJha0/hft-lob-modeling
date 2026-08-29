"""Shared fixtures: bundled data, replayed states, trained models (cached)."""

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import pytest

import lobmod as lm

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")


@pytest.fixture(scope="session")
def golden() -> Dict[str, dict]:
    with open(os.path.join(DATA_DIR, "golden", "golden.json")) as fh:
        raw = json.load(fh)
    return {c["name"]: c for c in raw["cases"]}


@pytest.fixture(scope="session")
def dgp() -> dict:
    with open(os.path.join(DATA_DIR, "dgp_params.json")) as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def cfg() -> lm.FeatureConfig:
    return lm.load_feature_config(os.path.join(DATA_DIR, "features_config.json"))


@pytest.fixture(scope="session")
def events() -> List[lm.Event]:
    return lm.read_events_csv(os.path.join(DATA_DIR, "lob_events.csv"))


@pytest.fixture(scope="session")
def states(events, cfg) -> lm.BookStates:
    return lm.replay(events, depth_k=cfg.depth_k)


@pytest.fixture(scope="session")
def uni_times() -> np.ndarray:
    return np.loadtxt(os.path.join(DATA_DIR, "hawkes_uni.csv"), skiprows=1)


@pytest.fixture(scope="session")
def biv_data():
    raw = np.loadtxt(os.path.join(DATA_DIR, "hawkes_biv.csv"), skiprows=1, delimiter=",")
    return raw[:, 0], raw[:, 1].astype(int)


@pytest.fixture(scope="session")
def uni_fit(uni_times, dgp) -> lm.HawkesUniFit:
    return lm.fit_uni(uni_times, dgp["hawkes_uni"]["T"])


@pytest.fixture(scope="session")
def biv_fit(biv_data, dgp) -> lm.HawkesBivFit:
    t, k = biv_data
    return lm.fit_biv(t, k, dgp["hawkes_biv"]["T"])


@pytest.fixture(scope="session")
def feature_set(states, cfg) -> lm.FeatureSet:
    return lm.compute_features(states, cfg)


@pytest.fixture(scope="session")
def pipeline(feature_set, cfg):
    """Standardized split + trained NN and logistic models (trained once)."""
    fs = feature_set
    train_idx, test_idx = lm.chronological_split(len(fs), cfg.train_frac)
    std = lm.Standardizer().fit(fs.X[train_idx])
    x_train, x_test = std.transform(fs.X[train_idx]), std.transform(fs.X[test_idx])
    net = lm.MLP(fs.X.shape[1], cfg.hidden, 3)
    history = net.fit(x_train, fs.y[train_idx], cfg.learning_rate, cfg.epochs)
    logistic = lm.MLP(fs.X.shape[1], (), 3)
    logistic.fit(x_train, fs.y[train_idx], cfg.learning_rate, cfg.epochs)
    return {
        "train_idx": train_idx, "test_idx": test_idx, "std": std,
        "x_train": x_train, "x_test": x_test,
        "y_train": fs.y[train_idx], "y_test": fs.y[test_idx],
        "net": net, "logistic": logistic, "history": history,
    }
