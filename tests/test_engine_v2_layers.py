#!/usr/bin/env python3
import pytest

from btcmi.engine_v2 import (
    level_signal,
    router_weights,
    combine_levels,
    nagr,
)
from btcmi.feature_processing import normalize_features, weighted_score


pytestmark = pytest.mark.smoke

def test_normalize_features_handles_empty_and_non_numeric_and_extreme():
    assert normalize_features({}, {"a": 1.0}) == {}
    feats = {"x": 1e6, "y": "bad"}
    scales = {"x": 1.0}
    norm = normalize_features(feats, scales)
    assert set(norm.keys()) == {"x"}
    assert norm["x"] == pytest.approx(1.0)


def test_weighted_score_zero_weights_and_missing_features():
    norm = {"a": 0.5}
    weights = {"a": 0.0, "b": 1.0}
    score, contrib = weighted_score(norm, weights)
    assert score == 0.0
    assert contrib == {"a": 0.0}


def test_weighted_score_clips_extreme_values():
    norm = {"a": 5.0}
    weights = {"a": 1.0}
    score, contrib = weighted_score(norm, weights)
    assert score == 1.0
    assert contrib["a"] == 5.0


def test_level_signal_empty_inputs():
    sig, contrib = level_signal({}, {}, [])
    assert sig == 0.0
    assert contrib == {}


def test_level_signal_zero_weights_uses_nagr():
    norm = {"a": 1.0}
    weights = {"a": 0.0}
    nodes = [{"weight": 1.0, "score": 1.0}]
    sig, contrib = level_signal(norm, weights, nodes)
    assert sig == pytest.approx(0.2)
    assert contrib == {"a": 0.0}


def test_level_signal_extreme_inputs_clipped():
    norm = {"a": 5.0}
    weights = {"a": 1.0}
    nodes = [{"weight": 1.0, "score": 2.0}]
    sig, _ = level_signal(norm, weights, nodes)
    assert sig == 1.0


@pytest.mark.parametrize(
    "vol_pctl, expected",
    [
        (0.0, "low"),
        (0.2, "mid"),
        (0.6, "high"),
        (1.0, "high"),
    ],
)
def test_router_weights_boundaries(vol_pctl, expected):
    regime, w = router_weights(vol_pctl)
    assert regime == expected
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_combine_levels_zero_weights():
    with pytest.raises(ValueError):
        combine_levels(1.0, -1.0, 0.5, weights={"L1": 0.0, "L2": 0.0, "L3": 0.0})


def test_combine_levels_normalizes_weights():
    sig = combine_levels(1.0, 0.0, 0.0, weights={"L1": 2.0, "L2": 1.0, "L3": 1.0})
    assert sig == pytest.approx(0.5)


def test_combine_levels_missing_weight_raises():
    with pytest.raises(ValueError):
        combine_levels(0.0, 0.0, 0.0, weights={"L1": 0.5, "L2": 0.5})


def test_combine_levels_extreme_values_clipped():
    sig = combine_levels(10.0, -10.0, 10.0, weights={"L1": 0.3, "L2": 0.3, "L3": 0.4})
    assert sig == 1.0
    sig_neg = combine_levels(
        -10.0, -10.0, -10.0, weights={"L1": 0.3, "L2": 0.3, "L3": 0.4}
    )
    assert sig_neg == -1.0


def test_nagr_skips_non_numeric_nodes():
    nodes = [
        {"weight": 1.0, "score": 0.5},
        {"weight": None, "score": 1.0},
        {"weight": 1.0, "score": "bad"},
    ]
    assert nagr(nodes) == pytest.approx(0.5)
