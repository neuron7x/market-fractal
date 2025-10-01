#!/usr/bin/env python3
import pytest

from btcmi.engine_v1 import completeness, base_signal, nagr_score, combine
from btcmi.config import NORM_SCALE, SCENARIO_WEIGHTS
from btcmi.feature_processing import normalize_features


pytestmark = pytest.mark.smoke

@pytest.fixture
def raw_features():
    return {
        "price_change_pct": 1e6,
        "volume_change_pct": -1e6,
        "funding_rate_bps": "bad",
        "oi_change_pct": 1.0,
        "extra": 123,
    }


@pytest.fixture
def norm_extreme():
    return {k: 5.0 for k in NORM_SCALE.keys()}


def test_normalize_handles_non_numeric_and_extreme(raw_features):
    assert normalize_features({}, NORM_SCALE) == {}
    norm = normalize_features(raw_features, NORM_SCALE)
    assert set(norm.keys()) == {
        "price_change_pct",
        "volume_change_pct",
        "oi_change_pct",
        "extra",
    }
    assert norm["price_change_pct"] == pytest.approx(1.0)
    assert norm["volume_change_pct"] == pytest.approx(-1.0)


def test_completeness_counts_numeric_only(raw_features):
    assert completeness({}) == 0.0
    comp = completeness(raw_features)
    assert comp == pytest.approx(3 / len(NORM_SCALE))


def test_base_signal_empty_inputs():
    res = base_signal("intraday", {})
    assert res.score == 0.0
    assert res.contributions == {}
    assert res.weights == SCENARIO_WEIGHTS["intraday"]


def test_base_signal_uses_present_features():
    res = base_signal("intraday", {"price_change_pct": 1.0})
    assert res.score == 1.0
    assert res.contributions == {"price_change_pct": pytest.approx(0.35)}


def test_base_signal_clips_extreme(norm_extreme):
    res = base_signal("intraday", norm_extreme)
    assert res.score == 1.0
    neg = {k: -v for k, v in norm_extreme.items()}
    res_neg = base_signal("intraday", neg)
    assert res_neg.score == -1.0


def test_nagr_score_handles_empty_and_zero_weights():
    assert nagr_score([]) == 0.0
    assert nagr_score([{"weight": 0.0, "score": 1.0}]) == 0.0


def test_nagr_score_clips_extreme():
    nodes = [{"weight": 1.0, "score": 2.0}]
    assert nagr_score(nodes) == 1.0


def test_nagr_score_skips_non_numeric_nodes():
    nodes = [
        {"weight": 1.0, "score": 0.5},
        {"weight": "bad", "score": 1.0},
        {"weight": 1.0, "score": "bad"},
    ]
    assert nagr_score(nodes) == pytest.approx(0.5)


def test_combine_zero_scores():
    assert combine(0.0, 0.0) == 0.0


@pytest.mark.parametrize(
    "base, nagr, expected",
    [
        (10.0, 10.0, 1.0),
        (-10.0, -10.0, -1.0),
    ],
)
def test_combine_extreme_values_clipped(base, nagr, expected):
    assert combine(base, nagr) == expected
