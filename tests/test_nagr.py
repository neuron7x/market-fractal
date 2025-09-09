import pytest

from btcmi.nagr import nagr


def test_nagr_handles_empty_and_zero_weights():
    assert nagr([]) == 0.0
    assert nagr([{"weight": 0.0, "score": 1.0}]) == 0.0


def test_nagr_clips_extreme():
    nodes = [{"weight": 1.0, "score": 2.0}]
    assert nagr(nodes) == 1.0


def test_nagr_skips_non_numeric_nodes():
    nodes = [
        {"weight": 1.0, "score": 0.5},
        {"weight": "bad", "score": 1.0},
        {"weight": 1.0, "score": "bad"},
    ]
    assert nagr(nodes) == pytest.approx(0.5)
