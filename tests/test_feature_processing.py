import logging
import math

import pytest

from btcmi.feature_processing import normalize_features


def test_normalize_features_skips_zero_scale(caplog):
    feats = {"a": 1.0, "b": 2.0}
    scales = {"a": 0.0, "b": 2.0}
    with caplog.at_level(logging.DEBUG):
        norm = normalize_features(feats, scales)
    assert "a" not in norm
    assert norm["b"] == pytest.approx(math.tanh(1.0))
    assert any("zero scale" in m for m in caplog.messages)

