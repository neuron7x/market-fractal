"""Second generation layered signal engine utilities."""

from __future__ import annotations
from typing import Dict, List
import math
from btcmi.config import SCALES as CONFIG_SCALES
from btcmi.feature_processing import normalize_features, weighted_score
from btcmi.nagr import nagr

SCALES = CONFIG_SCALES


def normalize_layer(
    feats: Dict[str, float], scales: Dict[str, float]
) -> Dict[str, float]:
    """Apply normalization to a layer's feature set."""

    return normalize_features(feats, scales)


def level_signal(
    norm: Dict[str, float],
    weights: Dict[str, float],
    nagr_nodes: List[dict],
) -> tuple[float, Dict[str, float]]:
    """Blend linear feature score with network rating for one level.

    Args:
        norm: Normalized feature values.
        weights: Weight mapping for the level.
        nagr_nodes: Network graph nodes contributing to NAGR score.

    Returns:
        Tuple of combined level signal and feature contributions.

    """
    base, contrib = weighted_score(norm, weights)
    return 0.8 * base + 0.2 * nagr(nagr_nodes), contrib


def router_weights(vol_pctl: float):
    """Select level weights based on volume percentile.

    Args:
        vol_pctl: Market volume percentile in [0, 1].

    Returns:
        A tuple of descriptor string and weight mapping for levels.

    """
    if vol_pctl < 0.2:
        return "low", {"L1": 0.15, "L2": 0.35, "L3": 0.50}
    if vol_pctl < 0.6:
        return "mid", {"L1": 0.25, "L2": 0.40, "L3": 0.35}
    return "high", {"L1": 0.40, "L2": 0.40, "L3": 0.20}


def combine_levels(L1: float, L2: float, L3: float, w):
    """Merge signals from all levels using provided weights.

    Args:
        L1: Level-one signal value.
        L2: Level-two signal value.
        L3: Level-three signal value.
        w: Weight mapping for each level.

    Returns:
        Combined score clipped to [-1, 1].

    """
    req = {"L1", "L2", "L3"}
    if not req.issubset(w):
        missing = ", ".join(sorted(req - w.keys()))
        raise ValueError(f"missing weights for: {missing}")

    total = w["L1"] + w["L2"] + w["L3"]
    if math.isclose(total, 0.0, abs_tol=1e-12):
        raise ValueError("sum of weights must be non-zero")
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        w = {k: v / total for k, v in w.items()}

    s = w["L1"] * L1 + w["L2"] * L2 + w["L3"] * L3
    return max(-1.0, min(1.0, s))


def layer_equal_weights(norm: Dict[str, float]) -> Dict[str, float]:
    """Generate equal weights for a layer's features.

    Args:
        norm: Normalized feature values for a layer.

    Returns:
        Mapping assigning an equal weight to each feature.

    """
    if not norm:
        return {}
    n = len(norm)
    w = 1.0 / n
    return {k: w for k in norm.keys()}
