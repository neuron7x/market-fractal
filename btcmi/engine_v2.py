"""Second generation layered signal engine utilities."""

from __future__ import annotations
from typing import Dict, List
import math
import logging
from btcmi.config import SCALES as CONFIG_SCALES
from btcmi.feature_processing import normalize_features, weighted_score

SCALES = CONFIG_SCALES
logger = logging.getLogger(__name__)


def normalize_layer(
    feats: Dict[str, float], scales: Dict[str, float]
) -> Dict[str, float]:
    """Apply normalization to a layer's feature set."""

    return normalize_features(feats, scales)


def nagr(nodes: List[dict]) -> float:
    """Aggregate network graph ratings.

    Args:
        nodes: Sequence of node dicts with ``weight`` and ``score``.

    Returns:
        Weighted average score clipped to [-1, 1].

    """
    if not nodes:
        return 0.0
    num = 0.0
    den = 0.0
    for n in nodes:
        try:
            w = float(n.get("weight", 0.0))
            sc = float(n.get("score", 0.0))
        except (TypeError, ValueError) as exc:
            logger.debug("Skipping node with non-numeric data %s: %s", n, exc)
            continue
        num += w * sc
        den += abs(w)
    den = den or 1.0
    return max(-1.0, min(1.0, num / den))


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


def combine_levels(l1: float, l2: float, l3: float, weights: dict[str, float]) -> float:
    """Merge signals from all levels using provided weights.

    Args:
        l1: Level-one signal value.
        l2: Level-two signal value.
        l3: Level-three signal value.
        weights: Weight mapping for each level.

    Returns:
        Combined score clipped to [-1, 1].

    """
    req = {"L1", "L2", "L3"}
    if not req.issubset(weights):
        missing = ", ".join(sorted(req - weights.keys()))
        raise ValueError(f"missing weights for: {missing}")

    total = weights["L1"] + weights["L2"] + weights["L3"]
    if math.isclose(total, 0.0, abs_tol=1e-12):
        raise ValueError("sum of weights must be non-zero")
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        weights = {k: v / total for k, v in weights.items()}

    s = weights["L1"] * l1 + weights["L2"] * l2 + weights["L3"] * l3
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
