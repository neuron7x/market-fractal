"""Shared utilities for NAGR (network aggregated graph rating) computation."""

from __future__ import annotations

from typing import Iterable, Mapping, Any
import logging

logger = logging.getLogger(__name__)


def nagr(nodes: Iterable[Mapping[str, Any]]) -> float:
    """Aggregate network graph ratings.

    Args:
        nodes: Iterable of node dictionaries with ``weight`` and ``score``.

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


__all__ = ["nagr"]
