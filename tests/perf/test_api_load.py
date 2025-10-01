from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import pytest


def load_values(path: Path) -> List[float]:
    with path.open() as f:
        data = json.load(f)
    return data["values"]


pytestmark = pytest.mark.regression


def summarize(values: List[float]) -> Tuple[float, int]:
    if not values:
        return 0.0, 0
    avg = sum(values) / len(values)
    return avg, len(values)


def test_baseline_average() -> None:
    root = Path(__file__).resolve().parents[2]
    baseline_path = root / "docs" / "api_load_baseline.json"
    values = load_values(baseline_path)
    avg, count = summarize(values)
    assert count == len(values)
    assert abs(avg - (sum(values) / len(values))) < 1e-9
