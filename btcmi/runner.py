from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict

from btcmi import engine_v1 as v1
from btcmi import engine_v2 as v2
from btcmi import engine_nf3p as nf3p
from btcmi.enums import Scenario, Window
from btcmi.io import write_output as write_output  # noqa: F401


def _validate_scenario_window(data: dict[str, Any]) -> tuple[Scenario, Window]:
    """Return the scenario and window ensuring both are valid."""

    scenario = data.get("scenario")
    if scenario is None:
        raise ValueError("'scenario' field is required")
    try:
        scenario_enum = (
            scenario if isinstance(scenario, Scenario) else Scenario(scenario)
        )
    except ValueError as exc:  # pragma: no cover - defensive
        allowed = ", ".join(sorted(s.value for s in Scenario))
        raise ValueError("'scenario' must be one of: " + allowed) from exc

    window = data.get("window")
    if window is None:
        raise ValueError("'window' field is required")
    try:
        window_enum = window if isinstance(window, Window) else Window(window)
    except ValueError as exc:  # pragma: no cover - defensive
        allowed = ", ".join(sorted(w.value for w in Window))
        raise ValueError("'window' must be one of: " + allowed) from exc
    return scenario_enum, window_enum


def run_v1(
    data: dict[str, Any],
    fixed_ts: str | None,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the v1 engine and optionally persist the output.

    Parameters
    ----------
    data:
        Input payload conforming to the input schema.
    fixed_ts:
        Timestamp used for the ``asof`` field.  When ``None`` the current
        UTC time is used.
    out_path:
        Optional path where the rendered JSON output should be written.  When
        ``None`` (the default) the output is only returned and no file is
        created.
    """
    scenario, window = _validate_scenario_window(data)
    feats: Dict[str, float] = data.get("features", {})
    norm = v1.normalize(feats)
    base_res = v1.base_signal(scenario.value, norm)
    ng = v1.nagr_score(data.get("nagr_nodes", []))
    overall = v1.combine(base_res.score, ng)
    comp = v1.completeness(feats)
    conf = round(0.5 + 0.5 * comp, 3)
    notes: list[str] = []
    constraints = False
    if comp < 0.6:
        notes.append("low_feature_completeness")
    asof = fixed_ts or dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    out: dict[str, Any] = {
        "schema_version": data.get("schema_version", "2.0.0"),
        "lineage": data.get("lineage", {}),
        "asof": asof,
        "summary": {
            "scenario": scenario.value,
            "window": window.value,
            "overall_signal": round(overall, 6),
            "confidence": conf,
            "router_path": f"{scenario.value}/v1",
            "nagr_score": round(ng, 6),
            "advisories": notes,
        },
        "details": {
            "normalized_features": {k: round(v, 6) for k, v in norm.items()},
            "weights": base_res.weights,
            "contributions": {
                k: round(v, 6) for k, v in base_res.contributions.items()
            },
            "constraints_applied": constraints,
            "diagnostics": {"completeness": round(comp, 3), "notes": notes},
        },
    }
    if out_path is not None:
        write_output(out, out_path)
    return out


def run_v2(
    data: dict[str, Any],
    fixed_ts: str | None,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the v2 fractal engine and optionally persist the output."""
    scenario, window = _validate_scenario_window(data)
    f1 = data.get("features_micro", {})
    f2 = data.get("features_mezo", {})
    f3 = data.get("features_macro", {})
    try:
        vol_pctl = float(data.get("vol_regime_pctl", 0.5))
    except (TypeError, ValueError) as exc:
        raise ValueError("'vol_regime_pctl' must be a number in [0, 1]") from exc
    if not 0.0 <= vol_pctl <= 1.0:
        raise ValueError("'vol_regime_pctl' must be a number in [0, 1]")
    n1 = v2.normalize_layer(f1, v2.SCALES["L1"])
    n2 = v2.normalize_layer(f2, v2.SCALES["L2"])
    n3 = v2.normalize_layer(f3, v2.SCALES["L3"])
    w1 = v2.layer_equal_weights(n1)
    w2 = v2.layer_equal_weights(n2)
    w3 = v2.layer_equal_weights(n3)
    s1, _ = v2.level_signal(n1, w1, data.get("nagr_nodes", []))
    s2, _ = v2.level_signal(n2, w2, data.get("nagr_nodes", []))
    s3, _ = v2.level_signal(n3, w3, data.get("nagr_nodes", []))
    regime, weights = v2.router_weights(vol_pctl)
    overall = v2.combine_levels(s1, s2, s3, weights=weights)
    coverage = sum(len(x) > 0 for x in [n1, n2, n3]) / 3.0
    conf = round(0.5 + 0.5 * min(coverage, 1.0), 3)
    notes: list[str] = []
    asof = fixed_ts or dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    out: dict[str, Any] = {
        "schema_version": data.get("schema_version", "2.0.0"),
        "lineage": data.get("lineage", {}),
        "asof": asof,
        "summary": {
            "scenario": scenario.value,
            "window": window.value,
            "overall_signal": round(overall, 6),
            "confidence": conf,
            "router_path": f"{scenario.value}/v2.fractal",
            "nagr_score": 0.0,
            "advisories": notes,
            "overall_signal_L1": round(s1, 6),
            "overall_signal_L2": round(s2, 6),
            "overall_signal_L3": round(s3, 6),
            "level_weights": weights,
        },
        "details": {
            "normalized_micro": {k: round(v, 6) for k, v in n1.items()},
            "normalized_mezo": {k: round(v, 6) for k, v in n2.items()},
            "normalized_macro": {k: round(v, 6) for k, v in n3.items()},
            "router_regime": regime,
            "diagnostics": {"completeness": round(coverage, 3), "notes": notes},
        },
    }
    if out_path is not None:
        write_output(out, out_path)
    return out


def run_nf3p(
    data: dict[str, Any],
    fixed_ts: str | None,
    out_path: str | Path | None = None,
) -> dict[str, Any]:  # noqa: D401 - short wrapper
    """Run the NF3P engine and optionally persist the output."""
    scenario, window = _validate_scenario_window(data)
    f1 = data.get("features_micro", {})
    f2 = data.get("features_mezo", {})
    f3 = data.get("features_macro", {})
    predictions, backtest = nf3p.predictions_and_backtest(f1, f2, f3)
    asof = fixed_ts or dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    out: dict[str, Any] = {
        "schema_version": data.get("schema_version", "2.0.0"),
        "lineage": data.get("lineage", {}),
        "asof": asof,
        "scenario": scenario.value,
        "window": window.value,
        "predictions": predictions,
        "backtest": backtest,
    }
    if out_path is not None:
        write_output(out, out_path)
    return out


__all__ = ["run_v1", "run_v2", "run_nf3p"]
