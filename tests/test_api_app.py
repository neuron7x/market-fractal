import json
import pathlib
import logging
from collections import deque

from fastapi.testclient import TestClient
from prometheus_client import CONTENT_TYPE_LATEST
from prometheus_client.parser import text_string_to_metric_families

import btcmi.api as api
from btcmi.api import app, load_runners, REQUEST_COUNTER, _req_times
from btcmi.logging_cfg import JsonFormatter

R = pathlib.Path(__file__).resolve().parents[1]


def _load_example(name: str) -> dict:
    return json.loads((R / "examples" / f"{name}.json").read_text())


HEADERS = {"X-API-Key": "changeme"}


def test_run_success():
    client = TestClient(app)
    payload = _load_example("intraday")
    resp = client.post("/run", json=payload, headers=HEADERS)
    assert resp.status_code == 200
    assert "summary" in resp.json()


def test_run_invalid_payload():
    client = TestClient(app)
    resp = client.post("/run", json={"mode": "v1"}, headers=HEADERS)
    assert resp.status_code == 422


def test_run_unknown_mode():
    client = TestClient(app)
    payload = _load_example("intraday")
    payload["mode"] = "foo"
    resp = client.post("/run", json=payload, headers=HEADERS)
    assert resp.status_code == 400
    assert "unknown mode" in resp.json()["detail"]


def test_run_runner_exception(monkeypatch, caplog):
    caplog.handler.setFormatter(JsonFormatter())
    caplog.set_level(logging.ERROR)

    def bad_runner(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("boom")

    monkeypatch.setitem(load_runners(), "v1", bad_runner)
    client = TestClient(app)
    payload = _load_example("intraday")
    resp = client.post("/run", json=payload, headers=HEADERS)
    assert resp.status_code == 500

    record = json.loads(caplog.text)
    assert record["msg"] == "runner_error"
    assert record["level"] == "error"
    assert "ts" in record


def test_run_out_path_none(monkeypatch):
    seen = {}

    def runner(p, _t, *, out_path: str | pathlib.Path | None = None):
        seen["out_path"] = out_path
        return {
            "schema_version": "2.0.0",
            "lineage": {},
            "summary": {"scenario": "intraday", "window": "1h"},
            "details": {},
            "asof": "1970-01-01T00:00:00Z",
        }

    monkeypatch.setitem(load_runners(), "v1", runner)
    client = TestClient(app)
    payload = _load_example("intraday")
    resp = client.post("/run", json=payload, headers=HEADERS)
    assert resp.status_code == 200
    assert seen["out_path"] is None


def test_validate_input_valid():
    client = TestClient(app)
    payload = _load_example("intraday")
    resp = client.post("/validate/input", json=payload, headers=HEADERS)
    assert resp.status_code == 200
    assert resp.json() == {"valid": True}


def test_validate_input_invalid():
    client = TestClient(app)
    resp = client.post("/validate/input", json={"schema_version": "2.0.0"}, headers=HEADERS)
    assert resp.status_code == 400


def test_metrics_prometheus_text_and_counters():
    client = TestClient(app)
    base_validate = REQUEST_COUNTER.labels(endpoint="/validate/input")._value.get()

    payload = _load_example("intraday")
    client.post("/validate/input", json=payload, headers=HEADERS)
    client.post("/validate/input", json={"schema_version": "2.0.0"}, headers=HEADERS)

    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == CONTENT_TYPE_LATEST

    metrics = {mf.name: mf for mf in text_string_to_metric_families(resp.text)}
    samples = {s.labels["endpoint"]: s.value for s in metrics["btcmi_requests"].samples}

    assert samples["/validate/input"] == base_validate + 2


def test_run_requires_auth():
    client = TestClient(app)
    payload = _load_example("intraday")
    resp = client.post("/run", json=payload)
    assert resp.status_code == 401


def test_rate_limit(monkeypatch):
    monkeypatch.setenv("BTCMI_RATE_LIMIT", "1")
    _req_times.clear()
    client = TestClient(app)
    payload = _load_example("intraday")
    assert client.post("/run", json=payload, headers=HEADERS).status_code == 200
    assert client.post("/run", json=payload, headers=HEADERS).status_code == 429


def test_throttle_prunes_stale_clients(monkeypatch):
    monkeypatch.setenv("BTCMI_RATE_LIMIT", "10")
    monkeypatch.setenv("BTCMI_RATE_LIMIT_WINDOW", "60")
    _req_times.clear()
    _req_times["old"] = deque([0.0])
    monkeypatch.setattr(api, "monotonic", lambda: 100.0)
    client = TestClient(app)
    payload = _load_example("intraday")
    assert client.post("/run", json=payload, headers=HEADERS).status_code == 200
    assert "old" not in _req_times


def test_throttle_limits_client_cache(monkeypatch):
    monkeypatch.setenv("BTCMI_RATE_LIMIT", "10")
    monkeypatch.setenv("BTCMI_RATE_LIMIT_WINDOW", "60")
    monkeypatch.setenv("BTCMI_RATE_LIMIT_MAX_CLIENTS", "2")
    _req_times.clear()
    _req_times["c1"] = deque([90.0])
    _req_times["c2"] = deque([95.0])
    monkeypatch.setattr(api, "monotonic", lambda: 100.0)
    client = TestClient(app)
    payload = _load_example("intraday")
    assert client.post("/run", json=payload, headers=HEADERS).status_code == 200
    assert len(_req_times) <= 2
    assert "c1" not in _req_times
