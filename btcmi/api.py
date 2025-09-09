"""Web API for running BTC market intelligence scenarios and utilities.

This FastAPI application exposes several endpoints:

* ``POST /run`` – execute a scenario and return the results.
* ``POST /validate/{schema_name}`` – validate a payload against a schema.
* ``GET /metrics`` – expose Prometheus metrics about the service.
* ``GET /healthz`` – basic health check endpoint.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
from collections import defaultdict, deque
from functools import lru_cache
from time import monotonic
from typing import Any, Callable, Dict

from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security
from fastapi.security import APIKeyHeader
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest
from pydantic import BaseModel, ConfigDict

from btcmi.enums import Scenario, Window
from btcmi.runner import run_nf3p, run_v1, run_v2
from btcmi.schema_util import SCHEMA_REGISTRY, validate_json

logger = logging.getLogger(__name__)

app = FastAPI()


@lru_cache()
def load_runners() -> Dict[str, Callable]:
    """Return a mapping of mode names to runner implementations."""
    return {
        "v1": run_v1,
        "v2.fractal": run_v2,
        "v2.nf3p": run_nf3p,
    }


REQUEST_COUNTER = Counter("btcmi_requests", "Total HTTP requests", ["endpoint"])

# store recent request timestamps per client for throttling
_req_times: defaultdict[str, deque] = defaultdict(deque)

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def get_api_key(api_key: str = Security(api_key_header)) -> str:
    """Validate API key from request headers."""
    expected = os.getenv("BTCMI_API_KEY", "changeme")
    if api_key and hmac.compare_digest(api_key, expected):
        return api_key
    raise HTTPException(status_code=401, detail="invalid or missing API key")


@app.middleware("http")
async def count_requests(request: Request, call_next: Callable):
    """Increment a Prometheus counter for each incoming HTTP request."""
    try:
        response = await call_next(request)
    finally:
        REQUEST_COUNTER.labels(endpoint=request.url.path).inc()
    return response


@app.middleware("http")
async def throttle_requests(request: Request, call_next: Callable):
    """Naive rate limiter to reduce brute-force attempts."""
    limit = int(os.getenv("BTCMI_RATE_LIMIT", "60"))
    window = int(os.getenv("BTCMI_RATE_LIMIT_WINDOW", "60"))
    client = request.client.host if request.client else "unknown"
    now = monotonic()
    q = _req_times[client]
    while q and q[0] <= now - window:
        q.popleft()
    if len(q) >= limit:
        return Response(status_code=429, content="too many requests")
    q.append(now)
    return await call_next(request)


class RunRequest(BaseModel):
    scenario: Scenario
    window: Window
    mode: str = "v1"

    # Allow additional, unmodelled fields in the request payload. Using
    # ``ConfigDict`` avoids deprecation warnings from Pydantic v2 where the
    # old ``class Config`` style is no longer supported.
    model_config = ConfigDict(extra="allow")


class Summary(BaseModel):
    scenario: Scenario
    window: Window

    # Permit additional keys when parsing summary data.
    model_config = ConfigDict(extra="allow")


class RunResponse(BaseModel):
    schema_version: str
    lineage: Dict[str, str]
    summary: Summary
    details: Dict[str, Any]
    asof: str


class ValidateRequest(BaseModel):
    # Permit arbitrary fields during validation requests.
    model_config = ConfigDict(extra="allow")


@app.post("/run", response_model=RunResponse)
async def run_endpoint(
    payload: RunRequest, api_key: str = Depends(get_api_key)
) -> RunResponse:
    data = payload.model_dump()
    mode = data.get("mode", "v1")
    runner = load_runners().get(mode)
    if runner is None:
        raise HTTPException(status_code=400, detail=f"unknown mode: {mode}")
    try:
        await asyncio.to_thread(validate_json, data, SCHEMA_REGISTRY["input"])
    except Exception as exc:  # noqa: BLE001
        logger.exception("validation_failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        # API requests should not leave artifacts on disk; explicitly disable
        # writing the output file.
        result = await asyncio.to_thread(runner, data, None, out_path=None)
    except (KeyError, ValueError) as exc:
        logger.exception("runner_error")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("runner_error")
        raise HTTPException(status_code=500, detail="internal error") from exc
    return result


@app.post("/validate/{schema_name}")
async def validate_endpoint(
    schema_name: str, payload: ValidateRequest, api_key: str = Depends(get_api_key)
):
    schema_path = SCHEMA_REGISTRY.get(schema_name)
    if schema_path is None:
        raise HTTPException(status_code=404, detail="schema not found")
    try:
        await asyncio.to_thread(
            validate_json, payload.model_dump(), schema_path
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("validation_failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"valid": True}


@app.get("/metrics")
async def metrics() -> Response:
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


__all__ = ["app", "load_runners", "REQUEST_COUNTER"]

