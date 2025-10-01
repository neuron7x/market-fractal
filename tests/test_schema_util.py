import json
from pathlib import Path

import pytest

from btcmi import schema_util
from btcmi.schema_util import validate_json


pytestmark = pytest.mark.smoke

def test_validate_json_additional_properties(tmp_path):
    pytest.importorskip("jsonschema")
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"foo": {"type": "string"}},
        "required": ["foo"],
        "additionalProperties": False,
    }
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema))
    data = {"foo": "bar", "extra": "baz"}
    with pytest.raises(ValueError):
        validate_json(data, schema_path)


def test_validate_json_without_jsonschema(monkeypatch, tmp_path):
    """Raises a clear ImportError when jsonschema is absent."""
    import sys

    # Simulate jsonschema being unavailable
    monkeypatch.setitem(sys.modules, "jsonschema", None)

    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["foo"],
    }
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema))

    with pytest.raises(ImportError) as err:
        validate_json({"foo": "bar"}, schema_path)
    assert "jsonschema" in str(err.value)


def test_schema_cached(monkeypatch, tmp_path):
    pytest.importorskip("jsonschema")
    schema_util._load_schema.cache_clear()
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
    }
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema))

    calls = 0
    original = Path.read_text

    def fake_read_text(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    validate_json({}, schema_path)
    validate_json({}, schema_path)
    assert calls == 1
