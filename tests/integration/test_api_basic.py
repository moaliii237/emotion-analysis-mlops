"""
Smoke tests for the basic API surface, /health, /api, /docs, /openapi.json.
These do not call the model, so they're fast (~1s total).
"""

from __future__ import annotations

import requests


def test_health_returns_200(base_url):
    r = requests.get(f"{base_url}/health")
    assert r.status_code == 200


def test_health_payload_shape(base_url):
    data = requests.get(f"{base_url}/health").json()
    assert set(["status", "model_loaded", "model_version"]).issubset(data.keys())


def test_health_reports_model_loaded(base_url, model_ready):
    data = requests.get(f"{base_url}/health").json()
    assert data["model_loaded"] is True
    assert data["model_version"]


def test_api_metadata_endpoint(base_url):
    r = requests.get(f"{base_url}/api")
    assert r.status_code == 200
    data = r.json()
    assert "endpoints" in data
    assert "/predict/text" in data["endpoints"]


def test_docs_endpoint_is_available(base_url):
    """FastAPI auto-docs should be served at /docs."""
    r = requests.get(f"{base_url}/docs")
    assert r.status_code == 200
    assert "swagger" in r.text.lower() or "openapi" in r.text.lower()


def test_openapi_json_is_valid(base_url):
    r = requests.get(f"{base_url}/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    assert spec["openapi"].startswith("3.")
    assert "/predict/text" in spec["paths"]


def test_unknown_endpoint_returns_404(base_url):
    r = requests.get(f"{base_url}/this/route/does/not/exist")
    assert r.status_code == 404


def test_static_frontend_is_served(base_url):
    """The HTML frontend should be reachable at /app.html."""
    r = requests.get(f"{base_url}/app.html")
    assert r.status_code == 200
    assert "EmotionScope" in r.text or "emotionscope" in r.text.lower()
