"""Smoke tests against the in-process TestClient app."""


def test_health_returns_200(client):
    r = client.get("/health")
    assert r.status_code == 200


def test_health_reports_loaded_model(client):
    data = client.get("/health").json()
    assert data["model_loaded"] is True
    assert data["model_version"] == "fake-v1"


def test_health_payload_shape(client):
    data = client.get("/health").json()
    assert {"status", "model_loaded", "model_version"}.issubset(data.keys())


def test_api_metadata_endpoint(client):
    r = client.get("/api")
    assert r.status_code == 200
    assert "/predict/text" in r.json()["endpoints"]


def test_docs_endpoint_is_available(client):
    r = client.get("/docs")
    assert r.status_code == 200


def test_openapi_json_is_valid(client):
    r = client.get("/openapi.json").json()
    assert r["openapi"].startswith("3.")
    assert "/predict/text" in r["paths"]


def test_unknown_endpoint_returns_404(client):
    r = client.get("/this/route/does/not/exist")
    assert r.status_code == 404
