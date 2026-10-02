from backend.app.config import settings
from backend.tests._testclient import TestClient


def _client(monkeypatch):
    from backend.app.main import app
    monkeypatch.setattr(settings, "auth_enabled", False)
    return TestClient(app)  # without `with`: the lifespan (DB, GPU) does not start


def test_static_files_are_revalidated_after_every_deploy(monkeypatch):
    c = _client(monkeypatch)
    for path in ("/login.html", "/js/login.js", "/css/styles.css", "/i18n/es.json"):
        r = c.get(path)
        assert r.status_code == 200, path
        assert r.headers["cache-control"] == "no-cache", path
        assert r.headers.get("etag"), path


def test_a_conditional_request_still_gets_a_304(monkeypatch):
    c = _client(monkeypatch)
    etag = c.get("/login.html").headers["etag"]
    r = c.get("/login.html", headers={"If-None-Match": etag})
    assert r.status_code == 304
    assert r.headers["cache-control"] == "no-cache"


def test_api_responses_are_left_alone(monkeypatch):
    r = _client(monkeypatch).get("/api/health")
    assert r.status_code == 200
    assert "cache-control" not in r.headers
