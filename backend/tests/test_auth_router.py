import pytest
from fastapi import FastAPI

from backend.app import auth
from backend.app.config import settings
from backend.app.routers import auth as auth_router
from backend.tests._testclient import TestClient

PASSWORD = "correcta-123"
KEY = "k" * 48


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "auth_username", "aldo")
    monkeypatch.setattr(settings, "auth_password_hash", auth.hash_password(PASSWORD))
    monkeypatch.setattr(settings, "auth_secret_key", KEY)
    monkeypatch.setattr(settings, "auth_session_hours", 12)
    monkeypatch.setattr(auth_router, "limiter", auth.LoginRateLimiter())


def _client(**kw):
    app = FastAPI()
    app.include_router(auth_router.router, prefix="/api/auth")
    return TestClient(app, **kw)


def _login(client, username="aldo", password=PASSWORD, headers=None):
    return client.post("/api/auth/login", json={"username": username, "password": password},
                       headers=headers or {})


def test_status_reports_whether_login_is_on(configured, monkeypatch):
    c = _client()
    assert c.get("/api/auth/status").json() == {"enabled": True}
    monkeypatch.setattr(settings, "auth_enabled", False)
    assert c.get("/api/auth/status").json() == {"enabled": False}


def test_login_sets_the_session_cookie(configured):
    r = _login(_client())
    assert r.status_code == 204
    cookie = r.headers["set-cookie"]
    low = cookie.lower()
    assert cookie.startswith("el_session=")
    for attr in ("httponly", "samesite=strict", "max-age=43200", "path=/"):
        assert attr in low, attr
    assert "secure" not in low
    token = cookie.split(";")[0].split("=", 1)[1]
    assert auth.token_is_valid(token, username="aldo", secret_key=KEY,
                               password_hash=settings.auth_password_hash)


def test_cookie_is_secure_behind_the_https_proxy(configured):
    r = _login(_client(client=("127.0.0.1", 40000)), headers={"X-Forwarded-Proto": "https"})
    assert "secure" in r.headers["set-cookie"].lower()


def test_forwarded_proto_from_outside_is_ignored(configured):
    r = _login(_client(client=("203.0.113.9", 40000)), headers={"X-Forwarded-Proto": "https"})
    assert "secure" not in r.headers["set-cookie"].lower()


def test_wrong_user_and_wrong_password_look_the_same(configured):
    c = _client()
    a = _login(c, password="mala-clave")
    b = _login(c, username="otro")
    assert a.status_code == b.status_code == 401
    assert a.json() == b.json() == {"detail": "invalid_credentials"}
    assert "set-cookie" not in a.headers


def test_sixth_attempt_is_rate_limited_even_with_the_right_password(configured):
    c = _client()
    for _ in range(5):
        assert _login(c, password="mala-clave").status_code == 401
    r = _login(c)
    assert r.status_code == 429
    body = r.json()
    assert body["detail"] == "too_many_attempts"
    assert 0 < body["retry_after"] <= 900
    assert r.headers["retry-after"] == str(body["retry_after"])


def test_rate_limit_is_per_real_ip_behind_the_proxy(configured):
    c = _client(client=("127.0.0.1", 40000))
    for _ in range(5):
        _login(c, password="mala-clave", headers={"X-Real-IP": "198.51.100.1"})
    assert _login(c, headers={"X-Real-IP": "198.51.100.1"}).status_code == 429
    assert _login(c, headers={"X-Real-IP": "198.51.100.2"}).status_code == 204


def test_non_ascii_credentials(configured, monkeypatch):
    monkeypatch.setattr(settings, "auth_username", "ñandú")
    monkeypatch.setattr(settings, "auth_password_hash", auth.hash_password("contraseña-ñ"))
    assert _login(_client(), username="ñandú", password="contraseña-ñ").status_code == 204


def test_unencodable_username_is_a_normal_failure(configured):
    """Lone surrogate in username should trigger rate limiting, not crash."""
    c = _client()
    # Send a lone surrogate \ud800 as JSON escape, which can't be encoded to UTF-8.
    for _ in range(5):
        r = c.post("/api/auth/login", content='{"username":"\\ud800","password":"x"}',
                   headers={"Content-Type": "application/json"})
        assert r.status_code == 401
        assert r.json() == {"detail": "invalid_credentials"}
    # 6th attempt should be rate limited
    r = c.post("/api/auth/login", content='{"username":"\\ud800","password":"x"}',
               headers={"Content-Type": "application/json"})
    assert r.status_code == 429
    assert r.json()["detail"] == "too_many_attempts"


def test_logout_clears_the_cookie(configured):
    r = _client().post("/api/auth/logout")
    assert r.status_code == 204
    cookie = r.headers["set-cookie"].lower()
    assert cookie.startswith("el_session=")
    assert "max-age=0" in cookie or "expires=" in cookie


def test_login_is_a_noop_when_disabled(configured, monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", False)
    r = _login(_client(), password="lo-que-sea")
    assert r.status_code == 204
    assert "set-cookie" not in r.headers
