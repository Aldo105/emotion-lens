import json

import pytest

from backend.app import auth
from backend.app.config import BASE_DIR, settings
from backend.tests._testclient import TestClient

LOGIN_KEYS = ("title", "subtitle", "tagline", "note", "username", "password", "submit",
              "submitting", "showPassword", "hidePassword", "invalid", "locked",
              "unavailable", "theme")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "auth_username", "aldo")
    monkeypatch.setattr(settings, "auth_password_hash", auth.hash_password("correcta-123"))
    monkeypatch.setattr(settings, "auth_secret_key", "k" * 48)
    from backend.app.main import app
    return TestClient(app)


def test_login_page_and_its_assets_are_reachable_without_a_session(client):
    r = client.get("/login.html")
    assert r.status_code == 200
    for needle in ('id="login-form"', 'autocomplete="username"',
                   'autocomplete="current-password"', 'src="/js/login.js"'):
        assert needle in r.text, needle
    for path in ("/css/styles.css", "/js/login.js", "/js/i18n.js", "/js/icons.js",
                 "/i18n/es.json", "/i18n/en.json"):
        assert client.get(path).status_code == 200, path


def test_login_strings_exist_in_both_languages():
    es = json.loads((BASE_DIR / "frontend/i18n/es.json").read_text(encoding="utf-8"))
    en = json.loads((BASE_DIR / "frontend/i18n/en.json").read_text(encoding="utf-8"))
    assert set(es["login"]) == set(en["login"])
    for key in LOGIN_KEYS:
        assert es["login"][key] and en["login"][key], key
