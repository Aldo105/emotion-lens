import asyncio

import pytest

from backend.app.config import Settings


def test_auth_defaults_are_on_and_empty(monkeypatch):
    for key in ("AUTH_ENABLED", "AUTH_USERNAME", "AUTH_PASSWORD_HASH",
                "AUTH_SECRET_KEY", "AUTH_SESSION_HOURS"):
        monkeypatch.delenv(key, raising=False)
    s = Settings(_env_file=None)
    assert s.auth_enabled is True
    assert (s.auth_username, s.auth_password_hash, s.auth_secret_key) == ("", "", "")
    assert s.auth_session_hours == 12


def test_app_refuses_to_start_with_login_on_and_no_credentials(monkeypatch):
    from backend.app import main
    monkeypatch.setattr(main.settings, "auth_enabled", True)
    monkeypatch.setattr(main.settings, "auth_username", "aldo")
    monkeypatch.setattr(main.settings, "auth_password_hash", "")
    monkeypatch.setattr(main.settings, "auth_secret_key", "")

    async def start():
        async with main.lifespan(main.app):
            pass

    with pytest.raises(RuntimeError, match="crear-credenciales"):
        asyncio.run(start())
