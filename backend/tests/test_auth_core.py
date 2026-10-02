import pytest

from backend.app import auth

KEY = "k" * 48
HASH = auth.hash_password("correcta-123")


def test_hash_and_verify_roundtrip():
    stored = auth.hash_password("correcta-123")
    assert stored.startswith("scrypt:16384:8:1:")
    assert "$" not in stored
    assert auth.verify_password("correcta-123", stored)
    assert not auth.verify_password("otra-cosa", stored)


def test_hash_uses_a_random_salt():
    assert auth.hash_password("x" * 8) != auth.hash_password("x" * 8)


def test_verify_rejects_malformed_hashes():
    assert not auth.verify_password("x", "")
    assert not auth.verify_password("x", "bcrypt:1:2:3:aa:bb")
    assert not auth.verify_password("x", "scrypt:notanint:8:1:aa:bb")


def test_non_ascii_password():
    stored = auth.hash_password("contraseña-ñandú")
    assert auth.verify_password("contraseña-ñandú", stored)


def _tok(user="aldo", now=1_000_000.0, ttl=3600):
    return auth.make_token(user, secret_key=KEY, password_hash=HASH, ttl_seconds=ttl, now=now)


def _valid(token, user="aldo", now=1_000_000.0, key=KEY, ph=HASH):
    return auth.token_is_valid(token, username=user, secret_key=key, password_hash=ph, now=now)


def test_token_valid_until_expiry():
    t = _tok()
    assert _valid(t, now=1_000_000.0 + 3599)
    assert not _valid(t, now=1_000_000.0 + 3600)


def test_token_rejects_tampering_and_garbage():
    payload, sig = _tok().split(".")
    forged = auth._b64e(b"aldo|9999999999") + "." + sig
    assert not _valid(forged)
    assert not _valid(payload + "." + sig[:-1] + ("B" if sig[-1] != "B" else "C"))
    assert not _valid("basura")
    assert not _valid("")


def test_token_bound_to_username_secret_and_password():
    t = _tok()
    assert not _valid(t, user="otro")
    assert not _valid(t, key="z" * 48)
    assert not _valid(t, ph=auth.hash_password("nueva-4567"))


def test_non_ascii_username_token():
    assert _valid(_tok(user="ñandú"), user="ñandú")


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_limiter_locks_after_five_failures_for_fifteen_minutes():
    c = Clock()
    lim = auth.LoginRateLimiter(clock=c)
    for _ in range(4):
        lim.record_failure("1.1.1.1")
        assert lim.retry_after("1.1.1.1") == 0
    lim.record_failure("1.1.1.1")
    assert lim.retry_after("1.1.1.1") == 900
    c.t = 899.5
    assert lim.retry_after("1.1.1.1") == 1
    c.t = 900
    assert lim.retry_after("1.1.1.1") == 0


def test_limiter_forgets_failures_outside_the_window():
    c = Clock()
    lim = auth.LoginRateLimiter(clock=c)
    for _ in range(4):
        lim.record_failure("ip")
    c.t = 901
    lim.record_failure("ip")
    assert lim.retry_after("ip") == 0


def test_limiter_success_resets_and_ips_are_independent():
    c = Clock()
    lim = auth.LoginRateLimiter(clock=c)
    for _ in range(4):
        lim.record_failure("a")
    lim.record_success("a")
    lim.record_failure("a")
    assert lim.retry_after("a") == 0
    for _ in range(5):
        lim.record_failure("b")
    assert lim.retry_after("b") > 0
    assert lim.retry_after("a") == 0


def _scope(client="203.0.113.5", headers=(), scheme="http"):
    return {"type": "http", "client": (client, 5000), "scheme": scheme,
            "headers": [(k.encode(), v.encode()) for k, v in headers]}


def test_client_ip_trusts_real_ip_only_from_local_proxy():
    assert auth.client_ip(_scope("127.0.0.1", [("x-real-ip", "198.51.100.7")])) == "198.51.100.7"
    assert auth.client_ip(_scope("203.0.113.5", [("x-real-ip", "198.51.100.7")])) == "203.0.113.5"
    assert auth.client_ip(_scope("127.0.0.1")) == "127.0.0.1"


def test_is_https_trusts_forwarded_proto_only_from_local_proxy():
    assert auth.is_https(_scope("127.0.0.1", [("x-forwarded-proto", "https")]))
    assert not auth.is_https(_scope("203.0.113.5", [("x-forwarded-proto", "https")]))
    assert auth.is_https(_scope("203.0.113.5", scheme="https"))
    assert not auth.is_https(_scope("127.0.0.1"))


def test_credential_lines_validate_input():
    with pytest.raises(ValueError):
        auth.credential_lines("", "larga-123")
    with pytest.raises(ValueError):
        auth.credential_lines("con espacio", "larga-123")
    with pytest.raises(ValueError):
        auth.credential_lines("aldo", "corta")
    v = auth.credential_lines("aldo", "larga-123", secret_key="s" * 48)
    assert v["AUTH_ENABLED"] == "true"
    assert v["AUTH_USERNAME"] == "aldo"
    assert v["AUTH_SECRET_KEY"] == "s" * 48
    assert auth.verify_password("larga-123", v["AUTH_PASSWORD_HASH"])


def test_upsert_env_replaces_existing_and_keeps_the_rest(tmp_path):
    env = tmp_path / ".env"
    env.write_text("DEBUG=false\nAUTH_ENABLED=false\nAUTH_USERNAME=viejo\n", encoding="utf-8")
    auth.upsert_env(env, {"AUTH_ENABLED": "true", "AUTH_USERNAME": "aldo", "AUTH_SECRET_KEY": "k"})
    assert env.read_text(encoding="utf-8").splitlines() == [
        "DEBUG=false", "AUTH_ENABLED=true", "AUTH_USERNAME=aldo", "AUTH_SECRET_KEY=k",
    ]


def test_upsert_env_creates_the_file(tmp_path):
    env = tmp_path / ".env"
    auth.upsert_env(env, {"AUTH_ENABLED": "true"})
    assert env.read_text(encoding="utf-8") == "AUTH_ENABLED=true\n"


def test_missing_auth_settings():
    class S:
        auth_enabled = True
        auth_username = "aldo"
        auth_password_hash = ""
        auth_secret_key = "k"

    assert auth.missing_auth_settings(S) == ["AUTH_PASSWORD_HASH"]
    S.auth_enabled = False
    assert auth.missing_auth_settings(S) == []


def test_cli_writes_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    monkeypatch.setattr("builtins.input", lambda prompt="": "aldo")
    answers = iter(["larga-123", "larga-123"])
    monkeypatch.setattr(auth.getpass, "getpass", lambda prompt="": next(answers))
    assert auth.main(["crear-credenciales", "--env", str(env)]) == 0
    text = env.read_text(encoding="utf-8")
    assert "AUTH_USERNAME=aldo" in text and "AUTH_ENABLED=true" in text


def test_cli_rejects_mismatched_passwords(tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "aldo")
    answers = iter(["larga-123", "otra-1234"])
    monkeypatch.setattr(auth.getpass, "getpass", lambda prompt="": next(answers))
    assert auth.main(["crear-credenciales", "--env", str(tmp_path / ".env")]) == 1
    assert not (tmp_path / ".env").exists()
