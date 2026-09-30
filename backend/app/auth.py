"""
EmotionLens — single-user login primitives.

Standard library only: the CLI at the bottom runs on a fresh VPS during
deploy/setup_vps.sh, and none of this needs FastAPI. The middleware
(auth_middleware.py) and the /api/auth router build on these pieces.

Usage:
    python -m backend.app.auth crear-credenciales [--env RUTA]
"""
import argparse
import base64
import binascii
import getpass
import hashlib
import hmac
import math
import secrets
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

COOKIE_NAME = "el_session"

SCRYPT_N, SCRYPT_R, SCRYPT_P = 2 ** 14, 8, 1
MIN_PASSWORD_LENGTH = 8

MAX_FAILURES = 5
WINDOW_SECONDS = 15 * 60
LOCKOUT_SECONDS = 15 * 60

# Nginx on the same host; only then are X-Real-IP / X-Forwarded-Proto trusted.
TRUSTED_PROXIES = frozenset({"127.0.0.1", "::1"})


def _b64e(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


# ── Password hashing ─────────────────────────────────────────────────

def hash_password(password: str, *, salt: bytes | None = None) -> str:
    """scrypt, stored as scrypt:n:r:p:salt:hash. ':' rather than '$' so the
    value survives .env parsing untouched."""
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt,
                            n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=32)
    return f"scrypt:{SCRYPT_N}:{SCRYPT_R}:{SCRYPT_P}:{_b64e(salt)}:{_b64e(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, digest = stored.split(":")
        if scheme != "scrypt":
            return False
        expected = _b64d(digest)
        actual = hashlib.scrypt(password.encode("utf-8"), salt=_b64d(salt),
                                n=int(n), r=int(r), p=int(p), dklen=len(expected))
    except (ValueError, TypeError, binascii.Error):
        return False
    return hmac.compare_digest(actual, expected)


# ── Session token ────────────────────────────────────────────────────

def _signing_key(secret_key: str, password_hash: str) -> bytes:
    # Mixing in the password hash means a password change logs out every
    # open session without any server-side session store.
    return hashlib.sha256(f"{secret_key}\x00{password_hash}".encode("utf-8")).digest()


def _sign(payload: str, secret_key: str, password_hash: str) -> str:
    mac = hmac.new(_signing_key(secret_key, password_hash), payload.encode("ascii"), hashlib.sha256)
    return _b64e(mac.digest())


def make_token(username: str, *, secret_key: str, password_hash: str,
               ttl_seconds: int, now: float | None = None) -> str:
    now = time.time() if now is None else now
    payload = _b64e(f"{username}|{int(now + ttl_seconds)}".encode("utf-8"))
    return f"{payload}.{_sign(payload, secret_key, password_hash)}"


def token_is_valid(token: str, *, username: str, secret_key: str, password_hash: str,
                   now: float | None = None) -> bool:
    now = time.time() if now is None else now
    try:
        payload, sig = token.split(".")
        if not hmac.compare_digest(sig.encode("ascii"),
                                   _sign(payload, secret_key, password_hash).encode("ascii")):
            return False
        user, expires = _b64d(payload).decode("utf-8").rsplit("|", 1)
        return (hmac.compare_digest(user.encode("utf-8"), username.encode("utf-8"))
                and now < int(expires))
    except (ValueError, UnicodeError, binascii.Error):
        return False


# ── Login attempt limiter ────────────────────────────────────────────

class LoginRateLimiter:
    """In memory, per IP: MAX_FAILURES within WINDOW_SECONDS locks that IP
    out for LOCKOUT_SECONDS. One process, one user: a restart forgetting the
    counters is acceptable."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._failures: dict[str, deque] = defaultdict(deque)
        self._locked_until: dict[str, float] = {}

    def retry_after(self, ip: str) -> int:
        until = self._locked_until.get(ip)
        if until is None:
            return 0
        remaining = until - self._clock()
        if remaining <= 0:
            del self._locked_until[ip]
            return 0
        return int(math.ceil(remaining))

    def record_failure(self, ip: str) -> None:
        now = self._clock()
        attempts = self._failures[ip]
        attempts.append(now)
        while attempts and now - attempts[0] > WINDOW_SECONDS:
            attempts.popleft()
        if len(attempts) >= MAX_FAILURES:
            self._locked_until[ip] = now + LOCKOUT_SECONDS
            attempts.clear()

    def record_success(self, ip: str) -> None:
        self._failures.pop(ip, None)
        self._locked_until.pop(ip, None)


# ── Request context behind Nginx ─────────────────────────────────────

def header(scope: dict, name: bytes) -> str | None:
    for key, value in scope.get("headers", []):
        if key == name:
            return value.decode("latin-1")
    return None


def _peer(scope: dict) -> str:
    client = scope.get("client")
    return client[0] if client else ""


def client_ip(scope: dict) -> str:
    peer = _peer(scope)
    if peer in TRUSTED_PROXIES:
        real = header(scope, b"x-real-ip")
        if real:
            return real.strip()
    return peer or "unknown"


def is_https(scope: dict) -> bool:
    if _peer(scope) in TRUSTED_PROXIES:
        proto = header(scope, b"x-forwarded-proto")
        if proto:
            return proto.strip().lower() == "https"
    return scope.get("scheme") in ("https", "wss")


# ── Credentials in .env ──────────────────────────────────────────────

def credential_lines(username: str, password: str, *, secret_key: str | None = None) -> dict[str, str]:
    username = username.strip()
    if not username or any(ch.isspace() for ch in username):
        raise ValueError("El usuario no puede estar vacío ni tener espacios")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"La contraseña debe tener al menos {MIN_PASSWORD_LENGTH} caracteres")
    return {
        "AUTH_ENABLED": "true",
        "AUTH_USERNAME": username,
        "AUTH_PASSWORD_HASH": hash_password(password),
        "AUTH_SECRET_KEY": secret_key or secrets.token_urlsafe(48),
    }


def upsert_env(path: Path, values: dict[str, str]) -> None:
    """Replace those keys in a .env file (or append them), keeping every
    other line as it was."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    kept = [line for line in lines if line.split("=", 1)[0].strip() not in values]
    kept += [f"{key}={value}" for key, value in values.items()]
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")


def missing_auth_settings(s) -> list[str]:
    if not s.auth_enabled:
        return []
    return [name.upper() for name in ("auth_username", "auth_password_hash", "auth_secret_key")
            if not getattr(s, name)]


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m backend.app.auth")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("crear-credenciales",
                            help="Pide usuario y contraseña y genera las variables AUTH_* del .env")
    create.add_argument("--env", type=Path, default=None,
                        help="Escribe las variables en este archivo en vez de imprimirlas")
    args = parser.parse_args(argv)

    username = input("Usuario: ")
    password = getpass.getpass("Contraseña: ")
    if password != getpass.getpass("Repite la contraseña: "):
        print("[ERROR] Las contraseñas no coinciden", file=sys.stderr)
        return 1
    try:
        values = credential_lines(username, password)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if args.env:
        upsert_env(args.env, values)
        print(f"[OK] Credenciales escritas en {args.env}. Reinicia el servicio para aplicarlas.")
    else:
        print("\n".join(f"{key}={value}" for key, value in values.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
