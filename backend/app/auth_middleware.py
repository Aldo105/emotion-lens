# backend/app/auth_middleware.py
"""
EmotionLens — gate every request behind the single-user session cookie.

Pure ASGI rather than BaseHTTPMiddleware, because the live analysis runs over
a WebSocket and BaseHTTPMiddleware never sees websocket scopes. Settings are
read per request, so tests (and a restart with a new .env) take effect
without rebuilding the app.
"""
from urllib.parse import quote

from starlette.requests import cookie_parser

from backend.app import auth
from backend.app.config import settings

LOGIN_PAGE = "/login.html"

# Everything the login page itself needs, plus health checks.
PUBLIC_PATHS = frozenset({
    LOGIN_PAGE,
    "/css/styles.css",
    "/js/login.js",
    "/js/i18n.js",
    "/js/icons.js",
    "/favicon.svg",
    "/api/auth/login",
    "/api/auth/status",
    "/api/health",
})
PUBLIC_PREFIXES = ("/i18n/",)

# Custom close code the frontend recognises as "session expired".
WS_UNAUTHORIZED = 4401


def is_public(path: str) -> bool:
    # A dot segment could otherwise ride a public prefix into the rest of
    # the static tree (curl --path-as-is /i18n/../index.html). Backslash too:
    # on Windows StaticFiles treats it as a separator, so /i18n/..\index.html
    # walks out of the prefix without ever having a ".." segment.
    if "\\" in path or ".." in path.split("/"):
        return False
    return path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES)


def safe_next(raw: str | None) -> str:
    """Where to go after logging in: local paths only, never another site."""
    if not raw or not raw.startswith("/") or raw.startswith("//") or raw.startswith("/\\"):
        return "/"
    # Browsers drop tabs and newlines inside a URL, so "/\t/evil.com" would
    # navigate as "//evil.com".
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in raw):
        return "/"
    return raw


def _session_token(scope: dict) -> str | None:
    raw = auth.header(scope, b"cookie")
    if not raw:
        return None
    # Starlette's own parser (what Request.cookies uses): it skips a malformed
    # neighbouring cookie instead of discarding the whole header the way
    # http.cookies.SimpleCookie does.
    return cookie_parser(raw).get(auth.COOKIE_NAME) or None


def has_valid_session(scope: dict) -> bool:
    # With empty credentials a token signed with an empty key for the empty
    # user would verify. The lifespan check normally prevents that state, but
    # `uvicorn --lifespan off` skips it, so fail closed here too.
    if not (settings.auth_username and settings.auth_password_hash and settings.auth_secret_key):
        return False
    token = _session_token(scope)
    return bool(token) and auth.token_is_valid(
        token, username=settings.auth_username, secret_key=settings.auth_secret_key,
        password_hash=settings.auth_password_hash)


class AuthMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket") or not settings.auth_enabled:
            return await self.app(scope, receive, send)

        path = scope.get("path", "")
        if (scope["type"] == "http" and is_public(path)) or has_valid_session(scope):
            return await self.app(scope, receive, send)

        if scope["type"] == "websocket":
            # Accept, then close with our own code: closing before accept
            # reaches the browser as a generic 1006 it can't tell apart from a
            # dropped network.
            await receive()  # websocket.connect
            await send({"type": "websocket.accept"})
            await send({"type": "websocket.close", "code": WS_UNAUTHORIZED})
            return

        if path.startswith("/api/"):
            body = b'{"detail":"not_authenticated"}'
            await send({"type": "http.response.start", "status": 401, "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ]})
            await send({"type": "http.response.body", "body": body})
            return

        query = scope.get("query_string", b"").decode("latin-1")
        target = path + (f"?{query}" if query else "")
        location = f"{LOGIN_PAGE}?next={quote(target, safe='')}"
        await send({"type": "http.response.start", "status": 302, "headers": [
            (b"location", location.encode("latin-1")),
            (b"content-length", b"0"),
        ]})
        await send({"type": "http.response.body", "body": b""})
