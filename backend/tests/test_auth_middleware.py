# backend/tests/test_auth_middleware.py
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocketDisconnect

from backend.app import auth
from backend.app.auth_middleware import AuthMiddleware, is_public, safe_next
from backend.app.config import settings
from backend.tests._testclient import TestClient

KEY = "k" * 48


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "auth_username", "aldo")
    monkeypatch.setattr(settings, "auth_password_hash", auth.hash_password("correcta-123"))
    monkeypatch.setattr(settings, "auth_secret_key", KEY)


def _cookie(now=None):
    token = auth.make_token("aldo", secret_key=KEY, password_hash=settings.auth_password_hash,
                            ttl_seconds=3600, now=now)
    return {"cookie": f"{auth.COOKIE_NAME}={token}"}


async def _ok(request):
    return JSONResponse({"ok": True})


async def _page(request):
    return PlainTextResponse("pagina")


async def _ws(websocket):
    await websocket.accept()
    await websocket.send_text("hola")
    await websocket.close()


def _client():
    inner = Starlette(routes=[
        Route("/api/private", _ok), Route("/api/health", _ok), Route("/api/auth/status", _ok),
        Route("/dashboard.html", _page), Route("/login.html", _page), Route("/i18n/es.json", _ok),
        WebSocketRoute("/ws/emotion", _ws),
    ])
    return TestClient(AuthMiddleware(inner))


def test_public_paths_need_no_cookie(configured):
    c = _client()
    for path in ("/api/health", "/api/auth/status", "/login.html", "/i18n/es.json"):
        assert c.get(path).status_code == 200, path


def test_api_without_cookie_is_401(configured):
    r = _client().get("/api/private")
    assert r.status_code == 401
    assert r.json() == {"detail": "not_authenticated"}


def test_page_without_cookie_redirects_with_next(configured):
    r = _client().get("/dashboard.html?tab=2", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "/login.html?next=%2Fdashboard.html%3Ftab%3D2"


def test_valid_cookie_lets_requests_through(configured):
    c = _client()
    assert c.get("/api/private", headers=_cookie()).json() == {"ok": True}
    assert c.get("/dashboard.html", headers=_cookie()).text == "pagina"


def test_expired_or_forged_cookies_are_rejected(configured):
    c = _client()
    assert c.get("/api/private", headers=_cookie(now=0)).status_code == 401
    assert c.get("/api/private", headers={"cookie": f"{auth.COOKIE_NAME}=basura.firma"}).status_code == 401
    assert c.get("/api/private", headers={"cookie": "otra=cosa"}).status_code == 401


def test_a_malformed_foreign_cookie_does_not_hide_the_session(configured):
    # Other apps on localhost (cookies ignore ports) or a parent domain can
    # leave cookies with spaces, quotes or no "=" next to ours.
    token = _cookie()["cookie"]
    c = _client()
    for header in (f"x=a b; {token}", f"{token}; foo", f"foo; {token}",
                   f'prefs={{"a":1}}; {token}'):
        assert c.get("/api/private", headers={"cookie": header}).status_code == 200, header
    with c.websocket_connect("/ws/emotion", headers={"cookie": f"x=a b; {token}"}) as ws:
        assert ws.receive_text() == "hola"
    # ...while a malformed neighbour alone still gets nobody in.
    assert c.get("/api/private", headers={"cookie": "x=a b; foo"}).status_code == 401


def test_websocket_without_cookie_is_closed_with_4401(configured):
    with _client().websocket_connect("/ws/emotion") as ws:
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
    assert exc.value.code == 4401


def test_websocket_with_cookie_works(configured):
    with _client().websocket_connect("/ws/emotion", headers=_cookie()) as ws:
        assert ws.receive_text() == "hola"


def test_disabled_lets_everything_through(configured, monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", False)
    c = _client()
    assert c.get("/api/private").status_code == 200
    with c.websocket_connect("/ws/emotion") as ws:
        assert ws.receive_text() == "hola"


def test_dot_segments_never_count_as_public():
    assert is_public("/i18n/es.json")
    assert not is_public("/i18n/../index.html")
    assert not is_public("/i18n/../api/sessions/")
    assert not is_public("/i18n/..\\index.html")  # backslash is a separator on Windows


def test_safe_next_only_allows_local_paths():
    assert safe_next("/sessions?x=1") == "/sessions?x=1"
    for bad in (None, "", "https://evil.com", "//evil.com", "/\\evil.com", "evil.com",
                "javascript:alert(1)", "/\t/evil.com", "/\n/evil.com", "/\r/evil.com"):
        assert safe_next(bad) == "/", bad


def test_the_real_app_is_protected(configured):
    from backend.app.main import app
    c = TestClient(app)  # without `with`: the lifespan (DB, GPU) does not start
    assert c.get("/api/sessions/").status_code == 401
    assert c.get("/api/auth/status").json() == {"enabled": True}
    assert c.get("/", follow_redirects=False).headers["location"] == "/login.html?next=%2F"
    # Static files walking out of the public /i18n/ prefix (backslash works on Windows).
    assert c.get("/i18n/..%5Cindex.html", follow_redirects=False).status_code == 302
