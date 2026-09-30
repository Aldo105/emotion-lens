import json
import re

from backend.app.config import BASE_DIR

FRONT = BASE_DIR / "frontend"


def _read(rel):
    return (FRONT / rel).read_text(encoding="utf-8")


def test_header_has_a_hidden_logout_button_with_its_icon():
    html = _read("index.html")
    assert 'id="btn-logout"' in html
    assert 'data-icon="logout"' in html
    # It must ship hidden: only the status check may reveal it.
    start_tag = re.search(r'<button\b[^>]*\bid="btn-logout"[^>]*>', html)
    assert start_tag is not None
    class_attr = re.search(r'\bclass="([^"]*)"', start_tag.group(0))
    assert class_attr is not None
    assert "hidden" in class_attr.group(1).split()
    assert "logout:" in _read("js/icons.js")


def test_app_sends_expired_sessions_to_the_login_page():
    app_js = _read("js/app.js")
    assert "function goToLogin" in app_js
    assert "response.status === 401" in app_js
    assert "4401" in _read("js/websocket.js")
    assert "goToLogin" in _read("js/video-upload.js")


def test_logout_string_exists_in_both_languages():
    for lang in ("es", "en"):
        data = json.loads(_read(f"i18n/{lang}.json"))
        assert data["auth"]["logout"]
