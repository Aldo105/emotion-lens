"""
EmotionLens — /api/auth: single-user login, logout and status.

The password check always runs, even when the username is wrong, and both
failures answer the same, so the response gives no hint about which half
was right. See backend/app/auth.py for the primitives.
"""
import hmac

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.app import auth
from backend.app.config import settings

router = APIRouter()
limiter = auth.LoginRateLimiter()


class LoginRequest(BaseModel):
    # Length limits are checked in login(), not with Field(max_length=...):
    # a pydantic length constraint also rejects lone surrogates as a
    # validation error whose echoed input then crashes the 422 response.
    username: str
    password: str

MAX_USERNAME_LENGTH = 256
MAX_PASSWORD_LENGTH = 1024


def _username_matches(username: str, expected_username: str) -> bool:
    """Compare username in constant time, treating unencodable characters as mismatch."""
    try:
        username_bytes = username.encode("utf-8")
    except UnicodeEncodeError:
        return False

    try:
        expected_bytes = expected_username.encode("utf-8")
    except UnicodeEncodeError:
        # Should not happen in normal operation, but be safe
        return False

    return hmac.compare_digest(username_bytes, expected_bytes)


@router.get("/status")
async def status():
    return {"enabled": settings.auth_enabled}


@router.post("/login", status_code=204)
async def login(body: LoginRequest, request: Request):
    if not settings.auth_enabled:
        return Response(status_code=204)

    if len(body.username) > MAX_USERNAME_LENGTH or len(body.password) > MAX_PASSWORD_LENGTH:
        return JSONResponse({"detail": "credentials_too_long"}, status_code=422)

    ip = auth.client_ip(request.scope)
    wait = limiter.retry_after(ip)
    if wait:
        return JSONResponse({"detail": "too_many_attempts", "retry_after": wait},
                            status_code=429, headers={"Retry-After": str(wait)})

    user_ok = _username_matches(body.username, settings.auth_username)
    password_ok = auth.verify_password(body.password, settings.auth_password_hash)
    if not (user_ok and password_ok):
        limiter.record_failure(ip)
        return JSONResponse({"detail": "invalid_credentials"}, status_code=401)

    limiter.record_success(ip)
    ttl = settings.auth_session_hours * 3600
    token = auth.make_token(settings.auth_username, secret_key=settings.auth_secret_key,
                            password_hash=settings.auth_password_hash, ttl_seconds=ttl)
    response = Response(status_code=204)
    response.set_cookie(auth.COOKIE_NAME, token, max_age=ttl, path="/", httponly=True,
                        samesite="strict", secure=auth.is_https(request.scope))
    return response


@router.post("/logout", status_code=204)
async def logout():
    response = Response(status_code=204)
    response.delete_cookie(auth.COOKIE_NAME, path="/", httponly=True, samesite="strict")
    return response
