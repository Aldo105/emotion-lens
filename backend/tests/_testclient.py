"""starlette.testclient without the deprecation notice Starlette 1.6 emits on
import (it now wants httpx2; this env ships httpx 0.28). Keeps test output clean."""
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=r".*httpx.*starlette\.testclient.*")
    from starlette.testclient import TestClient  # noqa: E402

__all__ = ["TestClient"]
