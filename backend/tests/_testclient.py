"""starlette.testclient without deprecation notices (httpx in Starlette 1.6,
anyio.abc.BlockingPortal). Keeps test output clean."""
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=r".*httpx.*starlette\.testclient.*")
    warnings.filterwarnings("ignore", message=r"The anyio\.abc\.BlockingPortal alias is deprecated", category=DeprecationWarning)
    from starlette.testclient import TestClient  # noqa: E402

__all__ = ["TestClient"]
