"""
EmotionLens — FastAPI Application Entry Point

Sets up the FastAPI app, middleware, routes, static file serving,
and startup/shutdown lifecycle events.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.config import settings, BASE_DIR, initialize_gpu
from backend.app.models.database import init_db
from backend.app.routers import sessions, notes, reports, websocket, videos, feedback


# ── Lifecycle ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    # ── Startup ──
    print(f"\n[*] Starting {settings.app_name} v{settings.app_version}")
    
    # Create data directories
    for dir_path in [
        settings.reports_dir,
        settings.recordings_dir,
        settings.uploads_dir,
        settings.model_dir,
        str(BASE_DIR / "data"),
    ]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Initialize database
    await init_db()

    # Initialize GPU with optimizations
    device = initialize_gpu()
    app.state.device = device

    print(f"[+] Dashboard: http://localhost:{settings.port}")
    print(f"[+] WebSocket: ws://localhost:{settings.port}/ws/emotion")
    print(f"[+] API Docs:  http://localhost:{settings.port}/docs")
    print(f"{'=' * 50}\n")

    yield

    # ── Shutdown ──
    print(f"\n[*] Shutting down {settings.app_name}")


# ── App Instance ──────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    description=(
        "Real-time facial emotion & micro-expression recognition system "
        "for HR interview analysis. Detects emotions, micro-expressions, "
        "and provides trustworthiness scoring."
    ),
    version=settings.app_version,
    lifespan=lifespan,
)


# ── CORS Middleware ───────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API Routes ────────────────────────────────────────────────────────

app.include_router(
    websocket.router,
    tags=["WebSocket"],
)

app.include_router(
    sessions.router,
    prefix="/api/sessions",
    tags=["Sessions"],
)

app.include_router(
    notes.router,
    prefix="/api/notes",
    tags=["Notes"],
)

app.include_router(
    reports.router,
    prefix="/api/reports",
    tags=["Reports"],
)

app.include_router(
    videos.router,
    prefix="/api/videos",
    tags=["Videos"],
)

app.include_router(
    feedback.router,
    prefix="/api/feedback",
    tags=["Feedback"],
)



# ── Health Check ──────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "device": getattr(app.state, "device", "unknown"),
    }


# ── Static Files (Frontend) ──────────────────────────────────────────

frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
