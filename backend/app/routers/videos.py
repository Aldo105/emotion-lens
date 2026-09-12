"""
EmotionLens — Video Upload Router

Endpoints for uploading video files and tracking offline analysis progress.

Routes:
  POST /api/videos/upload       — Upload a video and start background processing
  GET  /api/videos/{id}/progress — Poll processing progress
"""

import asyncio
import os
import threading
import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import settings
from backend.app.models.database import get_db, async_session_factory
from backend.app.services import session_manager
from backend.app.services.video_processor import VideoProcessor

router = APIRouter()

# Single processor instance so progress tracking is shared
_processor = VideoProcessor()


# ── Upload Video ─────────────────────────────────────────────────────

@router.post("/upload")
async def upload_video(
    file: UploadFile = File(..., description="Video file to analyze"),
    session_name: str = "Video Analysis",
    candidate_name: str = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a video file for offline emotion analysis.

    The video is saved to disk, a new session is created (input_type='video_upload'),
    and the analysis pipeline is launched in a background thread.

    Returns the session_id for progress polling.
    """
    # Validate file type
    if file.content_type and not file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Expected a video file.",
        )

    # Validate file size (read content length from header if available)
    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    # Generate unique filename to avoid collisions
    ext = Path(file.filename).suffix if file.filename else ".mp4"
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(settings.uploads_dir, unique_name)

    # Save uploaded file to disk
    os.makedirs(settings.uploads_dir, exist_ok=True)
    total_written = 0

    async with aiofiles.open(save_path, "wb") as out_file:
        while True:
            chunk = await file.read(1024 * 1024)  # 1 MB chunks
            if not chunk:
                break
            total_written += len(chunk)
            if total_written > max_bytes:
                # Clean up partial file
                await out_file.close()
                os.remove(save_path)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds maximum size of {settings.max_upload_size_mb} MB.",
                )
            await out_file.write(chunk)

    # Create session
    name = session_name or file.filename or "Video Analysis"
    session = await session_manager.create_session(
        db=db,
        name=name,
        candidate_name=candidate_name,
        input_type="video_upload",
    )

    # Store video path on the session
    session.video_path = save_path
    await db.flush()
    await db.refresh(session)

    session_id = session.id

    # Launch background processing in a separate thread with its own event loop
    def _run_processing():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                _processor.process_video(
                    video_path=save_path,
                    session_id=session_id,
                )
            )
        finally:
            loop.close()

    thread = threading.Thread(target=_run_processing, daemon=True)
    thread.start()

    return {
        "session_id": session_id,
        "filename": file.filename,
        "status": "processing",
        "message": "Video uploaded successfully. Processing started.",
    }


# ── Processing Progress ─────────────────────────────────────────────

@router.get("/{session_id}/progress")
async def get_video_progress(session_id: int):
    """
    Poll the processing progress for a video analysis session.

    Returns:
        {"progress": 0.0-1.0, "status": "initializing|processing|completed|error"}
    """
    progress_info = VideoProcessor.progress.get(session_id)

    if progress_info is None:
        raise HTTPException(
            status_code=404,
            detail=f"No active processing found for session {session_id}.",
        )

    return {
        "session_id": session_id,
        "progress": progress_info.get("progress", 0.0),
        "status": progress_info.get("status", "unknown"),
        "error": progress_info.get("error"),
    }
