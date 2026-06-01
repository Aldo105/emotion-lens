"""
EmotionLens — Sessions Router

CRUD operations for interview sessions.
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.database import get_db, Session, SessionSummary
from backend.app.models.schemas import (
    SessionCreate, SessionUpdate, SessionResponse,
    SessionListResponse, SessionComparisonRequest,
    SessionComparisonResponse, SessionSummaryResponse,
)

router = APIRouter()


# ── Create Session ────────────────────────────────────────────────────

@router.post("/", response_model=SessionResponse, status_code=201)
async def create_session(data: SessionCreate, db: AsyncSession = Depends(get_db)):
    """Start a new analysis session."""
    session = Session(
        name=data.name,
        candidate_name=data.candidate_name,
        input_type=data.input_type,
        status="active",
    )
    db.add(session)
    await db.flush()
    await db.refresh(session)
    return session


# ── List Sessions ─────────────────────────────────────────────────────

@router.get("/", response_model=SessionListResponse)
async def list_sessions(
    status: Optional[str] = Query(None, pattern="^(active|paused|completed|cancelled)$"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """List all sessions, optionally filtered by status."""
    query = select(Session).order_by(Session.created_at.desc())
    
    if status:
        query = query.where(Session.status == status)
    
    # Get total count
    count_query = select(func.count(Session.id))
    if status:
        count_query = count_query.where(Session.status == status)
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Get paginated results
    query = query.offset(offset).limit(limit)
    result = await db.execute(query)
    sessions = result.scalars().all()

    return SessionListResponse(sessions=sessions, total=total)


# ── Get Session ───────────────────────────────────────────────────────

@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: int, db: AsyncSession = Depends(get_db)):
    """Get a specific session by ID."""
    result = await db.execute(select(Session).where(Session.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


# ── Update Session ────────────────────────────────────────────────────

@router.patch("/{session_id}", response_model=SessionResponse)
async def update_session(
    session_id: int,
    data: SessionUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update session details (name, status, notes)."""
    result = await db.execute(select(Session).where(Session.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Apply updates
    update_data = data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(session, field, value)

    # If completing the session, set end time and duration
    if data.status == "completed" and session.ended_at is None:
        session.ended_at = datetime.now(timezone.utc)
        if session.created_at:
            delta = session.ended_at - session.created_at
            session.duration_seconds = delta.total_seconds()

    await db.flush()
    await db.refresh(session)
    return session


# ── Delete Session ────────────────────────────────────────────────────

@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a session and all its associated data."""
    result = await db.execute(select(Session).where(Session.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    await db.delete(session)


# ── Get Session Summary ──────────────────────────────────────────────

@router.get("/{session_id}/summary", response_model=SessionSummaryResponse)
async def get_session_summary(session_id: int, db: AsyncSession = Depends(get_db)):
    """Get the aggregate summary for a completed session."""
    result = await db.execute(
        select(SessionSummary).where(SessionSummary.session_id == session_id)
    )
    summary = result.scalar_one_or_none()
    if not summary:
        raise HTTPException(status_code=404, detail="Summary not found. Session may still be active.")
    return summary


# ── Compare Sessions ─────────────────────────────────────────────────

@router.post("/compare", response_model=SessionComparisonResponse)
async def compare_sessions(
    data: SessionComparisonRequest,
    db: AsyncSession = Depends(get_db),
):
    """Compare two sessions side by side."""
    # Fetch both sessions
    result_a = await db.execute(select(Session).where(Session.id == data.session_id_a))
    result_b = await db.execute(select(Session).where(Session.id == data.session_id_b))

    session_a = result_a.scalar_one_or_none()
    session_b = result_b.scalar_one_or_none()

    if not session_a or not session_b:
        raise HTTPException(status_code=404, detail="One or both sessions not found")

    # Fetch summaries
    sum_a = await db.execute(
        select(SessionSummary).where(SessionSummary.session_id == data.session_id_a)
    )
    sum_b = await db.execute(
        select(SessionSummary).where(SessionSummary.session_id == data.session_id_b)
    )

    return SessionComparisonResponse(
        session_a=session_a,
        session_b=session_b,
        summary_a=sum_a.scalar_one_or_none(),
        summary_b=sum_b.scalar_one_or_none(),
    )
