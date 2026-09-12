"""
EmotionLens — Session Feedback Router

Endpoints for submitting and retrieving subject post-session surveys.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.database import get_db, SessionFeedback, Session
from backend.app.models.schemas import SessionFeedbackCreate, SessionFeedbackResponse

router = APIRouter()


@router.post("/{session_id}", response_model=SessionFeedbackResponse, status_code=201)
async def create_feedback(
    session_id: int,
    data: SessionFeedbackCreate,
    db: AsyncSession = Depends(get_db),
):
    """Submit post-session survey feedback from the subject."""
    # Verify session exists
    result = await db.execute(select(Session).where(Session.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Check if feedback already exists for this session
    existing_result = await db.execute(
        select(SessionFeedback).where(SessionFeedback.session_id == session_id)
    )
    existing_feedback = existing_result.scalar_one_or_none()
    if existing_feedback:
        # Update existing feedback
        existing_feedback.overall_accuracy_rating = data.overall_accuracy_rating
        existing_feedback.self_reported_emotion = data.self_reported_emotion
        existing_feedback.attempted_suppression = data.attempted_suppression
        existing_feedback.moment_validations = data.moment_validations
        existing_feedback.free_text_comments = data.free_text_comments
        await db.flush()
        await db.refresh(existing_feedback)
        return existing_feedback

    # Create new feedback record
    feedback = SessionFeedback(
        session_id=session_id,
        overall_accuracy_rating=data.overall_accuracy_rating,
        self_reported_emotion=data.self_reported_emotion,
        attempted_suppression=data.attempted_suppression,
        moment_validations=data.moment_validations,
        free_text_comments=data.free_text_comments,
    )
    db.add(feedback)
    await db.flush()
    await db.refresh(feedback)
    return feedback


@router.get("/{session_id}", response_model=SessionFeedbackResponse)
async def get_feedback(
    session_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Retrieve feedback for a specific session."""
    result = await db.execute(
        select(SessionFeedback).where(SessionFeedback.session_id == session_id)
    )
    feedback = result.scalar_one_or_none()
    if not feedback:
        raise HTTPException(status_code=404, detail="Feedback not found for this session")
    return feedback


@router.get("/accuracy-stats/global")
async def get_global_accuracy_stats(
    db: AsyncSession = Depends(get_db),
):
    """Get global aggregated accuracy statistics across all feedback sessions."""
    result = await db.execute(
        select(
            func.avg(SessionFeedback.overall_accuracy_rating).label("avg_accuracy"),
            func.count(SessionFeedback.id).label("total_feedback_count"),
            func.sum(func.cast(SessionFeedback.attempted_suppression, Integer)).label("total_suppressions"),
        )
    )
    row = result.fetchone()
    
    avg_acc = row[0] if row and row[0] is not None else 0.0
    total_count = row[1] if row and row[1] is not None else 0
    total_supp = row[2] if row and row[2] is not None else 0

    return {
        "average_accuracy": float(avg_acc),
        "total_feedback_count": int(total_count),
        "total_suppression_reported": int(total_supp),
    }
