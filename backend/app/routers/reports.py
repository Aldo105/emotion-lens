"""
EmotionLens — Reports Router

Endpoints for generating and downloading session reports (PDF/CSV).

Routes:
  GET /api/reports/{session_id}/data — Full session data as JSON
  GET /api/reports/{session_id}/pdf  — Generate and download PDF report
  GET /api/reports/{session_id}/csv  — Generate and download CSV export
"""

import io
import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import settings
from backend.app.models.database import (
    get_db, Session, SessionSummary,
    EmotionRecord, MicroExpression, InterviewerNote,
)
from backend.app.services.report_generator import (
    generate_pdf_report, generate_csv_string,
)

router = APIRouter()


# ── Helper: Build report data dict ───────────────────────────────────

async def _get_report_data(session_id: int, db: AsyncSession) -> dict:
    """
    Fetch all session data needed for report generation.
    Returns a dict with session, summary, emotion_timeline,
    micro_expressions, and notes.
    """
    # Fetch session
    result = await db.execute(select(Session).where(Session.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Fetch summary
    sum_result = await db.execute(
        select(SessionSummary).where(SessionSummary.session_id == session_id)
    )
    summary = sum_result.scalar_one_or_none()

    # Fetch emotion timeline
    emo_result = await db.execute(
        select(EmotionRecord)
        .where(EmotionRecord.session_id == session_id)
        .order_by(EmotionRecord.timestamp.asc())
    )
    emotion_records = emo_result.scalars().all()

    # Fetch micro-expressions
    micro_result = await db.execute(
        select(MicroExpression)
        .where(MicroExpression.session_id == session_id)
        .order_by(MicroExpression.timestamp.asc())
    )
    micro_expressions = micro_result.scalars().all()

    # Fetch notes
    notes_result = await db.execute(
        select(InterviewerNote)
        .where(InterviewerNote.session_id == session_id)
        .order_by(InterviewerNote.timestamp.asc())
    )
    notes = notes_result.scalars().all()

    return {
        "session": {
            "id": session.id,
            "name": session.name,
            "candidate_name": session.candidate_name,
            "created_at": session.created_at.isoformat() if session.created_at else None,
            "ended_at": session.ended_at.isoformat() if session.ended_at else None,
            "duration_seconds": session.duration_seconds,
            "status": session.status,
            "input_type": session.input_type,
        },
        "summary": {
            "emotion_distribution": summary.emotion_distribution if summary else None,
            "dominant_emotion": summary.dominant_emotion if summary else None,
            "average_congruence": summary.average_congruence if summary else None,
            "total_micro_expressions": summary.total_micro_expressions if summary else 0,
            "contradictory_micro_expressions": summary.contradictory_micro_expressions if summary else 0,
            "average_nervousness": summary.average_nervousness if summary else None,
            "average_confidence": summary.average_confidence if summary else None,
            "nervousness_peaks": summary.nervousness_peaks if summary else 0,
            "average_model_confidence": summary.average_model_confidence if summary else None,
            "key_moments": summary.key_moments if summary else [],
        } if summary else None,
        "emotion_timeline": [
            {
                "timestamp": r.timestamp,
                "emotion": r.emotion,
                "confidence": r.confidence,
                "congruence_score": r.congruence_score,
                "model_confidence": r.model_confidence,
                "emotion_probabilities": r.emotion_probabilities,
                "action_units": r.action_units,
            }
            for r in emotion_records
        ],
        "micro_expressions": [
            {
                "timestamp": m.timestamp,
                "duration_ms": m.duration_ms,
                "detected_emotion": m.detected_emotion,
                "dominant_emotion_at_time": m.dominant_emotion_at_time,
                "action_units_involved": m.action_units_involved,
                "relevance_score": m.relevance_score,
                "is_contradictory": m.is_contradictory,
                "description": m.description,
            }
            for m in micro_expressions
        ],
        "notes": [
            {
                "timestamp": n.timestamp,
                "content": n.content,
                "tag": n.tag,
                "emotion_at_time": n.emotion_at_time,
                "congruence_at_time": n.congruence_at_time,
            }
            for n in notes
        ],
    }


# ── Get Report Data (JSON) ───────────────────────────────────────────

@router.get("/{session_id}/data")
async def get_report_data(session_id: int, db: AsyncSession = Depends(get_db)):
    """
    Get all data needed for a session report.
    This endpoint returns the full dataset — the frontend or
    report_generator can format it into PDF/CSV.
    """
    return await _get_report_data(session_id, db)


# ── Download PDF Report ──────────────────────────────────────────────

@router.get("/{session_id}/pdf")
async def download_pdf_report(session_id: int, db: AsyncSession = Depends(get_db)):
    """
    Generate a professional PDF report for a session and return it
    as a downloadable file.
    """
    session_data = await _get_report_data(session_id, db)

    # Build filename
    session_name = session_data["session"].get("name", f"session_{session_id}")
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in session_name)
    filename = f"EmotionLens_{safe_name}.pdf"

    output_path = os.path.join(settings.reports_dir, filename)
    generate_pdf_report(session_data, output_path)

    return FileResponse(
        path=output_path,
        media_type="application/pdf",
        filename=filename,
    )


# ── Download CSV Report ──────────────────────────────────────────────

@router.get("/{session_id}/csv")
async def download_csv_report(session_id: int, db: AsyncSession = Depends(get_db)):
    """
    Generate a CSV export of the emotion timeline for a session
    and return it as a streaming download.
    """
    session_data = await _get_report_data(session_id, db)

    session_name = session_data["session"].get("name", f"session_{session_id}")
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in session_name)
    filename = f"EmotionLens_{safe_name}.csv"

    csv_content = generate_csv_string(session_data)

    return StreamingResponse(
        io.StringIO(csv_content),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

