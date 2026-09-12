"""
EmotionLens — Session Manager Service

Manages the full lifecycle of analysis sessions:
  - Creating new sessions
  - Persisting per-frame emotion records and micro-expression events
  - Ending sessions and generating aggregate summaries
  - Computing emotion distribution, congruence stats, and key moments

All functions accept an async SQLAlchemy session for dependency injection
and are fully stateless — no global mutable state.
"""

from datetime import datetime, timezone
from typing import Optional

import numpy as np
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.database import (
    Session, SessionSummary, EmotionRecord,
    MicroExpression, InterviewerNote, async_session_factory,
)
from backend.app.models.schemas import (
    EmotionData, MicroExpressionEvent, SessionResponse,
)


# ═══════════════════════════════════════════════════════════════════════
# SESSION LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════

async def create_session(
    db: AsyncSession,
    name: str,
    candidate_name: Optional[str] = None,
    input_type: str = "webcam",
) -> Session:
    """
    Create a new analysis session in the database.

    Args:
        db: Async SQLAlchemy session.
        name: Human-readable session name (e.g., "John Doe — Backend Dev").
        candidate_name: Optional name of the person being analyzed.
        input_type: Source type — "webcam" or "video_upload".

    Returns:
        The newly created Session ORM instance.
    """
    session = Session(
        name=name,
        candidate_name=candidate_name,
        input_type=input_type,
        status="active",
    )
    db.add(session)
    await db.flush()
    await db.refresh(session)
    return session


async def end_session(db: AsyncSession, session_id: int) -> Session:
    """
    End an active session: set status to 'completed', record end time
    and duration, then generate the aggregate SessionSummary.

    Args:
        db: Async SQLAlchemy session.
        session_id: ID of the session to end.

    Returns:
        The updated Session ORM instance.

    Raises:
        ValueError: If the session is not found.
    """
    result = await db.execute(select(Session).where(Session.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise ValueError(f"Session {session_id} not found")

    session.status = "completed"
    session.ended_at = datetime.now(timezone.utc)

    if session.created_at:
        delta = session.ended_at - session.created_at
        session.duration_seconds = delta.total_seconds()

    # Generate the aggregate summary before committing
    await generate_summary(db, session_id)

    await db.flush()
    await db.refresh(session)
    return session


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY GENERATION
# ═══════════════════════════════════════════════════════════════════════

async def generate_summary(db: AsyncSession, session_id: int) -> SessionSummary:
    """
    Aggregate all EmotionRecord, MicroExpression, and InterviewerNote data
    for a session into a single SessionSummary row.

    Computes:
      - Emotion distribution (percentage of time in each emotion)
      - Dominant emotion
      - Congruence statistics (average, min, max)
      - Micro-expression counts (total, high-relevance, contradictory)
      - Nervousness / confidence averages
      - Nervousness peaks (frames where nervousness > 0.5)
      - Key moments (emotion shifts, high micro-expression relevance)
      - Average model confidence

    Args:
        db: Async SQLAlchemy session.
        session_id: ID of the session to summarize.

    Returns:
        The created or updated SessionSummary ORM instance.
    """
    # ── Fetch emotion records ────────────────────────────────────────
    emo_result = await db.execute(
        select(EmotionRecord)
        .where(EmotionRecord.session_id == session_id)
        .order_by(EmotionRecord.timestamp.asc())
    )
    emotion_records = emo_result.scalars().all()

    # ── Fetch micro-expressions ──────────────────────────────────────
    micro_result = await db.execute(
        select(MicroExpression)
        .where(MicroExpression.session_id == session_id)
        .order_by(MicroExpression.timestamp.asc())
    )
    micro_expressions = micro_result.scalars().all()

    # ── Fetch interviewer notes ──────────────────────────────────────
    notes_result = await db.execute(
        select(InterviewerNote)
        .where(InterviewerNote.session_id == session_id)
        .order_by(InterviewerNote.timestamp.asc())
    )
    notes = notes_result.scalars().all()

    # ── Emotion distribution ─────────────────────────────────────────
    emotion_distribution: dict[str, float] = {}
    dominant_emotion: Optional[str] = None
    avg_congruence: Optional[float] = None
    min_congruence: Optional[float] = None
    max_congruence: Optional[float] = None
    avg_nervousness: Optional[float] = None
    avg_confidence: Optional[float] = None
    nervousness_peaks: int = 0
    avg_model_confidence: Optional[float] = None
    key_moments: list[dict] = []

    if emotion_records:
        total_records = len(emotion_records)

        # Count occurrences of each emotion
        emotion_counts: dict[str, int] = {}
        congruence_scores: list[float] = []
        nervousness_values: list[float] = []
        confidence_values: list[float] = []
        model_confidences: list[float] = []

        prev_emotion: Optional[str] = None

        for record in emotion_records:
            # Emotion counts
            emotion_counts[record.emotion] = emotion_counts.get(record.emotion, 0) + 1

            # Congruence
            if record.congruence_score is not None:
                congruence_scores.append(record.congruence_score)

            # Nervousness and confidence from emotion probabilities
            if record.emotion_probabilities:
                nervousness_val = record.emotion_probabilities.get("nervousness", 0.0)
                confidence_val = record.emotion_probabilities.get("confidence", 0.0)
                nervousness_values.append(nervousness_val)
                confidence_values.append(confidence_val)

                if nervousness_val > 0.5:
                    nervousness_peaks += 1

            # Model confidence
            if record.model_confidence is not None:
                model_confidences.append(record.model_confidence)

            # Detect emotion shifts for key moments
            if prev_emotion and record.emotion != prev_emotion:
                key_moments.append({
                    "timestamp": round(record.timestamp, 2),
                    "type": "emotion_shift",
                    "detail": f"{prev_emotion} → {record.emotion}",
                })
            prev_emotion = record.emotion

        # Compute percentages
        emotion_distribution = {
            emotion: round(count / total_records, 4)
            for emotion, count in emotion_counts.items()
        }

        # Dominant emotion = highest percentage
        dominant_emotion = max(emotion_distribution, key=emotion_distribution.get)

        # Congruence stats
        if congruence_scores:
            congruence_arr = np.array(congruence_scores)
            avg_congruence = round(float(np.mean(congruence_arr)), 2)
            min_congruence = round(float(np.min(congruence_arr)), 2)
            max_congruence = round(float(np.max(congruence_arr)), 2)

        # Nervousness & confidence averages
        if nervousness_values:
            avg_nervousness = round(float(np.mean(nervousness_values)), 4)
        if confidence_values:
            avg_confidence = round(float(np.mean(confidence_values)), 4)
        if model_confidences:
            avg_model_confidence = round(float(np.mean(model_confidences)), 4)

    # ── Micro-expression stats ───────────────────────────────────────
    total_micro = len(micro_expressions)
    high_relevance_micro = sum(1 for m in micro_expressions if m.relevance_score >= 80)
    contradictory_micro = sum(1 for m in micro_expressions if m.is_contradictory)

    # Add high-relevance micro-expressions as key moments
    for m in micro_expressions:
        if m.relevance_score >= 80:
            key_moments.append({
                "timestamp": round(m.timestamp, 2),
                "type": "micro_expression",
                "detail": f"{m.detected_emotion} (relevance: {m.relevance_score})",
            })

    # Add tagged interviewer notes as key moments
    for n in notes:
        if n.tag in ("key_question", "red_flag", "important"):
            key_moments.append({
                "timestamp": round(n.timestamp, 2),
                "type": "interviewer_note",
                "detail": f"[{n.tag}] {n.content[:80]}",
            })

    # Sort key moments chronologically
    key_moments.sort(key=lambda m: m["timestamp"])

    # ── Upsert SessionSummary ────────────────────────────────────────
    existing_result = await db.execute(
        select(SessionSummary).where(SessionSummary.session_id == session_id)
    )
    summary = existing_result.scalar_one_or_none()

    if summary is None:
        summary = SessionSummary(session_id=session_id)
        db.add(summary)

    summary.emotion_distribution = emotion_distribution
    summary.dominant_emotion = dominant_emotion
    summary.average_congruence = avg_congruence
    summary.min_congruence = min_congruence
    summary.max_congruence = max_congruence
    summary.total_micro_expressions = total_micro
    summary.high_relevance_micro_expressions = high_relevance_micro
    summary.contradictory_micro_expressions = contradictory_micro
    summary.average_nervousness = avg_nervousness
    summary.average_confidence = avg_confidence
    summary.nervousness_peaks = nervousness_peaks
    summary.key_moments = key_moments
    summary.average_model_confidence = avg_model_confidence

    await db.flush()
    await db.refresh(summary)
    return summary


# ═══════════════════════════════════════════════════════════════════════
# PER-FRAME DATA PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════

async def save_emotion_record(
    db: AsyncSession,
    session_id: int,
    data: dict,
) -> EmotionRecord:
    """
    Persist a single frame's emotion analysis result to the database.

    Args:
        db: Async SQLAlchemy session.
        session_id: ID of the parent session.
        data: Dict with keys matching EmotionRecord columns:
            - timestamp, emotion, confidence, emotion_probabilities,
              action_units, congruence_score, model_confidence

    Returns:
        The created EmotionRecord ORM instance.
    """
    record = EmotionRecord(
        session_id=session_id,
        timestamp=data.get("timestamp", 0.0),
        emotion=data.get("emotion", "neutral"),
        confidence=data.get("confidence", 0.0),
        emotion_probabilities=data.get("emotion_probabilities"),
        action_units=data.get("action_units"),
        congruence_score=data.get("congruence_score"),
        model_confidence=data.get("model_confidence"),
    )
    db.add(record)
    await db.flush()
    return record


async def save_micro_expression(
    db: AsyncSession,
    session_id: int,
    event: dict,
) -> MicroExpression:
    """
    Persist a micro-expression event to the database.

    Args:
        db: Async SQLAlchemy session.
        session_id: ID of the parent session.
        event: Dict with keys matching MicroExpression columns:
            - timestamp, duration_ms, detected_emotion,
              dominant_emotion_at_time, action_units_involved,
              relevance_score, is_contradictory, description

    Returns:
        The created MicroExpression ORM instance.
    """
    micro = MicroExpression(
        session_id=session_id,
        timestamp=event.get("timestamp", 0.0),
        duration_ms=event.get("duration_ms", 0.0),
        detected_emotion=event.get("detected_emotion", "unknown"),
        dominant_emotion_at_time=event.get("dominant_emotion_at_time"),
        action_units_involved=event.get("action_units_involved", []),
        relevance_score=event.get("relevance_score", 0),
        temporal_valid=event.get("temporal_valid", True),
        context_valid=event.get("context_valid", True),
        is_contradictory=event.get("is_contradictory", False),
        description=event.get("description"),
    )
    db.add(micro)
    await db.flush()
    return micro
