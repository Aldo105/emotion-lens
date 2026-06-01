"""
EmotionLens — Database Models (SQLAlchemy ORM)

Defines all database tables for sessions, emotion records, micro-expressions,
interviewer notes, and session summaries.
"""

import json
from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, String, Float, Text, Boolean,
    DateTime, ForeignKey, JSON, create_engine
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import (
    DeclarativeBase, relationship, sessionmaker
)

from backend.app.config import settings


# ── Base Model ────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


# ── Session (Interview Session) ──────────────────────────────────────

class Session(Base):
    """
    Represents a single interview/analysis session.
    Each session tracks one candidate's facial analysis over time.
    """
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)                  # e.g., "John Doe - Backend Dev Interview"
    candidate_name = Column(String(255), nullable=True)         # Name of the person being analyzed
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    ended_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)             # Total session duration
    status = Column(String(50), default="active")               # active, paused, completed, cancelled
    input_type = Column(String(20), default="webcam")           # webcam, video_upload
    video_path = Column(String(500), nullable=True)             # Path to uploaded video or recording
    notes_text = Column(Text, nullable=True)                    # General session notes

    # Relationships
    emotion_records = relationship("EmotionRecord", back_populates="session", cascade="all, delete-orphan")
    micro_expressions = relationship("MicroExpression", back_populates="session", cascade="all, delete-orphan")
    interviewer_notes = relationship("InterviewerNote", back_populates="session", cascade="all, delete-orphan")
    summary = relationship("SessionSummary", back_populates="session", uselist=False, cascade="all, delete-orphan")


# ── Emotion Record ───────────────────────────────────────────────────

class EmotionRecord(Base):
    """
    A single emotion reading captured at a specific timestamp.
    One record per processed frame.
    """
    __tablename__ = "emotion_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    timestamp = Column(Float, nullable=False)                   # Seconds from session start
    
    # Primary emotion prediction
    emotion = Column(String(50), nullable=False)                # e.g., "happy", "nervousness"
    confidence = Column(Float, nullable=False)                  # 0.0 - 1.0

    # Full probability distribution across all 9 emotions (JSON)
    emotion_probabilities = Column(JSON, nullable=True)
    # e.g., {"happy": 0.72, "neutral": 0.15, "nervousness": 0.08, ...}

    # Action Unit activations at this timestamp (JSON)
    action_units = Column(JSON, nullable=True)
    # e.g., {"AU1": 0.3, "AU6": 0.8, "AU12": 0.9, ...}

    # Congruence score at this moment
    congruence_score = Column(Float, nullable=True)             # 0 - 100

    # Model confidence (how sure the CNN is about its prediction)
    model_confidence = Column(Float, nullable=True)             # 0.0 - 1.0

    # Relationship
    session = relationship("Session", back_populates="emotion_records")


# ── Micro-Expression Event ───────────────────────────────────────────

class MicroExpression(Base):
    """
    A detected micro-expression event.
    Only events that pass the 3-layer smart filtering are stored.
    """
    __tablename__ = "micro_expressions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    timestamp = Column(Float, nullable=False)                   # Seconds from session start
    duration_ms = Column(Float, nullable=False)                 # Duration in milliseconds

    # What was detected
    detected_emotion = Column(String(50), nullable=False)       # The emotion the micro-expr suggests
    dominant_emotion_at_time = Column(String(50), nullable=True)  # What the face was showing overall
    action_units_involved = Column(JSON, nullable=False)        # e.g., ["AU1", "AU4", "AU15"]

    # Scoring from the 3-layer filter
    relevance_score = Column(Integer, nullable=False)           # 0-100
    temporal_valid = Column(Boolean, default=True)              # Passed temporal filter?
    context_valid = Column(Boolean, default=True)               # Passed context/baseline filter?
    is_contradictory = Column(Boolean, default=False)           # Contradicts dominant emotion?

    # Description for the log
    description = Column(Text, nullable=True)
    # e.g., "Brief flash of fear (AU1+AU4+AU20) while displaying happiness"

    # Relationship
    session = relationship("Session", back_populates="micro_expressions")


# ── Interviewer Notes ────────────────────────────────────────────────

class InterviewerNote(Base):
    """
    Notes and tags added by the interviewer during a session.
    Correlated with emotion data at the same timestamp.
    """
    __tablename__ = "interviewer_notes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    timestamp = Column(Float, nullable=False)                   # Seconds from session start
    content = Column(Text, nullable=False)                      # The note text
    tag = Column(String(100), nullable=True)                    # Optional tag: "key_question", "red_flag", etc.
    
    # Snapshot of emotion state when note was taken
    emotion_at_time = Column(String(50), nullable=True)
    congruence_at_time = Column(Float, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationship
    session = relationship("Session", back_populates="interviewer_notes")


# ── Session Summary ──────────────────────────────────────────────────

class SessionSummary(Base):
    """
    Aggregate summary generated when a session ends.
    Used for reports and session comparison.
    """
    __tablename__ = "session_summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, unique=True)

    # Emotion distribution (% of time in each emotion)
    emotion_distribution = Column(JSON, nullable=True)
    # e.g., {"happy": 0.42, "neutral": 0.30, "nervousness": 0.15, ...}

    dominant_emotion = Column(String(50), nullable=True)
    
    # Congruence / Trustworthiness
    average_congruence = Column(Float, nullable=True)           # 0-100
    min_congruence = Column(Float, nullable=True)
    max_congruence = Column(Float, nullable=True)

    # Micro-expression stats
    total_micro_expressions = Column(Integer, default=0)
    high_relevance_micro_expressions = Column(Integer, default=0)  # Score >= 80
    contradictory_micro_expressions = Column(Integer, default=0)

    # Nervousness & Confidence metrics
    average_nervousness = Column(Float, nullable=True)          # 0.0 - 1.0
    average_confidence = Column(Float, nullable=True)           # 0.0 - 1.0
    nervousness_peaks = Column(Integer, default=0)              # Number of nervousness spikes

    # Key moments (auto-detected notable timestamps)
    key_moments = Column(JSON, nullable=True)
    # e.g., [{"timestamp": 123.4, "type": "emotion_shift", "detail": "happy→fear"}, ...]

    # Model performance
    average_model_confidence = Column(Float, nullable=True)

    # Relationship
    session = relationship("Session", back_populates="summary")


# ── Database Engine & Session Factory ────────────────────────────────

engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    future=True
)

async_session_factory = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def init_db():
    """Create all tables if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("[OK] Database initialized")


async def get_db() -> AsyncSession:
    """Dependency injection for database sessions."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
