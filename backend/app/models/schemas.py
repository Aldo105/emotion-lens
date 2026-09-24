"""
EmotionLens — Pydantic Schemas

Request/response schemas for the REST API and WebSocket messages.
Separated from ORM models to keep API contracts clean.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════
# SESSION SCHEMAS
# ═══════════════════════════════════════════════════════════════════════

class SessionCreate(BaseModel):
    """Request body for creating a new session."""
    name: str = Field(..., min_length=1, max_length=255, description="Session name")
    candidate_name: Optional[str] = Field(None, max_length=255, description="Candidate's name")
    input_type: str = Field("webcam", pattern="^(webcam|video_upload)$")


class SessionUpdate(BaseModel):
    """Request body for updating session details."""
    name: Optional[str] = Field(None, max_length=255)
    candidate_name: Optional[str] = Field(None, max_length=255)
    status: Optional[str] = Field(None, pattern="^(active|paused|completed|cancelled)$")
    notes_text: Optional[str] = None


class SessionResponse(BaseModel):
    """Response schema for a session."""
    id: int
    name: str
    candidate_name: Optional[str] = None
    created_at: datetime
    ended_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    status: str
    input_type: str
    video_path: Optional[str] = None
    notes_text: Optional[str] = None
    # Copied from the session's summary row so the session list can show
    # congruence without one extra request per card.
    average_congruence: Optional[float] = None
    dominant_emotion: Optional[str] = None

    class Config:
        from_attributes = True


class SessionListResponse(BaseModel):
    """Response schema for listing sessions."""
    sessions: list[SessionResponse]
    total: int


# ═══════════════════════════════════════════════════════════════════════
# EMOTION RECORD SCHEMAS
# ═══════════════════════════════════════════════════════════════════════

class EmotionData(BaseModel):
    """Real-time emotion data sent via WebSocket."""
    timestamp: float
    emotion: str
    confidence: float
    emotion_probabilities: dict[str, float]
    action_units: dict[str, float]
    congruence_score: float
    model_confidence: float


# ═══════════════════════════════════════════════════════════════════════
# MICRO-EXPRESSION SCHEMAS
# ═══════════════════════════════════════════════════════════════════════

class MicroExpressionEvent(BaseModel):
    """A micro-expression event detected by the engine."""
    timestamp: float
    duration_ms: float
    detected_emotion: str
    dominant_emotion_at_time: Optional[str] = None
    action_units_involved: list[str]
    relevance_score: int = Field(..., ge=0, le=100)
    is_contradictory: bool = False
    description: str


class MicroExpressionResponse(BaseModel):
    """Response schema for stored micro-expression."""
    id: int
    session_id: int
    timestamp: float
    duration_ms: float
    detected_emotion: str
    dominant_emotion_at_time: Optional[str] = None
    action_units_involved: list[str]
    relevance_score: int
    is_contradictory: bool
    description: Optional[str] = None

    class Config:
        from_attributes = True


# ═══════════════════════════════════════════════════════════════════════
# INTERVIEWER NOTES SCHEMAS
# ═══════════════════════════════════════════════════════════════════════

class NoteCreate(BaseModel):
    """Request body for creating an interviewer note."""
    content: str = Field(..., min_length=1, description="Note text")
    tag: Optional[str] = Field(None, max_length=100, description="Optional tag")
    timestamp: float = Field(..., description="Seconds from session start")


class NoteResponse(BaseModel):
    """Response schema for an interviewer note."""
    id: int
    session_id: int
    timestamp: float
    content: str
    tag: Optional[str] = None
    emotion_at_time: Optional[str] = None
    congruence_at_time: Optional[float] = None
    created_at: datetime

    class Config:
        from_attributes = True


# ═══════════════════════════════════════════════════════════════════════
# SESSION SUMMARY & COMPARISON SCHEMAS
# ═══════════════════════════════════════════════════════════════════════

class SessionSummaryResponse(BaseModel):
    """Full session summary for reports."""
    id: int
    session_id: int
    emotion_distribution: Optional[dict[str, float]] = None
    dominant_emotion: Optional[str] = None
    average_congruence: Optional[float] = None
    min_congruence: Optional[float] = None
    max_congruence: Optional[float] = None
    total_micro_expressions: int = 0
    high_relevance_micro_expressions: int = 0
    contradictory_micro_expressions: int = 0
    average_nervousness: Optional[float] = None
    average_confidence: Optional[float] = None
    nervousness_peaks: int = 0
    key_moments: Optional[list[dict]] = None
    average_model_confidence: Optional[float] = None

    class Config:
        from_attributes = True


class SessionComparisonRequest(BaseModel):
    """Request to compare two sessions."""
    session_id_a: int
    session_id_b: int


class SessionComparisonResponse(BaseModel):
    """Side-by-side comparison of two sessions."""
    session_a: SessionResponse
    session_b: SessionResponse
    summary_a: Optional[SessionSummaryResponse] = None
    summary_b: Optional[SessionSummaryResponse] = None


# ═══════════════════════════════════════════════════════════════════════
# WEBSOCKET MESSAGE SCHEMAS
# ═══════════════════════════════════════════════════════════════════════

class WSFrameResult(BaseModel):
    """
    Complete result payload sent back to the frontend via WebSocket
    for each processed frame.
    """
    type: str = "frame_result"
    timestamp: float
    
    # Emotion
    emotion: str
    confidence: float
    emotion_probabilities: dict[str, float]
    model_confidence: float
    # A brief strong reading that differs from the dominant emotion, or None.
    # Expressions like surprise last a moment and never dominate the smoothed
    # average, so without this they are detected and then discarded.
    peak_emotion: Optional[dict] = None

    # Action Units
    action_units: dict[str, float]

    # Congruence
    congruence_score: float
    congruence_breakdown: Optional[dict[str, float]] = None

    # Running performance index. The report's overall score is only available
    # once the session closes, so the live panel had no equivalent number.
    live_score: Optional[dict] = None

    # Why no emotion is being reported, when that is the case. A blank panel
    # with no reason is what made the heart rate so slow to diagnose.
    emotion_blocked_reason: Optional[str] = None

    # Micro-expression (if one was just detected)
    micro_expression: Optional[MicroExpressionEvent] = None

    # Heart Rate (Eulerian Video Magnification / rPPG)
    heart_rate: Optional[dict] = None
    evm_frame: Optional[str] = None
    # {"bpm": float, "bpm_confidence": float, "signal_ready": bool, "stress_indicator": float}

    # Camera quality assessment
    camera_quality: Optional[dict] = None
    # {"score": float, "face_size_ratio": float, "brightness": float, "warnings": list[str]}

    # Calibration status
    is_calibrating: bool = False
    calibration_progress: float = 0.0  # 0.0 - 1.0
    # Which head pose the guided calibration is asking for, and whether the
    # subject is currently holding it. None once calibration is done.
    calibration_pose: Optional[dict] = None

    # Noise filter state
    noise_state: Optional[dict] = None
    # {"is_speaking": bool, "is_yawning": bool, "noise_type": str|null}


class WSStatusMessage(BaseModel):
    """Status/control message sent via WebSocket."""
    type: str  # "status", "error", "calibration_complete", "session_ended"
    message: str
    data: Optional[dict] = None


# ═══════════════════════════════════════════════════════════════════════
# FEEDBACK SCHEMAS
# ═══════════════════════════════════════════════════════════════════════

class SessionFeedbackCreate(BaseModel):
    """Request body for submitting post-session feedback."""
    overall_accuracy_rating: float = Field(..., ge=0.0, le=1.0)
    self_reported_emotion: Optional[str] = Field(None, max_length=50)
    attempted_suppression: bool = False
    moment_validations: Optional[list[dict]] = None
    free_text_comments: Optional[str] = None


class SessionFeedbackResponse(BaseModel):
    """Response schema for feedback."""
    id: int
    session_id: int
    overall_accuracy_rating: Optional[float] = None
    self_reported_emotion: Optional[str] = None
    attempted_suppression: bool
    moment_validations: Optional[list[dict]] = None
    free_text_comments: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


# ═══════════════════════════════════════════════════════════════════════
# INTERVIEW ANALYSIS SCHEMAS
# ═══════════════════════════════════════════════════════════════════════

class BehavioralPattern(BaseModel):
    """A detected behavioral pattern during the interview."""
    type: str  # e.g., "rapid_confusion", "nervous_spiral"
    timestamp: Optional[float] = None
    severity: str = Field(..., pattern="^(high|medium|low)$")
    description: str
    context_question: Optional[str] = None
    is_positive: bool = False  # True for patterns like stress_recovery, authentic_engagement


class RedFlag(BaseModel):
    """A behavioral red flag detected during analysis."""
    type: str
    evidence: str
    severity: str = Field(..., pattern="^(high|medium|low)$")


class QuestionCorrelation(BaseModel):
    """Correlation between an interviewer question and emotional reaction."""
    question_timestamp: float
    question_text: str
    pre_emotion: str
    post_emotion: str
    emotion_shift: str  # "positive", "negative", "neutral"
    nervousness_change: Optional[float] = None
    congruence_change: Optional[float] = None
    insight: str


class Recommendation(BaseModel):
    """An AI-generated recommendation for the hiring team."""
    category: str  # "technical", "behavioral", "emotional", "positive", "follow_up"
    priority: str = Field(..., pattern="^(high|medium|low)$")
    text: str


class DimensionScores(BaseModel):
    """Scores across 5 evaluation dimensions."""
    technical_mastery: float = Field(..., ge=0, le=100)
    emotional_stability: float = Field(..., ge=0, le=100)
    authenticity: float = Field(..., ge=0, le=100)
    self_confidence: float = Field(..., ge=0, le=100)
    communication: float = Field(..., ge=0, le=100)
    overall: float = Field(..., ge=0, le=100)


class NoiseFilterStats(BaseModel):
    """Statistics from the facial noise filter."""
    total_frames_analyzed: int = 0
    speaking_frames: int = 0
    yawn_events: int = 0
    tic_events: int = 0
    scratch_events: int = 0
    forced_blink_events: int = 0
    speaking_time_ratio: float = 0.0
    noise_events_filtered: int = 0


class InterviewAnalysisResponse(BaseModel):
    """Complete interview analysis response."""
    id: int
    session_id: int
    dimension_scores: Optional[DimensionScores] = None
    behavioral_patterns: Optional[list[BehavioralPattern]] = None
    red_flags: Optional[list[RedFlag]] = None
    question_correlations: Optional[list[QuestionCorrelation]] = None
    recommendations: Optional[list[Recommendation]] = None
    noise_stats: Optional[NoiseFilterStats] = None
    created_at: datetime

    class Config:
        from_attributes = True


class InterviewAnalysisSummary(BaseModel):
    """Executive summary of interview analysis."""
    overall_score: float
    top_strengths: list[str]
    areas_of_concern: list[str]
    key_recommendations: list[str]
    pattern_count: int
    red_flag_count: int
