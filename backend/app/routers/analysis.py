"""
EmotionLens — Analysis Router

Endpoints for generating and fetching interview analysis data.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.database import (
    get_db, Session, SessionSummary, InterviewAnalysis, 
    EmotionRecord, MicroExpression, InterviewerNote
)
from backend.app.models.schemas import InterviewAnalysisResponse, InterviewAnalysisSummary
from backend.app.services.interview_analyzer import InterviewBehaviorAnalyzer

router = APIRouter()

@router.get("/{session_id}", response_model=InterviewAnalysisResponse)
async def get_analysis(session_id: int, db: AsyncSession = Depends(get_db)):
    """Get the full interview analysis for a session."""
    result = await db.execute(
        select(InterviewAnalysis).where(InterviewAnalysis.session_id == session_id)
    )
    analysis = result.scalar_one_or_none()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Interview analysis not found for this session")
        
    return analysis

@router.post("/{session_id}/generate", response_model=InterviewAnalysisResponse)
async def regenerate_analysis(session_id: int, db: AsyncSession = Depends(get_db)):
    """Regenerate the interview analysis for a session."""
    # Fetch session
    session_result = await db.execute(select(Session).where(Session.id == session_id))
    session = session_result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Fetch summary
    sum_result = await db.execute(select(SessionSummary).where(SessionSummary.session_id == session_id))
    summary = sum_result.scalar_one_or_none()
    summary_dict = {
        "emotion_distribution": summary.emotion_distribution if summary else None,
        "average_congruence": summary.average_congruence if summary else 100,
        "total_micro_expressions": summary.total_micro_expressions if summary else 0,
        "contradictory_micro_expressions": summary.contradictory_micro_expressions if summary else 0,
        "average_nervousness": summary.average_nervousness if summary else 0,
        "average_confidence": summary.average_confidence if summary else 0,
        "nervousness_peaks": summary.nervousness_peaks if summary else 0,
    }

    # Fetch records and convert to dicts
    emo_result = await db.execute(
        select(EmotionRecord).where(EmotionRecord.session_id == session_id).order_by(EmotionRecord.timestamp)
    )
    emotion_records = [{"timestamp": r.timestamp, "emotion": r.emotion, "emotion_probabilities": r.emotion_probabilities, "action_units": r.action_units, "congruence_score": r.congruence_score} for r in emo_result.scalars().all()]

    micro_result = await db.execute(
        select(MicroExpression).where(MicroExpression.session_id == session_id).order_by(MicroExpression.timestamp)
    )
    micro_expressions = [{"timestamp": m.timestamp, "detected_emotion": m.detected_emotion} for m in micro_result.scalars().all()]

    notes_result = await db.execute(
        select(InterviewerNote).where(InterviewerNote.session_id == session_id).order_by(InterviewerNote.timestamp)
    )
    notes = [{"timestamp": n.timestamp, "content": n.content, "tag": n.tag} for n in notes_result.scalars().all()]

    # Mock noise stats
    noise_stats = {
        "noise_events_filtered": 0,
        "speaking_time_ratio": 0.0
    }

    # Run Analyzer
    analyzer = InterviewBehaviorAnalyzer()
    analysis_data = analyzer.analyze_session(
        emotion_records=emotion_records, 
        micro_expressions=micro_expressions, 
        interviewer_notes=notes,
        summary=summary_dict,
        noise_stats=noise_stats
    )

    # Upsert results into InterviewAnalysis table
    result = await db.execute(
        select(InterviewAnalysis).where(InterviewAnalysis.session_id == session_id)
    )
    analysis = result.scalar_one_or_none()
    
    if not analysis:
        analysis = InterviewAnalysis(session_id=session_id)
        db.add(analysis)

    # Update fields from analysis_data
    analysis.technical_mastery_score = analysis_data.dimension_scores.get("technical_mastery")
    analysis.emotional_stability_score = analysis_data.dimension_scores.get("emotional_stability")
    analysis.authenticity_score = analysis_data.dimension_scores.get("authenticity")
    analysis.self_confidence_score = analysis_data.dimension_scores.get("self_confidence")
    analysis.communication_score = analysis_data.dimension_scores.get("communication")
    analysis.overall_score = analysis_data.dimension_scores.get("overall_score")

    analysis.behavioral_patterns = analysis_data.behavioral_patterns
    analysis.red_flags = analysis_data.red_flags
    analysis.question_correlations = analysis_data.question_correlations
    analysis.recommendations = analysis_data.recommendations

    if analysis_data.noise_stats:
        analysis.noise_events_filtered = analysis_data.noise_stats.get("noise_events_filtered", 0)
        analysis.speaking_time_ratio = analysis_data.noise_stats.get("speaking_time_ratio", 0.0)

    await db.commit()
    await db.refresh(analysis)

    return analysis

@router.get("/{session_id}/summary", response_model=InterviewAnalysisSummary)
async def get_analysis_summary(session_id: int, db: AsyncSession = Depends(get_db)):
    """Get executive summary of interview analysis."""
    result = await db.execute(
        select(InterviewAnalysis).where(InterviewAnalysis.session_id == session_id)
    )
    analysis = result.scalar_one_or_none()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Interview analysis not found for this session")

    # Derive top_strengths
    top_strengths = []
    if analysis.authenticity_score and analysis.authenticity_score > 80:
        top_strengths.append("High authenticity")
    if analysis.emotional_stability_score and analysis.emotional_stability_score > 80:
        top_strengths.append("Strong emotional stability")
    if analysis.self_confidence_score and analysis.self_confidence_score > 80:
        top_strengths.append("High self-confidence")
    if analysis.behavioral_patterns:
        for p in analysis.behavioral_patterns:
            if p.get("is_positive"):
                top_strengths.append(p.get("description", p.get("type")))

    # Derive areas_of_concern
    areas_of_concern = []
    if analysis.authenticity_score and analysis.authenticity_score < 50:
        areas_of_concern.append("Low authenticity")
    if analysis.emotional_stability_score and analysis.emotional_stability_score < 50:
        areas_of_concern.append("Low emotional stability")
    if analysis.red_flags:
        for rf in analysis.red_flags:
            areas_of_concern.append(rf.get("evidence", rf.get("type")))
            
    # Key recommendations
    key_recommendations = []
    if analysis.recommendations:
        for r in analysis.recommendations:
            key_recommendations.append(r.get("text"))

    return {
        "overall_score": analysis.overall_score or 0.0,
        "top_strengths": top_strengths[:3],
        "areas_of_concern": areas_of_concern[:3],
        "key_recommendations": key_recommendations[:3],
        "pattern_count": len(analysis.behavioral_patterns) if analysis.behavioral_patterns else 0,
        "red_flag_count": len(analysis.red_flags) if analysis.red_flags else 0
    }
