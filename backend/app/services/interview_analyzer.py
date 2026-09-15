import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


def _format_video_time(seconds: float) -> str:
    """Format elapsed session time as MM:SS for human review."""
    seconds = max(0, int(round(seconds or 0)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


# Plain-language interpretation for notable emotion-state transitions.
# These are hypotheses, not conclusions — every entry is meant to point a
# human reviewer at a specific moment in the video, not to replace their
# judgment (see the project's ethical framework: results must never be the
# sole basis for an assessment).
TRANSITION_INTERPRETATIONS: dict[tuple[str, str], dict] = {
    ("happy", "angry"):        {"text": "Cambio abrupto de alegria a enojo — posible reaccion a un comentario o pregunta especifica.", "severity": "high", "positive": False},
    ("happy", "sad"):          {"text": "Cambio de alegria a tristeza — podria reflejar una respuesta genuina a un tema sensible.", "severity": "medium", "positive": False},
    ("happy", "nervousness"):  {"text": "De alegria a nerviosismo — posible incomodidad ante un cambio de tema o pregunta.", "severity": "medium", "positive": False},
    ("happy", "fear"):         {"text": "De alegria a temor — cambio marcado, revisar que lo provoco.", "severity": "high", "positive": False},
    ("neutral", "nervousness"):{"text": "De neutral a nerviosismo — posible senal de tension ante la pregunta actual.", "severity": "medium", "positive": False},
    ("neutral", "fear"):       {"text": "De neutral a miedo — reaccion notable, revisar el estimulo que la provoco.", "severity": "high", "positive": False},
    ("neutral", "angry"):      {"text": "De neutral a enojo — posible reaccion a una pregunta incomoda.", "severity": "medium", "positive": False},
    ("confidence", "nervousness"): {"text": "De confianza a nerviosismo — posible perdida de seguridad, comun ante preguntas dificiles.", "severity": "medium", "positive": False},
    ("confidence", "fear"):    {"text": "De confianza a temor — cambio marcado, podria indicar una pregunta inesperada o incomoda.", "severity": "high", "positive": False},
    ("confidence", "sad"):     {"text": "De confianza a tristeza — cambio de estado notable, revisar contexto.", "severity": "medium", "positive": False},
    ("nervousness", "confidence"): {"text": "De nerviosismo a confianza — posible recuperacion tras superar un momento de tension.", "severity": "low", "positive": True},
    ("nervousness", "fear"):   {"text": "De nerviosismo a miedo — escalada de tension, revisar contexto.", "severity": "high", "positive": False},
    ("fear", "nervousness"):   {"text": "De miedo a nerviosismo — estado de tension sostenido.", "severity": "medium", "positive": False},
    ("fear", "confidence"):    {"text": "De miedo a confianza — recuperacion notable tras un momento tenso.", "severity": "low", "positive": True},
    ("sad", "angry"):          {"text": "De tristeza a enojo — podria indicar frustracion creciente.", "severity": "medium", "positive": False},
    ("angry", "neutral"):      {"text": "De enojo a neutral — posible autorregulacion emocional.", "severity": "low", "positive": True},
    ("angry", "sad"):          {"text": "De enojo a tristeza — cambio de estado, revisar contexto.", "severity": "medium", "positive": False},
    ("surprise", "fear"):      {"text": "De sorpresa a miedo — reaccion intensa, revisar que la provoco.", "severity": "high", "positive": False},
    ("surprise", "angry"):     {"text": "De sorpresa a enojo — posible reaccion defensiva ante informacion inesperada.", "severity": "medium", "positive": False},
    ("surprise", "happy"):     {"text": "De sorpresa a alegria — reaccion positiva ante algo inesperado.", "severity": "low", "positive": True},
    ("disgust", "angry"):      {"text": "De disgusto a enojo — intensificacion emocional negativa.", "severity": "medium", "positive": False},
    ("sad", "neutral"):        {"text": "De tristeza a neutral — posible recuperacion emocional.", "severity": "low", "positive": True},
}


@dataclass
class InterviewAnalysisResult:
    dimension_scores: dict
    behavioral_patterns: list[dict]
    red_flags: list[dict]
    question_correlations: list[dict]
    recommendations: list[dict]
    noise_stats: Optional[dict]
    event_timeline: list[dict] = field(default_factory=list)
    task_analysis: list[dict] = field(default_factory=list)


# ── Event tag vocabulary ─────────────────────────────────────────────
# Markers the moderator drops during a session. The same timeline serves
# both use cases: UX testing marks tasks and usability errors, HR process
# review marks questions — the reaction analysis underneath is identical.
TAG_TASK_START = "task_start"
TAG_TASK_END = "task_end"
TAG_ERROR = "error"
TAG_CONFUSION = "confusion"
TAG_KEY_QUESTION = "key_question"

EVENT_TAG_LABELS = {
    TAG_TASK_START: "Inicio de tarea",
    TAG_TASK_END: "Fin de tarea",
    TAG_ERROR: "Error del usuario",
    TAG_CONFUSION: "Confusion observada",
    TAG_KEY_QUESTION: "Pregunta clave",
}

# Windows used to compare emotional state around a marked event.
PRE_EVENT_WINDOW_S = 3.0
POST_EVENT_WINDOW_S = 7.0

# A marked event counts as a friction point when the reaction after it is
# clearly worse than before — a nervousness jump or a congruence drop.
FRICTION_NERVOUSNESS_DELTA = 0.15
FRICTION_CONGRUENCE_DELTA = -10.0


class InterviewBehaviorAnalyzer:
    """
    Post-session analysis engine for interview behavior assessment.
    
    Analyzes emotion timelines, micro-expressions, and interviewer notes
    to generate behavioral insights, dimension scores, and recommendations.
    """

    def analyze_session(
        self,
        emotion_records: list,  # List of EmotionRecord-like dicts
        micro_expressions: list,  # List of MicroExpression-like dicts  
        interviewer_notes: list,  # List of InterviewerNote-like dicts
        summary: dict,  # SessionSummary data
        noise_stats: dict | None = None,  # From FacialNoiseFilter
    ) -> InterviewAnalysisResult:
        
        patterns = []
        red_flags = []
        correlations = []
        recommendations = []

        # Convert records to easier format for arrays
        timestamps = np.array([r.get("timestamp", 0) for r in emotion_records])
        emotions = [r.get("emotion", "neutral") for r in emotion_records]
        
        def get_prob(r, key):
            probs = r.get("emotion_probabilities") or {}
            return probs.get(key, 0.0)

        def get_au(r, key):
            aus = r.get("action_units") or {}
            return aus.get(key, 0.0)
            
        nervousness = np.array([get_prob(r, "nervousness") for r in emotion_records])
        confidence = np.array([get_prob(r, "confidence") for r in emotion_records])
        neutral_probs = np.array([get_prob(r, "neutral") for r in emotion_records])
        congruence_scores = np.array([r.get("congruence_score", 0) for r in emotion_records])
        gaze_stability = np.array([get_au(r, "gaze_stability") for r in emotion_records])

        key_questions = [n for n in interviewer_notes if n.get("tag") == "key_question"]

        # 1. rapid_confusion & 7. question_avoidance & Correlation
        tech_mastery_penalties = 0
        tech_mastery_rewards = 0

        for kq in key_questions:
            q_time = kq.get("timestamp", 0)
            
            # Windows
            pre_mask = (timestamps >= q_time - 3) & (timestamps <= q_time)
            post_mask = (timestamps > q_time) & (timestamps <= q_time + 7)
            
            pre_emos = [emotions[i] for i, m in enumerate(pre_mask) if m]
            post_emos = [emotions[i] for i, m in enumerate(post_mask) if m]
            
            pre_dom = max(set(pre_emos), key=pre_emos.count) if pre_emos else "neutral"
            post_dom = max(set(post_emos), key=post_emos.count) if post_emos else "neutral"
            
            nerv_change = np.mean(nervousness[post_mask]) - np.mean(nervousness[pre_mask]) if np.any(post_mask) and np.any(pre_mask) else 0
            cong_change = np.mean(congruence_scores[post_mask]) - np.mean(congruence_scores[pre_mask]) if np.any(post_mask) and np.any(pre_mask) else 0

            # 1. rapid_confusion
            rapid_mask = (timestamps > q_time) & (timestamps <= q_time + 2)
            if np.any(rapid_mask) and pre_dom in ["happy", "confidence"]:
                rapid_emos = [emotions[i] for i, m in enumerate(rapid_mask) if m]
                if "surprise" in rapid_emos or "confused" in rapid_emos or "confusion" in rapid_emos: # handle variations
                    rapid_nerv_spike = np.max(nervousness[rapid_mask]) - np.mean(nervousness[pre_mask])
                    rapid_conf_drop = np.mean(confidence[pre_mask]) - np.min(confidence[rapid_mask])
                    if rapid_nerv_spike > 0.2 and rapid_conf_drop > 0.2:
                        patterns.append({
                            "type": "rapid_confusion",
                            "timestamp": q_time,
                            "severity": "high",
                            "description": "Quick shift to confusion/surprise accompanied by confidence drop.",
                            "context_question": kq.get("content")
                        })
                        tech_mastery_penalties += 1

            # 7. question_avoidance
            avoid_mask = (timestamps > q_time) & (timestamps <= q_time + 5)
            if np.any(avoid_mask) and np.any(pre_mask):
                gaze_drop = np.mean(gaze_stability[pre_mask]) - np.min(gaze_stability[avoid_mask])
                nerv_spike = np.max(nervousness[avoid_mask]) - np.mean(nervousness[pre_mask])
                if gaze_drop > 0.2 and nerv_spike > 0.3:
                    patterns.append({
                        "type": "question_avoidance",
                        "timestamp": q_time,
                        "severity": "medium",
                        "description": "Gaze stability dropped and nervousness spiked shortly after question.",
                        "context_question": kq.get("content")
                    })

            if nerv_change < 0.1 and cong_change > -5:
                tech_mastery_rewards += 1
            else:
                tech_mastery_penalties += 0.5

            correlations.append({
                "question_timestamp": q_time,
                "question_text": kq.get("content"),
                "pre_emotion": pre_dom,
                "post_emotion": post_dom,
                "emotion_shift": f"{pre_dom} -> {post_dom}",
                "nervousness_change": float(nerv_change),
                "congruence_change": float(cong_change),
                "insight": f"Nervousness changed by {nerv_change:.2f}, dominant emotion shifted to {post_dom}."
            })

        # 2. nervous_spiral
        window_size = 30
        has_spiral = False
        for i in range(len(timestamps)):
            t_start = timestamps[i]
            w_mask = (timestamps >= t_start) & (timestamps <= t_start + window_size)
            if np.sum(w_mask) > 10:
                t_w = timestamps[w_mask]
                n_w = nervousness[w_mask]
                slope, _ = np.polyfit(t_w, n_w, 1)
                if slope > 0.02: # arbitrary threshold for positive slope
                    has_spiral = True
                    patterns.append({
                        "type": "nervous_spiral",
                        "timestamp": t_start,
                        "severity": "high",
                        "description": "Progressive increase in nervousness over a 30s window.",
                        "context_question": None
                    })
                    break # just record one pattern for simplicity

        # 3. social_masking
        total_micro = summary.get("total_micro_expressions", 0)
        contradictory_micro = summary.get("contradictory_micro_expressions", 0)
        avg_congruence = summary.get("average_congruence", 100)
        has_social_masking = False
        
        if total_micro > 0:
            contra_ratio = contradictory_micro / total_micro
            if contra_ratio > 0.4 and avg_congruence < 60:
                has_social_masking = True
                patterns.append({
                    "type": "social_masking",
                    "timestamp": 0,
                    "severity": "high",
                    "description": "High rate of contradictory micro-expressions with low overall congruence.",
                    "context_question": None
                })
                red_flags.append({
                    "type": "social_masking",
                    "evidence": f"{contra_ratio*100:.0f}% contradictory micro-expressions, congruence {avg_congruence}",
                    "severity": "high"
                })

        # 4. stress_recovery
        has_stress_recovery = False
        nervous_peaks_indices = np.where(nervousness > 0.5)[0]
        # simplified check
        for idx in nervous_peaks_indices:
            t_peak = timestamps[idx]
            recovery_mask = (timestamps > t_peak) & (timestamps <= t_peak + 15)
            if np.any(recovery_mask):
                rec_nerv = nervousness[recovery_mask]
                rec_conf = confidence[recovery_mask]
                if np.min(rec_nerv) < 0.3 and np.max(rec_conf) > 0.5:
                    has_stress_recovery = True
                    patterns.append({
                        "type": "stress_recovery",
                        "timestamp": t_peak,
                        "severity": "low", # positive pattern
                        "description": "Successfully recovered to confident/neutral state within 15s of a stress peak.",
                        "context_question": None
                    })
                    break

        # 5. emotional_flatline
        has_emotional_flatline = False
        neutral_ratio = summary.get("emotion_distribution", {}).get("neutral", 0)
        if neutral_ratio > 0.8 and len(neutral_probs) > 0:
            emo_var = np.var(neutral_probs)
            if emo_var < 0.05:
                has_emotional_flatline = True
                patterns.append({
                    "type": "emotional_flatline",
                    "timestamp": 0,
                    "severity": "medium",
                    "description": "Sustained neutral emotion with very low variance.",
                    "context_question": None
                })

        # 6. confidence_decline
        has_confidence_decline = False
        if len(timestamps) > 10:
            slope, _ = np.polyfit(timestamps, confidence, 1)
            if slope < -0.001:
                has_confidence_decline = True
                patterns.append({
                    "type": "confidence_decline",
                    "timestamp": 0,
                    "severity": "medium",
                    "description": "Overall decline in confidence throughout the session.",
                    "context_question": None
                })

        # 8. authentic_engagement
        has_authentic_engagement = False
        contra_ratio = (contradictory_micro / total_micro) if total_micro > 0 else 0
        happy_ratio = summary.get("emotion_distribution", {}).get("happy", 0)
        if happy_ratio > 0.1 and avg_congruence > 75 and contra_ratio < 0.2:
            conf_var = np.var(confidence) if len(confidence) > 0 else 0
            if conf_var < 0.05:
                has_authentic_engagement = True
                patterns.append({
                    "type": "authentic_engagement",
                    "timestamp": 0,
                    "severity": "low",
                    "description": "Consistent authentic engagement with stable confidence and high congruence.",
                    "context_question": None
                })

        # 9. emotion_transitions — mood-change log for human review.
        # Collapses the per-frame emotion stream into sustained segments
        # (>= min_segment_duration each) so brief flicker isn't reported,
        # then flags each transition between segments with a plain-language
        # hypothesis and the exact video timestamp to jump to.
        transition_events = self._detect_emotion_transitions(timestamps, emotions)
        patterns.extend(transition_events)

        # 10. Event timeline + task friction — the moderator's markers
        # (tasks, errors, confusion, key questions) correlated against what
        # the face was doing around each one.
        event_timeline = self._build_event_timeline(
            timestamps, emotions, nervousness, congruence_scores, interviewer_notes
        )
        task_analysis = self._build_task_analysis(
            timestamps, emotions, nervousness, congruence_scores, interviewer_notes
        )

        # Dimension Scoring (0-100)
        
        # 1. technical_mastery
        tm_score = 70 + (tech_mastery_rewards * 5) - (tech_mastery_penalties * 10)
        tm_score = float(np.clip(tm_score, 0, 100))
        
        # 2. emotional_stability
        transition_rate = 0
        if len(emotions) > 1:
            transitions = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i - 1])
            transition_rate = transitions / len(emotions)
        es_score = 100 - (summary.get("nervousness_peaks", 0) * 5) - (transition_rate * 200)
        es_score = float(np.clip(es_score, 0, 100))
        
        # 3. authenticity
        auth_score = avg_congruence - (contra_ratio * 50)
        auth_score = float(np.clip(auth_score, 0, 100))
        
        # 4. self_confidence
        avg_conf = summary.get("average_confidence", 0)
        avg_nerv = summary.get("average_nervousness", 0)
        sc_score = 50 + (avg_conf * 50) - (avg_nerv * 50)
        if has_confidence_decline:
            sc_score -= 15
        sc_score = float(np.clip(sc_score, 0, 100))
        
        # 5. communication
        comm_score = 80
        if has_emotional_flatline:
            comm_score -= 20
        if has_authentic_engagement:
            comm_score += 10
        comm_score = float(np.clip(comm_score, 0, 100))
        
        # overall
        overall = (tm_score * 0.20) + (es_score * 0.20) + (auth_score * 0.25) + (sc_score * 0.20) + (comm_score * 0.15)
        overall = float(np.clip(overall, 0, 100))

        dimension_scores = {
            "technical_mastery": tm_score,
            "emotional_stability": es_score,
            "authenticity": auth_score,
            "self_confidence": sc_score,
            "communication": comm_score,
            "overall_score": overall
        }

        # Recommendations
        if tm_score < 50:
            recommendations.append({
                "category": "technical",
                "priority": "high",
                "text": "Se recomienda una evaluación técnica adicional — se detectaron señales de confusión ante preguntas técnicas."
            })
        if has_social_masking:
            recommendations.append({
                "category": "behavioral",
                "priority": "high",
                "text": "Considerar profundizar en áreas donde se detectó posible enmascaramiento social."
            })
        if has_authentic_engagement:
            recommendations.append({
                "category": "communication",
                "priority": "low",
                "text": "El candidato demostró engagement genuino y comunicación auténtica."
            })
        if has_stress_recovery:
            recommendations.append({
                "category": "resilience",
                "priority": "medium",
                "text": "El candidato mostró buena capacidad de recuperación ante situaciones de presión."
            })
        if has_spiral and has_confidence_decline:
            recommendations.append({
                "category": "confidence",
                "priority": "medium",
                "text": "Se observó pérdida progresiva de seguridad durante la entrevista."
            })

        # Surface the worst task as a recommendation — it's the single most
        # actionable output for a usability review.
        if task_analysis:
            worst = max(task_analysis, key=lambda t: t["friction_score"])
            if worst["friction_score"] >= 35:
                recommendations.append({
                    "category": "usability",
                    "priority": "high" if worst["friction_score"] >= 60 else "medium",
                    "text": (
                        f"La tarea con mayor friccion fue \"{worst['task_name']}\" "
                        f"({worst['friction_score']:.0f}/100), entre "
                        f"{worst['start_video_time']} y {worst['end_video_time']}. "
                        f"Revisar ese tramo del video."
                    ),
                })

        return InterviewAnalysisResult(
            dimension_scores=dimension_scores,
            behavioral_patterns=patterns,
            red_flags=red_flags,
            question_correlations=correlations,
            recommendations=recommendations,
            noise_stats=noise_stats,
            event_timeline=event_timeline,
            task_analysis=task_analysis,
        )

    @staticmethod
    def _window_stats(
        timestamps: np.ndarray,
        emotions: list,
        nervousness: np.ndarray,
        congruence: np.ndarray,
        start: float,
        end: float,
    ) -> dict | None:
        """Aggregate emotional state over a time window. None if no frames fall in it."""
        mask = (timestamps >= start) & (timestamps <= end)
        if not np.any(mask):
            return None

        window_emotions = [emotions[i] for i, m in enumerate(mask) if m]
        return {
            "dominant_emotion": max(set(window_emotions), key=window_emotions.count),
            "nervousness": float(np.mean(nervousness[mask])),
            "peak_nervousness": float(np.max(nervousness[mask])),
            "peak_nervousness_time": float(timestamps[mask][int(np.argmax(nervousness[mask]))]),
            "congruence": float(np.mean(congruence[mask])),
        }

    def _build_event_timeline(
        self,
        timestamps: np.ndarray,
        emotions: list,
        nervousness: np.ndarray,
        congruence: np.ndarray,
        notes: list,
    ) -> list[dict]:
        """
        For every tagged marker, compare the emotional state just before it
        with the state just after, so a reviewer can jump straight to the
        moments where something actually changed.
        """
        events = []

        for note in notes:
            tag = note.get("tag")
            if tag not in EVENT_TAG_LABELS:
                continue

            t = float(note.get("timestamp") or 0.0)
            before = self._window_stats(
                timestamps, emotions, nervousness, congruence,
                t - PRE_EVENT_WINDOW_S, t,
            )
            after = self._window_stats(
                timestamps, emotions, nervousness, congruence,
                t, t + POST_EVENT_WINDOW_S,
            )
            if before is None or after is None:
                # Marker landed outside the analyzed window (e.g. during
                # calibration, before any emotion record was persisted).
                continue

            nerv_change = after["nervousness"] - before["nervousness"]
            cong_change = after["congruence"] - before["congruence"]
            is_friction = (
                nerv_change >= FRICTION_NERVOUSNESS_DELTA
                or cong_change <= FRICTION_CONGRUENCE_DELTA
                or tag in (TAG_ERROR, TAG_CONFUSION)
            )

            if nerv_change >= FRICTION_NERVOUSNESS_DELTA:
                interpretation = (
                    f"El nerviosismo subio {nerv_change:+.2f} despues de este momento — "
                    f"posible punto de friccion, revisar el video."
                )
            elif cong_change <= FRICTION_CONGRUENCE_DELTA:
                interpretation = (
                    f"La congruencia cayo {cong_change:+.1f} puntos despues de este momento — "
                    f"revisar que lo provoco."
                )
            elif tag in (TAG_ERROR, TAG_CONFUSION):
                interpretation = (
                    "Marcado manualmente por el moderador; la reaccion emocional "
                    "medida no fue pronunciada."
                )
            else:
                interpretation = "Sin cambio emocional relevante tras este momento."

            events.append({
                "timestamp": t,
                "video_time": _format_video_time(t),
                "tag": tag,
                "label": EVENT_TAG_LABELS[tag],
                "content": note.get("content") or "",
                "emotion_before": before["dominant_emotion"],
                "emotion_after": after["dominant_emotion"],
                "nervousness_change": round(nerv_change, 3),
                "congruence_change": round(cong_change, 1),
                "is_friction_point": bool(is_friction),
                "interpretation": interpretation,
            })

        events.sort(key=lambda e: e["timestamp"])
        return events

    def _build_task_analysis(
        self,
        timestamps: np.ndarray,
        emotions: list,
        nervousness: np.ndarray,
        congruence: np.ndarray,
        notes: list,
    ) -> list[dict]:
        """
        Pair task_start markers with the next task_end and score each task
        by how much friction the participant showed while working on it.

        A task_start with no matching task_end is still reported (marked
        incomplete) and measured up to the end of the session — an abandoned
        task is usually the most interesting one in a usability test.
        """
        tagged = sorted(
            [n for n in notes if n.get("tag") in EVENT_TAG_LABELS],
            key=lambda n: float(n.get("timestamp") or 0.0),
        )
        starts = [n for n in tagged if n.get("tag") == TAG_TASK_START]
        if not starts:
            return []

        session_end = float(timestamps[-1]) if len(timestamps) else 0.0
        tasks = []

        for start_note in starts:
            start_t = float(start_note.get("timestamp") or 0.0)

            end_note = next(
                (n for n in tagged
                 if n.get("tag") == TAG_TASK_END
                 and float(n.get("timestamp") or 0.0) > start_t),
                None,
            )
            completed = end_note is not None
            end_t = float(end_note.get("timestamp")) if completed else session_end
            if end_t <= start_t:
                continue

            stats = self._window_stats(
                timestamps, emotions, nervousness, congruence, start_t, end_t
            )
            if stats is None:
                continue

            in_task = [
                n for n in tagged
                if start_t <= float(n.get("timestamp") or 0.0) <= end_t
            ]
            error_count = sum(1 for n in in_task if n.get("tag") == TAG_ERROR)
            confusion_count = sum(1 for n in in_task if n.get("tag") == TAG_CONFUSION)

            # Friction score (0-100). Weights are split between what the face
            # showed (60) and what the moderator marked (40) so neither signal
            # alone can dominate the ranking.
            friction = (
                stats["nervousness"] * 35
                + stats["peak_nervousness"] * 25
                + min(error_count, 3) / 3 * 25
                + min(confusion_count, 2) / 2 * 15
            )
            friction = float(np.clip(friction, 0, 100))

            if friction >= 60:
                interpretation = "Friccion alta — candidata principal a rediseño/reformulacion."
            elif friction >= 35:
                interpretation = "Friccion moderada — vale la pena revisar el video."
            else:
                interpretation = "Sin señales de friccion relevantes."
            if not completed:
                interpretation = "Tarea sin marcar como completada. " + interpretation

            tasks.append({
                "task_name": (start_note.get("content") or "Tarea sin nombre").strip(),
                "start": start_t,
                "end": end_t,
                "start_video_time": _format_video_time(start_t),
                "end_video_time": _format_video_time(end_t),
                "duration_seconds": round(end_t - start_t, 1),
                "completed": completed,
                "dominant_emotion": stats["dominant_emotion"],
                "avg_nervousness": round(stats["nervousness"], 3),
                "peak_nervousness": round(stats["peak_nervousness"], 3),
                "peak_nervousness_time": _format_video_time(stats["peak_nervousness_time"]),
                "avg_congruence": round(stats["congruence"], 1),
                "error_count": error_count,
                "confusion_count": confusion_count,
                "friction_score": round(friction, 1),
                "interpretation": interpretation,
            })

        return tasks

    @staticmethod
    def _detect_emotion_transitions(
        timestamps: np.ndarray,
        emotions: list,
        min_segment_duration: float = 1.0,
        max_events: int = 40,
    ) -> list[dict]:
        """
        Turn the per-frame emotion stream into a log of sustained mood
        changes, each with a plain-language hypothesis and the exact video
        timestamp — meant to point a human reviewer at specific moments,
        not to stand on its own as a conclusion.

        Frame-level noise is filtered by first collapsing consecutive
        identical labels into segments and requiring both the outgoing and
        incoming segment to last at least `min_segment_duration` seconds —
        a single flickered frame won't produce an event.
        """
        if len(emotions) < 2:
            return []

        # Run-length encode into (emotion, start_time, end_time) segments.
        segments = []
        seg_emotion = emotions[0]
        seg_start = timestamps[0]
        for i in range(1, len(emotions)):
            if emotions[i] != seg_emotion:
                segments.append((seg_emotion, seg_start, timestamps[i - 1]))
                seg_emotion = emotions[i]
                seg_start = timestamps[i]
        segments.append((seg_emotion, seg_start, timestamps[-1]))

        events = []
        for i in range(1, len(segments)):
            prev_emotion, prev_start, prev_end = segments[i - 1]
            cur_emotion, cur_start, cur_end = segments[i]

            if prev_emotion == cur_emotion:
                continue
            if (prev_end - prev_start) < min_segment_duration:
                continue
            if (cur_end - cur_start) < min_segment_duration:
                continue

            info = TRANSITION_INTERPRETATIONS.get((prev_emotion, cur_emotion))
            if info is None:
                text = (
                    f"Cambio de estado emocional de {prev_emotion} a {cur_emotion} — "
                    f"se recomienda revision humana para interpretar el contexto."
                )
                severity, is_positive = "medium", False
            else:
                text, severity, is_positive = info["text"], info["severity"], info["positive"]

            video_time = _format_video_time(cur_start)
            events.append({
                "type": "emotion_transition",
                "timestamp": float(cur_start),
                "severity": severity,
                "description": (
                    f"[{video_time}] {prev_emotion} -> {cur_emotion}: {text} "
                    f"(minuto {video_time} del video, para revision humana)"
                ),
                "context_question": None,
                "is_positive": is_positive,
            })

            if len(events) >= max_events:
                break

        return events
