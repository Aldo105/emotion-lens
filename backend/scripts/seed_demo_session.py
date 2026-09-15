"""
EmotionLens — Demo Session Seeder

Creates a complete, realistic session in the database without needing a
webcam: emotion timeline, micro-expressions, moderator task/event markers,
and post-session human validation.

Use it to demo the dashboard, the reports and the task-friction analysis
when a live capture isn't practical (bad lighting, no camera, presenting
from a different machine), and as an end-to-end exercise of the analysis
pipeline after changing it.

The scenario is a usability test with three tasks, where the second one
goes badly — the participant gets stuck, makes errors and grows visibly
nervous, then recovers on the third task.

Usage:
    python -m backend.scripts.seed_demo_session
"""

import asyncio
import random
from datetime import datetime, timedelta, timezone

from backend.app.models.database import (
    InterviewerNote, SessionFeedback, async_session_factory, init_db,
)
from backend.app.services import session_manager

FPS = 2.0  # one record every 0.5s — enough resolution for the analysis


# (start, end, emotion, nervousness, congruence)
PHASES = [
    (0,   40,  "neutral",     0.15, 82),   # warm-up
    (40,  70,  "confidence",  0.12, 85),   # task 1 — goes well
    (70,  95,  "neutral",     0.25, 78),   # task 2 starts, mild hesitation
    (95,  150, "nervousness", 0.72, 55),   # task 2 — stuck, errors pile up
    (150, 175, "angry",       0.65, 48),   # task 2 — frustration peak
    (175, 210, "neutral",     0.30, 70),   # task 2 abandoned, reset
    (210, 260, "happy",       0.18, 84),   # task 3 — recovers
]

TASK_MARKERS = [
    (40.0,  "task_start",   "Crear una cuenta nueva"),
    (68.0,  "task_end",     "Completada"),
    (72.0,  "task_start",   "Cambiar el metodo de pago"),
    (103.0, "confusion",    "No encuentra el boton de configuracion"),
    (118.0, "error",        "Entra al menu equivocado"),
    (141.0, "error",        "Vuelve a entrar al menu equivocado"),
    (168.0, "confusion",    "Dice en voz alta que no sabe donde esta"),
    (205.0, "task_end",     "Abandonada, se le indica continuar"),
    (212.0, "task_start",   "Buscar un producto y agregarlo al carrito"),
    (255.0, "task_end",     "Completada sin problemas"),
]

MICRO_EXPRESSIONS = [
    (99.0,  180, "fear",     "neutral",     88, True),
    (121.0, 220, "angry",    "nervousness", 94, True),
    (145.0, 160, "disgust",  "nervousness", 79, True),
    (171.0, 200, "sad",      "angry",       72, False),
    (232.0, 140, "happy",    "happy",       65, False),
]

# What a reviewer said about each flagged moment afterwards.
MOMENT_VERDICTS = ["correct", "correct", "correct", "incorrect", "unsure"]


def _phase_at(t: float):
    for start, end, emotion, nervousness, congruence in PHASES:
        if start <= t < end:
            return emotion, nervousness, congruence
    return PHASES[-1][2], PHASES[-1][3], PHASES[-1][4]


async def seed() -> int:
    await init_db()
    random.seed(7)  # stable demo data across runs

    async with async_session_factory() as db:
        session = await session_manager.create_session(
            db=db,
            name="Sesión demo — Test de usabilidad",
            candidate_name="Participante demo",
            input_type="webcam",
        )
        session_id = session.id
        duration = PHASES[-1][1]

        # Backdate the start so the recorded duration matches the simulated
        # timeline — otherwise the demo shows a 0-second session containing
        # four minutes of data.
        session.created_at = datetime.now(timezone.utc) - timedelta(seconds=duration)
        await db.commit()

        steps = int(duration * FPS)

        for i in range(steps):
            t = i / FPS
            emotion, base_nerv, base_cong = _phase_at(t)
            nervousness = max(0.0, min(1.0, base_nerv + random.uniform(-0.06, 0.06)))
            confidence_prob = max(0.0, min(1.0, (1.0 - nervousness) * 0.8))
            congruence = max(0.0, min(100.0, base_cong + random.uniform(-5, 5)))

            await session_manager.save_emotion_record(
                db=db,
                session_id=session_id,
                data={
                    "timestamp": round(t, 2),
                    "emotion": emotion,
                    "confidence": round(random.uniform(0.6, 0.9), 3),
                    "emotion_probabilities": {
                        emotion: 0.6,
                        "nervousness": round(nervousness, 3),
                        "confidence": round(confidence_prob, 3),
                        "neutral": 0.2,
                    },
                    "action_units": {
                        "AU4": round(nervousness * 0.6, 3),
                        "AU7": round(nervousness * 0.4, 3),
                        "gaze_stability": round(1.0 - nervousness, 3),
                    },
                    "congruence_score": round(congruence, 1),
                    "model_confidence": round(random.uniform(0.5, 0.8), 3),
                },
            )

        for timestamp, tag, content in TASK_MARKERS:
            db.add(
                InterviewerNote(
                    session_id=session_id,
                    timestamp=timestamp,
                    content=content,
                    tag=tag,
                )
            )

        moment_validations = []
        for (ts, duration_ms, detected, dominant, relevance, contradictory), verdict in zip(
            MICRO_EXPRESSIONS, MOMENT_VERDICTS
        ):
            await session_manager.save_micro_expression(
                db=db,
                session_id=session_id,
                event={
                    "timestamp": ts,
                    "duration_ms": duration_ms,
                    "detected_emotion": detected,
                    "dominant_emotion_at_time": dominant,
                    "action_units_involved": ["AU4", "AU7"],
                    "relevance_score": relevance,
                    "is_contradictory": contradictory,
                    "description": f"{detected} detectada durante {dominant}",
                },
            )
            moment_validations.append({
                "timestamp": ts,
                "emotion": detected,
                "relevance_score": relevance,
                "is_contradictory": contradictory,
                "verdict": verdict,
            })

        db.add(
            SessionFeedback(
                session_id=session_id,
                overall_accuracy_rating=0.8,
                self_reported_emotion="nervousness",
                attempted_suppression=True,
                moment_validations=moment_validations,
                free_text_comments=(
                    "La segunda tarea fue muy confusa, no encontraba donde cambiar el pago."
                ),
            )
        )

        await db.commit()

        # Triggers summary generation + the full behavioral analysis.
        await session_manager.end_session(db=db, session_id=session_id)
        await db.commit()

    return session_id


if __name__ == "__main__":
    new_id = asyncio.run(seed())
    print(f"[OK] Demo session created with id={new_id}")
