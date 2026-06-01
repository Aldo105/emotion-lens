"""
EmotionLens — Video Processor Service

Offline video analysis pipeline that processes uploaded video files
frame-by-frame through the same ML pipeline used for live webcam analysis:
  Face Detection → Action Units → Emotion Classification →
  Micro-Expression Engine → Congruence Scoring → Heart Rate Estimation

Results are persisted to the database via session_manager and a real-time
progress dict is maintained for polling by the REST API.
"""

import time
import traceback
from typing import Optional

import cv2
import numpy as np

from backend.app.config import settings
from backend.app.models.database import async_session_factory, Session
from backend.app.services.face_detector import FaceDetector
from backend.app.services.action_units import ActionUnitAnalyzer
from backend.app.services.emotion_classifier import EmotionClassifier
from backend.app.services.micro_expressions import MicroExpressionEngine
from backend.app.services.congruence import CongruenceScorer
from backend.app.services.heart_rate import HeartRateEstimator
from backend.app.services import session_manager

from sqlalchemy import select


# ═══════════════════════════════════════════════════════════════════════
# VIDEO PROCESSOR
# ═══════════════════════════════════════════════════════════════════════

class VideoProcessor:
    """
    Processes an uploaded video file through the full EmotionLens
    analysis pipeline, persisting results frame-by-frame.

    Usage::

        processor = VideoProcessor()
        await processor.process_video("path/to/video.mp4", session_id=42)

    Track progress via the class-level ``progress`` dict::

        progress = VideoProcessor.progress.get(session_id)
        # => {"progress": 0.73, "status": "processing"}
    """

    # Class-level progress tracker — shared across all instances so the
    # REST endpoint can poll without holding a reference to the processor.
    progress: dict[int, dict] = {}

    async def process_video(
        self,
        video_path: str,
        session_id: int,
        device: str = "cpu",
    ) -> None:
        """
        Open a video file, iterate every other frame through the ML
        pipeline, persist each result, then finalize the session.

        Args:
            video_path: Absolute path to the video file.
            session_id: Database ID of the session to attach results to.
            device: Compute device for the emotion classifier ("cpu" or "cuda").
        """
        VideoProcessor.progress[session_id] = {
            "progress": 0.0,
            "status": "initializing",
        }

        cap = None
        face_detector = None

        try:
            # ── Open video ───────────────────────────────────────────
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video file: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

            if total_frames <= 0:
                raise RuntimeError("Video contains zero frames")

            print(f"[VIDEO] Processing {video_path}  "
                  f"({total_frames} frames @ {fps:.1f} fps)")

            # ── Initialize ML services ───────────────────────────────
            face_detector = FaceDetector(
                max_faces=settings.max_faces,
                detection_confidence=settings.face_detection_confidence,
                tracking_confidence=settings.face_tracking_confidence,
            )
            au_analyzer = ActionUnitAnalyzer()
            emotion_classifier = EmotionClassifier(device=device)
            micro_engine = MicroExpressionEngine()
            congruence_scorer = CongruenceScorer()
            heart_rate_estimator = HeartRateEstimator(fps=fps)

            baseline_calibrated = False
            frame_index = 0

            VideoProcessor.progress[session_id] = {
                "progress": 0.0,
                "status": "processing",
            }

            # ── Frame loop ───────────────────────────────────────────
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_index += 1

                # Skip every other frame for speed
                if frame_index % 2 != 0:
                    continue

                timestamp = frame_index / fps

                # ── Step 1: Face Detection ───────────────────────────
                detection = face_detector.detect(frame)
                if detection is None:
                    # Update progress even when no face is found
                    VideoProcessor.progress[session_id]["progress"] = round(
                        frame_index / total_frames, 4
                    )
                    continue

                # ── Step 2: Action Units ─────────────────────────────
                action_units = au_analyzer.compute(
                    landmarks=detection["landmarks"],
                    frame_shape=detection["frame_shape"],
                    timestamp=timestamp,
                )

                # ── Step 3: Baseline calibration ─────────────────────
                is_calibrating = timestamp < settings.baseline_calibration_seconds

                if not baseline_calibrated and not is_calibrating:
                    au_analyzer.set_baseline()
                    if au_analyzer.baseline_aus:
                        micro_engine.set_baseline(
                            baseline_aus=au_analyzer.baseline_aus,
                            variability=None,
                        )
                        congruence_scorer.set_baseline(au_analyzer.baseline_aus)
                    baseline_calibrated = True

                # ── Step 4: Emotion classification ───────────────────
                emotion_result = emotion_classifier.predict(
                    action_units=action_units,
                    blendshapes=detection.get("blendshapes"),
                )

                # ── Step 5: Micro-expression detection ───────────────
                micro_event = None
                if baseline_calibrated:
                    raw_event = micro_engine.analyze(
                        current_aus=action_units,
                        timestamp=timestamp,
                        dominant_emotion=emotion_result["emotion"],
                    )
                    if raw_event:
                        micro_event = {
                            "timestamp": raw_event.timestamp,
                            "duration_ms": raw_event.duration_ms,
                            "detected_emotion": raw_event.detected_emotion,
                            "dominant_emotion_at_time": raw_event.dominant_emotion_at_time,
                            "action_units_involved": raw_event.action_units_involved,
                            "relevance_score": raw_event.relevance_score,
                            "is_contradictory": raw_event.is_contradictory,
                            "description": raw_event.description,
                            "temporal_valid": raw_event.temporal_valid,
                            "context_valid": raw_event.context_valid,
                        }
                        congruence_scorer.add_micro_expression({
                            "timestamp": raw_event.timestamp,
                            "detected_emotion": raw_event.detected_emotion,
                            "is_contradictory": raw_event.is_contradictory,
                            "relevance_score": raw_event.relevance_score,
                        })

                # ── Step 5.5: Heart rate estimation ──────────────────
                hr_result = heart_rate_estimator.process_frame(
                    frame=frame,
                    landmarks=detection["landmarks"],
                    frame_shape=detection["frame_shape"],
                    timestamp=timestamp,
                )

                if hr_result["signal_ready"]:
                    action_units["hr_stress"] = hr_result["stress_indicator"]

                # ── Step 6: Congruence scoring ───────────────────────
                congruence_result = congruence_scorer.compute(
                    emotion=emotion_result["emotion"],
                    confidence=emotion_result["confidence"],
                    action_units=action_units,
                    timestamp=timestamp,
                )

                # ── Persist to database ──────────────────────────────
                async with async_session_factory() as db:
                    try:
                        emotion_data = {
                            "timestamp": round(timestamp, 3),
                            "emotion": emotion_result["emotion"],
                            "confidence": round(emotion_result["confidence"], 3),
                            "emotion_probabilities": {
                                k: round(v, 3)
                                for k, v in emotion_result["probabilities"].items()
                            },
                            "action_units": {
                                k: round(v, 3) for k, v in action_units.items()
                            },
                            "congruence_score": congruence_result["score"],
                            "model_confidence": round(
                                emotion_result["model_confidence"], 3
                            ),
                        }
                        await session_manager.save_emotion_record(
                            db, session_id, emotion_data
                        )

                        if micro_event:
                            await session_manager.save_micro_expression(
                                db, session_id, micro_event
                            )

                        await db.commit()
                    except Exception:
                        await db.rollback()
                        raise

                # ── Update progress ──────────────────────────────────
                VideoProcessor.progress[session_id]["progress"] = round(
                    frame_index / total_frames, 4
                )

            # ── Finalize session ─────────────────────────────────────
            async with async_session_factory() as db:
                try:
                    await session_manager.end_session(db, session_id)
                    await db.commit()
                except Exception:
                    await db.rollback()
                    raise

            VideoProcessor.progress[session_id] = {
                "progress": 1.0,
                "status": "completed",
            }
            print(f"[VIDEO] Session {session_id} processing complete.")

        except Exception as e:
            print(f"[ERR] Video processing failed for session {session_id}: {e}")
            traceback.print_exc()

            VideoProcessor.progress[session_id] = {
                "progress": VideoProcessor.progress.get(session_id, {}).get("progress", 0.0),
                "status": "error",
                "error": str(e),
            }

            # Mark session as error in the database
            try:
                async with async_session_factory() as db:
                    result = await db.execute(
                        select(Session).where(Session.id == session_id)
                    )
                    session = result.scalar_one_or_none()
                    if session:
                        session.status = "error"
                        await db.commit()
            except Exception as db_err:
                print(f"[ERR] Could not update session status: {db_err}")

        finally:
            if cap is not None:
                cap.release()
            if face_detector is not None:
                face_detector.close()
