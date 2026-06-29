"""
EmotionLens — WebSocket Router

Handles real-time webcam frame streaming and emotion analysis.
Full processing pipeline:
  Browser (webcam frame) -> WebSocket -> Face Detection -> Action Units
  -> Emotion Classification -> Micro-Expression Engine -> Congruence Score -> Response
"""

import asyncio
import base64
import json
import time
import traceback

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.app.config import settings
from backend.app.models.schemas import WSFrameResult, WSStatusMessage, MicroExpressionEvent
from backend.app.services.face_detector import FaceDetector
from backend.app.services.action_units import ActionUnitAnalyzer
from backend.app.services.emotion_classifier import EmotionClassifier
from backend.app.services.micro_expressions import MicroExpressionEngine
from backend.app.services.congruence import CongruenceScorer
from backend.app.services.heart_rate import HeartRateEstimator
from backend.app.services import session_manager
from backend.app.models.database import async_session_factory
from backend.app.utils.image_preprocessing import AdaptivePreprocessor

router = APIRouter()


class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[WS] Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"[WS] Client disconnected. Total: {len(self.active_connections)}")

    async def send_json(self, websocket: WebSocket, data: dict):
        await websocket.send_json(data)


manager = ConnectionManager()


def decode_frame(data: str) -> np.ndarray | None:
    """
    Decode a base64-encoded image frame into an OpenCV numpy array.
    Handles both raw base64 and data URL format (data:image/...;base64,...).
    """
    try:
        # Strip data URL prefix if present
        if "," in data:
            data = data.split(",", 1)[1]

        img_bytes = base64.b64decode(data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"[!!] Frame decode error: {e}")
        return None


def _compute_camera_quality(frame: np.ndarray, detection: dict) -> dict:
    """
    Evaluate camera/capture quality for reliable emotion detection.

    Checks: face size, brightness, position, sharpness, contrast,
    head pose, lighting balance, and skin tone.

    Returns quality metrics, warnings, actionable suggestions, and a
    quality gate decision (pass/degraded/fail).
    """
    h, w = frame.shape[:2]
    bbox = detection["bbox"]

    face_w = bbox["x_max"] - bbox["x_min"]
    face_h = bbox["y_max"] - bbox["y_min"]
    face_area = face_w * face_h
    frame_area = h * w

    warnings = []
    suggestions = []
    score = 1.0

    # ── Face Size Check ──────────────────────────────────────────
    face_size_ratio = face_area / max(frame_area, 1)
    if face_size_ratio < 0.03:
        warnings.append("Face too small — move closer to camera")
        suggestions.append("Acercate a la camara para mejor precision")
        score -= 0.4
    elif face_size_ratio < 0.08:
        warnings.append("Face is small — move slightly closer")
        suggestions.append("Acercate un poco mas a la camara")
        score -= 0.15

    # ── Brightness Check ─────────────────────────────────────────
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_roi = gray[bbox["y_min"]:bbox["y_max"], bbox["x_min"]:bbox["x_max"]]
    if face_roi.size > 0:
        brightness = float(np.mean(face_roi)) / 255.0
    else:
        brightness = 0.5

    if brightness < 0.20:
        warnings.append("Too dark — improve lighting")
        suggestions.append("Enciende una luz frente a ti o acercate a una ventana")
        score -= 0.35
    elif brightness < 0.30:
        warnings.append("Lighting is low — improve if possible")
        suggestions.append("Mejora la iluminacion si es posible")
        score -= 0.10
    elif brightness > 0.85:
        warnings.append("Too bright — reduce lighting or glare")
        suggestions.append("Reduce el brillo o alejate de la fuente de luz")
        score -= 0.20

    # ── Face Position Check ──────────────────────────────────────
    face_center_x = (bbox["x_min"] + bbox["x_max"]) / 2.0 / w
    face_center_y = (bbox["y_min"] + bbox["y_max"]) / 2.0 / h

    off_center = abs(face_center_x - 0.5) + abs(face_center_y - 0.5)
    if off_center > 0.5:
        warnings.append("Face is off-center — center yourself in the frame")
        suggestions.append("Centra tu rostro en la camara")
        score -= 0.15

    # ── Sharpness Check (Laplacian variance) ─────────────────────
    sharpness = 100.0
    if face_roi.size > 0:
        laplacian = cv2.Laplacian(face_roi, cv2.CV_64F)
        sharpness = float(laplacian.var())
        if sharpness < 50:
            warnings.append("Image blurry — clean camera or stay still")
            suggestions.append("Limpia el lente de tu camara o quedate quieto")
            score -= 0.30
        elif sharpness < 100:
            suggestions.append("Intenta mantenerte mas quieto")
            score -= 0.10

    # ── Contrast Check (std of luminance) ────────────────────────
    contrast = 40.0
    if face_roi.size > 0:
        contrast = float(np.std(face_roi))
        if contrast < 25:
            suggestions.append("Tu rostro se ve muy plano — agrega una segunda fuente de luz")
            score -= 0.10
        elif contrast > 80:
            suggestions.append("Hay mucha sombra en un lado — usa luz frontal difusa")
            score -= 0.15

    # ── Head Pose Check (from landmarks) ─────────────────────────
    yaw, pitch = _estimate_head_pose(detection["landmarks"], w, h)
    if abs(yaw) > 20:
        warnings.append("Face turned too far — look at the camera")
        suggestions.append("Gira tu cabeza hacia la camara")
        score -= 0.20
    elif abs(yaw) > 12:
        suggestions.append("Mira un poco mas de frente a la camara")
        score -= 0.08
    if abs(pitch) > 20:
        suggestions.append("Ajusta la altura de la camara para mirar de frente")
        score -= 0.15

    # ── Lighting Balance (left vs right) ─────────────────────────
    lighting_balance = 1.0
    if face_roi.size > 0 and face_w > 4:
        left_half = face_roi[:, :face_w // 2]
        right_half = face_roi[:, face_w // 2:]
        if left_half.size > 0 and right_half.size > 0:
            lb = float(np.mean(left_half))
            rb = float(np.mean(right_half))
            lighting_balance = 1.0 - abs(lb - rb) / max(lb, rb, 1)
            if lighting_balance < 0.70:
                suggestions.append("La luz viene de un solo lado — pon la fuente de luz frente a ti")
                score -= 0.15

    # ── Skin Tone Detection ──────────────────────────────────────
    skin_tone = _detect_skin_tone(frame, bbox)

    # ── Quality Gate Decision ────────────────────────────────────
    final_score = float(np.clip(score, 0.0, 1.0))
    if final_score >= 0.7:
        quality_gate = "pass"
    elif final_score >= 0.4:
        quality_gate = "degraded"
    else:
        quality_gate = "fail"

    return {
        "score": final_score,
        "face_size_ratio": round(face_size_ratio, 4),
        "brightness": round(brightness, 3),
        "contrast": round(contrast, 1),
        "sharpness": round(sharpness, 1),
        "head_pose": {"yaw": round(yaw, 1), "pitch": round(pitch, 1)},
        "skin_tone": skin_tone,
        "lighting_balance": round(lighting_balance, 2),
        "warnings": warnings,
        "suggestions": suggestions,
        "quality_gate": quality_gate,
    }


def _estimate_head_pose(landmarks: list, frame_w: int, frame_h: int) -> tuple:
    """
    Estimate yaw and pitch from face landmarks.
    Uses nose tip, forehead, chin, and eye corners.
    Returns (yaw_degrees, pitch_degrees).
    """
    try:
        # Nose tip = 1, left eye outer = 33, right eye outer = 263
        # Forehead = 10, Chin = 152
        nose = landmarks[1]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        forehead = landmarks[10]
        chin = landmarks[152]

        # Yaw: horizontal displacement of nose relative to eye midpoint
        eye_mid_x = (left_eye["x"] + right_eye["x"]) / 2.0
        eye_width = abs(right_eye["x"] - left_eye["x"])
        if eye_width > 0.001:
            yaw_ratio = (nose["x"] - eye_mid_x) / eye_width
            yaw = yaw_ratio * 60.0  # Scale to approximate degrees
        else:
            yaw = 0.0

        # Pitch: vertical displacement of nose relative to forehead-chin midpoint
        vert_mid_y = (forehead["y"] + chin["y"]) / 2.0
        vert_span = abs(chin["y"] - forehead["y"])
        if vert_span > 0.001:
            pitch_ratio = (nose["y"] - vert_mid_y) / vert_span
            pitch = pitch_ratio * 60.0
        else:
            pitch = 0.0

        return (float(yaw), float(pitch))
    except (IndexError, KeyError, TypeError):
        return (0.0, 0.0)


def _detect_skin_tone(frame: np.ndarray, bbox: dict) -> str:
    """
    Classify skin tone as 'light', 'medium', or 'dark'
    based on the forehead region brightness in HSV.
    """
    try:
        # Use upper 30% of face bbox as forehead region
        y1 = bbox["y_min"]
        y2 = y1 + (bbox["y_max"] - bbox["y_min"]) // 3
        x1 = bbox["x_min"] + (bbox["x_max"] - bbox["x_min"]) // 4
        x2 = bbox["x_max"] - (bbox["x_max"] - bbox["x_min"]) // 4

        forehead = frame[y1:y2, x1:x2]
        if forehead.size == 0:
            return "medium"

        hsv = cv2.cvtColor(forehead, cv2.COLOR_BGR2HSV)
        avg_value = float(np.mean(hsv[:, :, 2]))

        if avg_value > 170:
            return "light"
        elif avg_value > 100:
            return "medium"
        else:
            return "dark"
    except Exception:
        return "medium"


@router.websocket("/ws/emotion")
async def websocket_emotion_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for real-time emotion analysis.

    Protocol:
    - Client sends: base64-encoded JPEG frames (as text messages)
    - Server responds: JSON with emotion data, AUs, congruence, micro-expressions
    """
    await manager.connect(websocket)

    # ── Initialize ML services per connection ────────────────────────
    device = getattr(websocket.app.state, "device", "cpu")

    face_detector = FaceDetector(
        max_faces=settings.max_faces,
        detection_confidence=settings.face_detection_confidence,
        tracking_confidence=settings.face_tracking_confidence,
    )
    au_analyzer = ActionUnitAnalyzer()
    emotion_classifier = EmotionClassifier(device=device)
    micro_engine = MicroExpressionEngine()
    congruence_scorer = CongruenceScorer()
    heart_rate_estimator = HeartRateEstimator(fps=20.0)
    preprocessor = AdaptivePreprocessor()

    frame_count = 0
    processed_frame_count = 0
    session_start = time.time()
    baseline_calibrated = False
    calibration_start = None  # For recalibration timing
    current_skin_tone = "medium"  # Updated per-frame from quality check

    # ── Database session persistence ─────────────────────────────────
    db_session = None
    analysis_session_id = None
    emotion_record_batch = []  # Batch records for efficient DB writes
    BATCH_SIZE = 10  # Flush to DB every N processed frames

    try:
        # Create a database session and analysis session for persistence
        db_session = async_session_factory()
        analysis_session = await session_manager.create_session(
            db=db_session,
            name=f"Live Session — {time.strftime('%Y-%m-%d %H:%M')}",
            candidate_name=None,
            input_type="webcam",
        )
        analysis_session_id = analysis_session.id
        await db_session.commit()

        # Send initial status
        await manager.send_json(websocket, WSStatusMessage(
            type="status",
            message="Connected to EmotionLens analysis server",
            data={"device": device, "mode": emotion_classifier.mode, "session_id": analysis_session_id},
        ).model_dump())

        while True:
            # Receive frame data from client
            raw_data = await websocket.receive_text()
            frame_count += 1

            # Frame throttling -- skip frames for performance
            if frame_count % settings.ws_frame_skip != 0:
                continue

            # Parse JSON payload or raw base64 frame data
            request_evm = False
            request_recalibrate = False
            frame_data = raw_data
            if raw_data.startswith("{"):
                try:
                    payload = json.loads(raw_data)
                    frame_data = payload.get("frame", "")
                    request_evm = payload.get("evm", False)
                    request_recalibrate = payload.get("recalibrate", False)
                except Exception as e:
                    print(f"[WARN] Failed to parse JSON from WebSocket message: {e}")

            # Handle recalibration request
            if request_recalibrate and baseline_calibrated:
                print("[INFO] Recalibration requested by client")
                baseline_calibrated = False
                calibration_start = time.time()
                au_analyzer.baseline_set = False
                au_analyzer.baseline_aus = None
                au_analyzer.baseline_variability = None
                au_analyzer._baseline_buffer = []
                micro_engine.baseline_set = False
                micro_engine.baseline_aus = {}
                micro_engine.baseline_variability = {}
                await manager.send_json(websocket, WSStatusMessage(
                    type="status",
                    message="Recalibration started. Please maintain a neutral expression.",
                ).model_dump())
                continue

            # Decode the frame
            frame = decode_frame(frame_data)
            if frame is None:
                continue

            timestamp = time.time() - session_start

            # ═══════════════════════════════════════════════════════
            # PROCESSING PIPELINE
            # ═══════════════════════════════════════════════════════

            # Step 0: Adaptive Preprocessing (CLAHE + gamma + denoise)
            # Use preprocessed frame for detection; keep original for rPPG
            preprocessed_frame = preprocessor.process(frame, skin_tone=current_skin_tone)

            # Step 1: Face Detection (MediaPipe) — on preprocessed frame
            detection = face_detector.detect(preprocessed_frame)
            if detection is None:
                # No face detected -- send empty result
                await manager.send_json(websocket, WSStatusMessage(
                    type="status",
                    message="No face detected",
                ).model_dump())
                continue

            # Step 1.5: Camera Quality + Quality Gate
            camera_quality = _compute_camera_quality(preprocessed_frame, detection)
            current_skin_tone = camera_quality.get("skin_tone", "medium")

            # Quality gate — discard frames too poor for reliable analysis
            if camera_quality["quality_gate"] == "fail":
                await manager.send_json(websocket, {
                    "type": "quality_warning",
                    "camera_quality": camera_quality,
                    "message": "Frame descartado por baja calidad — sigue las sugerencias",
                })
                continue

            # Quality penalty for degraded frames
            quality_penalty = 1.0 if camera_quality["quality_gate"] == "pass" else 0.75

            # Step 2: Action Unit Analysis
            action_units = au_analyzer.compute(
                landmarks=detection["landmarks"],
                frame_shape=detection["frame_shape"],
                timestamp=timestamp,
            )

            # Step 3: Baseline Calibration Check
            # Support both initial calibration and recalibration
            if calibration_start is not None:
                # Recalibration mode — use time since recalibration started
                calib_elapsed = time.time() - calibration_start
                is_calibrating = calib_elapsed < settings.baseline_calibration_seconds
                calibration_progress = min(1.0, calib_elapsed / settings.baseline_calibration_seconds)
            else:
                is_calibrating = timestamp < settings.baseline_calibration_seconds
                calibration_progress = min(1.0, timestamp / settings.baseline_calibration_seconds)

            # Feed calibration frames to micro-expression engine for habitual tracking
            if not baseline_calibrated and is_calibrating:
                micro_engine.record_calibration_frame(action_units)

            if not baseline_calibrated and not is_calibrating:
                # Calibration period just ended -- set baselines
                au_analyzer.set_baseline()
                if au_analyzer.baseline_aus:
                    micro_engine.set_baseline(
                        baseline_aus=au_analyzer.baseline_aus,
                        variability=au_analyzer.baseline_variability,
                    )
                    congruence_scorer.set_baseline(au_analyzer.baseline_aus)
                baseline_calibrated = True
                calibration_start = None  # Reset recalibration timer

                await manager.send_json(websocket, WSStatusMessage(
                    type="calibration_complete",
                    message="Baseline calibration complete. Analysis active.",
                ).model_dump())

            # Step 4: Emotion Classification (with quality penalty)
            emotion_result = emotion_classifier.predict(
                action_units=action_units,
                blendshapes=detection.get("blendshapes"),
                quality_penalty=quality_penalty,
            )

            # Step 5: Micro-Expression Detection
            micro_event = None
            if baseline_calibrated:
                raw_event = micro_engine.analyze(
                    current_aus=action_units,
                    timestamp=timestamp,
                    dominant_emotion=emotion_result["emotion"],
                    camera_quality_score=camera_quality["score"],
                )
                if raw_event:
                    micro_event = MicroExpressionEvent(
                        timestamp=raw_event.timestamp,
                        duration_ms=raw_event.duration_ms,
                        detected_emotion=raw_event.detected_emotion,
                        dominant_emotion_at_time=raw_event.dominant_emotion_at_time,
                        action_units_involved=raw_event.action_units_involved,
                        relevance_score=raw_event.relevance_score,
                        is_contradictory=raw_event.is_contradictory,
                        description=raw_event.description,
                    )
                    # Record for congruence analysis
                    congruence_scorer.add_micro_expression({
                        "timestamp": raw_event.timestamp,
                        "detected_emotion": raw_event.detected_emotion,
                        "is_contradictory": raw_event.is_contradictory,
                        "relevance_score": raw_event.relevance_score,
                    })

            # Step 5.5: Heart Rate Estimation (Eulerian Video Magnification / rPPG)
            hr_result = heart_rate_estimator.process_frame(
                frame=frame,
                landmarks=detection["landmarks"],
                frame_shape=detection["frame_shape"],
                timestamp=timestamp,
            )

            # Feed HR stress into congruence (adds physiological data)
            if hr_result["signal_ready"]:
                action_units["hr_stress"] = hr_result["stress_indicator"]

            # Generate magnified frame if requested
            evm_frame_b64 = None
            if request_evm:
                try:
                    mag_frame = heart_rate_estimator.get_magnified_frame(
                        frame=frame,
                        landmarks=detection["landmarks"],
                        frame_shape=detection["frame_shape"],
                    )
                    # Encode magnified frame back to base64 JPEG
                    _, buffer = cv2.imencode('.jpg', mag_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    evm_frame_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
                except Exception as e:
                    print(f"[ERR] EVM magnification failed: {e}")

            # Encode preprocessed frame back to base64 JPEG (visible CLAHE/gamma)
            preprocessed_frame_b64 = None
            try:
                _, prep_buffer = cv2.imencode('.jpg', preprocessed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                preprocessed_frame_b64 = "data:image/jpeg;base64," + base64.b64encode(prep_buffer).decode('utf-8')
            except Exception as e:
                print(f"[ERR] Preprocessed frame encoding failed: {e}")

            # Step 6: Congruence Scoring
            congruence_result = congruence_scorer.compute(
                emotion=emotion_result["emotion"],
                confidence=emotion_result["confidence"],
                action_units=action_units,
                timestamp=timestamp,
            )

            # ═══════════════════════════════════════════════════════
            # BUILD RESPONSE
            # ═══════════════════════════════════════════════════════

            result = WSFrameResult(
                type="frame_result",
                timestamp=round(timestamp, 3),
                emotion=emotion_result["emotion"],
                confidence=round(emotion_result["confidence"], 3),
                emotion_probabilities={
                    k: round(v, 3) for k, v in emotion_result["probabilities"].items()
                },
                model_confidence=round(emotion_result["model_confidence"], 3),
                action_units={k: round(v, 3) for k, v in action_units.items()},
                congruence_score=congruence_result["score"],
                congruence_breakdown=congruence_result["breakdown"],
                micro_expression=micro_event,
                heart_rate=hr_result if hr_result["signal_ready"] else None,
                evm_frame=evm_frame_b64,
                preprocessed_frame=preprocessed_frame_b64,
                camera_quality=camera_quality,
                is_calibrating=is_calibrating,
                calibration_progress=round(calibration_progress, 2),
            )

            await manager.send_json(websocket, result.model_dump())

            # ── Persist frame data to database (batched) ─────────────
            if analysis_session_id and baseline_calibrated:
                processed_frame_count += 1
                emotion_record_batch.append({
                    "timestamp": round(timestamp, 3),
                    "emotion": emotion_result["emotion"],
                    "confidence": round(emotion_result["confidence"], 3),
                    "emotion_probabilities": {
                        k: round(v, 3) for k, v in emotion_result["probabilities"].items()
                    },
                    "action_units": {k: round(v, 3) for k, v in action_units.items()},
                    "congruence_score": congruence_result["score"],
                    "model_confidence": round(emotion_result["model_confidence"], 3),
                })

                # Save micro-expression immediately (rare event)
                if micro_event:
                    try:
                        await session_manager.save_micro_expression(
                            db=db_session,
                            session_id=analysis_session_id,
                            event={
                                "timestamp": micro_event.timestamp,
                                "duration_ms": micro_event.duration_ms,
                                "detected_emotion": micro_event.detected_emotion,
                                "dominant_emotion_at_time": micro_event.dominant_emotion_at_time,
                                "action_units_involved": micro_event.action_units_involved,
                                "relevance_score": micro_event.relevance_score,
                                "is_contradictory": micro_event.is_contradictory,
                                "description": micro_event.description,
                            },
                        )
                        await db_session.commit()
                    except Exception as e:
                        print(f"[WARN] Failed to save micro-expression: {e}")

                # Batch-flush emotion records every N frames
                if len(emotion_record_batch) >= BATCH_SIZE:
                    try:
                        for record_data in emotion_record_batch:
                            await session_manager.save_emotion_record(
                                db=db_session,
                                session_id=analysis_session_id,
                                data=record_data,
                            )
                        await db_session.commit()
                    except Exception as e:
                        print(f"[WARN] Failed to batch-save emotion records: {e}")
                    emotion_record_batch.clear()

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"[ERR] WebSocket error: {e}")
        traceback.print_exc()
        manager.disconnect(websocket)
    finally:
        # ── Finalize session on disconnect ────────────────────────────
        if db_session and analysis_session_id:
            try:
                # Flush any remaining batched records
                if emotion_record_batch:
                    for record_data in emotion_record_batch:
                        await session_manager.save_emotion_record(
                            db=db_session,
                            session_id=analysis_session_id,
                            data=record_data,
                        )
                    emotion_record_batch.clear()

                # End session and generate summary
                await session_manager.end_session(
                    db=db_session,
                    session_id=analysis_session_id,
                )
                await db_session.commit()
            except Exception as e:
                print(f"[WARN] Failed to finalize session: {e}")
            finally:
                await db_session.close()

        face_detector.close()
