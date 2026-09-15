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
import os
import threading
import time
import traceback

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from backend.app.config import settings
from backend.app.models.schemas import WSFrameResult, WSStatusMessage, MicroExpressionEvent
from backend.app.services.face_detector import FaceDetector
from backend.app.services.action_units import ActionUnitAnalyzer
from backend.app.services.emotion_classifier import EmotionClassifier
from backend.app.services.micro_expressions import MicroExpressionEngine
from backend.app.services.congruence import CongruenceScorer
from backend.app.services.heart_rate import HeartRateEstimator
from backend.app.services.noise_filter import FacialNoiseFilter
from backend.app.services.video_recorder import SessionVideoRecorder
from backend.app.services.evm_renderer import EVMRenderer
from backend.app.services.micro_expression_highlight_renderer import MicroExpressionHighlightRenderer
from backend.app.services import session_manager
from backend.app.models.database import async_session_factory, MicroExpression as MicroExpressionRow
from backend.app.utils.image_preprocessing import AdaptivePreprocessor


def _render_session_videos(
    raw_path: str,
    meta_path: str,
    micro_events: list[dict],
    evm_output_path: str,
    highlights_output_path: str,
    delete_raw: bool,
) -> None:
    """
    Runs both offline renders (pulse EVM + micro-expression highlight reel)
    against the same raw session recording, then cleans up the raw files.

    Must run BEFORE any raw video cleanup — both renderers need the same
    raw_path/meta_path, so deletion is deferred to this function regardless
    of each renderer's own delete_raw setting.
    """
    evm_renderer = EVMRenderer(
        amplification=settings.evm_amplification,
        freq_low=settings.evm_freq_low,
        freq_high=settings.evm_freq_high,
        pyramid_levels=settings.evm_pyramid_levels,
        delete_raw=False,
    )
    try:
        evm_renderer.render(raw_path, meta_path, evm_output_path)
    except Exception as e:
        print(f"[WARN] Pulse EVM rendering failed: {e}")

    highlight_renderer = MicroExpressionHighlightRenderer(
        amplification=settings.micro_evm_amplification,
        freq_low=settings.micro_evm_freq_low,
        freq_high=settings.micro_evm_freq_high,
        pyramid_levels=settings.micro_evm_pyramid_levels,
        clip_padding_sec=settings.micro_evm_clip_padding_sec,
        max_clips=settings.micro_evm_max_clips,
        slowmo_factor=settings.micro_evm_slowmo_factor,
    )
    try:
        # Handles the empty-events case internally (writes "not_available"
        # progress immediately instead of leaving the frontend polling).
        highlight_renderer.render(raw_path, meta_path, micro_events, highlights_output_path)
    except Exception as e:
        print(f"[WARN] Micro-expression highlight rendering failed: {e}")

    if delete_raw:
        for path in [raw_path, meta_path, meta_path.replace("_meta.npz", "_hr.json")]:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"[OK] Cleaned up: {path}")
            except Exception as e:
                print(f"[WARN] Failed to delete {path}: {e}")

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
        warnings.append("Rostro muy pequeño — acercate a la camara")
        suggestions.append("Acercate a la camara para mejor precision")
        score -= 0.4
    elif face_size_ratio < 0.08:
        warnings.append("Rostro pequeño — acercate un poco")
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
        warnings.append("Muy oscuro — mejora la iluminacion")
        suggestions.append("Enciende una luz frente a ti o acercate a una ventana")
        score -= 0.35
    elif brightness < 0.30:
        warnings.append("Poca luz — mejorala si es posible")
        suggestions.append("Mejora la iluminacion si es posible")
        score -= 0.10
    elif brightness > 0.85:
        warnings.append("Demasiado brillo — reduce la luz o el reflejo")
        suggestions.append("Reduce el brillo o alejate de la fuente de luz")
        score -= 0.20

    # ── Face Position Check ──────────────────────────────────────
    face_center_x = (bbox["x_min"] + bbox["x_max"]) / 2.0 / w
    face_center_y = (bbox["y_min"] + bbox["y_max"]) / 2.0 / h

    off_center = abs(face_center_x - 0.5) + abs(face_center_y - 0.5)
    if off_center > 0.5:
        warnings.append("Rostro descentrado — centrate en el cuadro")
        suggestions.append("Centra tu rostro en la camara")
        score -= 0.15

    # ── Sharpness Check (Laplacian variance) ─────────────────────
    sharpness = 100.0
    if face_roi.size > 0:
        laplacian = cv2.Laplacian(face_roi, cv2.CV_64F)
        sharpness = float(laplacian.var())
        if sharpness < 50:
            warnings.append("Imagen borrosa — limpia la camara o quedate quieto")
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
        warnings.append("Rostro muy girado — mira hacia la camara")
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
    noise_filter = FacialNoiseFilter()

    frame_count = 0
    processed_frame_count = 0
    session_start = time.time()
    baseline_calibrated = False
    calibration_start = None  # For recalibration timing
    current_skin_tone = "medium"  # Updated per-frame from quality check

    # ── Throttle counters ─────────────────────────────────────────────
    # Recomputing camera quality on every frame (20/s) is wasteful —
    # lighting and position change slowly. Run it at ~2 Hz instead.
    _last_quality_result: dict | None = None
    _quality_check_interval = 10  # recompute every N processed frames

    # ── Database session persistence ─────────────────────────────────
    db_session = None
    analysis_session_id = None
    emotion_record_batch = []  # Batch records for efficient DB writes
    BATCH_SIZE = 10  # Flush to DB every N processed frames

    # ── Video recorder (for EVM post-processing) ──────────────────────
    video_recorder: SessionVideoRecorder | None = None

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

        # Initialize video recorder if enabled
        if settings.enable_video_recording:
            video_recorder = SessionVideoRecorder(
                session_id=analysis_session_id,
                output_dir=settings.recordings_dir,
                fps=20.0,
                frame_size=(640, 480),
            )

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
            # Pass real timestamp for accurate temporal tracking
            real_ts_ms = int(timestamp * 1000)
            detection = face_detector.detect(preprocessed_frame, timestamp_ms=real_ts_ms)
            if detection is None:
                # No face detected -- still record the frame (with no bbox)
                if video_recorder and video_recorder.is_open:
                    video_recorder.write_frame(frame, timestamp=timestamp, face_bbox=None)

                # No bbox means we can't run the full camera-quality check,
                # but overall frame brightness is cheap (one grayscale mean)
                # and this is exactly when lighting/framing problems are
                # worst -- give actionable feedback instead of a silent
                # "no face" the frontend otherwise ignores.
                frame_brightness = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))) / 255.0
                if frame_brightness < 0.15:
                    warnings = ["No se detecta tu rostro — el ambiente esta muy oscuro"]
                    suggestions = ["Enciende una luz frente a ti o acercate a una ventana"]
                elif frame_brightness > 0.90:
                    warnings = ["No se detecta tu rostro — hay demasiada luz o reflejo"]
                    suggestions = ["Reduce el brillo o alejate de la fuente de luz"]
                else:
                    warnings = ["No se detecta tu rostro"]
                    suggestions = ["Centra tu rostro frente a la camara, a un brazo de distancia"]

                await manager.send_json(websocket, {
                    "type": "quality_warning",
                    "camera_quality": {
                        "score": 0.0,
                        "brightness": round(frame_brightness, 3),
                        "warnings": warnings,
                        "suggestions": suggestions,
                        "quality_gate": "fail",
                    },
                    "message": warnings[0],
                })
                continue

            # Record raw frame with face bbox for EVM post-processing
            if video_recorder and video_recorder.is_open:
                video_recorder.write_frame(
                    frame, timestamp=timestamp, face_bbox=detection["bbox"]
                )

            # Step 1.5: Camera Quality + Quality Gate
            # Throttled: recompute every _quality_check_interval frames (~2 Hz).
            # Camera quality (brightness, sharpness, pose) changes slowly.
            if _last_quality_result is None or frame_count % _quality_check_interval == 0:
                _last_quality_result = _compute_camera_quality(preprocessed_frame, detection)
            camera_quality = _last_quality_result
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
                    noise_filter.set_baseline(au_analyzer.baseline_aus)

                    # Set emotion classifier baseline from blendshapes (Problem 5 fix)
                    # so resting facial morphology is subtracted from predictions
                    if detection.get("blendshapes"):
                        emotion_classifier.set_baseline(detection["blendshapes"])

                baseline_calibrated = True
                calibration_start = None  # Reset recalibration timer

                await manager.send_json(websocket, WSStatusMessage(
                    type="calibration_complete",
                    message="Baseline calibration complete. Analysis active.",
                ).model_dump())

            # Step 3.5: Noise Filtering (speech, yawns, tics, scratching)
            noise_state = noise_filter.analyze(
                current_aus=action_units,
                timestamp=timestamp,
            )

            # Step 4: Emotion Classification (with quality penalty)
            # Only pay for the face-crop + normalize when a CNN is actually
            # loaded -- blendshape/heuristic mode doesn't need face_image.
            face_image = None
            if emotion_classifier.mode == "cnn":
                face_image = EmotionClassifier.preprocess_face(
                    preprocessed_frame, detection["bbox"]
                )

            emotion_result = emotion_classifier.predict(
                face_image=face_image,
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
                    noise_state=noise_state,
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

                # Store HR reading for EVM video overlay
                if video_recorder and video_recorder.is_open:
                    video_recorder.record_hr(
                        timestamp=timestamp,
                        bpm=hr_result.get("bpm", 0.0),
                        confidence=hr_result.get("confidence", 0.0),
                    )

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
                camera_quality=camera_quality,
                is_calibrating=is_calibrating,
                calibration_progress=round(calibration_progress, 2),
                noise_state={
                    "is_speaking": noise_state.is_speaking,
                    "is_yawning": noise_state.is_yawning,
                    "is_tic": noise_state.is_tic,
                    "noise_type": noise_state.noise_type,
                } if noise_state else None,
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
        micro_events: list[dict] = []
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

                # Fetch top micro-expressions for the highlight video, as plain
                # dicts — extracted before closing the session so they can be
                # handed to the background render thread.
                result = await db_session.execute(
                    select(MicroExpressionRow)
                    .where(MicroExpressionRow.session_id == analysis_session_id)
                    .order_by(MicroExpressionRow.relevance_score.desc())
                    .limit(settings.micro_evm_max_clips)
                )
                micro_events = [
                    {
                        "timestamp": row.timestamp,
                        "duration_ms": row.duration_ms,
                        "detected_emotion": row.detected_emotion,
                        "dominant_emotion_at_time": row.dominant_emotion_at_time,
                        "action_units_involved": row.action_units_involved,
                        "relevance_score": row.relevance_score,
                        "is_contradictory": row.is_contradictory,
                    }
                    for row in result.scalars().all()
                ]

                await db_session.commit()
            except Exception as e:
                print(f"[WARN] Failed to finalize session: {e}")
            finally:
                await db_session.close()

        # ── Close video recorder and launch offline rendering ─────────
        if video_recorder and video_recorder.is_open:
            try:
                raw_path, meta_path = video_recorder.close()
                evm_output_path = os.path.join(
                    settings.recordings_dir,
                    f"{analysis_session_id}_evm.mp4",
                )
                highlights_output_path = os.path.join(
                    settings.recordings_dir,
                    f"{analysis_session_id}_micro_highlights.mp4",
                )
                # Run both renders in one background thread (non-blocking).
                # Raw video cleanup is deferred until both finish — see
                # _render_session_videos.
                threading.Thread(
                    target=_render_session_videos,
                    args=(
                        raw_path, meta_path, micro_events,
                        evm_output_path, highlights_output_path,
                        settings.evm_delete_raw_after_render,
                    ),
                    daemon=True,
                    name=f"session-render-{analysis_session_id}",
                ).start()
                print(f"[OK] Offline rendering started in background for session {analysis_session_id}")
            except Exception as e:
                print(f"[WARN] Failed to start offline rendering: {e}")

        face_detector.close()
