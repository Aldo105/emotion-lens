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
    Evaluate camera/capture quality for reliable micro-expression detection.
    
    Checks:
      - Face size: Is the face large enough in the frame for precise landmarks?
      - Brightness: Is the frame too dark or too bright for reliable color extraction?
      - Face position: Is the face centered and not at the edge of the frame?
    
    Returns:
        {
            "score": float (0.0 - 1.0),
            "face_size_ratio": float,
            "brightness": float,
            "warnings": list[str],
        }
    """
    h, w = frame.shape[:2]
    bbox = detection["bbox"]
    
    face_w = bbox["x_max"] - bbox["x_min"]
    face_h = bbox["y_max"] - bbox["y_min"]
    face_area = face_w * face_h
    frame_area = h * w
    
    warnings = []
    score = 1.0

    # ── Face Size Check ──────────────────────────────────────────
    face_size_ratio = face_area / max(frame_area, 1)
    if face_size_ratio < 0.03:
        warnings.append("Face too small — move closer to camera")
        score -= 0.4
    elif face_size_ratio < 0.08:
        warnings.append("Face is small — move slightly closer")
        score -= 0.15

    # ── Brightness Check ─────────────────────────────────────────
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Only measure brightness inside the face bounding box
    face_roi = gray[bbox["y_min"]:bbox["y_max"], bbox["x_min"]:bbox["x_max"]]
    if face_roi.size > 0:
        brightness = float(np.mean(face_roi)) / 255.0
    else:
        brightness = 0.5
    
    if brightness < 0.20:
        warnings.append("Too dark — improve lighting")
        score -= 0.35
    elif brightness < 0.30:
        warnings.append("Lighting is low — improve if possible")
        score -= 0.10
    elif brightness > 0.85:
        warnings.append("Too bright — reduce lighting or glare")
        score -= 0.20

    # ── Face Position Check ──────────────────────────────────────
    face_center_x = (bbox["x_min"] + bbox["x_max"]) / 2.0 / w
    face_center_y = (bbox["y_min"] + bbox["y_max"]) / 2.0 / h
    
    off_center = abs(face_center_x - 0.5) + abs(face_center_y - 0.5)
    if off_center > 0.5:
        warnings.append("Face is off-center — center yourself in the frame")
        score -= 0.15

    return {
        "score": float(np.clip(score, 0.0, 1.0)),
        "face_size_ratio": round(face_size_ratio, 4),
        "brightness": round(brightness, 3),
        "warnings": warnings,
    }


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
    heart_rate_estimator = HeartRateEstimator(fps=15.0)

    frame_count = 0
    session_start = time.time()
    baseline_calibrated = False
    calibration_start = None  # For recalibration timing

    try:
        # Send initial status
        await manager.send_json(websocket, WSStatusMessage(
            type="status",
            message="Connected to EmotionLens analysis server",
            data={"device": device, "mode": emotion_classifier.mode},
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

            # Step 1: Face Detection (MediaPipe)
            detection = face_detector.detect(frame)
            if detection is None:
                # No face detected -- send empty result
                await manager.send_json(websocket, WSStatusMessage(
                    type="status",
                    message="No face detected",
                ).model_dump())
                continue

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

            # ── Camera Quality Check ──────────────────────────────────
            camera_quality = _compute_camera_quality(frame, detection)

            # Step 4: Emotion Classification
            emotion_result = emotion_classifier.predict(
                action_units=action_units,
                blendshapes=detection.get("blendshapes"),
            )

            # Step 5: Micro-Expression Detection
            micro_event = None
            if baseline_calibrated:
                raw_event = micro_engine.analyze(
                    current_aus=action_units,
                    timestamp=timestamp,
                    dominant_emotion=emotion_result["emotion"],
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
            )

            await manager.send_json(websocket, result.model_dump())

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"[ERR] WebSocket error: {e}")
        traceback.print_exc()
        manager.disconnect(websocket)
    finally:
        face_detector.close()
