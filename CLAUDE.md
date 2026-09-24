# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup (Conda env, includes PyTorch + CUDA 12.1, MediaPipe, FastAPI, pytest)
conda env create -f environment.yml
conda activate emotion-lens

# Run the dev server (serves API + static frontend from one process)
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
# Dashboard: http://localhost:8000  |  API docs: http://localhost:8000/docs

# Run all tests
pytest backend/tests/

# Run a single test file / test
pytest backend/tests/test_event_timeline.py
pytest backend/tests/test_validation_metrics.py -k test_no_validations_returns_none_rate_not_zero
```

There is no build step for the frontend — it's vanilla HTML/CSS/JS served directly by FastAPI's `StaticFiles` mount (`backend/app/main.py`), no bundler/transpiler involved. No linter is configured.

## Architecture

**Single FastAPI process** serves both the JSON/WebSocket API and the static frontend (`backend/app/main.py` mounts `frontend/` at `/` after all `/api/*` and `/ws/*` routes). Config is a Pydantic `Settings` singleton in `backend/app/config.py`, overridable via `.env`; it also owns GPU/CUDA init (TF32, FP16, CuDNN benchmark, memory fraction).

### Per-frame analysis pipeline (services, in `backend/app/services/`)

Each webcam/video frame flows through this chain, tied together by `websocket.py` (live) or `video_processor.py` (offline upload):

1. `face_detector.py` — MediaPipe Face Mesh → 468 landmarks + 52 blendshapes
2. `action_units.py` — derives 16 FACS Action Units + blink rate, gaze stability, head tilt, symmetry from the landmarks
3. `emotion_classifier.py` — 3-tier fallback: PyTorch CNN (MobileNetV2, weights in `backend/ml/saved_models/emotion_cnn.pt`) → blendshape mapping → heuristic rules. Emotion label set is defined in `config.py` (`emotion_labels`) — note `fear` was deliberately dropped (see comment there and `backend/app/references.py`)
4. `micro_expressions.py` — 3-layer filter: temporal (`scipy.signal.find_peaks`, 40-500ms onset→apex→offset), context (personalized baseline + variability threshold, habitual-movement exclusion), relevance scoring (0-100, min threshold from `config.micro_expr_relevance_threshold`)
5. `heart_rate.py` — rPPG via CHROM (de Haan & Jeanne), multi-channel chrominance, motion-artifact rejection, nasal reference ROI, SNR-based signal quality
6. `congruence.py` — combines stability + micro-expression alignment + baseline deviation + physiological signal into a 0-100 trustworthiness score, weighted per `config.py` (`congruence_weight_*`)

Downstream of the pipeline:
- `session_manager.py` — session lifecycle, calibration (baseline capture, `config.baseline_calibration_seconds`), auto-summary generation
- `interview_analyzer.py` (`InterviewBehaviorAnalyzer`) — builds the moderator event timeline / per-task friction analysis from a session's emotion/congruence/notes series; backs the `/api/analysis` router
- `validation_metrics.py` — human-vs-model agreement metrics for post-session subject feedback; backs `/api/feedback`
- `noise_filter.py`, `ux_metrics.py` — additional signal cleanup / UX telemetry used by the above
- `evm_renderer.py` / `live_evm.py` — Eulerian Video Magnification: `live_evm` drives the real-time "show pulse" overlay (causal IIR filtering, `config.live_evm_amplification`), `evm_renderer` does the offline post-session render (zero-phase, `config.evm_*`) plus micro-expression highlight clips (`micro_expression_highlight_renderer.py`, `config.micro_evm_*`)
- `video_recorder.py` — optional raw session recording, feeding the offline EVM render
- `report_generator.py` — builds PDF (ReportLab) / CSV reports, including the emotions-vs-micro-expressions comparative table and contradiction summary

### API surface (`backend/app/routers/`)

`websocket.py` (`/ws/emotion`, live analysis + batched DB persistence), `sessions.py`, `notes.py`, `reports.py`, `videos.py` (upload + offline processing), `feedback.py` (post-session subject survey), `analysis.py` (interview behavior analysis). All registered in `main.py`.

### Data layer

`backend/app/models/database.py` — async SQLAlchemy ORM (SQLite dev via `aiosqlite`, Postgres prod via `asyncpg`), entities include `Session`, `SessionSummary`, `EmotionRecord`, `MicroExpression`, `InterviewerNote`, `InterviewAnalysis`, `SessionFeedback`. `models/schemas.py` holds the Pydantic request/response models. DB and file paths (`data/emotion_lens.db`, `data/uploads`, `data/reports`, `data/recordings`) are created at startup in `main.py`'s lifespan handler and are git-ignored.

### Frontend (`frontend/`, vanilla JS, no framework/bundler)

`app.js` (nav, i18n, ethical disclaimer gate) orchestrates feature modules: `webcam.js` (capture + canvas overlay) → `websocket.js` (client for `/ws/emotion`) → `dashboard.js` (live UI updates), plus `sessions.js` (history), `comparison.js` (side-by-side + radar chart), `video-upload.js`, `post-session-survey.js`, `charts.js` (Chart.js wrappers), `icons.js`. `i18n.js` + `i18n/en.json`/`es.json` drive full EN/ES bilingual UI — new user-facing strings must be added to both locale files.

### Tooling outside the app

`backend/ml/train_emotion_cnn.py` / `evaluate_model.py` — trains/evaluates the emotion CNN checkpoint. `tools/hr_validation/` — Arduino pulse sensor sketch + scripts to validate the rPPG heart-rate estimate against ground-truth hardware. `backend/scripts/` — one-off maintenance scripts (demo session seeding, bibliography/plan PDF generation).

## Domain notes

This is a research/educational HR-interview analysis tool, not a medical device — see the README's "Known Limitations" and "Ethical Notice" sections before changing anything in the emotion/congruence/heart-rate scoring logic, since those thresholds encode explicit ethical and accuracy trade-offs (e.g., the `fear` label removal in `config.py`, the FACS Western-centric bias caveat, the rPPG-is-not-clinical disclaimer).
