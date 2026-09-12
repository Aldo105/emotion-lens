# EmotionLens 🎯

**Real-time facial emotion & micro-expression recognition for HR interviews.**

EmotionLens analyzes facial expressions in real-time to detect emotions, micro-expressions, and emotional congruence — helping HR teams gain deeper insights during candidate interviews. Powered by MediaPipe face tracking, FACS-based Action Unit analysis, and a multi-layer micro-expression engine.

---

## ✨ Features

### Core Analysis
- 🧠 **9 Emotion Detection** — Happy, Sad, Angry, Surprise, Disgust, Fear, Neutral + Nervousness + Confidence
- 🔍 **Smart Micro-Expression Detection** — 3-layer filtering (temporal, context, relevance) using FACS Action Units with `scipy.signal.find_peaks`, dynamic per-AU thresholds, EMA smoothing, and habitual movement tracking
- 🎯 **Trustworthiness Score** — 4-component congruence analysis: stability, micro-alignment, baseline deviation, and physiological correlation
- ❤️ **Heart Rate (rPPG CHROM)** — Multi-channel chrominance-based remote photoplethysmography with motion artifact rejection, nasal reference ROI, and signal quality (SNR) reporting

### Workflow
- 📹 **Dual Input** — Live webcam analysis or offline video file upload
- 📂 **Session Management** — Persistent session history with search and review (live sessions now auto-persist to database)
- 🔄 **Candidate Comparison** — Side-by-side comparison with radar chart metrics
- 📝 **Interviewer Notes** — Timestamped notes tagged with emotional context
- 📄 **PDF/CSV Reports** — Professional reports with comparative analysis table (Emotions vs Micro-Expressions) and contradiction summaries

### Platform
- 🌐 **Bilingual (EN/ES)** — Full English and Spanish localization
- ⚡ **GPU Accelerated** — NVIDIA CUDA support with TF32, FP16, and CuDNN optimization
- ⚖️ **Ethical Framework** — Built-in disclaimer and usage guidelines

---

## 🚀 Quick Start

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- NVIDIA GPU + CUDA 12.1 (optional, for acceleration)
- A webcam (for live analysis)

### Setup

```bash
# 1. Clone or navigate to the project
cd emotion-lens

# 2. Create the conda environment
conda env create -f environment.yml

# 3. Activate the environment
conda activate emotion-lens

# 4. Start the development server
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

# 5. Open your browser
# Dashboard: http://localhost:8000
# API Docs:  http://localhost:8000/docs
```

### First Run

1. The **ethical disclaimer** will appear on first load — read and acknowledge it
2. Click **Start Session** to begin live webcam analysis
3. The system calibrates a neutral baseline for ~10 seconds (builds personalized AU thresholds and habitual movement tracking)
4. Explore the dashboard: emotion timeline, congruence gauge, micro-expression log, heart rate monitor

---

## 🏗️ Architecture

```
emotion-lens/
├── environment.yml              # Conda dependencies
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point + lifecycle
│   │   ├── config.py            # Settings, GPU init, env vars
│   │   ├── models/
│   │   │   ├── database.py      # SQLAlchemy ORM (Session, Emotion, Micro, Notes)
│   │   │   └── schemas.py       # Pydantic request/response models
│   │   ├── services/
│   │   │   ├── face_detector.py        # MediaPipe Face Mesh (468 landmarks + 52 blendshapes)
│   │   │   ├── action_units.py         # 16 FACS Action Units + derived metrics
│   │   │   ├── emotion_classifier.py   # 3-tier: CNN (MobileNetV2) → Blendshapes → Heuristic
│   │   │   ├── micro_expressions.py    # 3-layer engine (temporal + context + relevance)
│   │   │   ├── congruence.py           # 4-component trustworthiness scoring
│   │   │   ├── heart_rate.py           # rPPG CHROM estimator + motion rejection + SNR
│   │   │   ├── session_manager.py      # Session lifecycle + auto-summary generation
│   │   │   ├── video_processor.py      # Offline video analysis pipeline
│   │   │   └── report_generator.py     # PDF/CSV reports + comparative analysis table
│   │   ├── routers/
│   │   │   ├── websocket.py     # Live analysis WebSocket + DB persistence
│   │   │   ├── sessions.py      # Session CRUD + comparison
│   │   │   ├── notes.py         # Interviewer notes
│   │   │   ├── reports.py       # Report data + PDF/CSV downloads
│   │   │   └── videos.py        # Video upload + progress tracking
│   │   └── utils/
│   │       └── image_processing.py  # Frame preprocessing
│   └── ml/
│       └── saved_models/        # Trained model weights (.pt)
├── frontend/
│   ├── index.html               # Main dashboard (4 views)
│   ├── css/styles.css           # Dark glassmorphism design system
│   ├── js/
│   │   ├── config.js            # API URLs + emotion mappings
│   │   ├── i18n.js              # Internationalization engine
│   │   ├── charts.js            # Chart.js (timeline, gauge, donut)
│   │   ├── webcam.js            # Webcam capture + canvas overlay
│   │   ├── websocket.js         # WebSocket client
│   │   ├── dashboard.js         # Live dashboard UI updates
│   │   ├── sessions.js          # Session history cards
│   │   ├── comparison.js        # Side-by-side comparison
│   │   ├── video-upload.js      # Video upload + progress
│   │   └── app.js               # Navigation, i18n, disclaimer
│   └── i18n/
│       ├── en.json              # English translations
│       └── es.json              # Spanish translations
├── data/                        # Runtime data (auto-created)
│   ├── emotion_lens.db          # SQLite database
│   ├── uploads/                 # Uploaded videos
│   └── reports/                 # Generated PDF reports
└── docs/
    └── deployment.md            # Production deployment guide
```

---

## 🧪 Tech Stack

| Component | Technology |
|:----------|:-----------|
| **Backend** | Python 3.11 / FastAPI / Uvicorn |
| **Face Detection** | MediaPipe Face Mesh (468 landmarks + 52 blendshapes) |
| **Emotion Model** | 3-tier: PyTorch CNN (MobileNetV2) → Blendshape mapping → Heuristic fallback |
| **Action Units** | 16 FACS AUs + blink rate, gaze stability, head tilt, face symmetry |
| **Micro-Expressions** | 3-layer engine: `find_peaks` temporal → dynamic threshold context → multi-factor relevance |
| **Heart Rate** | CHROM rPPG (de Haan & Jeanne 2013) + motion rejection + nasal reference ROI + SNR |
| **Frontend** | Vanilla HTML/CSS/JS + Chart.js |
| **Database** | SQLite (dev) / PostgreSQL (prod) via SQLAlchemy async |
| **Real-time** | WebSockets (base64 frame streaming) + batched DB persistence |
| **Reports** | ReportLab (PDF with comparative analysis) + CSV export |
| **GPU** | NVIDIA CUDA 12.1 + CuDNN + TF32/FP16 |

---

## 🔬 Analysis Pipeline

The full pipeline processes each video frame through 6 stages:

```
📷 Camera Frame
 │
 ▼
🔍 MediaPipe FaceLandmarker ─────────────────────── 468 landmarks + 52 blendshapes
 │
 ▼
📐 Action Unit Analyzer (FACS) ──────────────────── 16 AUs + blink, gaze, tilt, symmetry
 │
 ├──▶ 🧠 Emotion Classifier ────────────────────── 7 emotions + nervousness + confidence
 │     (CNN → Blendshapes → Heuristic)
 │
 ├──▶ ⚡ Micro-Expression Engine ────────────────── onset→apex→offset detection (40-500ms)
 │     (temporal filter → context filter → relevance scoring)
 │
 ├──▶ ❤️ Heart Rate Estimator ──────────────────── BPM + stress indicator + signal quality
 │     (CHROM multi-channel + motion rejection)
 │
 └──▶ 📊 Congruence Scorer ─────────────────────── Trust score 0-100
       (stability + micro-alignment + baseline + physiological)
```

### Heart Rate Algorithm (CHROM)

| Condition | Mean Absolute Error (MAE) | Effectiveness |
|-----------|:-------------------------:|:-------------:|
| Controlled (still, good light) | 1.5 – 2.5 BPM | ~95-97% |
| General (office, mixed light) | 3.0 – 6.0 BPM | ~90-94% |
| Adverse (movement, low light) | 8.0+ BPM | ~80-85% |

> **Note:** rPPG is NOT a medical device. It provides estimates for stress analysis, not clinical diagnostics.

### Micro-Expression Detection

The engine uses a **3-layer validation system** to minimize false positives:

1. **Temporal Filter** — Requires onset→apex→offset pattern within 40-500ms, detected via `scipy.signal.find_peaks`
2. **Context Filter** — AU activation must exceed personalized baseline + (variability × 2.5), and not be a habitual movement (tracked during calibration)
3. **Relevance Scoring** — Multi-factor score (0-100): AU count, FACS pattern match, contradiction with displayed emotion, optimal duration, baseline deviation — weighted by camera quality. Minimum threshold: 60/100

---

## 📄 PDF Report Contents

Generated reports include the following sections:

1. **Session Metadata** — Name, candidate, date, duration, status
2. **Emotion Distribution** — Percentage breakdown of all detected emotions
3. **Congruence Summary** — Average, min, max congruence scores + nervousness/confidence metrics
4. **Micro-Expression Log** — Timestamped event table with emotion, duration, and contradiction flag
5. **Interviewer Notes** — Timestamped notes with emotional context
6. **Key Moments** — Significant emotion shifts and high-relevance events
7. **Comparative Analysis: Emotions vs Micro-Expressions** — Correlates each micro-expression with the emotion being displayed at that moment, color-coded by contradiction status (red = contradictory, green = congruent)
8. **Contradiction Summary** — Frequency table of emotion suppression patterns (e.g., "Happiness → Anger: 3 occurrences — possible social masking")

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `WS` | `/ws/emotion` | Live emotion analysis WebSocket (auto-persists to DB) |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/sessions/` | List all sessions |
| `POST` | `/api/sessions/` | Create a session |
| `GET` | `/api/sessions/{id}` | Get session details |
| `DELETE` | `/api/sessions/{id}` | Delete a session |
| `POST` | `/api/sessions/compare` | Compare two sessions |
| `GET` | `/api/reports/{id}/data` | Full report data (JSON) |
| `GET` | `/api/reports/{id}/pdf` | Download PDF report |
| `GET` | `/api/reports/{id}/csv` | Download CSV export |
| `POST` | `/api/videos/upload` | Upload video for analysis |
| `GET` | `/api/videos/{id}/progress` | Poll processing progress |
| `POST` | `/api/notes/` | Add interviewer note |
| `GET` | `/api/notes/{session_id}` | Get session notes |

Interactive API documentation available at `/docs` (Swagger UI).

---

## ⚙️ Configuration

All settings can be overridden via environment variables or `.env` file:

```bash
# GPU
USE_GPU=true
GPU_MEMORY_FRACTION=0.7
ENABLE_HALF_PRECISION=true

# Server
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Database (PostgreSQL for production)
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/emotionlens

# Uploads
MAX_UPLOAD_SIZE_MB=500
```

See [docs/deployment.md](docs/deployment.md) for full deployment instructions.

---

## ⚠️ Known Limitations & Non-Candidate Populations

This software is calibrated for **neurotypical adult facial expression patterns** based on the FACS system (Ekman & Friesen, 1978). Results may be inaccurate for:

| Population | Risk Level | Reason |
|-----------|:----------:|--------|
| Facial paralysis (Bell's Palsy, Moebius) | 🔴 Critical | Physical inability to produce standard AUs |
| Post-facial reconstruction surgery | 🔴 Critical | Altered facial geometry outside normal ranges |
| Extensive Botox treatments | 🟠 High | Paralyzed muscles bias toward "neutral" classification |
| Autism Spectrum (TEA) | 🟠 High | Atypical expression patterns, masking, alexithymia |
| Very dark skin tones | 🟡 Medium | rPPG heart rate has lower signal-to-noise ratio (emotion detection unaffected) |
| Dense beard covering mouth | 🟡 Medium | Occluded lip landmarks; eye/brow AUs still functional |
| Children under 6 | 🟡 Medium | Facial proportions differ from adult training data |
| Cross-cultural expressions | 🟡 Medium | AU→emotion mappings are Western-centric |

---

## ⚖️ Ethical Notice

EmotionLens is designed as an **analytical aid for research and educational purposes**.

> ⚠️ Emotion detection technology has inherent limitations and biases. Results should **never** be used as the sole basis for hiring decisions, character assessment, or truthfulness determination. All analysis should be interpreted by qualified professionals within proper ethical frameworks.

> This software is NOT a medical device. Heart rate estimates via rPPG are for stress analysis purposes only and should not be used for clinical diagnostics.

---

## 📄 License

This project is for educational and research purposes.

---

Built with ❤️ by the EmotionLens team.
