# EmotionLens 🎯

**Real-time facial emotion & micro-expression recognition for HR interviews.**

EmotionLens analyzes facial expressions in real-time to detect emotions, micro-expressions, and emotional congruence — helping HR teams gain deeper insights during candidate interviews.

---

## ✨ Features

- 🧠 **9 Emotion Detection** — Happy, Sad, Angry, Surprise, Disgust, Fear, Neutral + Nervousness + Confidence
- 🔍 **Smart Micro-Expression Detection** — 3-layer filtering (temporal, context, relevance) using FACS Action Units
- 🎯 **Trustworthiness Score** — Congruence analysis between expressed and underlying emotions
- ❤️ **Heart Rate (rPPG)** — Remote photoplethysmography estimates pulse from facial video + stress indicator
- 📹 **Dual Input** — Live webcam analysis or offline video file upload
- 📂 **Session Management** — Persistent session history with search and review
- 🔄 **Candidate Comparison** — Side-by-side comparison with radar chart metrics
- 📝 **Interviewer Notes** — Timestamped notes tagged with emotional context
- 📄 **PDF/CSV Reports** — Professional report export with full analytics
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
3. The system calibrates a neutral baseline for 30 seconds
4. Explore the dashboard: emotion timeline, congruence gauge, micro-expression log

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
│   │   │   ├── face_detector.py        # MediaPipe Face Mesh (468 landmarks)
│   │   │   ├── action_units.py         # 16 FACS Action Units + derived metrics
│   │   │   ├── emotion_classifier.py   # CNN (MobileNetV2) + heuristic fallback
│   │   │   ├── micro_expressions.py    # 3-layer smart filtering engine
│   │   │   ├── congruence.py           # 4-component trustworthiness scoring
│   │   │   ├── heart_rate.py           # rPPG heart rate estimator (EVM)
│   │   │   ├── session_manager.py      # Session lifecycle + auto-summary
│   │   │   ├── video_processor.py      # Offline video analysis pipeline
│   │   │   └── report_generator.py     # PDF/CSV report generation
│   │   ├── routers/
│   │   │   ├── websocket.py     # Live analysis WebSocket
│   │   │   ├── sessions.py      # Session CRUD + comparison
│   │   │   ├── notes.py         # Interviewer notes
│   │   │   ├── reports.py       # Report data + PDF/CSV downloads
│   │   │   └── videos.py        # Video upload + progress
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
| **Face Detection** | MediaPipe Face Mesh (468 landmarks) |
| **Emotion Model** | PyTorch CNN (MobileNetV2) + Heuristic fallback |
| **Action Units** | 16 FACS AUs + blink, gaze, tilt, symmetry |
| **Micro-Expressions** | 3-layer engine (temporal + context + relevance) |
| **Heart Rate** | Remote PPG (green channel FFT, 0.7–4.0 Hz) |
| **Frontend** | Vanilla HTML/CSS/JS + Chart.js |
| **Database** | SQLite (dev) / PostgreSQL (prod) |
| **Real-time** | WebSockets (binary frame streaming) |
| **Reports** | ReportLab (PDF) + CSV export |
| **GPU** | NVIDIA CUDA 12.1 + CuDNN + TF32/FP16 |

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `WS` | `/ws/emotion` | Live emotion analysis WebSocket |
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

## ⚖️ Ethical Notice

EmotionLens is designed as an **analytical aid for research and educational purposes**.

> ⚠️ Emotion detection technology has inherent limitations and biases. Results should **never** be used as the sole basis for hiring decisions, character assessment, or truthfulness determination. All analysis should be interpreted by qualified professionals within proper ethical frameworks.

---

## 📄 License

This project is for educational and research purposes.

---

Built with ❤️ by the EmotionLens team.
