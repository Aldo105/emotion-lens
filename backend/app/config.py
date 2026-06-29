"""
EmotionLens — Application Configuration

Centralizes all configurable settings. Values can be overridden via environment
variables or a .env file at the project root.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


# Project root directory (emotion-lens/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # ── App ──────────────────────────────────────────────────────────
    app_name: str = "EmotionLens"
    app_version: str = "0.1.0"
    debug: bool = True

    # ── Server ───────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["*"]  # Restrict in production

    # ── Database ─────────────────────────────────────────────────────
    database_url: str = f"sqlite+aiosqlite:///{BASE_DIR / 'data' / 'emotion_lens.db'}"

    # ── ML Models ────────────────────────────────────────────────────
    model_dir: str = str(BASE_DIR / "backend" / "ml" / "saved_models")
    emotion_model_name: str = "emotion_cnn.pt"
    # ── GPU / CUDA Optimization ───────────────────────────────────────
    use_gpu: bool = True          # Auto-fallback to CPU if CUDA unavailable
    gpu_device_id: int = 0        # Which GPU to use (multi-GPU systems)
    gpu_memory_fraction: float = 0.7  # Max fraction of GPU memory to use
    enable_half_precision: bool = True  # Use FP16 for faster inference
    enable_cudnn_benchmark: bool = True  # CuDNN auto-tuner for fixed input sizes

    # ── Face Detection ───────────────────────────────────────────────
    max_faces: int = 1  # Single person analysis (expandable later)
    face_detection_confidence: float = 0.7
    face_tracking_confidence: float = 0.5

    # ── Emotion Classification ───────────────────────────────────────
    emotion_confidence_threshold: float = 0.3  # Min confidence to report
    emotion_labels: list[str] = [
        "happy", "sad", "angry", "surprise",
        "disgust", "fear", "neutral",
        "nervousness", "confidence"
    ]

    # ── Micro-Expressions ────────────────────────────────────────────
    micro_expr_min_duration_ms: int = 40     # Min duration (ms)
    micro_expr_max_duration_ms: int = 500    # Max duration (ms)
    micro_expr_relevance_threshold: int = 60  # Min relevance score (0-100)
    baseline_calibration_seconds: int = 30   # Baseline capture duration
    baseline_deviation_multiplier: float = 2.0  # Flag if > 2x baseline

    # ── Congruence / Trustworthiness ─────────────────────────────────
    congruence_weight_stability: float = 0.30
    congruence_weight_micro_alignment: float = 0.35
    congruence_weight_baseline: float = 0.20
    congruence_weight_physiological: float = 0.15

    # ── WebSocket ────────────────────────────────────────────────────
    ws_frame_skip: int = 1  # Process every frame (needed for micro-expression detection at 20 FPS)
    ws_max_connections: int = 10

    # ── Video Recording (Optional) ───────────────────────────────────
    enable_video_recording: bool = False
    recordings_dir: str = str(BASE_DIR / "data" / "recordings")

    # ── Reports ──────────────────────────────────────────────────────
    reports_dir: str = str(BASE_DIR / "data" / "reports")

    # ── Uploads ──────────────────────────────────────────────────────
    uploads_dir: str = str(BASE_DIR / "data" / "uploads")
    max_upload_size_mb: int = 500

    class Config:
        env_file = str(BASE_DIR / ".env")
        env_file_encoding = "utf-8"


# Singleton settings instance
settings = Settings()


# ═══════════════════════════════════════════════════════════════════════
# GPU / CUDA Helpers
# ═══════════════════════════════════════════════════════════════════════

def get_device() -> str:
    """Determine the compute device (CUDA GPU or CPU)."""
    if settings.use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(settings.gpu_device_id)
                vram = torch.cuda.get_device_properties(settings.gpu_device_id).total_mem
                vram_gb = vram / (1024 ** 3)
                print(f"[OK] GPU detected: {device_name} ({vram_gb:.1f} GB VRAM)")
                return "cuda"
        except Exception as e:
            print(f"[!!] Could not load PyTorch: {e}")
    print("[!!] Running on CPU (no CUDA available or GPU disabled)")
    return "cpu"


def initialize_gpu() -> str:
    """
    Full GPU initialization with performance optimizations.
    Call once at startup. Returns the device string.
    """
    device = get_device()

    if device == "cuda":
        try:
            import torch

            # Set default device
            torch.cuda.set_device(settings.gpu_device_id)

            # Limit GPU memory usage to prevent OOM
            if settings.gpu_memory_fraction < 1.0:
                total_mem = torch.cuda.get_device_properties(
                    settings.gpu_device_id
                ).total_mem
                max_mem = int(total_mem * settings.gpu_memory_fraction)
                torch.cuda.set_per_process_memory_fraction(
                    settings.gpu_memory_fraction,
                    settings.gpu_device_id,
                )
                print(f"[GPU] Memory limit: {max_mem / (1024**3):.1f} GB "
                      f"({settings.gpu_memory_fraction * 100:.0f}%)")

            # Enable CuDNN benchmark mode for consistent input sizes
            if settings.enable_cudnn_benchmark:
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                print("[GPU] CuDNN benchmark mode enabled")

            # Enable TF32 for Ampere+ GPUs (30xx, A100, etc.)
            if hasattr(torch.backends, 'cuda'):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("[GPU] TF32 precision enabled")

            # Enable automatic mixed precision globally
            if settings.enable_half_precision:
                print("[GPU] Half-precision (FP16) inference enabled")

            # Warm up GPU with a small allocation
            _ = torch.zeros(1, device="cuda")
            print("[GPU] GPU initialized and warmed up successfully")

        except Exception as e:
            print(f"[!!] GPU optimization failed: {e}")
            device = "cpu"

    return device
