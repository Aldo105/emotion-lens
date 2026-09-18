import cv2
import numpy as np
import pytest

from backend.app.routers import websocket as ws


def _frame(size=(480, 640), seed=0):
    """Frame con textura: sin detalle, la varianza del Laplaciano sería ~0."""
    rng = np.random.default_rng(seed)
    return rng.integers(60, 200, size=(*size, 3), dtype=np.uint8)


def _detection(frame):
    h, w = frame.shape[:2]
    # Rostro centrado que ocupa una fracción sana del cuadro.
    return {
        "bbox": {
            "x_min": int(w * 0.35), "x_max": int(w * 0.65),
            "y_min": int(h * 0.25), "y_max": int(h * 0.75),
        },
        "landmarks": [],
        "frame_shape": (h, w),
    }


@pytest.fixture
def pose(monkeypatch):
    """Fija el ángulo de cabeza que verá la función de calidad."""
    def _set(yaw, pitch=0.0):
        monkeypatch.setattr(ws, "_estimate_head_pose", lambda *a, **k: (yaw, pitch))
    return _set


# ── La pose pedida no debe contar como defecto ────────────────────────

def test_girar_sin_que_nadie_lo_pida_penaliza(pose):
    pose(yaw=22.0)
    f = _frame()
    q = ws._compute_camera_quality(f, _detection(f))
    assert any("girado" in x for x in q["warnings"])


def test_la_misma_pose_no_penaliza_si_es_la_pedida(pose):
    pose(yaw=22.0)
    f = _frame()
    q = ws._compute_camera_quality(f, _detection(f), requested_pose="right")
    assert not any("girado" in x for x in q["warnings"])


def test_pose_pedida_puntua_igual_que_estar_de_frente(pose):
    f = _frame()
    pose(yaw=0.0)
    frontal = ws._compute_camera_quality(f, _detection(f), requested_pose="center")
    pose(yaw=22.0)
    girado = ws._compute_camera_quality(f, _detection(f), requested_pose="right")
    assert girado["score"] == pytest.approx(frontal["score"])


def test_izquierda_tambien(pose):
    pose(yaw=-22.0)
    f = _frame()
    q = ws._compute_camera_quality(f, _detection(f), requested_pose="left")
    assert not any("girado" in x for x in q["warnings"])


def test_pose_vertical_pedida_no_penaliza(pose):
    f = _frame()
    pose(yaw=0.0, pitch=18.0)
    pedida = ws._compute_camera_quality(f, _detection(f), requested_pose="down")
    pose(yaw=0.0, pitch=0.0)
    frontal = ws._compute_camera_quality(f, _detection(f), requested_pose="center")
    assert pedida["score"] == pytest.approx(frontal["score"])


def test_no_alcanzar_la_pose_pedida_si_penaliza(pose):
    # Le piden girar a la derecha (+22) y mira al lado contrario.
    pose(yaw=-25.0)
    f = _frame()
    q = ws._compute_camera_quality(f, _detection(f), requested_pose="right")
    assert any("postura" in x for x in q["warnings"])
    assert q["score"] < 1.0


def test_mensaje_de_pose_no_contradice_la_instruccion(pose):
    # Mientras se pide una pose girada, nunca debe decirse "mira a la camara".
    pose(yaw=-25.0)
    f = _frame()
    q = ws._compute_camera_quality(f, _detection(f), requested_pose="right")
    todo = " ".join(q["warnings"] + q["suggestions"])
    assert "Gira tu cabeza hacia la camara" not in todo


def test_pose_desconocida_cae_a_frontal(pose):
    pose(yaw=22.0)
    f = _frame()
    q = ws._compute_camera_quality(f, _detection(f), requested_pose="no_existe")
    assert any("girado" in x for x in q["warnings"])


# ── La nitidez debe medirse sobre el frame crudo ──────────────────────

def test_el_preprocesado_baja_la_nitidez_medida():
    """
    Justifica medir sobre el frame crudo: el box filter del preprocesador es
    un pasa-bajos, y la nitidez se define como varianza del Laplaciano.
    """
    crudo = _frame()
    det = _detection(crudo)
    suavizado = cv2.blur(crudo, (5, 5))

    q_crudo = ws._compute_camera_quality(crudo, det)
    q_suave = ws._compute_camera_quality(suavizado, det)

    assert q_suave["sharpness"] < q_crudo["sharpness"]
    assert q_suave["score"] <= q_crudo["score"]


def test_frame_crudo_con_textura_no_se_marca_borroso():
    f = _frame()
    q = ws._compute_camera_quality(f, _detection(f))
    assert not any("borrosa" in x for x in q["warnings"])
