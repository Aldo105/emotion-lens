import numpy as np
import pytest

from backend.app.routers import websocket as ws
from backend.app.services.face_detector import FaceDetector
from backend.app.services.pose_calibration import POSE_ANCHORS, pose_matches


def _rot_y(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), 0, np.sin(t)],
                     [0, 1, 0],
                     [-np.sin(t), 0, np.cos(t)]])


def _rot_x(deg):
    t = np.radians(deg)
    return np.array([[1, 0, 0],
                     [0, np.cos(t), -np.sin(t)],
                     [0, np.sin(t), np.cos(t)]])


def _matrix(rot):
    m = np.eye(4)
    m[:3, :3] = rot
    return m


# ── Decomposición de la matriz ────────────────────────────────────────

@pytest.mark.parametrize("angle", [-40, -22, -10, 0, 10, 22, 40])
def test_yaw_se_recupera_en_grados_reales(angle):
    r = FaceDetector._head_pose_from_matrix(_matrix(_rot_y(angle)))
    assert r["yaw"] == pytest.approx(angle, abs=0.1)
    assert r["pitch"] == pytest.approx(0.0, abs=0.1)


@pytest.mark.parametrize("angle", [-18, 0, 18])
def test_pitch_se_recupera_en_grados_reales(angle):
    r = FaceDetector._head_pose_from_matrix(_matrix(_rot_x(angle)))
    assert r["pitch"] == pytest.approx(angle, abs=0.1)
    assert r["yaw"] == pytest.approx(0.0, abs=0.1)


def test_rotacion_combinada_recupera_ambos():
    r = FaceDetector._head_pose_from_matrix(_matrix(_rot_x(12) @ _rot_y(20)))
    assert r["yaw"] == pytest.approx(20, abs=1.0)
    assert r["pitch"] == pytest.approx(12, abs=1.0)


def test_identidad_es_frontal():
    r = FaceDetector._head_pose_from_matrix(np.eye(4))
    assert r["yaw"] == pytest.approx(0.0, abs=0.01)
    assert r["pitch"] == pytest.approx(0.0, abs=0.01)
    assert r["roll"] == pytest.approx(0.0, abs=0.01)


def test_matriz_con_forma_invalida_no_rompe():
    assert FaceDetector._head_pose_from_matrix(np.eye(3)) is None


# ── Selección de fuente ───────────────────────────────────────────────

def _landmarks_para_giro(theta_deg, nose_depth=0.02, eye_half_span=0.045):
    """
    Landmarks proyectados para un giro REAL de theta grados.

    La nariz sobresale `nose_depth` del plano de los ojos, así que al girar se
    desplaza `nose_depth·sen θ` mientras el ancho entre ojos se encoge por
    `cos θ`. Es la geometría de la que depende el heurístico.
    """
    t = np.radians(theta_deg)
    pts = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(478)]
    pts[1] = {"x": 0.5 + nose_depth * np.sin(t), "y": 0.50, "z": 0.0}   # nariz
    pts[33] = {"x": 0.5 - eye_half_span * np.cos(t), "y": 0.45, "z": 0.0}
    pts[263] = {"x": 0.5 + eye_half_span * np.cos(t), "y": 0.45, "z": 0.0}
    pts[10] = {"x": 0.5, "y": 0.30, "z": 0.0}    # frente
    pts[152] = {"x": 0.5, "y": 0.70, "z": 0.0}   # barbilla
    return pts


def _detection(theta_deg=0.0, head_pose=None):
    return {
        "landmarks": _landmarks_para_giro(theta_deg),
        "frame_shape": (480, 640),
        "head_pose": head_pose,
    }


def test_prefiere_la_matriz_sobre_el_heuristico():
    d = _detection(theta_deg=0.0, head_pose={"yaw": 22.0, "pitch": -3.0, "roll": 0.0})
    yaw, pitch = ws._head_pose(d)
    assert yaw == 22.0
    assert pitch == -3.0


def test_cae_al_heuristico_si_no_hay_matriz():
    d = _detection(theta_deg=30.0, head_pose=None)
    yaw, _ = ws._head_pose(d)
    assert yaw != 0.0  # produce algo, aunque su escala no sean grados


# ── La regresión que motivó el cambio ─────────────────────────────────

def test_el_heuristico_subestima_muchisimo_el_giro_real():
    """
    Con la geometría de un giro real de 22°, el heurístico reporta un número
    muy por debajo de 22: por eso el ancla de la calibración era inalcanzable
    sin girar muchísimo más de lo pedido.
    """
    yaw, _ = ws._head_pose(_detection(theta_deg=22.0, head_pose=None))
    assert yaw < 10.0


def test_el_giro_real_pedido_no_alcanzaba_el_ancla_con_el_heuristico():
    yaw, pitch = ws._head_pose(_detection(theta_deg=22.0, head_pose=None))
    assert not pose_matches("right", yaw, pitch)


def test_con_la_matriz_el_giro_pedido_si_cuenta():
    target_yaw, target_pitch = POSE_ANCHORS["right"]
    d = _detection(head_pose={"yaw": target_yaw, "pitch": target_pitch, "roll": 0.0})
    yaw, pitch = ws._head_pose(d)
    assert pose_matches("right", yaw, pitch)


def test_giro_comodo_de_25_grados_ya_cuenta_como_la_pose():
    # Con grados reales basta acercarse: 25° cae dentro de la tolerancia de 12.
    # Girar a la derecha de la persona es yaw negativo (imagen sin espejo).
    d = _detection(head_pose={"yaw": -25.0, "pitch": 0.0, "roll": 0.0})
    yaw, pitch = ws._head_pose(d)
    assert pose_matches("right", yaw, pitch)


def test_seguir_de_frente_no_cuenta_como_giro():
    d = _detection(head_pose={"yaw": 2.0, "pitch": 0.0, "roll": 0.0})
    yaw, pitch = ws._head_pose(d)
    assert not pose_matches("right", yaw, pitch)
    assert pose_matches("center", yaw, pitch)


# ── Convención física: lo que dice la pantalla vs. el signo del yaw ──

def test_girar_a_tu_derecha_cuenta_como_la_pose_derecha():
    """
    La cámara no está en espejo: si la persona gira a SU derecha, la nariz se
    mueve hacia la izquierda de la imagen (x disminuye). Eso debe alinear la
    pose "right", que es la que la pantalla pide con "Gira la cabeza a tu
    derecha". Antes el ancla tenía el signo opuesto y nunca se alineaba.
    """
    # theta negativo en _landmarks_para_giro = nariz hacia la izquierda del encuadre
    yaw_heuristico, _ = ws._head_pose(_detection(theta_deg=-40.0, head_pose=None))
    assert yaw_heuristico < 0
    assert pose_matches("right", -22.0, 0.0)
    assert not pose_matches("left", -22.0, 0.0)


def test_girar_a_tu_izquierda_cuenta_como_la_pose_izquierda():
    yaw_heuristico, _ = ws._head_pose(_detection(theta_deg=40.0, head_pose=None))
    assert yaw_heuristico > 0
    assert pose_matches("left", 22.0, 0.0)
    assert not pose_matches("right", 22.0, 0.0)
