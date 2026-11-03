import numpy as np
import pytest

from backend.app.services.action_units import ActionUnitAnalyzer

W, H = 640, 480

# Pixel positions of the landmarks the symmetry check reads, on an upright,
# perfectly symmetric face.
FACE = {
    33: (250, 200), 263: (390, 200),    # outer eye corners
    133: (295, 200), 362: (345, 200),   # inner eye corners (inter-ocular distance 50 px)
    61: (285, 300), 291: (355, 300),    # lip corners
    107: (300, 165), 336: (340, 165),   # brows
}


def _landmarks(roll_deg=0.0, raise_right_corner_px=0.0):
    pts = {i: np.array(p, dtype=float) for i, p in FACE.items()}
    pts[291][1] -= raise_right_corner_px
    center = np.array([320.0, 240.0])
    a = np.radians(roll_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    lm = np.zeros((468, 3), dtype=np.float32)
    for i, p in pts.items():
        x, y = rot @ (p - center) + center
        lm[i] = (x / W, y / H, 0.0)
    return lm


@pytest.mark.parametrize("roll", [0.0, 8.0, -12.0])
def test_tilted_symmetric_face_reads_as_symmetric(roll):
    # Head roll used to read as asymmetry, so the contempt check, which skips
    # faces at >= 0.85, let nearly every smile through as contempt.
    sym = ActionUnitAnalyzer()._compute_face_symmetry(_landmarks(roll_deg=roll), W, H)
    assert sym >= 0.95


def test_one_sided_lip_corner_reads_as_asymmetric():
    # A unilateral corner raise of 15 px (30% of the inter-ocular distance) is
    # the contempt cue and must fall below the 0.85 cut, tilted or not.
    an = ActionUnitAnalyzer()
    assert an._compute_face_symmetry(_landmarks(raise_right_corner_px=15), W, H) < 0.85
    assert an._compute_face_symmetry(_landmarks(roll_deg=10, raise_right_corner_px=15), W, H) < 0.85
