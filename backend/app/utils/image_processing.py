"""
EmotionLens — Image Processing Utilities

Preprocessing functions for frames before they enter the ML pipeline.
"""

import cv2
import numpy as np


def preprocess_face_for_cnn(
    face_img: np.ndarray,
    target_size: tuple[int, int] = (224, 224),
    grayscale: bool = False,
) -> np.ndarray:
    """
    Preprocess a cropped face image for CNN input.

    Args:
        face_img: Cropped face region (BGR format from OpenCV).
        target_size: Target dimensions (width, height).
        grayscale: If True, convert to single-channel grayscale.

    Returns:
        Preprocessed image as float32 numpy array, normalized to [0, 1].
    """
    if face_img is None or face_img.size == 0:
        raise ValueError("Empty face image provided")

    # Convert color space
    if grayscale:
        if len(face_img.shape) == 3:
            img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            img = face_img
    else:
        img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

    # Resize to target dimensions
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    # Add batch and channel dimensions for PyTorch
    if grayscale:
        # (H, W) → (1, 1, H, W)
        img = np.expand_dims(img, axis=(0, 1))
    else:
        # (H, W, C) → (1, C, H, W)  — PyTorch expects channels-first
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

    return img


def crop_face_from_landmarks(
    frame: np.ndarray,
    landmarks: list[dict],
    padding: float = 0.2,
) -> np.ndarray | None:
    """
    Crop the face region from a frame using MediaPipe landmarks.

    Args:
        frame: Full video frame (BGR).
        landmarks: List of landmark dicts with 'x', 'y' keys (normalized 0-1).
        padding: Extra padding around the face as a fraction of face size.

    Returns:
        Cropped face image, or None if landmarks are insufficient.
    """
    if not landmarks or len(landmarks) < 10:
        return None

    h, w = frame.shape[:2]

    # Get bounding box from landmarks
    xs = [lm["x"] * w for lm in landmarks]
    ys = [lm["y"] * h for lm in landmarks]

    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))

    # Add padding
    face_w = x_max - x_min
    face_h = y_max - y_min
    pad_x = int(face_w * padding)
    pad_y = int(face_h * padding)

    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(w, x_max + pad_x)
    y_max = min(h, y_max + pad_y)

    # Crop
    face_crop = frame[y_min:y_max, x_min:x_max]

    if face_crop.size == 0:
        return None

    return face_crop


def draw_face_mesh_overlay(
    frame: np.ndarray,
    landmarks: list[dict],
    color: tuple[int, int, int] = (0, 255, 200),
    thickness: int = 1,
    draw_points: bool = True,
    draw_connections: bool = False,
) -> np.ndarray:
    """
    Draw face mesh landmarks on a frame for the live preview overlay.

    Args:
        frame: Video frame (BGR).
        landmarks: MediaPipe landmarks with 'x', 'y' keys.
        color: BGR color for the overlay.
        thickness: Line/point thickness.
        draw_points: Whether to draw landmark points.
        draw_connections: Whether to draw connections between landmarks.

    Returns:
        Frame with overlay drawn on it.
    """
    overlay = frame.copy()
    h, w = overlay.shape[:2]

    if draw_points and landmarks:
        for lm in landmarks:
            x = int(lm["x"] * w)
            y = int(lm["y"] * h)
            cv2.circle(overlay, (x, y), 1, color, thickness)

    return overlay
