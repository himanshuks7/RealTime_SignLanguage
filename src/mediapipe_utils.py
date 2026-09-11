"""
MediaPipe utility functions for sign language detection.
Uses the new MediaPipe Tasks API (HolisticLandmarker) for
pose, face, and hand landmark detection.

On first run, automatically downloads the required model file.
"""

import cv2
import numpy as np
import os
import urllib.request

# MediaPipe Tasks API imports
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HolisticLandmarker,
    HolisticLandmarkerOptions,
    RunningMode,
    drawing_utils,
    drawing_styles,
)
from mediapipe.tasks.python.vision import (
    HandLandmarksConnections,
    PoseLandmarksConnections,
    FaceLandmarksConnections,
)

# ─── Model Download ─────────────────────────────────────────────────────────
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'holistic_landmarker.task')


def ensure_model_downloaded():
    """Download the HolisticLandmarker model if not present."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading HolisticLandmarker model (~15MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"✅ Model saved to: {MODEL_PATH}")
    return MODEL_PATH


def get_holistic_model():
    """
    Create and return a HolisticLandmarker instance using Tasks API.
    Returns a context manager compatible object.
    """
    model_path = ensure_model_downloaded()

    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        output_face_blendshapes=False,
        output_segmentation_masks=False,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
    return HolisticLandmarker.create_from_options(options)


class HolisticResults:
    """
    Adapter that wraps HolisticLandmarkerResult to provide a
    consistent interface similar to the old solutions API.
    """
    def __init__(self, result):
        self._result = result

    @property
    def pose_landmarks(self):
        """Return pose landmarks or None."""
        lm = self._result.pose_landmarks
        if lm and len(lm) > 0:
            return lm[0]  # List of NormalizedLandmark
        return None

    @property
    def face_landmarks(self):
        """Return face landmarks or None."""
        lm = self._result.face_landmarks
        if lm and len(lm) > 0:
            return lm[0]
        return None

    @property
    def left_hand_landmarks(self):
        """Return left hand landmarks or None."""
        lm = self._result.left_hand_landmarks
        if lm and len(lm) > 0:
            return lm[0]
        return None

    @property
    def right_hand_landmarks(self):
        """Return right hand landmarks or None."""
        lm = self._result.right_hand_landmarks
        if lm and len(lm) > 0:
            return lm[0]
        return None


def mediapipe_detection(image, landmarker):
    """
    Process a BGR frame through MediaPipe HolisticLandmarker.

    Args:
        image: BGR image from OpenCV
        landmarker: HolisticLandmarker instance

    Returns:
        tuple: (original BGR image, HolisticResults wrapper)
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    result = landmarker.detect(mp_image)
    return image, HolisticResults(result)


def draw_styled_landmarks(image, results):
    """
    Draw styled MediaPipe landmarks on the image.
    Uses the Tasks API drawing utilities.
    """
    h, w, _ = image.shape

    # Draw pose landmarks
    if results.pose_landmarks:
        _draw_landmarks_list(
            image, results.pose_landmarks, w, h,
            PoseLandmarksConnections.POSE_LANDMARKS,
            drawing_utils.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            drawing_utils.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
        )

    # Draw face landmarks (simplified — just dots, no mesh)
    if results.face_landmarks:
        for lm in results.face_landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            cv2.circle(image, (px, py), 1, (80, 256, 121), -1)

    # Draw left hand landmarks
    if results.left_hand_landmarks:
        _draw_landmarks_list(
            image, results.left_hand_landmarks, w, h,
            HandLandmarksConnections.HAND_CONNECTIONS,
            drawing_utils.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            drawing_utils.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
        )

    # Draw right hand landmarks
    if results.right_hand_landmarks:
        _draw_landmarks_list(
            image, results.right_hand_landmarks, w, h,
            HandLandmarksConnections.HAND_CONNECTIONS,
            drawing_utils.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            drawing_utils.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )


def _draw_landmarks_list(image, landmarks, width, height, connections,
                         landmark_spec, connection_spec):
    """
    Draw landmarks and connections manually from a list of NormalizedLandmark.
    """
    points = []
    for lm in landmarks:
        px = int(lm.x * width)
        py = int(lm.y * height)
        points.append((px, py))
        cv2.circle(image, (px, py), landmark_spec.circle_radius,
                   landmark_spec.color, -1)

    # Draw connections
    if connections:
        for connection in connections:
            start_idx = connection.start
            end_idx = connection.end
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(image, points[start_idx], points[end_idx],
                         connection_spec.color, connection_spec.thickness)


def extract_keypoints(results):
    """
    Extract and flatten all keypoints from HolisticResults.

    Returns a 1D numpy array of shape (1662,):
        - Pose: 33 landmarks × 4 values (x, y, z, visibility) = 132
        - Face: 468 landmarks × 3 values (x, y, z) = 1404
        - Left hand: 21 landmarks × 3 values (x, y, z) = 63
        - Right hand: 21 landmarks × 3 values (x, y, z) = 63
    """
    # Pose landmarks (33 × 4)
    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, 'visibility') else 0.0]
             for lm in results.pose_landmarks]
        ).flatten()
    else:
        pose = np.zeros(33 * 4)

    # Face landmarks (468 × 3)
    if results.face_landmarks:
        face_lm = results.face_landmarks
        # MediaPipe face has 478 landmarks in Tasks API, but we use first 468 for compatibility
        face_list = list(face_lm)[:468]
        face = np.array(
            [[lm.x, lm.y, lm.z] for lm in face_list]
        ).flatten()
        # Pad if fewer than 468
        if len(face) < 468 * 3:
            face = np.pad(face, (0, 468 * 3 - len(face)))
    else:
        face = np.zeros(468 * 3)

    # Left hand landmarks (21 × 3)
    if results.left_hand_landmarks:
        lh = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks]
        ).flatten()
    else:
        lh = np.zeros(21 * 3)

    # Right hand landmarks (21 × 3)
    if results.right_hand_landmarks:
        rh = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks]
        ).flatten()
    else:
        rh = np.zeros(21 * 3)

    return np.concatenate([pose, face, lh, rh])


def check_hands_visible(results):
    """Check if at least one hand is detected in the frame."""
    return (results.left_hand_landmarks is not None or
            results.right_hand_landmarks is not None)


def check_body_visible(results):
    """Check if the pose (body) is detected in the frame."""
    return results.pose_landmarks is not None
