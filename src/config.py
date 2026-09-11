"""
Central configuration for the Sign Language Translation project.
All paths, hyperparameters, and vocabulary are defined here.
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'MP_Data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'Logs')
SAVED_SENTENCES_PATH = os.path.join(BASE_DIR, 'saved_sentences.txt')

# ─── Data Collection ────────────────────────────────────────────────────────
NO_SEQUENCES = 30          # Number of video sequences per sign
SEQUENCE_LENGTH = 30       # Number of frames per sequence

# ─── Sign Vocabulary ────────────────────────────────────────────────────────
# Phase 1: Basic signs (start here)
BASIC_SIGNS = ['hello', 'thanks', 'help', 'yes', 'no', 'please']

# Phase 2: Add subjects & verbs
INTERMEDIATE_SIGNS = BASIC_SIGNS + ['i', 'you', 'want', 'need', 'sorry']

# Phase 3: Full vocabulary
ALL_SIGNS = ['i', 'you', 'we', 'want', 'need', 'help', 'food', 'water',
             'cat', 'dog', 'hello', 'thanks', 'please', 'sorry', 'yes', 'no']

# Active vocabulary — change this as you progress
ACTIVE_SIGNS = BASIC_SIGNS

# ─── Word Categories (for sentence generation) ──────────────────────────────
WORD_CATEGORIES = {
    'subject':     ['i', 'you', 'we'],
    'verb':        ['want', 'need'],
    'object':      ['help', 'food', 'water', 'cat', 'dog'],
    'greeting':    ['hello', 'thanks'],
    'modifier':    ['please', 'sorry'],
    'affirmation': ['yes', 'no'],
}

# ─── Model Hyperparameters ──────────────────────────────────────────────────
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 128
LSTM_UNITS_3 = 64
DENSE_UNITS_1 = 64
DENSE_UNITS_2 = 32
DROPOUT_RATE = 0.3
EPOCHS = 200
BATCH_SIZE = 16

# ─── Detection Thresholds ───────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.7     # Minimum confidence to accept a prediction
DETECTION_COOLDOWN = 1.0       # Seconds between accepting the same sign
SENTENCE_TIMEOUT = 3.0         # Seconds of inactivity before sentence resets

# ─── MediaPipe Keypoint Dimensions ──────────────────────────────────────────
POSE_LANDMARKS = 33 * 4       # 33 landmarks × (x, y, z, visibility) = 132
FACE_LANDMARKS = 468 * 3      # 468 landmarks × (x, y, z) = 1404
HAND_LANDMARKS = 21 * 3       # 21 landmarks × (x, y, z) = 63 (per hand)
TOTAL_KEYPOINTS = POSE_LANDMARKS + FACE_LANDMARKS + (HAND_LANDMARKS * 2)  # 1662

# ─── Ensure directories exist ───────────────────────────────────────────────
for d in [DATA_PATH, MODEL_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)
