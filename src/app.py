"""
Flask Web Application for Real-time Sign Language Translation.
Provides a modern web interface with live webcam feed,
sign detection overlay, and sentence display.

Usage:
    python src/app.py
    Then open http://localhost:5000 in your browser.
"""

import cv2
import numpy as np
import os
import sys
import time
import json
import threading
from flask import Flask, render_template, Response, jsonify, request

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    MODEL_DIR, SEQUENCE_LENGTH, CONFIDENCE_THRESHOLD,
    DETECTION_COOLDOWN, SAVED_SENTENCES_PATH, ACTIVE_SIGNS, BASE_DIR
)
from src.mediapipe_utils import (
    get_holistic_model, mediapipe_detection,
    draw_styled_landmarks, extract_keypoints
)
from src.sentence_generator import SentenceGenerator

# ─── Flask App ───────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)

# ─── Global State ────────────────────────────────────────────────────────────
class DetectionState:
    """Thread-safe shared state between video stream and API endpoints."""
    def __init__(self):
        self.lock = threading.Lock()
        self.current_sign = ""
        self.confidence = 0.0
        self.current_sentence = ""
        self.sentence_history = []
        self.signs_list = []
        self.is_running = False
        self.fps = 0
        self.model = None
        self.label_map = {}
        self.idx_to_sign = {}
        self.sentence_gen = SentenceGenerator()
        self.detection_active = True
    
    def update(self, sign, confidence, sentence):
        with self.lock:
            self.current_sign = sign
            self.confidence = confidence
            self.current_sentence = sentence
            self.sentence_history = self.sentence_gen.get_history()
    
    def get_state(self):
        with self.lock:
            return {
                'sign': self.current_sign,
                'confidence': self.confidence,
                'sentence': self.current_sentence,
                'history': self.sentence_history[-10:],  # Last 10 sentences
                'is_running': self.is_running,
                'fps': self.fps,
                'detection_active': self.detection_active,
                'signs': self.signs_list
            }

state = DetectionState()


def load_model():
    """Load the trained LSTM model and label map."""
    import tensorflow as tf
    
    model_path = os.path.join(MODEL_DIR, 'sign_language_model.keras')
    label_map_path = os.path.join(MODEL_DIR, 'label_map.npy')
    
    if not os.path.exists(model_path):
        print("⚠️  No trained model found. Web UI will run in preview mode.")
        print("   To train: python src/collect_data.py && python src/train_model.py")
        return False
    
    print("🧠 Loading model...")
    state.model = tf.keras.models.load_model(model_path)
    
    if os.path.exists(label_map_path):
        state.label_map = np.load(label_map_path, allow_pickle=True).item()
    else:
        state.label_map = {sign: i for i, sign in enumerate(ACTIVE_SIGNS)}
    
    state.idx_to_sign = {v: k for k, v in state.label_map.items()}
    state.signs_list = [state.idx_to_sign[i] for i in sorted(state.idx_to_sign.keys())]
    
    print(f"✅ Model loaded with signs: {', '.join(state.signs_list)}")
    return True


def generate_frames():
    """
    Generator function that yields MJPEG frames from webcam.
    Runs MediaPipe detection and LSTM prediction on each frame.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    state.is_running = True
    sequence = []
    last_detection_time = 0
    last_detected_sign = ""
    
    # FPS tracking
    frame_count = 0
    fps_start_time = time.time()
    
    holistic = get_holistic_model()
    
    try:
        while state.is_running:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            
            # Detection logic
            if state.model is not None and state.detection_active:
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-SEQUENCE_LENGTH:]
                
                if len(sequence) == SEQUENCE_LENGTH:
                    input_data = np.expand_dims(sequence, axis=0)
                    prediction = state.model.predict(input_data, verbose=0)[0]
                    predicted_idx = np.argmax(prediction)
                    confidence = float(prediction[predicted_idx])
                    predicted_sign = state.idx_to_sign.get(predicted_idx, "?")
                    
                    current_time = time.time()
                    
                    if confidence >= CONFIDENCE_THRESHOLD:
                        if (predicted_sign != last_detected_sign or
                            current_time - last_detection_time > DETECTION_COOLDOWN):
                            sentence = state.sentence_gen.add_word(predicted_sign)
                            state.update(predicted_sign, confidence, sentence)
                            last_detected_sign = predicted_sign
                            last_detection_time = current_time
                    else:
                        state.update("", 0.0, state.sentence_gen.current_sentence)
                
                # Check timeout
                state.sentence_gen.check_timeout()
            
            # FPS
            frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                state.fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.time()
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    finally:
        holistic.close()
        cap.release()
        state.is_running = False


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main web page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """MJPEG video stream endpoint."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/state')
def get_state():
    """Get current detection state as JSON."""
    return jsonify(state.get_state())


@app.route('/api/clear', methods=['POST'])
def clear_sentence():
    """Clear the current sentence buffer."""
    state.sentence_gen.clear()
    state.update("", 0.0, "")
    return jsonify({'status': 'ok', 'message': 'Sentence cleared'})


@app.route('/api/save', methods=['POST'])
def save_sentences():
    """Save sentence history to file."""
    count = state.sentence_gen.save_sentences(SAVED_SENTENCES_PATH)
    return jsonify({'status': 'ok', 'message': f'Saved {count} sentences', 'count': count})


@app.route('/api/toggle_detection', methods=['POST'])
def toggle_detection():
    """Toggle detection on/off (pause/resume)."""
    state.detection_active = not state.detection_active
    return jsonify({
        'status': 'ok',
        'detection_active': state.detection_active
    })


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("🌐  SIGN LANGUAGE TRANSLATION — WEB UI")
    print("=" * 60)
    
    model_loaded = load_model()
    
    if not model_loaded:
        print("\n⚠️  Running without model — webcam preview only")
        print("   Collect data and train a model first:\n")
        print("   1. python src/collect_data.py")
        print("   2. python src/train_model.py")
        print("   3. python src/app.py\n")
    
    print("\n🚀 Starting server at http://localhost:8080")
    print("   Press Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)


if __name__ == '__main__':
    main()
