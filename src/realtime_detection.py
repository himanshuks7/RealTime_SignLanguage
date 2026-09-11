"""
Real-time Sign Language Detection using OpenCV.
Uses the trained LSTM model to detect signs from webcam feed
and generates sentences using the SentenceGenerator.

Usage:
    python src/realtime_detection.py
"""

import cv2
import numpy as np
import os
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    MODEL_DIR, SEQUENCE_LENGTH, CONFIDENCE_THRESHOLD,
    DETECTION_COOLDOWN, SAVED_SENTENCES_PATH, ACTIVE_SIGNS
)
from src.mediapipe_utils import (
    get_holistic_model, mediapipe_detection,
    draw_styled_landmarks, extract_keypoints
)
from src.sentence_generator import SentenceGenerator


def draw_overlay(image, sign, confidence, sentence, sentence_history, fps):
    """
    Draw a styled, semi-transparent overlay on the frame.
    Shows current detection, confidence, sentence, and controls.
    """
    h, w = image.shape[:2]
    
    # ── Top bar: current detection ──────────────────────────────────────
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)
    
    if sign:
        # Sign name
        cv2.putText(image, f"Sign: {sign.upper()}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 200), 2)
        
        # Confidence bar
        bar_width = int(confidence * 250)
        bar_color = (0, 255, 128) if confidence > 0.85 else (0, 200, 255) if confidence > 0.7 else (0, 100, 255)
        cv2.rectangle(image, (320, 15), (570, 40), (60, 60, 60), -1)
        cv2.rectangle(image, (320, 15), (320 + bar_width, 40), bar_color, -1)
        cv2.putText(image, f"{confidence:.0%}", (580, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    else:
        cv2.putText(image, "Waiting for sign...", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
    
    # FPS
    cv2.putText(image, f"FPS: {fps:.0f}", (w - 100, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    # ── Sentence bar ────────────────────────────────────────────────────
    overlay2 = image.copy()
    cv2.rectangle(overlay2, (0, 50), (w, 90), (40, 40, 80), -1)
    cv2.addWeighted(overlay2, 0.6, image, 0.4, 0, image)
    
    sentence_display = sentence if sentence else "..."
    cv2.putText(image, f"Sentence: {sentence_display}", (20, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # ── Bottom: controls + history ──────────────────────────────────────
    overlay3 = image.copy()
    cv2.rectangle(overlay3, (0, h - 80), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay3, 0.75, image, 0.25, 0, image)
    
    cv2.putText(image, "Q=Quit  C=Clear  S=Save sentence",
                (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    
    # Show last sentence from history
    if sentence_history:
        last = sentence_history[-1]
        if len(last) > 60:
            last = last[:57] + "..."
        cv2.putText(image, f"Last: {last}",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)
    
    return image


def run_detection():
    """Main real-time detection loop."""
    import tensorflow as tf
    
    # Load model
    model_path = os.path.join(MODEL_DIR, 'sign_language_model.keras')
    label_map_path = os.path.join(MODEL_DIR, 'label_map.npy')
    
    if not os.path.exists(model_path):
        print("❌ No trained model found!")
        print("   Steps to get started:")
        print("   1. Collect data: python src/collect_data.py")
        print("   2. Train model:  python src/train_model.py")
        print("   3. Run detection: python src/realtime_detection.py")
        return
    
    print("🧠 Loading model...")
    model = tf.keras.models.load_model(model_path)
    
    # Load label map
    if os.path.exists(label_map_path):
        label_map = np.load(label_map_path, allow_pickle=True).item()
    else:
        label_map = {sign: i for i, sign in enumerate(ACTIVE_SIGNS)}
    
    idx_to_sign = {v: k for k, v in label_map.items()}
    signs = [idx_to_sign[i] for i in sorted(idx_to_sign.keys())]
    
    print(f"📋 Loaded signs: {', '.join(signs)}")
    
    # Initialize components
    sentence_gen = SentenceGenerator()
    sequence = []
    current_sign = ""
    current_confidence = 0.0
    last_detection_time = 0
    last_detected_sign = ""
    
    # FPS tracking
    fps = 0
    frame_count = 0
    fps_start_time = time.time()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("🎥 Starting real-time detection...")
    print("   Controls: Q=Quit, C=Clear, S=Save")
    
    holistic = get_holistic_model()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            
            # Extract keypoints and build sequence
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-SEQUENCE_LENGTH:]  # Keep last N frames
            
            # Predict when we have enough frames
            if len(sequence) == SEQUENCE_LENGTH:
                input_data = np.expand_dims(sequence, axis=0)
                prediction = model.predict(input_data, verbose=0)[0]
                predicted_idx = np.argmax(prediction)
                confidence = prediction[predicted_idx]
                predicted_sign = idx_to_sign.get(predicted_idx, "?")
                
                current_time = time.time()
                
                # Apply confidence threshold and cooldown
                if confidence >= CONFIDENCE_THRESHOLD:
                    if (predicted_sign != last_detected_sign or 
                        current_time - last_detection_time > DETECTION_COOLDOWN):
                        current_sign = predicted_sign
                        current_confidence = confidence
                        sentence_gen.add_word(predicted_sign)
                        last_detected_sign = predicted_sign
                        last_detection_time = current_time
                else:
                    current_sign = ""
                    current_confidence = 0.0
            
            # Check sentence timeout
            sentence_gen.check_timeout()
            
            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.time()
            
            # Draw overlay
            image = draw_overlay(
                image, current_sign, current_confidence,
                sentence_gen.current_sentence,
                sentence_gen.get_history(), fps
            )
            
            cv2.imshow('Sign Language Detection', image)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                sentence_gen.clear()
                current_sign = ""
                print("🗑️  Sentence cleared")
            elif key == ord('s'):
                count = sentence_gen.save_sentences(SAVED_SENTENCES_PATH)
                print(f"💾 Saved {count} sentences to {SAVED_SENTENCES_PATH}")
    finally:
        holistic.close()
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Detection stopped.")


if __name__ == '__main__':
    run_detection()
