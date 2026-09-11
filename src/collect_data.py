"""
Interactive Sign Language Data Collection Script.
Collects webcam video sequences of sign language gestures and saves
the extracted MediaPipe keypoints as numpy arrays.

Usage:
    python src/collect_data.py                    # Collect all active signs
    python src/collect_data.py --signs hello yes  # Collect specific signs
    python src/collect_data.py --status           # Check collection progress
"""

import cv2
import numpy as np
import os
import sys
import time
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_PATH, ACTIVE_SIGNS, ALL_SIGNS, NO_SEQUENCES, SEQUENCE_LENGTH
from src.mediapipe_utils import (
    get_holistic_model, mediapipe_detection,
    draw_styled_landmarks, extract_keypoints, check_hands_visible
)


def check_progress(signs=None):
    """
    Check and display data collection progress.
    
    Args:
        signs: List of signs to check. If None, checks all signs.
    
    Returns:
        tuple: (collected_signs, missing_signs, partial_signs)
    """
    if signs is None:
        signs = ALL_SIGNS
    
    collected = []
    missing = []
    partial = []
    
    print("\n" + "=" * 60)
    print("📊  DATA COLLECTION PROGRESS")
    print("=" * 60)
    
    for sign in signs:
        sign_path = os.path.join(DATA_PATH, sign)
        if not os.path.exists(sign_path):
            missing.append(sign)
            print(f"  ❌  {sign:12s}  —  No data collected")
            continue
        
        # Count sequences that have all frames
        complete_sequences = 0
        total_sequences = 0
        for seq_dir in sorted(os.listdir(sign_path)):
            seq_path = os.path.join(sign_path, seq_dir)
            if os.path.isdir(seq_path):
                total_sequences += 1
                frame_count = len([f for f in os.listdir(seq_path) if f.endswith('.npy')])
                if frame_count >= SEQUENCE_LENGTH:
                    complete_sequences += 1
        
        if complete_sequences >= NO_SEQUENCES:
            collected.append(sign)
            print(f"  ✅  {sign:12s}  —  {complete_sequences}/{NO_SEQUENCES} sequences")
        elif complete_sequences > 0:
            partial.append(sign)
            print(f"  🔶  {sign:12s}  —  {complete_sequences}/{NO_SEQUENCES} sequences (incomplete)")
        else:
            missing.append(sign)
            print(f"  ❌  {sign:12s}  —  No complete sequences")
    
    print("=" * 60)
    print(f"  ✅ Collected: {len(collected)}  |  🔶 Partial: {len(partial)}  |  ❌ Missing: {len(missing)}")
    print("=" * 60 + "\n")
    
    return collected, missing, partial


def create_folders(signs):
    """Create folder structure for data collection."""
    for sign in signs:
        for seq in range(NO_SEQUENCES):
            seq_path = os.path.join(DATA_PATH, sign, str(seq))
            os.makedirs(seq_path, exist_ok=True)
    print(f"✅ Created folders for {len(signs)} signs × {NO_SEQUENCES} sequences")


def collect_sign_data(signs_to_collect):
    """
    Interactively collect sign language data via webcam.
    
    For each sign, collects NO_SEQUENCES video sequences,
    each containing SEQUENCE_LENGTH frames of MediaPipe keypoints.
    """
    if not signs_to_collect:
        print("No signs to collect. Use --status to check progress.")
        return
    
    # Create folders
    create_folders(signs_to_collect)
    
    print(f"\n🎯 Collecting data for: {', '.join(signs_to_collect)}")
    print(f"   Sequences per sign: {NO_SEQUENCES}")
    print(f"   Frames per sequence: {SEQUENCE_LENGTH}")
    print(f"\n   Controls:")
    print(f"   ───────────────────────────────────")
    print(f"   SPACE  = Start collecting current sign")
    print(f"   S      = Skip current sign")
    print(f"   Q      = Quit and save progress")
    print(f"   ───────────────────────────────────\n")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam!")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    holistic = get_holistic_model()
    
    try:
        for sign_idx, sign in enumerate(signs_to_collect):
            print(f"\n📋 Sign {sign_idx + 1}/{len(signs_to_collect)}: '{sign}'")
            
            # Wait for user to be ready
            waiting = True
            while waiting:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)  # Mirror for natural interaction
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)
                
                # Draw instruction overlay
                overlay = image.copy()
                cv2.rectangle(overlay, (0, 0), (640, 80), (45, 45, 45), -1)
                cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
                
                cv2.putText(image, f"NEXT SIGN: '{sign.upper()}'",
                           (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(image, "Press SPACE when ready | S=Skip | Q=Quit",
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Show hand detection status
                hands_ok = check_hands_visible(results)
                status_color = (0, 255, 0) if hands_ok else (0, 0, 255)
                status_text = "Hands detected" if hands_ok else "No hands detected"
                cv2.putText(image, status_text, (20, 460),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                
                cv2.imshow('Sign Language Data Collection', image)
                
                key = cv2.waitKey(10) & 0xFF
                if key == ord(' '):
                    waiting = False
                elif key == ord('s'):
                    print(f"   ⏭️  Skipped '{sign}'")
                    waiting = False
                    continue
                elif key == ord('q'):
                    print("\n💾 Progress saved. You can resume later.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
            
            # Countdown before collection
            for countdown in range(3, 0, -1):
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    image, _ = mediapipe_detection(frame, holistic)
                    
                    overlay = image.copy()
                    cv2.rectangle(overlay, (0, 0), (640, 480), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
                    
                    cv2.putText(image, str(countdown), (280, 280),
                               cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 8)
                    cv2.putText(image, f"Get ready for '{sign}'",
                               (120, 380), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    cv2.imshow('Sign Language Data Collection', image)
                    cv2.waitKey(1000)
            
            # Collect sequences
            for sequence in range(NO_SEQUENCES):
                print(f"   📹 Collecting sequence {sequence + 1}/{NO_SEQUENCES}...", end='\r')
                
                for frame_num in range(SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame = cv2.flip(frame, 1)
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)
                    
                    # Draw recording indicator
                    overlay = image.copy()
                    cv2.rectangle(overlay, (0, 0), (640, 50), (45, 45, 45), -1)
                    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
                    
                    # Recording dot
                    cv2.circle(image, (20, 25), 8, (0, 0, 255), -1)
                    cv2.putText(image, f"REC  '{sign}'  Seq {sequence+1}/{NO_SEQUENCES}  Frame {frame_num+1}/{SEQUENCE_LENGTH}",
                               (40, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Progress bar
                    progress = int((frame_num / SEQUENCE_LENGTH) * 600)
                    cv2.rectangle(image, (20, 460), (620, 475), (60, 60, 60), -1)
                    cv2.rectangle(image, (20, 460), (20 + progress, 475), (0, 255, 128), -1)
                    
                    cv2.imshow('Sign Language Data Collection', image)
                    
                    # Extract and save keypoints
                    keypoints = extract_keypoints(results)
                    save_path = os.path.join(DATA_PATH, sign, str(sequence), str(frame_num))
                    np.save(save_path, keypoints)
                    
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        print("\n💾 Progress saved. You can resume later.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                
                # Brief pause between sequences
                time.sleep(0.3)
            
            print(f"   ✅ Completed '{sign}' — {NO_SEQUENCES} sequences collected     ")
    finally:
        holistic.close()
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n🎉 Data collection complete for: {', '.join(signs_to_collect)}")
    print("   Run 'python src/train_model.py' to train the model.")


def main():
    parser = argparse.ArgumentParser(description="Sign Language Data Collection")
    parser.add_argument('--signs', nargs='+', help='Specific signs to collect')
    parser.add_argument('--status', action='store_true', help='Check collection progress')
    parser.add_argument('--all', action='store_true', help='Collect all active signs')
    args = parser.parse_args()
    
    if args.status:
        check_progress(ACTIVE_SIGNS)
        return
    
    if args.signs:
        # Validate requested signs
        invalid = [s for s in args.signs if s not in ALL_SIGNS]
        if invalid:
            print(f"❌ Unknown signs: {', '.join(invalid)}")
            print(f"   Available: {', '.join(ALL_SIGNS)}")
            return
        signs_to_collect = args.signs
    elif args.all:
        signs_to_collect = ACTIVE_SIGNS
    else:
        # Default: collect signs that are missing or incomplete
        _, missing, partial = check_progress(ACTIVE_SIGNS)
        signs_to_collect = missing + partial
        if not signs_to_collect:
            print("✅ All active signs have been collected!")
            print("   Run 'python src/train_model.py' to train the model.")
            return
    
    collect_sign_data(signs_to_collect)


if __name__ == '__main__':
    main()
