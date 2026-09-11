"""
LSTM Model Training Script for Sign Language Detection.
Loads collected keypoint data, trains an LSTM model, and saves it.

Usage:
    python src/train_model.py                    # Train with active signs
    python src/train_model.py --epochs 300       # Custom epochs
    python src/train_model.py --evaluate         # Evaluate existing model
"""

import os
import sys
import argparse
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    DATA_PATH, MODEL_DIR, LOG_DIR, ACTIVE_SIGNS, ALL_SIGNS,
    NO_SEQUENCES, SEQUENCE_LENGTH, TOTAL_KEYPOINTS,
    LSTM_UNITS_1, LSTM_UNITS_2, LSTM_UNITS_3,
    DENSE_UNITS_1, DENSE_UNITS_2, DROPOUT_RATE,
    EPOCHS, BATCH_SIZE
)


def load_data(signs):
    """
    Load collected keypoint data and create training arrays.
    
    Args:
        signs: List of sign names to load data for.
    
    Returns:
        tuple: (X, y, label_map) where X is input sequences and y is labels.
    """
    label_map = {sign: idx for idx, sign in enumerate(signs)}
    
    sequences = []
    labels = []
    skipped = []
    
    print("\n📂 Loading training data...")
    
    for sign in signs:
        sign_path = os.path.join(DATA_PATH, sign)
        if not os.path.exists(sign_path):
            skipped.append(sign)
            print(f"  ⚠️  {sign}: No data found, skipping")
            continue
        
        loaded_count = 0
        for seq in range(NO_SEQUENCES):
            window = []
            seq_path = os.path.join(sign_path, str(seq))
            
            if not os.path.exists(seq_path):
                continue
            
            valid = True
            for frame_num in range(SEQUENCE_LENGTH):
                frame_path = os.path.join(seq_path, f"{frame_num}.npy")
                if os.path.exists(frame_path):
                    keypoints = np.load(frame_path)
                    window.append(keypoints)
                else:
                    valid = False
                    break
            
            if valid and len(window) == SEQUENCE_LENGTH:
                sequences.append(window)
                labels.append(label_map[sign])
                loaded_count += 1
        
        print(f"  ✅  {sign}: {loaded_count} sequences loaded")
    
    if not sequences:
        print("\n❌ No training data found!")
        print("   Run 'python src/collect_data.py' first to collect sign data.")
        return None, None, None
    
    X = np.array(sequences)
    y = np.array(labels)
    
    # Remove skipped signs from label map
    active_label_map = {s: i for s, i in label_map.items() if s not in skipped}
    
    print(f"\n📊 Dataset: {X.shape[0]} sequences, {len(active_label_map)} signs")
    print(f"   Input shape: {X.shape}")
    
    return X, y, active_label_map


def build_model(num_classes, input_shape=(SEQUENCE_LENGTH, TOTAL_KEYPOINTS)):
    """
    Build and compile the LSTM model architecture.
    
    Architecture:
        LSTM(64) → LSTM(128) → LSTM(64) → Dense(64) → Dense(32) → Dense(num_classes)
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    
    model = Sequential([
        # First LSTM layer — returns sequences for stacking
        LSTM(LSTM_UNITS_1, return_sequences=True, activation='relu',
             input_shape=input_shape),
        Dropout(DROPOUT_RATE),
        
        # Second LSTM layer — deeper feature extraction
        LSTM(LSTM_UNITS_2, return_sequences=True, activation='relu'),
        Dropout(DROPOUT_RATE),
        
        # Third LSTM layer — final temporal features
        LSTM(LSTM_UNITS_3, return_sequences=False, activation='relu'),
        Dropout(DROPOUT_RATE),
        
        # Dense classification head
        Dense(DENSE_UNITS_1, activation='relu'),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),
        
        Dense(DENSE_UNITS_2, activation='relu'),
        BatchNormalization(),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='Adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    
    return model


def train(signs=None, epochs=None):
    """
    Full training pipeline: load data → build model → train → save.
    """
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
    from sklearn.model_selection import train_test_split
    
    if signs is None:
        signs = ACTIVE_SIGNS
    if epochs is None:
        epochs = EPOCHS
    
    print("=" * 60)
    print("🧠  SIGN LANGUAGE MODEL TRAINING")
    print("=" * 60)
    print(f"   Signs: {', '.join(signs)}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {BATCH_SIZE}")
    
    # Load data
    X, y, label_map = load_data(signs)
    if X is None:
        return
    
    # One-hot encode labels
    num_classes = len(set(y))
    y_cat = to_categorical(y, num_classes=num_classes)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.15, random_state=42, stratify=y
    )
    
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Test:  {X_test.shape[0]} samples")
    print(f"   Classes: {num_classes}")
    
    # Build model
    model = build_model(num_classes)
    model.summary()
    
    # Callbacks
    tb_callback = TensorBoard(log_dir=LOG_DIR)
    early_stop = EarlyStopping(
        monitor='val_categorical_accuracy',
        patience=30,
        restore_best_weights=True,
        verbose=1
    )
    
    model_path = os.path.join(MODEL_DIR, 'sign_language_model.keras')
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_categorical_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Train
    print("\n🚀 Training started...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=[tb_callback, early_stop, checkpoint],
        verbose=1
    )
    
    # Save final model
    model.save(model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Save label map
    label_map_path = os.path.join(MODEL_DIR, 'label_map.npy')
    np.save(label_map_path, label_map)
    print(f"💾 Label map saved to: {label_map_path}")
    
    # Evaluate
    evaluate_model(model, X_test, y_test, label_map)
    
    return model, history, label_map


def evaluate_model(model, X_test, y_test, label_map):
    """Evaluate model and print metrics."""
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\n" + "=" * 60)
    print("📈  EVALUATION RESULTS")
    print("=" * 60)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Reverse label map
    idx_to_sign = {v: k for k, v in label_map.items()}
    target_names = [idx_to_sign[i] for i in sorted(idx_to_sign.keys())]
    
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=target_names))
    
    # Overall accuracy
    accuracy = np.mean(y_pred_classes == y_true_classes)
    print(f"\n🎯 Overall Accuracy: {accuracy:.1%}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Pretty print confusion matrix
    header = "        " + "  ".join(f"{name[:6]:>6}" for name in target_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{val:>6}" for val in row)
        print(f"  {target_names[i][:6]:>6}  {row_str}")


def main():
    parser = argparse.ArgumentParser(description="Train Sign Language LSTM Model")
    parser.add_argument('--signs', nargs='+', help='Signs to train on')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Training epochs')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate existing model')
    args = parser.parse_args()
    
    signs = args.signs if args.signs else ACTIVE_SIGNS
    
    # Validate signs
    invalid = [s for s in signs if s not in ALL_SIGNS]
    if invalid:
        print(f"❌ Unknown signs: {', '.join(invalid)}")
        return
    
    if args.evaluate:
        import tensorflow as tf
        model_path = os.path.join(MODEL_DIR, 'sign_language_model.keras')
        if not os.path.exists(model_path):
            print("❌ No trained model found. Train first with: python src/train_model.py")
            return
        model = tf.keras.models.load_model(model_path)
        X, y, label_map = load_data(signs)
        if X is not None:
            from tensorflow.keras.utils import to_categorical
            y_cat = to_categorical(y, num_classes=len(label_map))
            evaluate_model(model, X, y_cat, label_map)
    else:
        train(signs=signs, epochs=args.epochs)


if __name__ == '__main__':
    main()
