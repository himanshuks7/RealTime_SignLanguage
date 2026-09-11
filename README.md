# 🤟 Real-Time Sign Language Translator

> AI-powered sign language detection that converts gestures into grammatically correct English sentences in real time.

Built with **TensorFlow LSTM**, **MediaPipe**, **OpenCV**, and a **Flask web UI**.

---

## ✨ Features

- **Real-time Detection** — Uses your webcam to detect sign language gestures instantly
- **Sentence Generation** — Combines individual signs into grammatically correct sentences (SVO structure)
- **16 Sign Vocabulary** — Subjects, verbs, objects, greetings, modifiers, and affirmations
- **Web Interface** — Beautiful dark-themed Flask UI with live video feed
- **Confidence Thresholding** — Reduces false positives with adjustable confidence levels
- **Progressive Learning** — Start with 6 basic signs, expand to 16 as you improve
- **Save & Export** — Save generated sentences to a text file

## 📋 Sign Vocabulary

| Category | Signs |
|----------|-------|
| **Subjects** | I, You, We |
| **Verbs** | Want, Need |
| **Objects** | Help, Food, Water, Cat, Dog |
| **Greetings** | Hello, Thanks |
| **Modifiers** | Please, Sorry |
| **Affirmations** | Yes, No |

**Example sentences:** "I want food.", "You need help please.", "Hello!", "Thanks!"

---

## 🚀 Quick Start

### 1. Set Up Environment (macOS with Apple Silicon)

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Collect Sign Data

```bash
# Check status
python src/collect_data.py --status

# Collect basic signs (recommended starting set)
python src/collect_data.py --signs hello thanks help yes no please

# Collect all active signs
python src/collect_data.py --all
```

**Tips for good data:**
- Use bright, even lighting
- Plain background
- Perform each sign 30 times with slight variations
- Keep both hands visible

### 3. Train the Model

```bash
python src/train_model.py
```

### 4. Run the Web UI

```bash
python src/app.py
# Open http://localhost:5000
```

Or use the OpenCV-based detector:
```bash
python src/realtime_detection.py
```

---

## 📁 Project Structure

```
RealTime_SignLanguage/
├── src/
│   ├── config.py              # Central configuration
│   ├── mediapipe_utils.py     # MediaPipe helper functions
│   ├── collect_data.py        # Interactive data collection
│   ├── train_model.py         # LSTM model training
│   ├── sentence_generator.py  # Grammar-aware sentence builder
│   ├── realtime_detection.py  # OpenCV-based detection
│   └── app.py                 # Flask web application
├── templates/
│   └── index.html             # Web UI template
├── static/
│   ├── css/style.css          # Premium dark theme
│   └── js/app.js              # Frontend logic
├── models/                    # Trained model files
├── MP_Data/                   # Collected keypoint data
├── Logs/                      # TensorBoard logs
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Configuration

Edit `src/config.py` to customize:

- **`ACTIVE_SIGNS`** — Which signs to use (start with `BASIC_SIGNS`)
- **`CONFIDENCE_THRESHOLD`** — Min confidence to accept detection (default: 0.7)
- **`DETECTION_COOLDOWN`** — Seconds between same-sign detections (default: 1.0)
- **`EPOCHS`** — Training epochs (default: 200)

---

## ⌨️ Web UI Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `C` | Clear current sentence |
| `S` | Save sentences to file |
| `P` | Pause/Resume detection |

---

## 📊 Progressive Training Schedule

| Phase | Signs | Time |
|-------|-------|------|
| **Phase 1** | hello, thanks, help, yes, no, please | ~45 min |
| **Phase 2** | + i, you, want, need, sorry | ~40 min |
| **Phase 3** | + we, food, water, cat, dog | ~40 min |

---

## 🙏 Credits

Inspired by [Quasar-025/Realtime_SLT](https://github.com/Quasar-025/Realtime_SLT). Rebuilt with a modular architecture, Flask web UI, and several improvements.

## 📄 License

MIT License
