# 🏀 Basketball Player Detector

A real-time basketball detection system powered by **YOLOv8** that detects players, tracks them across frames, classifies them by team using jersey colors, and detects the ball — all in one pipeline.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧍 Player Detection | Bounding boxes on every player using YOLOv8 |
| 🏀 Ball Detection | Detects the basketball (COCO class 32) |
| 🎽 Team Classification | Auto-classifies players by jersey color (K-Means clustering) |
| 🔢 Player Tracking | Consistent IDs across frames using IoU tracking |
| 🖼️ Image Support | Run on single images |
| 🎬 Video Support | Process full video files |
| 📷 Webcam Support | Real-time live detection |

---

## 🗂️ Project Structure

```
basketball-detector/
├── src/
│   ├── detector.py          # Main detection pipeline
│   ├── team_classifier.py   # Jersey color → team assignment
│   ├── tracker.py           # IoU-based multi-object tracker
│   └── utils.py             # Drawing, video I/O helpers
├── output/
│   ├── images/              # Detected image outputs
│   └── videos/              # Detected video outputs
├── notebooks/
│   └── demo.ipynb           # Jupyter demo notebook
├── tests/
│   └── test_tracker.py      # Unit tests
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/basketball-detector.git
cd basketball-detector
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> YOLOv8 weights (`yolov8n.pt`) are downloaded automatically on first run.

---

## 🚀 Usage

### Detect on an image
```bash
python src/detector.py --source path/to/image.jpg --output output/images/
```

### Detect on a video
```bash
python src/detector.py --source path/to/game.mp4 --output output/videos/result.mp4
```

### Live webcam detection
```bash
python src/detector.py --source webcam
```

### All options
```
--source         Input: image path, video path, or 'webcam'  (required)
--output         Output path or directory
--model          YOLOv8 model variant (default: yolov8n.pt)
--conf           Confidence threshold 0–1 (default: 0.4)
--device         cpu | cuda | mps (default: cpu)
--no-tracking    Disable player tracking
--no-teams       Disable team classification
--preview        Show live preview window (video mode)
--max-frames     Limit frames processed (video mode)
```

### Use a larger model for better accuracy
```bash
python src/detector.py --source game.mp4 --model yolov8m.pt --device cuda
```

---

## 🏷️ Model Options

| Model | Size | Speed | Accuracy |
|---|---|---|---|
| `yolov8n.pt` | 6 MB | ⚡ Fastest | Good |
| `yolov8s.pt` | 22 MB | Fast | Better |
| `yolov8m.pt` | 52 MB | Medium | Great |
| `yolov8l.pt` | 87 MB | Slower | Excellent |
| `yolov8x.pt` | 131 MB | Slowest | Best |

> For real-time use, `yolov8n` or `yolov8s` are recommended.

---

## 🎽 How Team Classification Works

1. For each detected player, the upper torso region is cropped (jersey area)
2. K-Means clustering extracts the dominant jersey color in HSV space
3. Over the first few frames, color samples are collected
4. Players are clustered into **Team A** and **Team B** automatically
5. Team labels and colors persist consistently across frames

> No manual color configuration needed — it adapts to any teams automatically.

---

## 🔍 How Tracking Works

A lightweight **IoU-based tracker** assigns consistent IDs:
- Detections are matched to existing tracks using Intersection over Union
- Tracks persist for up to 30 frames without a match before being dropped
- New players are assigned new IDs automatically

For production-grade tracking, consider enabling [ByteTrack](https://github.com/ifzhang/ByteTrack) via `ultralytics` built-in tracking:
```python
results = model.track(frame, persist=True, tracker="bytetrack.yaml")
```

---

## 🧪 Running Tests

```bash
python -m pytest tests/
```

---

## 🖥️ Requirements

- Python 3.9+
- OpenCV
- PyTorch
- Ultralytics (YOLOv8)
- scikit-learn

GPU is optional but greatly speeds up video processing.

---

## 📸 Output Preview

Annotated output includes:
- Colored bounding boxes per team
- Player track IDs (`#1`, `#2`, ...)
- Ball circle overlay
- Live stats panel (player count, team count, ball detected)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙌 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- COCO dataset for pretrained weights
