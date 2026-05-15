# Basketball Player Detection & Performance Analytics

I built this because I was curious whether YOLOv8 could do more than just draw boxes around players. Turns out it can — this project detects players and the ball, figures out which team each player is on just from their jersey color, tracks every player across the entire video with a consistent ID, and at the end spits out a dashboard with distance covered, speed, and heatmaps showing where each player spent their time on the court.

The goal was to take something that coaches usually spend 3-4 hours doing manually (rewatching footage, logging stats) and get it down to a few minutes.

---

## What it does

- Detects players and the basketball in any video or image
- Draws bounding boxes with team colors and player IDs that stay consistent frame to frame
- Automatically figures out which player belongs to which team by looking at jersey colors — no manual setup needed
- Tracks how far each player ran and how fast they were going
- Maps out which areas of the court each player used most (Left Paint, Right Paint, Mid Court etc.)
- Generates a heatmap per player and per team
- Puts everything into a `dashboard.html` you can open in your browser — charts, stats table, heatmaps, all in one place
- Exports a CSV with the full per-player stats table built with Pandas

Works on images, video files, and live webcam.

---

## What the output looks like

After running on a match video, you get:

```
output/
├── detected_game.mp4       ← annotated video with boxes, IDs, team colors
├── match_stats.csv         ← per-player distance, speed, zone breakdown
├── dashboard.html          ← open this in your browser
└── heatmaps/
    ├── player_1_heatmap.png
    ├── player_2_heatmap.png
    ├── team_a_heatmap.png
    └── team_b_heatmap.png
```

The stats table in the terminal looks like this:

```
Player ID  Team    Time(s)  Distance(m)  Avg Speed  Max Speed  Dominant Zone
#2         Team A  6.1      3.28         1.77       5.54       Right Corner
#14        Team A  5.8      4.44         2.32       7.16       Left Corner
#9         Team A  1.4      6.91         4.98       14.98      Right Paint
```

---

## Project structure

```
Basketball_player_Detection/
│
├── detector.py          # entry point — runs the full pipeline
├── analytics.py         # distance, speed, zone tracking per player
├── heatmap.py           # generates court heatmap images
├── dashboard.py         # builds the HTML report
├── team_classifier.py   # jersey color → team (K-Means on HSV)
├── tracker.py           # keeps player IDs consistent across frames
├── utils.py             # drawing boxes, saving video, overlays
│
├── finetune/
│   ├── train.py             # fine-tune on your own basketball dataset
│   ├── download_dataset.py  # pulls dataset from Roboflow
│   ├── evaluate.py          # check mAP, precision, recall after training
│   └── tune_hyperparams.py  # automated hyperparameter search
│
├── tests/
│   └── test_tracker.py
│
├── output/              # everything gets saved here after running
│
├── Basketball_Finetune_Colab.ipynb  # Google Colab notebook for GPU training
├── requirements.txt
├── requirements-finetune.txt
└── FINE_TUNING.md
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/basketball-detector.git
cd basketball-detector
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

YOLOv8 weights download automatically the first time you run it.

---

## Running it

**On a video — runs full pipeline and generates the report**
```bash
python detector.py --source game.mp4 --output output/
```

**On a single image**
```bash
python detector.py --source game.jpg --output output/
```

**Webcam**
```bash
python detector.py --source webcam
```

**Quick test without analytics (faster)**
```bash
python detector.py --source game.mp4 --output output/ --no-analytics
```

**Open the dashboard after processing**
```bash
# Windows
start output\dashboard.html

# Mac/Linux
open output/dashboard.html
```

**All flags**
```
--source          video, image, or 'webcam'
--output          where to save results (default: output/)
--model           which weights to use (default: yolov8n.pt)
--conf            detection confidence 0-1 (default: 0.4)
--device          cpu / cuda / mps
--no-tracking     turn off player tracking
--no-teams        turn off team classification
--no-analytics    skip the stats, heatmaps and dashboard
--preview         show a live preview window while processing
--max-frames      stop after N frames — useful for quick tests
```

> **Note:** Tested on Google Colab (T4 GPU) and locally on CPU. For local CPU testing,
> use `--max-frames 100` to process a short clip quickly.

---

## The stats it tracks

After processing a video, `match_stats.csv` has one row per player:

| Column | What it means |
|---|---|
| Player ID | The tracking ID assigned to that player (#1, #2 ...) |
| Team | Team A or Team B |
| Time on Court (s) | How long they were visible in the video |
| Distance (m) | Total metres covered |
| Avg Speed (m/s) | Average pace across the match |
| Max Speed (m/s) | Fastest moment recorded |
| Dominant Zone | Where they spent the most time |
| Zone: X (%) | Breakdown across all 7 court zones |

The court is split into 7 zones — Left Paint, Left Wing, Left Corner, Mid Court, Right Wing, Right Corner, Right Paint.

---

## How team classification works

It looks at the upper half of each player's bounding box (where the jersey is), extracts the dominant color using K-Means clustering in HSV color space, and groups players into two clusters. The first few frames are used to build up enough samples before assignments stabilize. No manual color setup needed — it adapts automatically to any two teams.

---

## How tracking works

Uses IoU (Intersection over Union) matching between frames. Each detection in the current frame gets matched to the closest existing track based on how much the boxes overlap. Tracks survive up to 30 frames without a match before being dropped. New detections that don't match anything start a fresh track with a new ID.

---

## Model options

| Model | Size | Notes |
|---|---|---|
| yolov8n.pt | 6 MB | fastest, good for real-time and CPU |
| yolov8s.pt | 22 MB | better accuracy, still fast |
| yolov8m.pt | 52 MB | solid balance |
| yolov8l.pt | 87 MB | high accuracy |
| yolov8x.pt | 131 MB | best accuracy, slowest |

The base model is pretrained on COCO so confidence scores on basketball footage will be in the 40-60% range. Fine-tuning pushes this up to 75-90%.

---

## Fine-tuning

Fine-tuning on actual basketball footage makes the model significantly better — it learns to ignore coaches and crowd, gets better at detecting the basketball specifically, and handles crowded court situations more reliably.

Full guide in [FINE_TUNING.md](FINE_TUNING.md).

Quick version:
```bash
pip install -r requirements-finetune.txt
python finetune/download_dataset.py   # needs a free Roboflow API key
python finetune/train.py --data data/basketball/data.yaml --model yolov8s.pt --device 0
```

If you don't have a GPU locally, `Basketball_Finetune_Colab.ipynb` runs the whole thing on Google Colab for free in about 30 minutes.

---

## Known issues

- `cv2.destroyAllWindows()` throws an error on some Windows setups after processing finishes — this is a harmless OpenCV bug, everything is already saved before it happens
- Confidence scores will be low (40-50%) with the base model on basketball footage — this is expected and improves significantly after fine-tuning
- Team classification works best when the two teams wear clearly different jersey colors

---

## Tests

```bash
python -m pytest test_tracker.py -v
```

---

## Requirements

- Python 3.9+
- PyTorch
- Ultralytics (YOLOv8)
- OpenCV
- scikit-learn
- Pandas
- NumPy

GPU not required but makes a big difference for training and speeds up inference on longer videos.

---

## What's next

Things I want to add:
- Ball possession detection (which player has the ball at any given moment)
- Jersey number recognition for identifying specific players by name
- PDF export of the match report
- Live camera integration for courtside use

---

## License

MIT
