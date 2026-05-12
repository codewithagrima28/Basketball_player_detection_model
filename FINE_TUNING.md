# 🏀 Fine-Tuning Guide — Basketball Player Detector

This guide walks you through fine-tuning YOLOv8 on a public basketball dataset to get **significantly better accuracy** on players and the basketball specifically (vs. the generic COCO pretrained weights).

---

## 📋 What You'll Get After Fine-Tuning

| | Pretrained (COCO) | Fine-Tuned |
|---|---|---|
| Player detection | Generic "person" class | Basketball-specific players |
| Ball detection | Generic "sports ball" | Basketball specifically |
| False positives | Detects all people | Ignores coaches, audience |
| mAP (typical) | ~45–55% on basketball | **~75–90%** on basketball |

---

## 🔧 Setup

```bash
pip install -r requirements-finetune.txt
```

---

## 📦 Step 1 — Download the Dataset

We use **[Basketball Players Detection](https://universe.roboflow.com/roboflow-100/basketball-players-fy4c2)** from Roboflow Universe (free, ~5,000 labeled images).

### Get your free Roboflow API key
1. Go to [roboflow.com](https://roboflow.com) → Sign up (free)
2. Settings → API Keys → copy your key
3. Set it as an environment variable:

```bash
# macOS / Linux
export ROBOFLOW_API_KEY=your_key_here

# Windows PowerShell
$env:ROBOFLOW_API_KEY="your_key_here"
```

### Download
```bash
python finetune/download_dataset.py
```

This saves the dataset to `data/basketball/` in YOLOv8 format:
```
data/basketball/
├── train/
│   ├── images/   ← training images
│   └── labels/   ← YOLO format .txt labels
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml     ← dataset config used by trainer
```

### Alternative: Use a different dataset
Browse [Roboflow Universe Basketball](https://universe.roboflow.com/search?q=basketball) and swap the `--dataset` flag:
```bash
python finetune/download_dataset.py \
  --workspace your-workspace \
  --dataset your-dataset-slug \
  --version 3
```

---

## 🚀 Step 2 — Fine-Tune the Model

### Quick start (recommended for first run)
```bash
python finetune/train.py --data data/basketball/data.yaml
```

### Full training with GPU (best results)
```bash
python finetune/train.py \
  --data data/basketball/data.yaml \
  --model yolov8s.pt \
  --epochs 80 \
  --device 0 \
  --project runs/basketball \
  --name v1
```

### Model size recommendations

| Scenario | Model | Command |
|---|---|---|
| Fast laptop / CPU | `yolov8n.pt` | `--model yolov8n.pt --batch 4` |
| Good laptop GPU | `yolov8s.pt` | `--model yolov8s.pt` |
| Workstation / Colab | `yolov8m.pt` | `--model yolov8m.pt --epochs 100` |
| Max accuracy | `yolov8l.pt` | `--model yolov8l.pt --device 0` |

### Resume interrupted training
```bash
python finetune/train.py --resume runs/basketball/v1/weights/last.pt
```

### Monitor with TensorBoard
```bash
tensorboard --logdir runs/basketball
# Open http://localhost:6006
```

---

## 📊 Step 3 — Evaluate Your Model

```bash
python finetune/evaluate.py \
  --model runs/basketball/v1/weights/best.pt \
  --data data/basketball/data.yaml
```

### With visual samples
```bash
python finetune/evaluate.py \
  --model runs/basketball/v1/weights/best.pt \
  --data data/basketball/data.yaml \
  --visualize \
  --n 30
```

Output saved to `output/eval_samples/`.

### Reading the metrics

| Metric | What it means | Target |
|---|---|---|
| **mAP@0.5** | % boxes correctly detected (IoU ≥ 0.5) | > 75% |
| **mAP@0.5:0.95** | Stricter average across IoU thresholds | > 55% |
| **Precision** | Of all detections, % that are correct | > 80% |
| **Recall** | Of all real objects, % that are found | > 75% |

---

## 🎯 Step 4 — Use the Fine-Tuned Model

Replace the default model in the detector:

```bash
python src/detector.py \
  --source your_game.mp4 \
  --model runs/basketball/v1/weights/best.pt \
  --output output/videos/result.mp4
```

Or in Python:
```python
from src.detector import BasketballDetector

detector = BasketballDetector(
    model_path="runs/basketball/v1/weights/best.pt",
    conf_threshold=0.35,
)
detector.process_video("game.mp4", "output/result.mp4")
```

---

## ⚙️ Hyperparameter Tuning (Optional)

Run automated hyperparameter search for even better results:

```bash
# Install Ray Tune first
pip install "ray[tune]"

python finetune/tune_hyperparams.py \
  --data data/basketball/data.yaml \
  --iterations 30 \
  --epochs 15
```

This runs 30 short experiments and finds the best learning rate, augmentation settings, etc.

---

## 🔍 Troubleshooting

### Low mAP on ball class
The ball is small and fast — it's the hardest to detect.
- Use a larger model (`yolov8m` or `yolov8l`)
- Increase image size: `--imgsz 1280`
- Train longer: `--epochs 120`
- Try a ball-specific dataset from Roboflow Universe

### CUDA out of memory
Reduce batch size:
```bash
python finetune/train.py --batch 4
```
Or use `--batch -1` for auto batch size.

### Model detects audience/coaches as players
Lower the confidence threshold or add more diverse training data with non-player people labeled as background (negative examples).

### Training loss not decreasing
- Verify `data.yaml` paths are correct
- Check that labels exist for all images
- Try a lower learning rate: `--lr0 0.0005`

---

## 📁 Output Structure After Training

```
runs/basketball/v1/
├── weights/
│   ├── best.pt        ← Use this for inference
│   └── last.pt        ← Resume from this
├── results.png        ← Training curves
├── confusion_matrix.png
├── PR_curve.png
├── F1_curve.png
└── val_batch*.jpg     ← Sample validation predictions
```

---

## 💡 Tips for Best Results

1. **More data wins** — if accuracy is low, add more labeled images
2. **Roboflow augmentation** — enable augmentation in Roboflow before downloading to 3× your dataset size for free
3. **Freeze backbone** — for very small datasets, freeze early layers: add `freeze=10` to `train()` call
4. **Two-stage training** — train with `yolov8n` first to verify pipeline, then switch to `yolov8m` for final model
5. **Export for deployment** — ONNX runs faster on CPU:
   ```bash
   python finetune/train.py --data data.yaml --export
   ```
