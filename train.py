"""
train.py
Fine-tunes YOLOv8 on a basketball dataset (player + ball classes).
Supports resuming, custom hyperparameters, and exports the best model.

Usage:
  # Quick start (auto downloads yolov8n.pt weights)
  python finetune/train.py --data data/basketball/data.yaml

  # Full control
  python finetune/train.py \
    --data data/basketball/data.yaml \
    --model yolov8s.pt \
    --epochs 100 \
    --batch 16 \
    --device cuda \
    --project runs/basketball \
    --name exp1

  # Resume from checkpoint
  python finetune/train.py --resume runs/basketball/exp1/weights/last.pt
"""

import argparse
import sys
import time
from pathlib import Path


def check_dependencies():
    missing = []
    for pkg in ["ultralytics", "torch", "yaml"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[ERROR] Missing packages: {', '.join(missing)}")
        print("        Run: pip install -r requirements.txt")
        sys.exit(1)


def get_device_info() -> str:
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"CUDA — {name} ({vram:.1f} GB VRAM)"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "Apple MPS (Metal)"
    return "CPU (slow — GPU recommended for training)"


def recommend_batch_size(device: str) -> int:
    """Suggest batch size based on available hardware."""
    import torch
    if device in ("cuda", "0") and torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram >= 16:
            return 32
        elif vram >= 8:
            return 16
        else:
            return 8
    return 4  # CPU / MPS


def load_data_config(yaml_path: str) -> dict:
    import yaml
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def train(args):
    from ultralytics import YOLO
    import torch

    print("\n" + "=" * 55)
    print("  🏀 Basketball YOLOv8 Fine-Tuning")
    print("=" * 55)
    print(f"  Device    : {get_device_info()}")
    print(f"  Base model: {args.model}")
    print(f"  Data      : {args.data}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Batch     : {args.batch}")
    print(f"  Image size: {args.imgsz}px")
    print("=" * 55 + "\n")

    # Load and show dataset info
    cfg = load_data_config(args.data)
    classes = cfg.get("names", [])
    print(f"[INFO] Classes ({len(classes)}): {classes}")

    # Load model
    if args.resume:
        print(f"[INFO] Resuming from: {args.resume}")
        model = YOLO(args.resume)
    else:
        print(f"[INFO] Loading base model: {args.model}")
        model = YOLO(args.model)

    # Auto batch size
    batch = args.batch
    if batch == -1:
        batch = recommend_batch_size(args.device)
        print(f"[INFO] Auto batch size: {batch}")

    # ── Training ────────────────────────────────────────────────
    start = time.time()

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,

        # Optimizer
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,

        # Augmentation — important for sports footage
        hsv_h=0.015,        # Hue shift (jersey color robustness)
        hsv_s=0.7,          # Saturation
        hsv_v=0.4,          # Brightness (arena lighting variation)
        degrees=5.0,         # Rotation
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,          # Mosaic augmentation
        mixup=0.1,           # MixUp for harder examples
        copy_paste=0.1,      # Copy-paste augmentation

        # Training quality
        patience=args.patience,
        save=True,
        save_period=10,      # Save checkpoint every 10 epochs
        val=True,
        plots=True,

        # Class weights — upweight ball (smaller, rarer object)
        # cls=1.0,  # uncomment to tune
        resume=bool(args.resume),
        verbose=True,
        workers=args.workers,
    )

    elapsed = time.time() - start
    print(f"\n[INFO] Training complete in {elapsed / 60:.1f} minutes")

    # ── Results ─────────────────────────────────────────────────
    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    last_weights = save_dir / "weights" / "last.pt"

    print(f"\n── Results ────────────────────────────────")
    print(f"  Saved to    : {save_dir}")
    print(f"  Best weights: {best_weights}")
    print(f"  Last weights: {last_weights}")

    # ── Validation ──────────────────────────────────────────────
    if best_weights.exists() and not args.skip_val:
        print("\n[INFO] Running final validation on best.pt...")
        val_model = YOLO(str(best_weights))
        metrics = val_model.val(data=args.data, imgsz=args.imgsz, device=args.device)

        print(f"\n── Validation Metrics ─────────────────────")
        print(f"  mAP50      : {metrics.box.map50:.4f}")
        print(f"  mAP50-95   : {metrics.box.map:.4f}")
        print(f"  Precision  : {metrics.box.mp:.4f}")
        print(f"  Recall     : {metrics.box.mr:.4f}")

        per_class = metrics.box.ap_class_index
        if per_class is not None:
            names = cfg.get("names", [])
            for idx, cls_idx in enumerate(per_class):
                name = names[cls_idx] if cls_idx < len(names) else f"class_{cls_idx}"
                ap = metrics.box.ap[idx]
                print(f"  AP [{name:<8}]: {ap:.4f}")

    # ── Export ──────────────────────────────────────────────────
    if args.export and best_weights.exists():
        print(f"\n[INFO] Exporting to ONNX...")
        export_model = YOLO(str(best_weights))
        export_model.export(format="onnx", imgsz=args.imgsz, simplify=True)
        print(f"[INFO] Exported: {best_weights.with_suffix('.onnx')}")

    print(f"\n✅ Done! Use your fine-tuned model:")
    print(f"   python src/detector.py --source your_video.mp4 --model {best_weights}")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for basketball detection")

    # Required
    parser.add_argument("--data", type=str, default="data/basketball/data.yaml",
                        help="Path to data.yaml (from Roboflow download)")

    # Model
    parser.add_argument("--model", type=str, default="yolov8s.pt",
                        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
                                 "yolov8l.pt", "yolov8x.pt"],
                        help="Base YOLOv8 model to fine-tune (default: yolov8s.pt)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from last.pt checkpoint path")

    # Training
    parser.add_argument("--epochs", type=int, default=80,
                        help="Number of training epochs (default: 80)")
    parser.add_argument("--batch", type=int, default=-1,
                        help="Batch size (-1 = auto based on GPU VRAM)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Training image size (default: 640)")
    parser.add_argument("--device", type=str, default="0",
                        help="Device: 0 (GPU), cpu, mps (Apple Silicon)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Dataloader workers (default: 4)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (default: 20)")

    # Optimizer
    parser.add_argument("--optimizer", type=str, default="AdamW",
                        choices=["SGD", "Adam", "AdamW", "auto"],
                        help="Optimizer (default: AdamW)")
    parser.add_argument("--lr0", type=float, default=0.001,
                        help="Initial learning rate (default: 0.001)")
    parser.add_argument("--lrf", type=float, default=0.01,
                        help="Final LR as fraction of lr0 (default: 0.01)")

    # Output
    parser.add_argument("--project", type=str, default="runs/basketball",
                        help="Project directory for saving results")
    parser.add_argument("--name", type=str, default="finetune",
                        help="Experiment name (creates runs/basketball/finetune/)")
    parser.add_argument("--exist-ok", action="store_true",
                        help="Overwrite existing experiment directory")

    # Post-training
    parser.add_argument("--skip-val", action="store_true",
                        help="Skip final validation after training")
    parser.add_argument("--export", action="store_true",
                        help="Export best.pt to ONNX after training")

    return parser.parse_args()


if __name__ == "__main__":
    check_dependencies()
    args = parse_args()
    train(args)
