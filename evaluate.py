"""
evaluate.py
Evaluates a fine-tuned model and generates visual reports.

Usage:
  python finetune/evaluate.py --model runs/basketball/finetune/weights/best.pt \
                              --data data/basketball/data.yaml

  # Also generate visual predictions on test set
  python finetune/evaluate.py --model best.pt --data data.yaml --visualize --n 20
"""

import argparse
import sys
from pathlib import Path
import random


def evaluate(args):
    try:
        from ultralytics import YOLO
        import yaml
    except ImportError:
        print("[ERROR] Run: pip install -r requirements.txt")
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)

    print(f"\n[INFO] Loading model: {model_path}")
    model = YOLO(str(model_path))

    # Load class names
    with open(args.data) as f:
        cfg = yaml.safe_load(f)
    class_names = cfg.get("names", [])

    # ── Validation metrics ──────────────────────────────────────
    print(f"[INFO] Running validation on: {args.data}")
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        plots=True,
        save_json=args.save_json,
    )

    print("\n" + "=" * 50)
    print("  📊 Evaluation Results")
    print("=" * 50)
    print(f"  mAP@0.5     : {metrics.box.map50:.4f}  ({metrics.box.map50 * 100:.1f}%)")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}  ({metrics.box.map * 100:.1f}%)")
    print(f"  Precision   : {metrics.box.mp:.4f}")
    print(f"  Recall      : {metrics.box.mr:.4f}")
    print(f"  F1 Score    : {_f1(metrics.box.mp, metrics.box.mr):.4f}")
    print()

    # Per-class breakdown
    print("  Per-class AP@0.5:")
    if hasattr(metrics.box, 'ap_class_index') and metrics.box.ap_class_index is not None:
        for idx, cls_idx in enumerate(metrics.box.ap_class_index):
            name = class_names[cls_idx] if cls_idx < len(class_names) else f"class_{cls_idx}"
            ap50 = metrics.box.ap50[idx] if hasattr(metrics.box, 'ap50') else metrics.box.ap[idx]
            status = "✅" if ap50 >= 0.75 else "⚠️ " if ap50 >= 0.5 else "❌"
            bar = "█" * int(ap50 * 20)
            print(f"  {status} {name:<12} {ap50:.4f}  |{bar:<20}|")

    print()
    save_dir = Path(metrics.save_dir)
    print(f"  Confusion matrix & curves saved to: {save_dir}")

    # ── Visual predictions ──────────────────────────────────────
    if args.visualize:
        _visualize_predictions(model, cfg, args, class_names)

    # ── Model info ─────────────────────────────────────────────
    if args.model_info:
        _print_model_info(model, model_path)

    return metrics


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _visualize_predictions(model, cfg: dict, args, class_names: list):
    """Run predictions on N random test images and save annotated results."""
    import cv2
    import yaml

    test_dir = Path(cfg.get("test", "")).parent
    if not test_dir.exists():
        # Try common paths
        for candidate in ["data/basketball/test/images", "test/images"]:
            if Path(candidate).exists():
                test_dir = Path(candidate)
                break
        else:
            print("[WARN] Could not find test images directory for visualization.")
            return

    # Resolve path relative to data.yaml if needed
    if not test_dir.is_absolute():
        test_dir = Path(args.data).parent / test_dir

    images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    if not images:
        print(f"[WARN] No images found in {test_dir}")
        return

    sample = random.sample(images, min(args.n, len(images)))
    out_dir = Path("output/eval_samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Generating {len(sample)} prediction samples → {out_dir}/")

    for img_path in sample:
        results = model.predict(
            str(img_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )
        annotated = results[0].plot()
        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), annotated)

    print(f"[INFO] Saved {len(sample)} annotated samples to: {out_dir}/")


def _print_model_info(model, model_path: Path):
    """Print model architecture summary."""
    import torch

    file_size = model_path.stat().st_size / 1e6
    print(f"\n── Model Info ────────────────────────────")
    print(f"  File size : {file_size:.1f} MB")

    try:
        info = model.info(verbose=False)
    except Exception:
        pass

    try:
        # Count parameters
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        print(f"  Parameters: {total_params:,} total / {trainable:,} trainable")
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned basketball detection model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to best.pt model weights")
    parser.add_argument("--data", type=str, default="data/basketball/data.yaml",
                        help="Path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for evaluation")
    parser.add_argument("--iou", type=float, default=0.7,
                        help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu, cuda, mps")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate annotated prediction samples")
    parser.add_argument("--n", type=int, default=20,
                        help="Number of test images to visualize (default: 20)")
    parser.add_argument("--save-json", action="store_true",
                        help="Save results in COCO JSON format")
    parser.add_argument("--model-info", action="store_true",
                        help="Print model architecture info")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
