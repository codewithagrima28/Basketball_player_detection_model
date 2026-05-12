"""
tune_hyperparams.py
Runs automated hyperparameter search using Ultralytics' built-in Ray Tune integration.
Finds the best learning rate, augmentation params, etc. for your specific dataset.

Usage:
  python finetune/tune_hyperparams.py --data data/basketball/data.yaml --iterations 30
"""

import argparse
import sys


def tune(args):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Run: pip install -r requirements.txt")
        sys.exit(1)

    print(f"\n[INFO] Starting hyperparameter tuning")
    print(f"  Model      : {args.model}")
    print(f"  Data       : {args.data}")
    print(f"  Iterations : {args.iterations}")
    print(f"  Epochs/run : {args.epochs}")
    print(f"\n  This will train {args.iterations} mini-experiments.")
    print(f"  Estimated time: {args.iterations * args.epochs * 2}–{args.iterations * args.epochs * 5} minutes on GPU\n")

    model = YOLO(args.model)

    # Ultralytics tune() runs Ray Tune under the hood
    result = model.tune(
        data=args.data,
        epochs=args.epochs,
        iterations=args.iterations,
        imgsz=args.imgsz,
        device=args.device,
        optimizer="AdamW",
        plots=True,
        save=True,
        val=True,
        # Search space is managed by ultralytics internally
        # It searches: lr0, lrf, momentum, weight_decay,
        # warmup_epochs, box, cls, dfl, hsv_h/s/v, degrees,
        # translate, scale, fliplr, mosaic, mixup, copy_paste
    )

    print("\n── Best Hyperparameters Found ──────────────")
    if hasattr(result, 'best_params'):
        for k, v in result.best_params.items():
            print(f"  {k:<20}: {v}")
    print()
    print("Next step: use these hyperparameters in train.py")
    print(f"  python finetune/train.py --data {args.data} --lr0 <best_lr0> ...")


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for basketball detector")
    parser.add_argument("--model", default="yolov8s.pt",
                        help="Base model to tune (default: yolov8s.pt)")
    parser.add_argument("--data", default="data/basketball/data.yaml",
                        help="Path to data.yaml")
    parser.add_argument("--iterations", type=int, default=30,
                        help="Number of tuning iterations (default: 30)")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Epochs per tuning run (default: 15, keep short)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size")
    parser.add_argument("--device", default="0",
                        help="Device: 0 (GPU), cpu, mps")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tune(args)
