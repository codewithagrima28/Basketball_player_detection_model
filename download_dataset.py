"""
download_dataset.py
Downloads a public basketball detection dataset from Roboflow Universe
and prepares it in YOLOv8 format for fine-tuning.

Public dataset used:
  "Basketball Detection" by Roboflow Universe
  https://universe.roboflow.com/roboflow-100/basketball-players-fy4c2
  Classes: player, ball

Usage:
  python finetune/download_dataset.py
  python finetune/download_dataset.py --dataset basketball-players-fy4c2 --version 2
"""

import os
import sys
import argparse
import zipfile
import shutil
import tempfile
import urllib.request
from pathlib import Path


def _check_zip(zip_path: Path) -> bool:
    """Return True if file is a valid zip."""
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.testzip()
        return True
    except Exception:
        return False


def _extract_zip(zip_path: Path, dest: Path):
    """Extract zip and flatten one extra nesting level if present."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)

    # Roboflow sometimes wraps everything in a single sub-folder — unwrap it
    children = [c for c in dest.iterdir() if c.is_dir()]
    top_files = [c for c in dest.iterdir() if c.is_file()]
    if len(children) == 1 and not top_files:
        inner = children[0]
        for item in inner.iterdir():
            shutil.move(str(item), str(dest / item.name))
        inner.rmdir()


def download_with_roboflow(workspace: str, dataset: str, version: int, output_dir: str):
    """
    Download dataset using the Roboflow Python SDK with a robust fallback:
    1. Try the SDK (fastest)
    2. If SDK produces a bad zip (common Windows bug), re-download via direct HTTP
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[ERROR] roboflow package not installed.")
        print("        Run: pip install roboflow")
        sys.exit(1)

    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        print("\n[ERROR] ROBOFLOW_API_KEY environment variable not set.")
        print("  1. Sign up free at https://roboflow.com")
        print("  2. Go to Settings → API Keys and copy your key")
        print("  3. On Windows CMD run:")
        print("       set ROBOFLOW_API_KEY=your_key_here")
        print("     On PowerShell run:")
        print("       $env:ROBOFLOW_API_KEY='your_key_here'")
        print("  Then re-run this script.\n")
        sys.exit(1)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Connecting to Roboflow...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(dataset)
    ver_obj = project.version(version)

    # ── Attempt 1: SDK download ──────────────────────────────────
    print(f"[INFO] Downloading via SDK: {workspace}/{dataset} v{version}")
    try:
        dataset_obj = ver_obj.download("yolov8", location=output_dir)
        location = Path(dataset_obj.location)

        # Verify it actually extracted correctly
        has_train = (location / "train" / "images").exists()
        if has_train:
            print(f"[INFO] SDK download successful: {location}")
            return str(location)
        else:
            print("[WARN] SDK extracted but folder structure missing — trying fallback...")
    except Exception as e:
        print(f"[WARN] SDK download failed ({e}) — trying direct HTTP fallback...")

    # ── Attempt 2: Direct HTTP download ─────────────────────────
    print("[INFO] Fetching direct download URL from Roboflow API...")
    try:
        import urllib.parse, json

        # Get the signed download URL via Roboflow REST API
        api_url = (
            f"https://api.roboflow.com/{workspace}/{dataset}/{version}/"
            f"yolov8?api_key={api_key}"
        )
        with urllib.request.urlopen(api_url, timeout=30) as resp:
            meta = json.loads(resp.read().decode())

        export_url = meta.get("export", {}).get("link") or meta.get("version", {}).get("export", {}).get("link")
        if not export_url:
            print("[ERROR] Could not get download link from API response.")
            print("        Please download manually — see FINE_TUNING.md for instructions.")
            sys.exit(1)

        print(f"[INFO] Downloading ZIP directly...")
        zip_path = out_path / "dataset.zip"

        def _reporthook(count, block_size, total_size):
            if total_size > 0:
                pct = min(100, count * block_size * 100 // total_size)
                print(f"\r  Progress: {pct}%", end="", flush=True)

        urllib.request.urlretrieve(export_url, zip_path, reporthook=_reporthook)
        print()  # newline after progress

        if not _check_zip(zip_path):
            raise ValueError("Downloaded file is not a valid ZIP")

        print(f"[INFO] Extracting ZIP...")
        _extract_zip(zip_path, out_path)
        zip_path.unlink()  # clean up zip

        print(f"[INFO] Dataset saved to: {out_path}")
        return str(out_path)

    except Exception as e:
        print(f"\n[ERROR] Both download methods failed: {e}")
        print("\nManual download instructions:")
        print(f"  1. Go to: https://universe.roboflow.com/{workspace}/{dataset}")
        print(f"  2. Click 'Download Dataset' → choose 'YOLOv8' format")
        print(f"  3. Extract the ZIP into:  {out_path}/")
        print(f"  4. Make sure the folder has: train/, valid/, test/, data.yaml")
        sys.exit(1)


def verify_dataset(data_dir: str):
    """Check that the dataset has the expected structure."""
    base = Path(data_dir)
    splits = ["train", "valid", "test"]
    issues = []

    for split in splits:
        img_dir = base / split / "images"
        lbl_dir = base / split / "labels"

        if not img_dir.exists():
            issues.append(f"Missing: {img_dir}")
            continue

        imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        lbls = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

        print(f"  [{split:5s}] {len(imgs):4d} images | {len(lbls):4d} labels")

        if len(imgs) == 0:
            issues.append(f"{split}: No images found")
        if len(lbls) == 0:
            issues.append(f"{split}: No labels found")

    if issues:
        print("\n[WARN] Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n[OK] Dataset structure looks good!")

    return len(issues) == 0


def patch_data_yaml(data_dir: str, class_names: list[str] | None = None):
    """
    Update data.yaml to ensure class names match what we expect.
    Roboflow sometimes uses different class name casing.
    """
    import yaml

    yaml_path = Path(data_dir) / "data.yaml"
    if not yaml_path.exists():
        print(f"[WARN] data.yaml not found at {yaml_path}")
        return

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    print(f"\n[INFO] data.yaml classes: {cfg.get('names', [])}")

    # Normalize class names if provided
    if class_names:
        cfg["names"] = class_names
        cfg["nc"] = len(class_names)
        with open(yaml_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        print(f"[INFO] Updated class names to: {class_names}")

    # Make paths absolute
    for key in ["train", "val", "test"]:
        if key in cfg:
            p = Path(cfg[key])
            if not p.is_absolute():
                cfg[key] = str(Path(data_dir) / p)

    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"[INFO] data.yaml updated: {yaml_path}")
    return str(yaml_path)


def print_dataset_summary(data_dir: str):
    """Print a summary of class distribution in the dataset."""
    from collections import Counter
    import yaml

    yaml_path = Path(data_dir) / "data.yaml"
    if not yaml_path.exists():
        return

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    names = cfg.get("names", [])
    counter = Counter()

    for split in ["train", "valid", "test"]:
        lbl_dir = Path(data_dir) / split / "labels"
        if not lbl_dir.exists():
            continue
        for lbl_file in lbl_dir.glob("*.txt"):
            for line in lbl_file.read_text().strip().splitlines():
                if line:
                    cls_id = int(line.split()[0])
                    counter[cls_id] += 1

    print("\n── Class Distribution ──────────────────")
    for cls_id, name in enumerate(names):
        count = counter.get(cls_id, 0)
        bar = "█" * min(40, count // max(counter.values(), 1) * 40 + 1) if counter else ""
        print(f"  [{cls_id}] {name:<12} {count:6d}  {bar}")
    print()


def parse_args():
    parser = argparse.ArgumentParser(description="Download basketball dataset from Roboflow")
    parser.add_argument("--workspace", default="roboflow-100",
                        help="Roboflow workspace name")
    parser.add_argument("--dataset", default="basketball-players-fy4c2",
                        help="Roboflow dataset slug")
    parser.add_argument("--version", type=int, default=1,
                        help="Dataset version number")
    parser.add_argument("--output", default="data/basketball",
                        help="Output directory for dataset")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 50)
    print("  Basketball Dataset Downloader")
    print("=" * 50)

    data_dir = download_with_roboflow(
        workspace=args.workspace,
        dataset=args.dataset,
        version=args.version,
        output_dir=args.output,
    )

    print("\n── Verifying dataset structure ─────────")
    ok = verify_dataset(data_dir)

    # Normalize class names to lowercase
    patch_data_yaml(data_dir, class_names=["player", "ball"])

    print_dataset_summary(data_dir)

    if ok:
        print(f"✅ Dataset ready at: {data_dir}")
        print(f"\nNext step:")
        print(f"  python finetune/train.py --data {data_dir}/data.yaml")
