"""
Basketball Player Detector
Detects players, ball, classifies teams by jersey color, tracks across frames,
computes per-player analytics (distance, speed, court zones), generates heatmaps,
and produces a coach dashboard HTML report.
Supports both image and video input.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent))

from team_classifier import TeamClassifier
from tracker import PlayerTracker
from utils import draw_detections, save_output, get_video_writer
from analytics import PlayerAnalytics
from heatmap import generate_team_heatmap
from dashboard import generate_dashboard


@dataclass
class Detection:
    bbox: tuple
    confidence: float
    class_name: str
    track_id: Optional[int] = None
    team: Optional[str] = None
    team_color: Optional[tuple] = None


class BasketballDetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.4,
        device: str = "cpu",
        enable_tracking: bool = True,
        enable_team_classification: bool = True,
        enable_analytics: bool = True,
    ):
        print(f"[INFO] Loading YOLOv8 model: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(device)

        self.conf_threshold = conf_threshold
        self.enable_tracking = enable_tracking
        self.enable_team_classification = enable_team_classification
        self.enable_analytics = enable_analytics

        self.PERSON_CLASS = 0
        self.SPORTS_BALL_CLASS = 32

        self.tracker = PlayerTracker() if enable_tracking else None
        self.team_classifier = TeamClassifier() if enable_team_classification else None
        self.analytics: Optional[PlayerAnalytics] = None

        self.stats = defaultdict(int)

    def detect_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list]:
        results = self.model(
            frame,
            conf=self.conf_threshold,
            classes=[self.PERSON_CLASS, self.SPORTS_BALL_CLASS],
            verbose=False,
        )

        detections = []
        result = results[0]

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = "player" if cls_id == self.PERSON_CLASS else "ball"

            det = Detection(bbox=(x1, y1, x2, y2), confidence=conf, class_name=class_name)

            if class_name == "player" and self.team_classifier:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    team, color = self.team_classifier.classify(crop)
                    det.team = team
                    det.team_color = color

            detections.append(det)
            self.stats[class_name] += 1

        if self.tracker and self.enable_tracking:
            player_dets = [d for d in detections if d.class_name == "player"]
            track_ids = self.tracker.update(player_dets, frame)
            for det, tid in zip(player_dets, track_ids):
                det.track_id = tid

        if self.analytics:
            self.analytics.update(detections)

        annotated = draw_detections(frame.copy(), detections)
        return annotated, detections

    def process_image(self, image_path: str, output_path: Optional[str] = None) -> np.ndarray:
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        print(f"[INFO] Processing image: {image_path}")
        annotated, detections = self.detect_frame(frame)

        players = [d for d in detections if d.class_name == "player"]
        balls   = [d for d in detections if d.class_name == "ball"]
        print(f"[INFO] Found {len(players)} player(s) and {len(balls)} ball(s)")

        if output_path:
            save_output(annotated, output_path)
            print(f"[INFO] Saved to: {output_path}")

        return annotated

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show_preview: bool = False,
        max_frames: Optional[int] = None,
        generate_report: bool = True,
    ) -> dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[INFO] Video: {width}x{height} @ {fps:.1f}fps — {total_frames} frames")

        if self.enable_analytics:
            self.analytics = PlayerAnalytics(fps=fps, frame_w=width, frame_h=height)

        out_dir = Path(output_path).parent if output_path else Path("output")
        out_dir.mkdir(parents=True, exist_ok=True)

        writer = get_video_writer(output_path, fps, width, height) if output_path else None

        frame_count = 0
        start_time  = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if max_frames and frame_count >= max_frames:
                    break

                annotated, _ = self.detect_frame(frame)

                if writer:
                    writer.write(annotated)

                if show_preview:
                    cv2.imshow("Basketball Detection", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[INFO] Preview stopped by user.")
                        break

                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    pct = int(frame_count / total_frames * 100) if total_frames else 0
                    print(f"[INFO] Frame {frame_count}/{total_frames} ({pct}%) — {frame_count/elapsed:.1f} FPS")

        finally:
            cap.release()
            if writer:
                writer.release()
            try:
              cv2.destroyAllWindows()
            except Exception:
              pass

        elapsed = time.time() - start_time
        processing_stats = {
            "frames_processed": frame_count,
            "elapsed_seconds": round(elapsed, 2),
            "avg_fps": round(frame_count / elapsed, 2) if elapsed > 0 else 0,
            "total_player_detections": self.stats["player"],
            "total_ball_detections":   self.stats["ball"],
        }

        print(f"\n[DONE] Processed {frame_count} frames in {elapsed:.1f}s ({processing_stats['avg_fps']} FPS)")

        if self.enable_analytics and self.analytics and generate_report:
            self._generate_match_report(out_dir, video_path)

        return processing_stats

    def _generate_match_report(self, out_dir: Path, video_path: str):
        print("\n[INFO] Generating match analytics report...")

        df = self.analytics.get_summary()
        if df.empty:
            print("[WARN] Not enough tracking data to generate report.")
            return

        # Save CSV
        csv_path = out_dir / "match_stats.csv"
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Stats CSV saved: {csv_path}")

        # Print summary table
        print("\n── Player Stats ──────────────────────────────────")
        cols = ["Player ID", "Team", "Distance (m)", "Avg Speed (m/s)", "Max Speed (m/s)", "Dominant Zone"]
        available = [c for c in cols if c in df.columns]
        print(df[available].to_string(index=False))

        # Heatmaps
        heatmap_dir = out_dir / "heatmaps"
        all_positions   = self.analytics.get_all_positions()
        team_assignments = {tid: p.team for tid, p in self.analytics.players.items()}

        if all_positions:
            generate_team_heatmap(
                all_positions=all_positions,
                team_assignments=team_assignments,
                output_dir=str(heatmap_dir),
                source_w=self.analytics.frame_w,
                source_h=self.analytics.frame_h,
            )

        # HTML Dashboard
        dashboard_path = out_dir / "dashboard.html"
        generate_dashboard(
            df=df,
            output_path=str(dashboard_path),
            heatmap_dir=str(heatmap_dir) if heatmap_dir.exists() else None,
            match_name=f"Match — {Path(video_path).stem}",
            video_path=video_path,
        )

        print(f"\n✅ Match report ready!")
        print(f"   📊 Stats CSV  : {csv_path}")
        print(f"   🗺️  Heatmaps  : {heatmap_dir}/")
        print(f"   🖥️  Dashboard : {dashboard_path}")
        print(f"\n   Open in browser: start {dashboard_path}")

    def process_webcam(self, camera_id: int = 0):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.enable_analytics:
            self.analytics = PlayerAnalytics(fps=fps, frame_w=width, frame_h=height)

        print("[INFO] Webcam started. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated, _ = self.detect_frame(frame)
            cv2.imshow("Basketball Detection — Webcam", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(description="Basketball Player Detector using YOLOv8")
    parser.add_argument("--source",      type=str,   required=True)
    parser.add_argument("--output",      type=str,   default=None)
    parser.add_argument("--model",       type=str,   default="yolov8n.pt")
    parser.add_argument("--conf",        type=float, default=0.4)
    parser.add_argument("--device",      type=str,   default="cpu",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--no-tracking",  action="store_true")
    parser.add_argument("--no-teams",     action="store_true")
    parser.add_argument("--no-analytics", action="store_true",
                        help="Skip analytics, heatmaps and dashboard")
    parser.add_argument("--preview",      action="store_true")
    parser.add_argument("--max-frames",   type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    detector = BasketballDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        device=args.device,
        enable_tracking=not args.no_tracking,
        enable_team_classification=not args.no_teams,
        enable_analytics=not args.no_analytics,
    )

    source = args.source
    output = args.output or "output"

    if source == "webcam":
        detector.process_webcam()
    elif Path(source).suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        out_path = str(Path(output) / f"detected_{Path(source).name}") if Path(output).is_dir() else output
        detector.process_image(source, out_path)
    else:
        out_dir  = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / f"detected_{Path(source).name}")
        detector.process_video(source, out_path, show_preview=args.preview, max_frames=args.max_frames)
