"""
Basketball Player Detector
Detects players, ball, classifies teams by jersey color, and tracks across frames.
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

from team_classifier import TeamClassifier
from tracker import PlayerTracker
from utils import draw_detections, save_output, get_video_writer


@dataclass
class Detection:
    bbox: tuple          # (x1, y1, x2, y2)
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
    ):
        print(f"[INFO] Loading YOLOv8 model: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(device)

        self.conf_threshold = conf_threshold
        self.enable_tracking = enable_tracking
        self.enable_team_classification = enable_team_classification

        # COCO class IDs relevant to basketball
        self.PERSON_CLASS = 0
        self.SPORTS_BALL_CLASS = 32

        self.tracker = PlayerTracker() if enable_tracking else None
        self.team_classifier = TeamClassifier() if enable_team_classification else None

        self.stats = defaultdict(int)

    def detect_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[Detection]]:
        """Run detection on a single frame. Returns annotated frame and detections."""
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

            det = Detection(
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                class_name=class_name,
            )

            # Team classification for players
            if class_name == "player" and self.team_classifier:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    team, color = self.team_classifier.classify(crop)
                    det.team = team
                    det.team_color = color

            detections.append(det)
            self.stats[class_name] += 1

        # Player tracking
        if self.tracker and self.enable_tracking:
            player_dets = [d for d in detections if d.class_name == "player"]
            track_ids = self.tracker.update(player_dets, frame)
            for det, tid in zip(player_dets, track_ids):
                det.track_id = tid

        annotated = draw_detections(frame.copy(), detections)
        return annotated, detections

    def process_image(self, image_path: str, output_path: Optional[str] = None) -> np.ndarray:
        """Detect on a single image file."""
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        print(f"[INFO] Processing image: {image_path}")
        annotated, detections = self.detect_frame(frame)

        players = [d for d in detections if d.class_name == "player"]
        balls = [d for d in detections if d.class_name == "ball"]
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
    ) -> dict:
        """Detect on a video file. Returns stats dict."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[INFO] Video: {width}x{height} @ {fps:.1f}fps — {total_frames} frames")

        writer = get_video_writer(output_path, fps, width, height) if output_path else None

        frame_count = 0
        start_time = time.time()

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
                    current_fps = frame_count / elapsed
                    print(f"[INFO] Frame {frame_count}/{total_frames} — {current_fps:.1f} FPS")

        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        stats = {
            "frames_processed": frame_count,
            "elapsed_seconds": round(elapsed, 2),
            "avg_fps": round(frame_count / elapsed, 2) if elapsed > 0 else 0,
            "total_player_detections": self.stats["player"],
            "total_ball_detections": self.stats["ball"],
        }

        print(f"\n[DONE] Processed {frame_count} frames in {elapsed:.1f}s ({stats['avg_fps']} FPS)")
        if output_path:
            print(f"[INFO] Saved to: {output_path}")

        return stats

    def process_webcam(self, camera_id: int = 0):
        """Real-time detection from webcam."""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")

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
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Basketball Player Detector using YOLOv8")
    parser.add_argument("--source", type=str, required=True,
                        help="Input: image path, video path, or 'webcam'")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (optional)")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="YOLOv8 model (default: yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Confidence threshold (default: 0.4)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"],
                        help="Inference device")
    parser.add_argument("--no-tracking", action="store_true",
                        help="Disable player tracking")
    parser.add_argument("--no-teams", action="store_true",
                        help="Disable team classification")
    parser.add_argument("--preview", action="store_true",
                        help="Show live preview window (video only)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit number of frames processed (video only)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    detector = BasketballDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        device=args.device,
        enable_tracking=not args.no_tracking,
        enable_team_classification=not args.no_teams,
    )

    source = args.source
    output = args.output or "output/"

    if source == "webcam":
        detector.process_webcam()
    elif Path(source).suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        out_path = str(Path(output) / f"detected_{Path(source).name}") if Path(output).is_dir() else output
        detector.process_image(source, out_path)
    else:
        out_path = str(Path(output) / f"detected_{Path(source).name}") if Path(output).is_dir() else output
        detector.process_video(source, out_path, show_preview=args.preview, max_frames=args.max_frames)
