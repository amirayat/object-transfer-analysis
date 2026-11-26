"""Track people and COCO objects in one video using YOLO11 + BoT-SORT/ByteTrack.

Run from the project root:
    python src/tracking/run_tracking.py --video data/raw_videos/IMG_1586.MOV

Outputs:
    results/tracking/<video_stem>_tracked.mp4
    results/tracking/<video_stem>_tracks.jsonl
    results/tracking/<video_stem>_summary.json
"""

import argparse
import json
import math
from pathlib import Path

import cv2
from ultralytics import YOLO


# COCO class IDs: person=0, bottle=39, cup=41, book=73
TARGET_CLASSES = [0, 39, 41, 73]


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO object detection and tracking")
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--output", type=Path, default=Path("results/tracking"))
    parser.add_argument("--classes", type=int, nargs="+", default=TARGET_CLASSES)
    parser.add_argument("--max-frames", type=int, default=0, help="Optional short smoke test; 0 processes whole video")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.video is None:
        raise ValueError("Pass --video, e.g. data/raw_videos/IMG_1586.MOV")
    if not args.video.is_file():
        raise FileNotFoundError(f"Input video not found: {args.video}")
    if not 0 < args.conf <= 1:
        raise ValueError("--conf must be between 0 and 1")

    args.output.mkdir(parents=True, exist_ok=True)
    video_id = args.video.stem
    video_path = args.output / f"{video_id}_tracked.mp4"
    jsonl_path = args.output / f"{video_id}_tracks.jsonl"
    summary_path = args.output / f"{video_id}_summary.json"

    model = YOLO(args.model)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Video has invalid FPS, width, or height")

    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create video output: {video_path}")

    frame_index = 0
    track_stats = {}
    try:
        with jsonl_path.open("w", encoding="utf-8") as output:
            while True:
                # The frame MUST be read before model.track(frame, ...).
                success, frame = cap.read()
                if not success:
                    break

                if args.max_frames and frame_index >= args.max_frames:
                    break
                result = model.track(
                    source=frame,
                    persist=True,  # carry tracks across video frames
                    tracker=args.tracker,
                    classes=args.classes,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    verbose=False,
                )[0]

                tracks = []
                boxes = result.boxes
                if boxes is not None and boxes.is_track and boxes.id is not None:
                    ids = boxes.id.int().cpu().tolist()
                    class_ids = boxes.cls.int().cpu().tolist()
                    confidences = boxes.conf.cpu().tolist()
                    coordinates = boxes.xyxy.cpu().tolist()

                    for tracker_id, class_id, confidence, xyxy in zip(
                        ids, class_ids, confidences, coordinates
                    ):
                        class_name = str(model.names[class_id])
                        track_id = f"{class_name}_{tracker_id}"
                        x1, y1, x2, y2 = map(float, xyxy)
                        bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

                        tracks.append(
                            {
                                "track_id": track_id,
                                "tracker_numeric_id": int(tracker_id),
                                "class_name": class_name,
                                "bbox": bbox,
                                "confidence": float(confidence),
                            }
                        )

                        # Diagnostics help reveal stationary shelf-book tracks
                        # versus moving book tracks. No automatic filtering.
                        center = ((x1 + x2) / 2, (y1 + y2) / 2)
                        if track_id not in track_stats:
                            track_stats[track_id] = {
                                "class_name": class_name,
                                "first_frame": frame_index,
                                "last_frame": frame_index,
                                "frames_detected": 0,
                                "sum_confidence": 0.0,
                                "path_length_px": 0.0,
                                "_last_center": None,
                                "_last_seen_frame": None,
                            }
                        stat = track_stats[track_id]
                        if stat["_last_center"] is not None:
                            if frame_index - stat["_last_seen_frame"] <= 2:
                                stat["path_length_px"] += math.dist(
                                    center, stat["_last_center"]
                                )
                        stat["last_frame"] = frame_index
                        stat["frames_detected"] += 1
                        stat["sum_confidence"] += float(confidence)
                        stat["_last_center"] = center
                        stat["_last_seen_frame"] = frame_index

                record = {
                    "video_id": video_id,
                    "frame_index": frame_index,
                    "timestamp_sec": round(frame_index / fps, 6),
                    "tracks": tracks,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")

                annotated_frame = result.plot()
                if annotated_frame.shape[:2] != (height, width):
                    annotated_frame = cv2.resize(annotated_frame, (width, height))
                writer.write(annotated_frame)

                frame_index += 1
                if frame_index % 30 == 0:
                    print(f"Processed {frame_index} frames ({frame_index / fps:.1f}s)")
    finally:
        cap.release()
        writer.release()

    summary_tracks = []
    for track_id, stat in sorted(track_stats.items()):
        summary_tracks.append(
            {
                "track_id": track_id,
                "class_name": stat["class_name"],
                "first_frame": stat["first_frame"],
                "last_frame": stat["last_frame"],
                "frames_detected": stat["frames_detected"],
                "mean_confidence": round(
                    stat["sum_confidence"] / stat["frames_detected"], 3
                ),
                "path_length_px": round(stat["path_length_px"], 1),
            }
        )
    summary = {
        "video_id": video_id,
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": frame_index,
        "duration_sec": round(frame_index / fps, 3),
        "model": args.model,
        "tracker": args.tracker,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "target_classes": args.classes,
        "tracks": summary_tracks,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nFinished!")
    print("Annotated video:", video_path)
    print("Frame tracking JSONL:", jsonl_path)
    print("Tracking summary:", summary_path)
    print(f"Total frames: {frame_index} | FPS: {fps:.2f}")
    for t in summary_tracks:
        print(
            f"  {t['track_id']:12} "
            f"frames={t['frames_detected']:4} "
            f"path={t['path_length_px']:7.1f}px "
            f"mean_conf={t['mean_confidence']:.2f}"
        )


if __name__ == "__main__":
    main()
