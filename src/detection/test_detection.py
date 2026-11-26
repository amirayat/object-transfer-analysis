"""Quick preflight for MOV/MP4 videos: decode frames and optionally run YOLO detection.

python test_detection.py --video data/raw_videos/IMG_1586.MOV
python test_detection.py --video-dir data/raw_videos --decode-only
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

EXTS = {'.mov', '.mp4', '.m4v', '.avi', '.mkv'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--video', type=Path)
    group.add_argument('--video-dir', type=Path)
    p.add_argument('--decode-only', action='store_true')
    p.add_argument('--frames', type=int, default=3, help='Number of frames to sample per video')
    p.add_argument('--model', default='yolo11s.pt')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--report', type=Path, default=Path('results/preflight.csv'))
    args = p.parse_args()
    if args.frames < 1:
        p.error('--frames must be >=1')
    paths = [args.video] if args.video else sorted(
        f for f in args.video_dir.iterdir() if f.suffix.lower() in EXTS and f.is_file())
    if not paths:
        p.error('No videos found')

    import cv2
    model = None
    if not args.decode_only:
        from ultralytics import YOLO
        model = YOLO(args.model)
    results = []
    for path in paths:
        row = {'filename': path.name, 'decodes': False, 'width': 0, 'height': 0,
               'fps': 0, 'sampled_frames': 0, 'person_detections': 0,
               'object_detections': 0, 'message': ''}
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            row['message'] = 'OpenCV cannot open this file; transcode with ffmpeg'
        else:
            row['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            row['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            row['fps'] = round(float(cap.get(cv2.CAP_PROP_FPS)), 3)
            for _ in range(args.frames):
                success, frame = cap.read()
                if not success:
                    break
                row['decodes'] = True
                row['sampled_frames'] += 1
                if model is not None:
                    r = model.predict(frame, classes=[0, 39, 41, 73], imgsz=args.imgsz,
                                      conf=0.20, verbose=False)[0]
                    if r.boxes is not None:
                        classes = r.boxes.cls.int().cpu().tolist()
                        row['person_detections'] += classes.count(0)
                        row['object_detections'] += sum(c in (39, 41, 73) for c in classes)
            if not row['decodes']:
                row['message'] = 'No readable frames; likely codec issue or corrupt recording'
        cap.release()
        results.append(row)
        print(f"{path.name:20} decode={row['decodes']} size={row['width']}x{row['height']} "
              f"people={row['person_detections']} objects={row['object_detections']} {row['message']}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print('Preflight report:', args.report)


if __name__ == '__main__':
    main()
