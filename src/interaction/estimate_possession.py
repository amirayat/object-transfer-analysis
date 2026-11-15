"""Pose + wrist-to-object proximity baseline for persistent identities.

Consumes the identity-resolved JSONL (demo2_entities.jsonl), not raw tracker IDs.
This is NOT proof of physical holding. Manual inspection is required.

Example from the project root:
python src/interaction/estimate_possession.py \
  --video data/videos/demo2.mp4 \
  --tracks results/tracking/demo2_entities.jsonl \
  --object-id object_001 \
  --pose-imgsz 960
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def xyxy(bbox):
    return [float(bbox[k]) for k in ("x1", "y1", "x2", "y2")]


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    area_i = max(0., x2 - x1) * max(0., y2 - y1)
    area_a = max(0., a[2] - a[0]) * max(0., a[3] - a[1])
    area_b = max(0., b[2] - b[0]) * max(0., b[3] - b[1])
    union = area_a + area_b - area_i
    return area_i / union if union > 0 else 0.0


def gap_to_box(x, y, box):
    x1, y1, x2, y2 = box
    return math.hypot(max(x1 - x, 0., x - x2), max(y1 - y, 0., y - y2))


def assign_pose_to_people(people, result, min_iou=0.15):
    """Match independent pose results to person identities using one-to-one IoU."""
    if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
        return {}
    pose_boxes = result.boxes.xyxy.cpu().numpy()
    wrist_xy = result.keypoints.xy.cpu().numpy()
    wrist_conf = (result.keypoints.conf.cpu().numpy()
                  if result.keypoints.conf is not None else None)

    matches = []
    for pose_index, pose_box in enumerate(pose_boxes):
        for person in people:
            match_score = box_iou(pose_box, xyxy(person["bbox"]))
            if match_score >= min_iou:
                matches.append((match_score, pose_index, person["object_id"]))
    matches.sort(reverse=True)

    used_poses, used_people, out = set(), set(), {}
    for _, pose_index, person_id in matches:
        if pose_index in used_poses or person_id in used_people:
            continue
        used_poses.add(pose_index)
        used_people.add(person_id)
        wrists = []
        for keypoint_index, name in [(9, "left"), (10, "right")]:
            x, y = map(float, wrist_xy[pose_index, keypoint_index])
            conf = (float(wrist_conf[pose_index, keypoint_index])
                    if wrist_conf is not None else None)
            if x > 0 and y > 0:
                wrists.append({"side": name, "x": round(x, 2),
                               "y": round(y, 2), "confidence": conf})
        out[person_id] = wrists
    return out


def estimate(obj_box, people, person_wrists, hand_gap_ratio=0.10,
             min_wrist_conf=0.35, ambiguity_margin=0.25):
    """Return a provisional spatial holder, or UNKNOWN if ambiguous/too far."""
    candidates = []
    for p in people:
        stable_id = p["object_id"]
        height = float(p["bbox"]["y2"] - p["bbox"]["y1"])
        max_gap = max(12.0, min(100.0, height * hand_gap_ratio))
        wrists = [w for w in person_wrists.get(stable_id, [])
                  if w["confidence"] is None or w["confidence"] >= min_wrist_conf]
        if not wrists:
            continue
        best = min(gap_to_box(w["x"], w["y"], obj_box) for w in wrists)
        candidates.append({"person_id": stable_id,
                           "wrist_gap_px": round(best, 2),
                           "relative_gap": round(best / max_gap, 3)})
    candidates.sort(key=lambda c: c["relative_gap"])

    if not candidates:
        return None, "UNKNOWN", "no_visible_wrist", candidates
    first = candidates[0]
    if first["relative_gap"] > 1.0:
        return None, "UNKNOWN", "wrists_not_near_object", candidates
    if (len(candidates) > 1 and candidates[1]["relative_gap"] <= 1.0
            and candidates[1]["relative_gap"] - first["relative_gap"] < ambiguity_margin):
        return None, "UNKNOWN", "ambiguous_two_people", candidates
    return first["person_id"], "HELD", "wrist_proximity_only", candidates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True,
                   help="Original, unannotated source video")
    p.add_argument("--tracks", type=Path, required=True,
                   help="Persistent-identity entities JSONL (not raw YOLO tracks)")
    p.add_argument("--object-id", action="append", default=[],
                   help="Stable object ID, e.g. object_001 (can repeat)")
    p.add_argument("--pose-model", default="yolo11n-pose.pt")
    p.add_argument("--pose-imgsz", type=int, default=960)
    p.add_argument("--min-iou", type=float, default=0.15)
    p.add_argument("--min-wrist-conf", type=float, default=0.35)
    p.add_argument("--hand-gap-ratio", type=float, default=0.10)
    p.add_argument("--output-dir", type=Path, default=Path("results/interaction"))
    args = p.parse_args()

    import cv2

    if not args.video.exists() or not args.tracks.exists():
        p.error("Input video or JSONL file not found; run from the project root")

    # Collect stable IDs first so every frame has an explicit UNKNOWN record
    # when a tracked object is temporarily not detected.
    object_ids = set(args.object_id)
    if not object_ids:
        with args.tracks.open("r", encoding="utf-8") as src:
            for line in src:
                if line.strip():
                    for obj in json.loads(line)["tracks"]:
                        if obj["class_name"] != "person" and obj.get("object_id"):
                            object_ids.add(obj["object_id"])
    if not object_ids:
        p.error("No stable object IDs found. Check identity_map.json and resolved entities JSONL")

    # Fail fast: an --object-id argument by itself must NOT count as evidence
    # that the corresponding object exists in the input. Check mapped tracks
    # before loading the pose model or processing the video.
    detected_objects = set()
    detected_people = set()
    matched_frames = 0
    with args.tracks.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            frame_data = json.loads(line)
            frame_has_requested_object = False
            for t in frame_data.get("tracks", []):
                stable_id = t.get("object_id")
                if stable_id is None:
                    continue
                if t.get("class_name") == "person":
                    detected_people.add(stable_id)
                elif stable_id in object_ids:
                    detected_objects.add(stable_id)
                    frame_has_requested_object = True
            matched_frames += int(frame_has_requested_object)

    absent_objects = object_ids - detected_objects
    if absent_objects:
        p.error(
            "No verified detections for requested object ID(s): "
            + ", ".join(sorted(absent_objects))
            + ". The JSONL may have object_id: null everywhere. "
            "Regenerate resolved identities and use demo2_entities.jsonl "
            "or demo2_resolved_verified.jsonl."
        )
    if not detected_people:
        p.error(
            "No persistent person identities found in --tracks. "
            "Create a verified person mapping before possession estimation."
        )
    print(
        "Validated mapped entities: "
        f"objects={sorted(detected_objects)}, "
        f"persons={sorted(detected_people)}, "
        f"frames_with_requested_object={matched_frames}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.output_dir / f"{args.video.stem}_possession.jsonl"
    out_video = args.output_dir / f"{args.video.stem}_possession_preview.mp4"

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width, height = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Bad video metadata")
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not open output VideoWriter")

    from ultralytics import YOLO
    model = YOLO(args.pose_model)
    count = 0
    try:
        with args.tracks.open("r", encoding="utf-8") as src, \
             out_jsonl.open("w", encoding="utf-8") as dst:
            for raw in src:
                if not raw.strip():
                    continue
                tracked = json.loads(raw)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Video ended before the tracking JSONL")
                if tracked["frame_index"] != count:
                    raise ValueError(f"Expected frame {count}, got {tracked['frame_index']}")
                people = [t for t in tracked["tracks"]
                          if t["class_name"] == "person" and t.get("object_id")]
                objects = {t["object_id"]: t for t in tracked["tracks"]
                           if t["class_name"] != "person" and t.get("object_id")
                           and t["object_id"] in object_ids}

                pose_result = model.predict(frame, conf=0.30,
                                            imgsz=args.pose_imgsz, verbose=False)[0]
                wrists = assign_pose_to_people(people, pose_result, args.min_iou)
                for person in people:
                    name = person["object_id"]
                    for w in wrists.get(name, []):
                        if w["confidence"] is not None and w["confidence"] < args.min_wrist_conf:
                            continue
                        cv2.circle(frame, (round(w["x"]), round(w["y"])),
                                   6, (0, 255, 255), -1)
                        cv2.putText(frame, name, (round(w["x"]), max(20, round(w["y"])-10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

                records = []
                for oid in sorted(object_ids):
                    if oid not in objects:
                        records.append({"object_id": oid, "holder_id": None,
                                        "status": "UNKNOWN", "confidence": None,
                                        "object_detected": False,
                                        "reason": "object_not_detected",
                                        "candidates": []})
                        continue
                    obj = objects[oid]
                    box = xyxy(obj["bbox"])
                    holder, status, reason, candidates = estimate(
                        box, people, wrists, args.hand_gap_ratio, args.min_wrist_conf)
                    records.append({"object_id": oid, "holder_id": holder,
                                    "status": status, "confidence": None,
                                    "object_detected": True,
                                    "reason": reason, "candidates": candidates})
                    a,b,c,d = map(round, box)
                    cv2.rectangle(frame, (a,b), (c,d), (80,220,80), 2)
                    cv2.putText(frame, f"{oid}: {holder or 'UNKNOWN'} (provisional)",
                                (max(a,0),max(20,b-10)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.46, (80,220,80), 2)

                dst.write(json.dumps({
                    "video_id": tracked["video_id"],
                    "frame_index": count,
                    "timestamp_sec": tracked["timestamp_sec"],
                    "persons": [{"person_id": p["object_id"],
                                 "wrists": wrists.get(p["object_id"], [])} for p in people],
                    "possessions": records,
                    "provisional": True
                }) + "\n")
                writer.write(frame)
                count += 1
    finally:
        cap.release()
        writer.release()
    print(f"Processed {count} frames")
    print(f"Saved proximity estimates: {out_jsonl}")
    print(f"Saved annotated preview: {out_video}")
    print("CAUTION: HELD is only a wrist-proximity hypothesis; verify visually.")


if __name__ == "__main__":
    main()
