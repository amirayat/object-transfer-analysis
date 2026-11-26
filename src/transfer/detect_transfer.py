"""Detect provisional object handovers, pickups and put-downs from possession JSONL.

Input: one JSON object per frame, as produced by estimate_possession.py.
This is a conservative temporal rule-based baseline, NOT a trained model.
It cannot recover a transfer when the previous holder was never observed.

Run:
    python src/transfer/detect_transfer.py \
      --input results/interaction/demo2_possession.jsonl

Outputs:
    results/transfer/demo2_events.json
    results/transfer/demo2_events.csv

Each HANDOVER is provisional unless an upstream interaction module supplies
an explicit boolean `direct_transfer_evidence` in a frame or possession entry.
Even supported events require validation before use as ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_frames(path: Path) -> tuple[str, list[dict]]:
    frames: list[dict] = []
    video_id = None
    prev_idx, prev_time = -1, -1.0
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            frame = json.loads(line)
            vid = frame["video_id"]
            index = int(frame["frame_index"])
            timestamp = float(frame["timestamp_sec"])
            if video_id is None:
                video_id = vid
            if vid != video_id:
                raise ValueError("Each JSONL must contain just one video_id")
            if index <= prev_idx or timestamp < prev_time:
                raise ValueError(f"Frames must be in ascending order (line {line_no})")
            prev_idx, prev_time = index, timestamp
            frames.append(frame)
    if not frames:
        raise ValueError(f"No frames in {path}")
    return video_id, frames


def get_observations(frames: list[dict]) -> dict[str, list[dict]]:
    """Include UNKNOWN records even when an object's entry is absent."""
    ids: set[str] = set()
    for frame in frames:
        for possession in frame.get("possessions", []):
            ids.add(possession["object_id"])
    observations = {}
    for object_id in sorted(ids):
        timeline = []
        for frame in frames:
            matches = [p for p in frame.get("possessions", [])
                       if p.get("object_id") == object_id]
            if len(matches) > 1:
                raise ValueError(f"Duplicate possession for {object_id} in frame {frame['frame_index']}")
            p = matches[0] if matches else {}
            status = p.get("status", "UNKNOWN")
            holder = p.get("holder_id") if status == "HELD" else None
            if status not in {"HELD", "UNHELD", "UNKNOWN"}:
                raise ValueError(f"Invalid status {status!r}")
            if status == "HELD" and not holder:
                raise ValueError(f"HELD requires holder_id (frame {frame['frame_index']})")
            timeline.append({
                "frame_index": int(frame["frame_index"]),
                "time": float(frame["timestamp_sec"]),
                "status": status,
                "holder": holder,
                "reason": p.get("reason"),
                # Future interaction module can supply a contact/motion cue.
                "direct_cue": bool(p.get("direct_transfer_evidence", False)
                                   or frame.get("direct_transfer_evidence", False)),
            })
        observations[object_id] = timeline
    return observations


def make_segments(timeline: list[dict], bridge_unknown_frames: int) -> list[dict]:
    """Group states, and tolerate a few UNKNOWN frames within the same holder."""
    segments = []
    for i, rec in enumerate(timeline):
        key = (rec["status"], rec["holder"])
        if segments and segments[-1]["key"] == key:
            segments[-1]["end"] = i
            segments[-1]["observed"] += 1
        else:
            segments.append({"key": key, "start": i, "end": i, "observed": 1})

    # Merge HELD(A), short UNKNOWN, HELD(A) to resist dropped frames.
    changed = True
    while changed:
        changed = False
        merged = []
        i = 0
        while i < len(segments):
            if i + 2 < len(segments):
                a, middle, b = segments[i:i+3]
                if (a["key"][0] == "HELD" and a["key"] == b["key"]
                        and middle["key"][0] == "UNKNOWN"
                        and middle["observed"] <= bridge_unknown_frames):
                    merged.append({"key": a["key"], "start": a["start"],
                                   "end": b["end"], "observed": a["observed"] + b["observed"]})
                    i += 3
                    changed = True
                    continue
            merged.append(segments[i])
            i += 1
        segments = merged
    return segments


def build_events(video_id: str, objects: dict[str, list[dict]], *,
                 min_held_frames: int, min_unheld_frames: int,
                 bridge_unknown_frames: int, max_transition_sec: float,
                 cooldown_sec: float) -> tuple[list[dict], dict]:
    events: list[dict] = []
    diagnostics = {}
    for object_id, timeline in objects.items():
        segments = make_segments(timeline, bridge_unknown_frames)
        stable = [s for s in segments if
                  (s["key"][0] == "HELD" and s["observed"] >= min_held_frames)
                  or (s["key"][0] == "UNHELD" and s["observed"] >= min_unheld_frames)]
        held_counts = Counter(r["holder"] for r in timeline if r["status"] == "HELD")
        diagnostics[object_id] = {
            "held_frame_counts": dict(sorted(held_counts.items())),
            "unknown_frames": sum(r["status"] == "UNKNOWN" for r in timeline),
            "stable_held_runs": [
                {"holder_id": s["key"][1],
                 "from_sec": round(timeline[s["start"]]["time"], 3),
                 "to_sec": round(timeline[s["end"]]["time"], 3),
                 "held_observations": s["observed"]}
                for s in stable if s["key"][0] == "HELD"
            ],
            "ignored_long_or_ambiguous_transitions": 0,
        }

        last_event_by_kind: dict[tuple, float] = {}
        for before, after in zip(stable, stable[1:]):
            status_a, holder_a = before["key"]
            status_b, holder_b = after["key"]
            if before["key"] == after["key"]:
                continue
            start_rec = timeline[before["end"]]
            end_rec = timeline[after["start"]]
            dt = end_rec["time"] - start_rec["time"]
            interval = timeline[before["end"] + 1:after["start"]]
            if dt < 0 or dt > max_transition_sec:
                diagnostics[object_id]["ignored_long_or_ambiguous_transitions"] += 1
                continue
            if any(r["status"] == "HELD" and r["holder"] not in {holder_a, holder_b}
                   for r in interval):
                diagnostics[object_id]["ignored_long_or_ambiguous_transitions"] += 1
                continue

            if status_a == "HELD" and status_b == "HELD" and holder_a != holder_b:
                # An observed UNHELD interval suggests put-down + pickup, not handover.
                if any(r["status"] == "UNHELD" for r in interval):
                    diagnostics[object_id]["ignored_long_or_ambiguous_transitions"] += 1
                    continue
                event_type = "HANDOVER"
                source, target = holder_a, holder_b
            elif status_a == "UNHELD" and status_b == "HELD":
                event_type, source, target = "PICKUP", None, holder_b
            elif status_a == "HELD" and status_b == "UNHELD":
                event_type, source, target = "PUT_DOWN", holder_a, None
            else:
                continue

            dedup_key = (event_type, source, target)
            if end_rec["time"] - last_event_by_kind.get(dedup_key, -1e9) < cooldown_sec:
                continue

            # This is supporting contextual evidence, not proof of physical contact.
            neighborhood = timeline[max(0, before["end"]-2):min(len(timeline), after["start"]+3)]
            both_near = any(r["reason"] == "ambiguous_two_people" for r in neighborhood)
            explicit_cue = any(r["direct_cue"] for r in neighborhood)
            # A heuristic score, NOT a calibrated probability.
            score = min(0.95, 0.55 + (0.10 if before["observed"] >= 10 else 0)
                        + (0.10 if after["observed"] >= 10 else 0)
                        + (0.05 if both_near else 0) + (0.10 if explicit_cue else 0))
            events.append({
                "event_id": "",  # Assigned after merging events from all objects.
                "video_id": video_id,
                "object_id": object_id,
                "event_type": event_type,
                "from_person_id": source,
                "to_person_id": target,
                "start_time_sec": round(start_rec["time"], 3),
                "end_time_sec": round(end_rec["time"], 3),
                "confidence": round(score, 3),
                "validation_status": "SUPPORTED" if explicit_cue else "PROVISIONAL",
                "review_required": True,
                "evidence": {
                    "stable_before_frames": before["observed"],
                    "stable_after_frames": after["observed"],
                    "transition_gap_sec": round(dt, 3),
                    "two_people_near_object": both_near,
                    "direct_transfer_evidence": explicit_cue,
                    "method": "temporal_rules",
                },
            })
            last_event_by_kind[dedup_key] = end_rec["time"]

    events.sort(key=lambda e: (e["end_time_sec"], e["object_id"], e["event_type"]))
    for i, event in enumerate(events, 1):
        event["event_id"] = f"event_{i:03d}"
    return events, diagnostics


def save_csv(path: Path, events: list[dict]) -> None:
    columns = ["event_id", "video_id", "object_id", "event_type", "from_person_id",
               "to_person_id", "start_time_sec", "end_time_sec", "confidence",
               "validation_status", "review_required"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for event in events:
            writer.writerow({key: event.get(key) for key in columns})


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal object transfer detector")
    parser.add_argument("--input", type=Path, required=True,
                        help="Possession JSONL from src/interaction/estimate_possession.py")
    parser.add_argument("--output-dir", type=Path, default=Path("results/transfer"))
    parser.add_argument("--min-held-frames", type=int, default=5)
    parser.add_argument("--min-unheld-frames", type=int, default=5)
    parser.add_argument("--bridge-unknown-frames", type=int, default=2)
    parser.add_argument("--max-transition-sec", type=float, default=1.2)
    parser.add_argument("--cooldown-sec", type=float, default=0.6)
    args = parser.parse_args()
    if args.min_held_frames <= 0 or args.min_unheld_frames <= 0 or args.bridge_unknown_frames < 0:
        parser.error("frame thresholds must be positive (bridge-unknown-frames may be zero)")
    if args.max_transition_sec <= 0 or args.cooldown_sec < 0:
        parser.error("max-transition-sec must be positive and cooldown-sec nonnegative")

    video_id, frames = load_frames(args.input)
    observations = get_observations(frames)
    if not observations:
        print("WARNING: no objects in possession file; reporting zero events")
    events, diagnostics = build_events(video_id, observations,
                                      min_held_frames=args.min_held_frames,
                                      min_unheld_frames=args.min_unheld_frames,
                                      bridge_unknown_frames=args.bridge_unknown_frames,
                                      max_transition_sec=args.max_transition_sec,
                                      cooldown_sec=args.cooldown_sec)
    output_json = args.output_dir / f"{video_id}_events.json"
    output_csv = args.output_dir / f"{video_id}_events.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "video_id": video_id,
        "method": "temporal_rule_based",
        "all_events_are_review_required": True,
        "settings": {
            "min_held_frames": args.min_held_frames,
            "min_unheld_frames": args.min_unheld_frames,
            "bridge_unknown_frames": args.bridge_unknown_frames,
            "max_transition_sec": args.max_transition_sec,
            "cooldown_sec": args.cooldown_sec,
        },
        "events": events,
        "diagnostics": diagnostics,
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_csv(output_csv, events)
    print(f"Read {len(frames)} frames for {video_id}")
    print(f"Events: {len(events)} | by type: {dict(Counter(e['event_type'] for e in events))}")
    if not events:
        print("No supported holder transition found; this is NOT evidence that no transfer occurred.")
    for object_id, diag in diagnostics.items():
        print(f"  {object_id}: HELD frames by person = {diag['held_frame_counts']}; "
              f"stable HELD runs = {len(diag['stable_held_runs'])}")
    print("JSON:", output_json)
    print("CSV: ", output_csv)


if __name__ == "__main__":
    main()
