"""Resolve verified YOLO/BoT-SORT identities and export verified entities.

Run from your project root:
    python src/tracking/resolve_identity.py

Optional overrides:
    python src/tracking/resolve_identity.py --tracks PATH --identity-map PATH \
        --output PATH --entities-output PATH

IMPORTANT: config/identity_map.json must refer to the SAME tracking run as --tracks.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracks", type=Path,
                   default=Path("results/tracking/demo2_tracks.jsonl"))
    p.add_argument("--identity-map", type=Path,
                   default=Path("config/identity_map.json"))
    p.add_argument("--output", type=Path,
                   default=Path("results/tracking/demo2_resolved.jsonl"))
    p.add_argument("--entities-output", type=Path,
                   default=Path("results/tracking/demo2_entities.jsonl"))
    return p.parse_args()


def mapping_for_verified(config):
    # raw tracker id -> list[(first_frame, last_frame, permanent_id, class_name)]
    lookup = defaultdict(list)
    for entity in config.get("identities", []):
        if entity.get("identity_status") != "verified":
            continue
        stable_id = entity["object_id"]
        class_name = entity["class_name"]
        tracker_ids = entity["tracker_ids"]
        ranges = entity.get("active_ranges") or [
            {"tracker_id": tid, "first_frame": 0, "last_frame": 10**12}
            for tid in tracker_ids
        ]
        for span in ranges:
            raw_id = span["tracker_id"]
            if raw_id not in tracker_ids:
                raise ValueError(f"{raw_id} is not in tracker_ids for {stable_id}")
            first = int(span["first_frame"])
            last = int(span["last_frame"])
            if first > last:
                raise ValueError(f"Invalid frame range for {raw_id}: {first}-{last}")
            lookup[raw_id].append((first, last, stable_id, class_name))
    return lookup


def main():
    args = arguments()
    if not args.tracks.is_file():
        raise FileNotFoundError(f"Tracking file not found: {args.tracks}")
    if not args.identity_map.is_file():
        raise FileNotFoundError(f"Identity map not found: {args.identity_map}")
    config = json.loads(args.identity_map.read_text(encoding="utf-8"))
    lookup = mapping_for_verified(config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.entities_output.parent.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    frames = 0

    with args.tracks.open("r", encoding="utf-8") as src, \
         args.output.open("w", encoding="utf-8") as all_out, \
         args.entities_output.open("w", encoding="utf-8") as entities_out:
        for line in src:
            if not line.strip():
                continue
            frame = json.loads(line)
            frame_idx = int(frame["frame_index"])
            if frame["video_id"] != config["video_id"]:
                raise ValueError("Video ID differs between tracking and identity map")

            verified_tracks = []
            seen_ids = set()
            for raw_track in frame["tracks"]:
                track = dict(raw_track)
                raw_id = track["track_id"]
                matches = [rule for rule in lookup.get(raw_id, [])
                           if rule[0] <= frame_idx <= rule[1]
                           and rule[3] == track["class_name"]]
                if len(matches) > 1:
                    raise ValueError(f"Ambiguous mapping: {raw_id} at frame {frame_idx}")
                if matches:
                    permanent_id = matches[0][2]
                    if permanent_id in seen_ids:
                        raise ValueError(
                            f"Two simultaneous detections mapped to {permanent_id} "
                            f"at frame {frame_idx}; review active_ranges"
                        )
                    seen_ids.add(permanent_id)
                    track["object_id"] = permanent_id
                    track["identity_status"] = "verified"
                    verified_tracks.append(track)
                    stats[permanent_id] += 1
                else:
                    track["object_id"] = None
                    track["identity_status"] = "unmapped"
                # Preserve original detections, including unmapped ones
                raw_track.update({"object_id": track["object_id"],
                                  "identity_status": track["identity_status"]})

            all_out.write(json.dumps(frame) + "\n")
            entities_out.write(json.dumps({
                "video_id": frame["video_id"],
                "frame_index": frame_idx,
                "timestamp_sec": frame["timestamp_sec"],
                "tracks": verified_tracks,
            }) + "\n")
            frames += 1

    print(f"Processed {frames} frames")
    for entity_id, count in sorted(stats.items()):
        print(f"  {entity_id}: {count} verified detections")
    print(f"All tracks: {args.output}")
    print(f"Verified entities only: {args.entities_output}")
    if stats.get("object_001", 0) == 0:
        print("WARNING: object_001 has zero verified detections. Make sure "
              "the identity map refers to THIS BoT-SORT tracking run.")
    if not any(id.startswith("person_") for id in stats):
        print("WARNING: No verified people. Map the giver and receiver before pose estimation.")


if __name__ == "__main__":
    main()
