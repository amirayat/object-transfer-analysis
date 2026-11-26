"""Create annotation templates for the videos listed in the dataset manifest.

The optional interaction label BOTH_TOUCHING is stored inside a
possession_intervals entry. No model or evaluator changes are needed.

Run from the Project2 root:
    python scripts/init_annotations.py
    python scripts/init_annotations.py --validate

Existing files are never overwritten.
"""

import argparse
import csv
import json
from pathlib import Path


def make_template(video_id, scenario):
    return {
        "video_id": video_id,
        "scenario": scenario or None,
        "annotation_status": "NOT_ANNOTATED",
        "people": None,
        "objects": None,
        "possession_intervals": None,
        "events": None,
        "notes": (
            "Watch the ORIGINAL video. Enter observed people, objects, possession "
            "intervals and event times. For each possession interval, you may "
            "add the optional field 'interaction': 'BOTH_TOUCHING' only when "
            "both people visibly touch the object. UNKNOWN means the holder "
            "is uncertain; it does not by itself mean both touching. Set "
            "events=[] only after confirming no relevant events."
        ),
    }


def validate_annotation(path, expected_video_id):
    data = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    if data.get("video_id") != expected_video_id:
        errors.append("video_id does not match manifest")
    if data.get("annotation_status") != "ANNOTATED":
        return ["annotation_status is not ANNOTATED"]
    for key in ("people", "objects", "possession_intervals", "events"):
        if not isinstance(data.get(key), list):
            errors.append(f"{key} must be a list (use [] when intentionally empty)")
    if errors:
        return errors

    person_ids = {p.get("person_id") for p in data["people"] if isinstance(p, dict)}
    object_ids = {o.get("object_id") for o in data["objects"] if isinstance(o, dict)}

    def valid_interval(item, name, index):
        try:
            start = float(item["start_sec"])
            end = float(item["end_sec"])
            if start < 0 or end <= start:
                errors.append(f"{name}[{index}]: expected 0 <= start_sec < end_sec")
        except (KeyError, TypeError, ValueError):
            errors.append(f"{name}[{index}]: invalid start_sec/end_sec")

    for i, interval in enumerate(data["possession_intervals"]):
        if not isinstance(interval, dict):
            errors.append(f"possession_intervals[{i}] must be an object")
            continue
        valid_interval(interval, "possession_intervals", i)
        if interval.get("object_id") not in object_ids:
            errors.append(f"possession_intervals[{i}]: unknown object_id")
        status, holder = interval.get("status"), interval.get("holder_id")
        if status not in ("HELD", "UNHELD", "UNKNOWN"):
            errors.append(f"possession_intervals[{i}]: invalid status")
        if status == "HELD" and holder not in person_ids:
            errors.append(f"possession_intervals[{i}]: HELD needs a known holder_id")
        if status != "HELD" and holder is not None:
            errors.append(f"possession_intervals[{i}]: non-HELD holder_id must be null")
        if interval.get("interaction") not in (None, "BOTH_TOUCHING"):
            errors.append(f"possession_intervals[{i}]: unsupported interaction label")

    for i, event in enumerate(data["events"]):
        if not isinstance(event, dict):
            errors.append(f"events[{i}] must be an object")
            continue
        valid_interval(event, "events", i)
        if event.get("event_type") not in ("HANDOVER", "PICKUP", "PUT_DOWN"):
            errors.append(f"events[{i}]: invalid event_type")
        if event.get("object_id") not in object_ids:
            errors.append(f"events[{i}]: unknown object_id")
        if event.get("from_person_id") is not None and event["from_person_id"] not in person_ids:
            errors.append(f"events[{i}]: unknown from_person_id")
        if event.get("to_person_id") is not None and event["to_person_id"] not in person_ids:
            errors.append(f"events[{i}]: unknown to_person_id")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/annotations"))
    parser.add_argument("--validate", action="store_true",
                        help="Validate completed annotations instead of creating templates")
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    failed = 0
    checked = 0
    for row in rows:
        video_id = row["video_id"]
        path = args.output_dir / f"{video_id}.json"
        if args.validate:
            if not path.exists():
                print(f"MISSING {video_id}")
                failed += 1
                continue
            problems = validate_annotation(path, video_id)
            if problems:
                print(f"REVIEW {video_id}: {'; '.join(problems)}")
                failed += 1
            else:
                print(f"OK {video_id}")
                checked += 1
        elif not path.exists():
            path.write_text(json.dumps(make_template(video_id, row.get("scenario", "")),
                                       indent=2) + "\n", encoding="utf-8")
            created += 1
    if args.validate:
        print(f"Validated {checked} completed annotations; {failed} missing/incomplete/invalid.")
        if failed:
            raise SystemExit(1)
    else:
        print(f"Created {created} templates; existing files preserved in {args.output_dir}.")


if __name__ == "__main__":
    main()
