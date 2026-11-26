"""Create unannotated JSON templates for every video in dataset manifest.

Does not overwrite existing ground-truth labels and does not invent events.
"""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=Path('data/splits/manifest.csv'))
    p.add_argument('--output-dir', type=Path, default=Path('data/annotations'))
    args = p.parse_args()
    with args.manifest.open(newline='', encoding='utf-8') as src:
        videos = list(csv.DictReader(src))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    for row in videos:
        path = args.output_dir / f"{row['video_id']}.json"
        if path.exists():
            continue
        payload = {
            'video_id': row['video_id'], 'scenario': row['scenario'] or None,
            'annotation_status': 'NOT_ANNOTATED',
            'people': None, 'objects': None, 'possession_intervals': None,
            'events': None,
            'notes': 'Watch the ORIGINAL video; enter actual event time intervals. events=[] means confirmed no events.'
        }
        path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        created += 1
    print(f'Created {created} annotation templates in {args.output_dir}; existing labels preserved.')

if __name__ == '__main__':
    main()
