"""Make a repeatable development/test split for locally recorded videos.

Initialize once:
    python src/dataset/prepare_dataset.py --input-dir data/raw_videos

IMPORTANT: Edit the 'scenario' column in data/splits/manifest.csv, then run:
    python src/dataset/prepare_dataset.py --resplit

Without scenarios the initial 11/6 split is PROVISIONAL, not stratified.
No deep-learning model training is performed by this script.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

EXTS = {'.mov', '.mp4', '.m4v', '.avi', '.mkv'}
FIELDS = ['video_id', 'filename', 'scenario', 'split', 'split_quality']


def provisional_test_indices(n: int, requested: int) -> set[int]:
    # Deterministic and spread through the recording order; no scenario assumptions.
    if requested <= 0:
        return set()
    return {round((i + 1) * n / requested) - 1 for i in range(requested)}


def choose_splits(rows: list[dict], *, allow_unlabeled: bool) -> list[dict]:
    if not rows:
        raise ValueError('No videos found in manifest')
    unlabeled = [r['filename'] for r in rows if not r.get('scenario', '').strip()]
    if unlabeled:
        if not allow_unlabeled:
            raise ValueError('Fill scenario column for every video before --resplit. Missing: ' + ', '.join(unlabeled))
        # Six hold-outs for 17 clips; one out of three, where possible.
        n_test = max(1, round(len(rows) / 3)) if len(rows) > 1 else 0
        selected = provisional_test_indices(len(rows), n_test)
        for i, row in enumerate(rows):
            row['split'] = 'test' if i in selected else 'development'
            row['split_quality'] = 'PROVISIONAL_UNLABELED'
    else:
        groups = defaultdict(list)
        for r in rows:
            groups[r['scenario'].strip()].append(r)
        for scenario, group in sorted(groups.items()):
            if len(group) < 2:
                raise ValueError(f'Scenario {scenario!r} has only one clip; cannot hold out a test clip')
            for r in group:
                r['split'] = 'development'
                r['split_quality'] = 'SCENARIO_STRATIFIED'
            group[-1]['split'] = 'test'
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('data/raw_videos'))
    parser.add_argument('--manifest', type=Path, default=Path('data/splits/manifest.csv'))
    parser.add_argument('--resplit', action='store_true', help='Require all scenarios to be labeled, then make one test per scenario')
    args = parser.parse_args()
    if not args.input_dir.is_dir():
        parser.error(f'Missing folder {args.input_dir}. Run this script from the project root.')
    paths = sorted(p for p in args.input_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS)
    if not paths:
        parser.error(f'No videos found in {args.input_dir}')

    previous = {}
    if args.manifest.exists():
        with args.manifest.open(newline='', encoding='utf-8') as src:
            previous = {r['filename']: r for r in csv.DictReader(src)}
    rows = []
    for p in paths:
        old = previous.get(p.name, {})
        rows.append({'video_id': p.stem, 'filename': p.name,
                     'scenario': old.get('scenario', '').strip(),
                     'split': old.get('split', ''),
                     'split_quality': old.get('split_quality', '')})

    if args.resplit:
        choose_splits(rows, allow_unlabeled=False)
    elif not previous or any(not r['split'] for r in rows):
        # First run (or newly added clips). Existing split decisions can be re-established after labeling.
        choose_splits(rows, allow_unlabeled=True)
    # Else preserve current split and annotations on rerun (avoid accidental test leakage).

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open('w', newline='', encoding='utf-8') as dst:
        w = csv.DictWriter(dst, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    for kind in ['development', 'test']:
        out = args.manifest.parent / f'{kind}.txt'
        out.write_text(''.join(r['filename'] + '\n' for r in rows if r['split'] == kind), encoding='utf-8')
    counts = Counter(r['split'] for r in rows)
    print(f'Found {len(rows)} videos: {dict(counts)}')
    print('Manifest:', args.manifest)
    if any(r['split_quality'] == 'PROVISIONAL_UNLABELED' for r in rows):
        print('IMPORTANT: split is provisional. Label each scenario then run --resplit before final evaluation.')
    else:
        print('Scenario-stratified: one held-out test clip per scenario. Do not tune with test videos.')


if __name__ == '__main__':
    main()
