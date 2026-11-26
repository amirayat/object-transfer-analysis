"""Evaluate transfer-event type and time against independently annotated videos.

This basic metric ignores object/person identity. Evaluate identity accuracy separately.
Matches event type and midpoint within a fixed time tolerance; one-to-one matching.
Never use held-out test clips to tune thresholds.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=Path('data/splits/manifest.csv'))
    p.add_argument('--annotations-dir', type=Path, default=Path('data/annotations'))
    p.add_argument('--results-dir', type=Path, default=Path('results'))
    p.add_argument('--split', choices=['development', 'test'], default='test')
    p.add_argument('--tolerance-sec', type=float, default=1.0)
    p.add_argument('--allow-provisional-test', action='store_true')
    args = p.parse_args()
    with args.manifest.open(newline='', encoding='utf-8') as f:
        manifest = [r for r in csv.DictReader(f) if r['split'] == args.split]
    if args.split == 'test' and any(r.get('split_quality') != 'SCENARIO_STRATIFIED' for r in manifest) \
            and not args.allow_provisional_test:
        p.error('Test set is provisional. Label scenario for all videos and run prepare_dataset.py --resplit.')
    if not manifest:
        p.error('No videos in selected split')
    TP = FP = FN = 0
    rows = []
    for row in manifest:
        vid = row['video_id']
        gt_path = args.annotations_dir / f'{vid}.json'
        predicted = args.results_dir / vid / 'transfer' / f'{vid}_events.json'
        if not gt_path.exists() or not predicted.exists():
            print(f'SKIP {vid}: ground truth or prediction missing')
            continue
        gt = json.loads(gt_path.read_text(encoding='utf-8'))
        if gt.get('annotation_status') != 'ANNOTATED' or gt.get('events') is None:
            print(f'SKIP {vid}: annotation not complete')
            continue
        expected = gt['events']
        got = json.loads(predicted.read_text(encoding='utf-8')).get('events', [])
        pairs = []
        for i, e in enumerate(expected):
            for j, pred in enumerate(got):
                if e['event_type'] == pred['event_type']:
                    ground_time = (float(e['start_sec']) + float(e['end_sec'])) / 2
                    pred_time = (float(pred['start_time_sec']) + float(pred['end_time_sec'])) / 2
                    delta = abs(ground_time - pred_time)
                    if delta <= args.tolerance_sec:
                        pairs.append((delta, i, j))
        matched_gt, matched_pred = set(), set()
        for _, i, j in sorted(pairs):
            if i not in matched_gt and j not in matched_pred:
                matched_gt.add(i); matched_pred.add(j)
        tp = len(matched_gt)
        fp = len(got) - tp
        fn = len(expected) - tp
        TP += tp; FP += fp; FN += fn
        rows.append({'video_id': vid, 'scenario': row.get('scenario'), 'TP': tp, 'FP': fp, 'FN': fn})
    if not rows:
        raise RuntimeError('No completed annotated videos with predictions; cannot calculate metrics')
    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    print(f'Evaluated {len(rows)} {args.split} videos | TP={TP} FP={FP} FN={FN}')
    print(f'Event precision={precision:.3f} recall={recall:.3f} F1={f1:.3f}')
    print('WARNING: type + time matching only, NOT object identity accuracy or exact event boundary accuracy.')
    out = args.results_dir / f'{args.split}_event_evaluation.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['video_id', 'scenario', 'TP', 'FP', 'FN'])
        w.writeheader(); w.writerows(rows)
    print('Details:', out)


if __name__ == '__main__':
    main()
