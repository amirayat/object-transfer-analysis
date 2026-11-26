"""Run detection -> provisional/verified IDs -> pose -> temporal events -> history on recorded clips.

First check one development clip:
    python scripts/run_batch.py --split development --limit 1
Then process all development clips:
    python scripts/run_batch.py --split development
Run test clips only after setting all scenario labels and freezing thresholds:
    python scripts/run_batch.py --split test

Outputs are separated by video: results/<video_id>/tracking, interaction, transfer, history.
"""
from __future__ import annotations
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def execute(label: str, cmd: list[str], output: Path, *, dry_run: bool, force: bool):
    if output.exists() and not force:
        print(f'  SKIP {label} (exists): {output}')
        return 'skipped'
    print('  ', label, ':', ' '.join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)
    return 'planned' if dry_run else 'completed'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=Path('data/splits/manifest.csv'))
    p.add_argument('--raw-dir', type=Path, default=Path('data/raw_videos'))
    p.add_argument('--converted-dir', type=Path, default=None, help='Use converted MP4 copy of each video when present')
    p.add_argument('--results-dir', type=Path, default=Path('results'))
    p.add_argument('--split', choices=['development', 'test', 'all'], default='development')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--model', default='yolo11s.pt')
    p.add_argument('--tracker', default='botsort.yaml')
    p.add_argument('--imgsz', type=int, default=960)
    p.add_argument('--conf', type=float, default=0.15)
    p.add_argument('--pose-model', default='yolo11n-pose.pt')
    p.add_argument('--pose-imgsz', type=int, default=640)
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--infer-unheld', action='store_true', help='Experimental provisional PUT_DOWN/PICKUP cue')
    p.add_argument('--force', action='store_true', help='Rerun and overwrite stage outputs')
    p.add_argument('--dry-run', action='store_true', help='Print commands; do not execute')
    p.add_argument('--allow-provisional-test', action='store_true',
                   help='Override restriction for checking unlabeled test files; do NOT report as final evaluation')
    args = p.parse_args()
    if not args.manifest.is_file():
        p.error('Manifest missing. First run python src/dataset/prepare_dataset.py')
    if args.limit < 0 or args.max_frames < 0:
        p.error('--limit and --max-frames must be >= 0')
    with args.manifest.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if args.split != 'all':
        rows = [r for r in rows if r['split'] == args.split]
    if args.split == 'test' and any(r.get('split_quality') != 'SCENARIO_STRATIFIED' for r in rows) \
            and not args.allow_provisional_test:
        p.error('Test split is provisional. Fill scenario column then run prepare_dataset.py --resplit.')
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        p.error('No matching videos in manifest')

    python = sys.executable
    summary = []
    failures = 0
    for i, row in enumerate(rows, 1):
        video_id, filename = row['video_id'], row['filename']
        video = args.raw_dir / filename
        if args.converted_dir and (args.converted_dir / f"{video_id}.mp4").is_file():
            video = args.converted_dir / f"{video_id}.mp4"
        output = args.results_dir / video_id
        tracking = output / 'tracking'
        interaction = output / 'interaction'
        transfer = output / 'transfer'
        history = output / 'history'
        raw_tracks = tracking / f'{video_id}_tracks.jsonl'
        entities = tracking / f'{video_id}_entities.jsonl'
        resolved = tracking / f'{video_id}_resolved.jsonl'
        possession = interaction / f'{video_id}_possession.jsonl'
        event_json = transfer / f'{video_id}_events.json'
        history_json = history / f'{video_id}_history.json'
        rec = {'video_id': video_id, 'filename': filename, 'scenario': row.get('scenario', ''),
               'split': row['split'], 'status': 'pending', 'events': '', 'error': ''}
        summary.append(rec)
        print(f'\n[{i}/{len(rows)}] {video_id} ({row["split"]}): {video}', flush=True)
        if not video.exists():
            rec['status'], rec['error'] = 'FAILED', 'Raw input file missing'
            failures += 1
            print('  ERROR:', rec['error'])
            continue
        commands = [
            ('track', [python, 'src/tracking/run_tracking.py', '--video', str(video),
                       '--output', str(tracking), '--model', args.model, '--tracker', args.tracker,
                       '--imgsz', str(args.imgsz), '--conf', str(args.conf),
                       '--classes', '0', '39', '41', '73', '--max-frames', str(args.max_frames)], raw_tracks),
        ]
        identity_cmd = [python, 'src/tracking/resolve_identity.py', '--tracks', str(raw_tracks),
                        '--output', str(resolved), '--entities-output', str(entities)]
        verified_map = Path('config/identity_maps') / f'{video_id}.json'
        if verified_map.exists():
            identity_cmd += ['--identity-map', str(verified_map)]
        commands += [('identity', identity_cmd, entities)]
        pose_cmd = [python, 'src/interaction/estimate_possession.py', '--video', str(video),
                    '--tracks', str(entities), '--output-dir', str(interaction),
                    '--pose-model', args.pose_model, '--pose-imgsz', str(args.pose_imgsz)]
        if args.infer_unheld:
            pose_cmd += ['--infer-unheld']
        commands += [
            ('possession', pose_cmd, possession),
            ('events', [python, 'src/transfer/detect_transfer.py', '--input', str(possession),
                        '--output-dir', str(transfer)], event_json),
            ('history', [python, 'src/history/build_history.py', '--possession', str(possession),
                         '--events', str(event_json), '--output', str(history_json)], history_json),
        ]
        try:
            for name, cmd, expected in commands:
                execute(name, cmd, expected, dry_run=args.dry_run, force=args.force)
            rec['status'] = 'PLANNED' if args.dry_run else 'DONE'
            if not args.dry_run and event_json.exists():
                rec['events'] = len(json.loads(event_json.read_text(encoding='utf-8')).get('events', []))
        except Exception as ex:
            failures += 1
            rec['status'], rec['error'] = 'FAILED', str(ex)
            print('  FAILED:', ex, flush=True)
            # Continue with the next video instead of losing the batch.
    if not args.dry_run:
        args.results_dir.mkdir(parents=True, exist_ok=True)
        out = args.results_dir / 'batch_summary.csv'
        with out.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['video_id', 'filename', 'scenario', 'split', 'status', 'events', 'error'])
            w.writeheader()
            w.writerows(summary)
        print('Batch summary:', out)
    print(f'Batch complete: {len(rows) - failures}/{len(rows)} successful or planned; {failures} failed')
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
