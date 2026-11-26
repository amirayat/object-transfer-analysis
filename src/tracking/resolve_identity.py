"""Assign provisional per-video identities, optionally applying a verified manual map.

No supplied identity map is reused by default. This prevents demo2 IDs being
silently applied to new recordings. Provisional IDs are NOT verified physical
identities; a tracker ID switch can split a real person/object into two IDs.

Example:
 python src/tracking/resolve_identity.py --tracks results/IMG_1586/tracking/IMG_1586_tracks.jsonl \
   --output results/IMG_1586/tracking/IMG_1586_resolved.jsonl \
   --entities-output results/IMG_1586/tracking/IMG_1586_entities.jsonl
"""
from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def raw_key(track: dict) -> tuple[str, int | str]:
    """Keep persons separate from objects, but tolerate book <-> cup switches."""
    n = track.get('tracker_numeric_id')
    if n is None:
        # Earlier tracking exports only had e.g. 'book_40'.
        n = track['track_id'].rsplit('_', 1)[-1]
    entity_type = 'person' if track['class_name'] == 'person' else 'object'
    return entity_type, str(n)


def verified_mapping(config: dict, video_id: str):
    if not config:
        return defaultdict(list)
    if config.get('video_id') != video_id:
        raise ValueError(f'Identity map belongs to {config.get("video_id")!r}, not {video_id!r}')
    lookup = defaultdict(list)
    for e in config.get('identities', []):
        if e.get('identity_status') != 'verified':
            continue
        intervals = e.get('active_ranges') or [
            {'tracker_id': tid, 'first_frame': 0, 'last_frame': 10**12}
            for tid in e['tracker_ids']]
        for span in intervals:
            raw = span['tracker_id']
            if raw not in e['tracker_ids']:
                raise ValueError(f'{raw} not in tracker_ids for {e["object_id"]}')
            lookup[raw].append((int(span['first_frame']), int(span['last_frame']),
                                e['object_id'], e['class_name']))
    return lookup


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tracks', type=Path, required=True)
    p.add_argument('--identity-map', type=Path, help='OPTIONAL verified map for this exact video and tracking run')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--entities-output', type=Path, required=True)
    p.add_argument('--min-track-frames', type=int, default=3, help='Exclude 1-2 frame spurious tracks (not a detection-quality guarantee)')
    args = p.parse_args()
    if not args.tracks.is_file():
        p.error(f'Missing tracking file: {args.tracks}')
    if args.min_track_frames < 1:
        p.error('--min-track-frames must be >= 1')

    # First pass, track ID counts and dominant object type per raw numeric ID.
    observations = Counter()
    class_votes = defaultdict(Counter)
    first_frame = {}
    video_id = None
    count_frames = 0
    with args.tracks.open(encoding='utf-8') as src:
        for line in src:
            if not line.strip():
                continue
            frame = json.loads(line)
            video_id = video_id or frame['video_id']
            if frame['video_id'] != video_id:
                raise ValueError('Tracking JSONL cannot contain multiple video IDs')
            count_frames += 1
            for tr in frame['tracks']:
                key = raw_key(tr)
                observations[key] += 1
                class_votes[key][tr['class_name']] += 1
                first_frame.setdefault(key, frame['frame_index'])
    if not count_frames:
        raise ValueError('Empty tracking file')

    config = json.loads(args.identity_map.read_text(encoding='utf-8')) if args.identity_map else {}
    manual = verified_mapping(config, video_id)
    eligible = sorted((key for key, n in observations.items() if n >= args.min_track_frames),
                      key=lambda key: (first_frame[key], key[0], int(key[1]) if key[1].isdigit() else key[1]))
    manual_ids = {rule[2] for rules in manual.values() for rule in rules}
    allocated = set(manual_ids)
    provisional = {}
    counter = Counter()
    for key in eligible:
        kind = key[0]
        while True:
            counter[kind] += 1
            name = f'person_{counter[kind]:03d}' if kind == 'person' else f'object_{counter[kind]:03d}'
            if name not in allocated:
                break
        allocated.add(name)
        provisional[key] = name

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.entities_output.parent.mkdir(parents=True, exist_ok=True)
    reported = Counter()
    verified_ids_seen = set()
    with args.tracks.open(encoding='utf-8') as src, \
         args.output.open('w', encoding='utf-8') as all_out, \
         args.entities_output.open('w', encoding='utf-8') as entity_out:
        for line in src:
            if not line.strip():
                continue
            frame = json.loads(line)
            seen = set()
            resolved = []
            entity_tracks = []
            idx = int(frame['frame_index'])
            for track in frame['tracks']:
                tr = dict(track)
                raw_id = tr['track_id']
                matches = [m for m in manual.get(raw_id, [])
                           if m[0] <= idx <= m[1] and m[3] == tr['class_name']]
                if len(matches) > 1:
                    raise ValueError(f'Ambiguous verified map for {raw_id} at frame {idx}')
                key = raw_key(tr)
                if matches:
                    tr['object_id'] = matches[0][2]
                    tr['identity_status'] = 'verified'
                    verified_ids_seen.add(tr['object_id'])
                elif key in provisional:
                    tr['object_id'] = provisional[key]
                    tr['identity_status'] = 'provisional'
                else:
                    tr['object_id'] = None
                    tr['identity_status'] = 'unmapped'
                tr['identity_strategy'] = 'manual_map' if matches else ('tracker_id_only' if tr['object_id'] else 'excluded_short_track')
                if tr['object_id']:
                    if tr['object_id'] in seen:
                        raise ValueError(f'Multiple tracks mapped to {tr["object_id"]} at frame {idx}')
                    seen.add(tr['object_id'])
                    entity_tracks.append(tr)
                    reported[tr['object_id']] += 1
                resolved.append(tr)
            all_out.write(json.dumps({**frame, 'tracks': resolved}, ensure_ascii=False) + '\n')
            entity_out.write(json.dumps({**frame, 'tracks': entity_tracks}, ensure_ascii=False) + '\n')

    summary_path = args.entities_output.with_name(f'{video_id}_identity_summary.json')
    summary = {
        'video_id': video_id,
        'frames': count_frames,
        'mapped_detection_counts': dict(sorted(reported.items())),
        'verified_ids_observed': sorted(verified_ids_seen),
        'warning': 'Provisional IDs correspond to tracker fragments, not necessarily stable physical identities.',
        'provisional_tracks': [
            {'raw_key': f'{kind}:{num}', 'object_id': provisional[(kind, num)],
             'observations': observations[(kind, num)],
             'dominant_class': class_votes[(kind, num)].most_common(1)[0][0]}
            for kind, num in eligible],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'{video_id}: {count_frames} frames; {len(provisional)} provisional track identities, '
          f'{len(verified_ids_seen)} verified identities observed')
    print('Entities:', args.entities_output)
    print('Identity summary:', summary_path)
    print('WARNING: automatic raw-track identities do not stitch ID switches.')


if __name__ == '__main__':
    main()
