"""Aggregate per-frame possession observations and detected events by object.

This is an observation history, not ground truth. UNKNOWN stays UNKNOWN.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--possession', type=Path, required=True)
    p.add_argument('--events', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    payload = json.loads(args.events.read_text(encoding='utf-8'))
    events = payload.get('events', [])
    stats = defaultdict(Counter)
    timelines = defaultdict(list)
    video_id = payload['video_id']
    for line in args.possession.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        frame = json.loads(line)
        if frame['video_id'] != video_id:
            raise ValueError('Event/possession video IDs do not match')
        for obs in frame.get('possessions', []):
            oid = obs['object_id']
            status = obs.get('status', 'UNKNOWN')
            holder = obs.get('holder_id') if status == 'HELD' else None
            key = f'{status}:{holder}' if holder else status
            stats[oid][key] += 1
            timelines[oid].append((frame['frame_index'], frame['timestamp_sec'],
                                   status, holder, obs.get('class_name')))
    for e in events:
        stats.setdefault(e['object_id'], Counter())
    objects = []
    for oid in sorted(stats):
        items = timelines.get(oid, [])
        intervals = []
        for index, t, status, holder, _ in items:
            if intervals and intervals[-1]['status'] == status and intervals[-1]['holder_id'] == holder \
                    and index == intervals[-1]['last_frame'] + 1:
                intervals[-1]['last_frame'] = index
                intervals[-1]['end_sec'] = t
            else:
                intervals.append({'status': status, 'holder_id': holder,
                                  'first_frame': index, 'last_frame': index,
                                  'start_sec': t, 'end_sec': t})
        classes = Counter(x[4] for x in items if x[4])
        objects.append({'object_id': oid, 'class_name': classes.most_common(1)[0][0] if classes else None,
                        'observation_counts': dict(stats[oid]), 'possession_intervals': intervals,
                        'detected_events': [e for e in events if e['object_id'] == oid]})
    out = {'video_id': video_id, 'review_required': True,
           'note': 'Events and possession intervals are model-derived; UNKNOWN is not absence of transfer.',
           'objects': objects}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'{video_id}: {len(objects)} object histories, {len(events)} detected events -> {args.output}')


if __name__ == '__main__':
    main()
