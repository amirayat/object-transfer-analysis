"""Optional: transcode iPhone .MOV videos to easily decoded H.264 MP4 copies.

Requires ffmpeg installed (brew install ffmpeg). Raw MOV files are preserved.
Use this only when preflight reports unreadable MOV files.
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,default=Path('data/raw_videos'))
    p.add_argument('--output-dir',type=Path,default=Path('data/processed_videos'))
    p.add_argument('--force',action='store_true')
    p.add_argument('--fps',type=int,default=30,help='Make constant-frame-rate MP4 proxies; annotate the version you process')
    args=p.parse_args()
    if not shutil.which('ffmpeg'):
        p.error('ffmpeg is not installed. On macOS: brew install ffmpeg')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    paths=sorted(p for p in args.source_dir.iterdir() if p.suffix.lower()=='.mov')
    for i,source in enumerate(paths,1):
        target=args.output_dir/(source.stem+'.mp4')
        if target.exists() and not args.force:
            print(f'[{i}/{len(paths)}] exists: {target}');continue
        cmd=['ffmpeg','-hide_banner','-loglevel','error','-y' if args.force else '-n',
             '-i',str(source),'-map','0:v:0','-an','-vf',f'fps={args.fps}','-c:v','libx264','-crf','20',
             '-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
        print(f'[{i}/{len(paths)}] {source} -> {target}',flush=True)
        subprocess.run(cmd,check=True)
    print('Done. For batch processing use --converted-dir data/processed_videos.')

if __name__=='__main__':
    main()
