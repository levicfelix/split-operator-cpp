#!/usr/bin/env python3
"""
Small FFmpeg helper for building an MP4 from numbered PNG frames.
"""

from __future__ import annotations
import subprocess
from pathlib import Path


def frames_to_mp4(frames_dir: Path, mp4_name: str, fps: int = 10, pattern: str = "frame_%05d.png") -> None:
    frames_dir = Path(frames_dir)
    inp = frames_dir / pattern

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(int(fps)),
        "-i", str(inp),
        # ensure even width/height for libx264
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(mp4_name),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

