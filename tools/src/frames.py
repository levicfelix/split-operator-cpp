#!/usr/bin/env python3
"""
Generate PNG frames (and optional MP4) from wave packet trajectory snapshots.

Run via generated wrapper:
  plot_trajectory filename
  plot_trajectory filename --MAKE_MP4 True

It expects:
  filename_snapshots/wave_header.bin
  filename_snapshots/wave_<k>.bin

It writes:
  filename_frames/frame_00000.png, ...
  optional MP4: filename.mp4
"""

from __future__ import annotations
from pathlib import Path
import argparse
import re
import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

import ffmpeg as ffmpeg_mod


_WAVE_RE = re.compile(r"^wave_(\d+)\.bin$")


# -------------------- defaults (from your original script) --------------------
OUTDIR_SUFFIX = "_frames"
SNAPDIR_SUFFIX = "_snapshots"

FPS = 10
MP4_SUFFIX = ".mp4"

CMAP_NAME = "jet"
N_BANDS = 9
INTERP = "nearest"

FIXED_SCALE_MODE = "initial"  # "initial" | "global" | None

USE_PERCENTILE_SCALE = True
PCTL_LO = 0.5
PCTL_HI = 99.5

VMIN = None
VMAX = 1e-5

PLOT_GRAIN_BOUNDARY = True
GRAIN_BOUNDARY_X = 0.0
GRAIN_BOUNDARY_STYLE = dict(linestyle="--", linewidth=2.0, color="k")
# -----------------------------------------------------------------------------


def _str2bool(x: str) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean (True/False), got: {x!r}")


def list_snapshot_files(snapdir: Path) -> list[Path]:
    if not snapdir.is_dir():
        raise SystemExit(f"Missing folder: {snapdir}")

    files = []
    for p in snapdir.iterdir():
        if p.is_file() and _WAVE_RE.match(p.name):
            files.append(p)
    files.sort(key=lambda p: int(_WAVE_RE.match(p.name).group(1)))
    return files


def extract_k(p: Path, fallback: int) -> int:
    m = _WAVE_RE.match(p.name)
    return int(m.group(1)) if m else fallback


def read_header(path: Path):
    with open(path, "rb") as f:
        raw = f.read(16)
    if len(raw) != 16:
        raise ValueError(f"{path}: expected 16 bytes, got {len(raw)}")
    Nx, Ny, dx, dy = struct.unpack("<ii ff", raw)
    return int(Nx), int(Ny), float(dx), float(dy)


def load_prob_bin(fname: Path, Nx: int, Ny: int) -> np.ndarray:
    arr = np.fromfile(fname, dtype="<f4")
    n_expected = Nx * Ny
    if arr.size != n_expected:
        raise ValueError(f"{fname}: expected {n_expected} float32 values, got {arr.size}")
    return arr.reshape((Nx, Ny), order="C")


def apply_caps(lo, hi):
    if VMIN is not None:
        lo = VMIN
    if VMAX is not None:
        hi = VMAX
    return lo, hi


def scale_from_frame(P: np.ndarray):
    if USE_PERCENTILE_SCALE:
        lo = float(np.nanpercentile(P, PCTL_LO))
        hi = float(np.nanpercentile(P, PCTL_HI))
    else:
        lo = float(np.nanmin(P))
        hi = float(np.nanmax(P))

    lo, hi = apply_caps(lo, hi)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(P))
        hi = float(np.nanmax(P))
        lo, hi = apply_caps(lo, hi)

    return lo, hi


def compute_scale_initial(files, Nx, Ny):
    P0 = load_prob_bin(files[0], Nx, Ny)
    return scale_from_frame(P0)


def compute_scale_global(files, Nx, Ny):
    lo = np.inf
    hi = -np.inf
    for f in files:
        P = load_prob_bin(f, Nx, Ny)
        a, b = scale_from_frame(P)
        lo = min(lo, a)
        hi = max(hi, b)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None, None
    return lo, hi


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="plot_trajectory",
        description="Generate PNG frames (and optional MP4) from filename_snapshots/wave_*.bin",
    )
    p.add_argument("filename", help="Base name; reads from filename_snapshots/")
    p.add_argument("--MAKE_MP4", type=_str2bool, default=False,
                   help="If True, also generate MP4 (default: False)")
    p.add_argument("--FPS", type=int, default=FPS, help=f"MP4 frames per second (default: {FPS})")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    base = str(args.filename)
    snapdir = Path(base + SNAPDIR_SUFFIX)
    outdir = Path(base + OUTDIR_SUFFIX)
    mp4_name = base + MP4_SUFFIX

    header_file = snapdir / "wave_header.bin"
    if not header_file.exists():
        raise SystemExit(f"Missing {header_file}. It must exist to decode .bin snapshots.")

    Nx, Ny, dx, dy = read_header(header_file)

    files = list_snapshot_files(snapdir)
    if not files:
        raise SystemExit(f"No snapshot files found in {snapdir} matching wave_<k>.bin (e.g. wave_0.bin).")

    outdir.mkdir(parents=True, exist_ok=True)

    # Coordinates (centered at 0)
    xs = (np.arange(Nx) - Nx / 2.0) * dx
    ys = (np.arange(Ny) - Ny / 2.0) * dy
    extent = [xs.min(), xs.max(), ys.min(), ys.max()]

    # Fixed scale
    vmin = vmax = None
    if FIXED_SCALE_MODE == "initial":
        vmin, vmax = compute_scale_initial(files, Nx, Ny)
    elif FIXED_SCALE_MODE == "global":
        vmin, vmax = compute_scale_global(files, Nx, Ny)

    cmap = plt.get_cmap(CMAP_NAME, N_BANDS)

    fixed_levels = None
    fixed_norm = None
    if vmin is not None and vmax is not None and np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
        fixed_levels = np.linspace(vmin, vmax, N_BANDS + 1)
        fixed_norm = BoundaryNorm(fixed_levels, cmap.N, clip=True)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.set_xlabel("x (Angstrom)")
    ax.set_ylabel("y (Angstrom)")
    ax.set_aspect("equal")

    if PLOT_GRAIN_BOUNDARY:
        ax.axvline(GRAIN_BOUNDARY_X, **GRAIN_BOUNDARY_STYLE)

    # Init first frame
    P0 = load_prob_bin(files[0], Nx, Ny)
    Zt0 = P0.T

    if fixed_norm is None:
        lo, hi = scale_from_frame(P0)
        levels = np.linspace(lo, hi, N_BANDS + 1)
        norm = BoundaryNorm(levels, cmap.N, clip=True)
    else:
        levels = fixed_levels
        norm = fixed_norm

    im = ax.imshow(
        Zt0,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=cmap,
        norm=norm,
        interpolation=INTERP,
    )

    cbar = fig.colorbar(im, ax=ax, boundaries=levels, ticks=levels)
    cbar.set_label(r"$|\psi(x,y)|^2$")

    # Frame loop
    for frame_idx, f in enumerate(files):
        k = extract_k(f, frame_idx)
        P = load_prob_bin(f, Nx, Ny)
        Zt = P.T

        if fixed_norm is None:
            lo, hi = scale_from_frame(P)
            levels = np.linspace(lo, hi, N_BANDS + 1)
            norm = BoundaryNorm(levels, cmap.N, clip=True)
            im.set_norm(norm)
            cbar.set_ticks(levels)
            cbar.update_normal(im)

        im.set_data(Zt)
        ax.set_title(f"{f.name} (k={k})")

        out_png = outdir / f"frame_{frame_idx:05d}.png"
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        print(f"Wrote {out_png}   (from {f.name}, k={k})")

    plt.close(fig)

    if args.MAKE_MP4:
        ffmpeg_mod.frames_to_mp4(frames_dir=outdir, mp4_name=mp4_name, fps=int(args.FPS))
        print(f"Wrote {mp4_name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

