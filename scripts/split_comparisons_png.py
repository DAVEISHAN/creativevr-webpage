#!/usr/bin/env python3
"""
Split the qualitative grid PNG in assets/comparisons.png into per-sample images.

Behavior:
- Detect horizontal separator "white bands" between samples
- Detect the header height near the top
- For each sample, export an image that is: header + that sample body

Usage:
  python3 scripts/split_comparisons_png.py \
    --input /Users/idave/Downloads/creativevr-webpage/assets/comparisons.png \
    --out-dir /Users/idave/Downloads/creativevr-webpage/assets/comparisons_samples

Optional overrides:
  --header-height 60
  --white-threshold 245
  --min-band-height 6
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Band:
    y0: int
    y1: int

    @property
    def height(self) -> int:
        return self.y1 - self.y0 + 1


def _find_true_bands(mask_1d: np.ndarray) -> List[Band]:
    bands: List[Band] = []
    start = None
    for y, v in enumerate(mask_1d.tolist()):
        if v and start is None:
            start = y
        if (not v) and start is not None:
            bands.append(Band(start, y - 1))
            start = None
    if start is not None:
        bands.append(Band(start, len(mask_1d) - 1))
    return bands


def detect_separator_bands(
    gray: np.ndarray, *, white_threshold: float, min_band_height: int
) -> List[Band]:
    row_mean = gray.mean(axis=1)
    bright = row_mean >= white_threshold
    bands = [b for b in _find_true_bands(bright) if b.height >= min_band_height]
    return bands


def detect_header_height(gray: np.ndarray) -> int:
    """
    Heuristics (in order):
    1) Detect the first sustained dark region (start of the first sample content). This works
       well for our qualitative grid where the header is bright and the first sample begins
       with darker video content.
    2) Fallback: find the first y where row std becomes "high" for a sustained window,
       indicating we've left the relatively flat header background and entered image content.
    """
    row_mean = gray.mean(axis=1)
    win_m = 15
    sm = np.convolve(row_mean, np.ones(win_m) / win_m, mode="same")

    # Dynamic threshold based on the top portion (mostly header). Clamp for safety.
    top_med = float(np.median(sm[20:260])) if gray.shape[0] > 280 else float(np.median(sm[:200]))
    thr_dark = max(120.0, min(200.0, top_med - 50.0))

    sustain_m = 5
    search_h = min(1200, gray.shape[0] - sustain_m - 1)
    for y in range(20, search_h):
        if (sm[y : y + sustain_m] < thr_dark).all():
            return int(y)

    row_std = gray.std(axis=1)
    win = 5
    roll = np.convolve(row_std, np.ones(win) / win, mode="same")

    # Tune: header is fairly uniform; content rows have lots of variance.
    # Require sustained high variance.
    thresh = 40.0
    sustain = 10
    for y in range(0, min(400, gray.shape[0] - sustain)):
        if (roll[y : y + sustain] > thresh).all():
            return max(1, y)
    # Fallback
    return 260


def compute_sample_ranges(
    *, height: int, header_height: int, separator_bands: List[Band]
) -> List[Tuple[int, int]]:
    """
    Returns body ranges [y0, y1] for each sample (excluding header).
    """
    # Filter bands below the header area (avoid accidental header whites)
    bands = [b for b in separator_bands if b.y0 > header_height + 5]
    bands = sorted(bands, key=lambda b: b.y0)

    ranges: List[Tuple[int, int]] = []
    cur = header_height
    for b in bands:
        y0 = cur
        y1 = b.y0 - 1
        if y1 > y0:
            ranges.append((y0, y1))
        cur = b.y1 + 1

    # Tail after last separator
    if cur < height - 1:
        ranges.append((cur, height - 1))
    return ranges


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to the comparisons PNG.")
    ap.add_argument("--out-dir", required=True, help="Output directory for samples.")
    ap.add_argument(
        "--header-height",
        type=int,
        default=0,
        help="Override header height in pixels (0 = auto-detect).",
    )
    ap.add_argument(
        "--white-threshold",
        type=float,
        default=245.0,
        help="Row mean grayscale threshold to consider a row as 'white'.",
    )
    ap.add_argument(
        "--min-band-height",
        type=int,
        default=6,
        help="Minimum consecutive white rows to consider a separator band.",
    )
    args = ap.parse_args()

    img = Image.open(args.input).convert("RGB")
    w, h = img.size
    gray = np.asarray(img.convert("L"), dtype=np.float32)

    header_h = args.header_height if args.header_height > 0 else detect_header_height(gray)
    header_h = int(max(1, min(header_h, h - 1)))

    sep_bands = detect_separator_bands(
        gray, white_threshold=args.white_threshold, min_band_height=args.min_band_height
    )
    ranges = compute_sample_ranges(height=h, header_height=header_h, separator_bands=sep_bands)

    os.makedirs(args.out_dir, exist_ok=True)
    header = img.crop((0, 0, w, header_h))

    print(f"Input: {args.input}")
    print(f"Size: {w}x{h}")
    print(f"Header height: {header_h}")
    print(f"Separator bands: {[ (b.y0,b.y1) for b in sep_bands ]}")
    print(f"Samples found: {len(ranges)}")

    for i, (y0, y1) in enumerate(ranges, start=1):
        body = img.crop((0, y0, w, y1 + 1))
        out = Image.new("RGB", (w, header.height + body.height), (255, 255, 255))
        out.paste(header, (0, 0))
        out.paste(body, (0, header.height))

        out_path = os.path.join(args.out_dir, f"sample_{i:02d}.png")
        out.save(out_path, optimize=True)
        print(f"Wrote: {out_path}  (body y={y0}:{y1})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


