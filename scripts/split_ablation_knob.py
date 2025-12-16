#!/usr/bin/env python3
"""
Split assets/ablation_knob.png (a grid) into individual per-cell images.

Grid structure (by default):
- Columns: [input, knob1, knob06, knob04, knob02, knob01]
- Rows: sample_01..N (header row is auto-detected + dropped by default)

The splitter detects near-white separator bands (gutters) and crops each cell.

Usage:
  python3 scripts/split_ablation_knob.py \
    --input /Users/idave/Downloads/creativevr-webpage/assets/ablation_knob.png \
    --out-dir /Users/idave/Downloads/creativevr-webpage/assets/ablation_knob_cells

Optional tuning:
  --white-threshold 245
  --min-band-thickness 6
  --trim 2
  --keep-header
  --expect-cols 6
  --expect-rows 6
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
    a0: int
    a1: int

    @property
    def thickness(self) -> int:
        return self.a1 - self.a0 + 1


def _find_true_bands(mask_1d: np.ndarray) -> List[Band]:
    bands: List[Band] = []
    start = None
    for i, v in enumerate(mask_1d.tolist()):
        if v and start is None:
            start = i
        if (not v) and start is not None:
            bands.append(Band(start, i - 1))
            start = None
    if start is not None:
        bands.append(Band(start, len(mask_1d) - 1))
    return bands


def _segments_from_bands(length: int, bands: List[Band], *, min_size: int) -> List[Tuple[int, int]]:
    """
    Convert separator bands into content segments, by taking gaps between bands.
    Returns segments as inclusive ranges [(s0,e0), ...].
    """
    if length <= 0:
        return []
    bands = sorted(bands, key=lambda b: b.a0)
    segs: List[Tuple[int, int]] = []
    cur = 0
    for b in bands:
        s0 = cur
        e0 = b.a0 - 1
        if e0 - s0 + 1 >= min_size:
            segs.append((s0, e0))
        cur = b.a1 + 1
    if cur <= length - 1 and (length - 1) - cur + 1 >= min_size:
        segs.append((cur, length - 1))
    return segs


def detect_separator_bands_1d(
    values_1d: np.ndarray, *, white_threshold: float, min_band_thickness: int
) -> List[Band]:
    bright = values_1d >= white_threshold
    return [b for b in _find_true_bands(bright) if b.thickness >= min_band_thickness]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to ablation_knob.png")
    ap.add_argument("--out-dir", required=True, help="Output directory for per-cell crops")
    ap.add_argument(
        "--white-threshold",
        type=float,
        default=245.0,
        help="Mean grayscale threshold to consider a row/col as 'white' (separator).",
    )
    ap.add_argument(
        "--min-band-thickness",
        type=int,
        default=6,
        help="Minimum consecutive white pixels to count as a separator band.",
    )
    ap.add_argument(
        "--trim",
        type=int,
        default=2,
        help="Trim this many pixels from each side of a cell to remove gutters.",
    )
    ap.add_argument(
        "--keep-header",
        action="store_true",
        help="Keep the top header row as row_00_* exports (default: drop header).",
    )
    ap.add_argument("--expect-cols", type=int, default=6, help="Expected number of columns.")
    ap.add_argument("--expect-rows", type=int, default=6, help="Expected number of rows (incl header).")
    args = ap.parse_args()

    img = Image.open(args.input).convert("RGB")
    w, h = img.size
    gray = np.asarray(img.convert("L"), dtype=np.float32)

    # Detect separators as near-white columns/rows (mean over the opposite axis).
    col_mean = gray.mean(axis=0)
    row_mean = gray.mean(axis=1)
    col_bands = detect_separator_bands_1d(
        col_mean, white_threshold=args.white_threshold, min_band_thickness=args.min_band_thickness
    )
    row_bands = detect_separator_bands_1d(
        row_mean, white_threshold=args.white_threshold, min_band_thickness=args.min_band_thickness
    )

    # Convert to content segments. min_size prunes the outer border and any tiny gaps.
    col_segs = _segments_from_bands(w, col_bands, min_size=max(50, w // 20))
    row_segs = _segments_from_bands(h, row_bands, min_size=max(50, h // 20))

    # Heuristic: drop header row if it's much shorter than the others.
    header_dropped = False
    if not args.keep_header and len(row_segs) >= 2:
        heights = [y1 - y0 + 1 for (y0, y1) in row_segs]
        med = float(np.median(heights[1:]))  # ignore the first when computing median
        if heights[0] < 0.60 * med:
            row_segs = row_segs[1:]
            header_dropped = True

    col_keys = ["input", "knob1", "knob06", "knob04", "knob02", "knob01"]
    if args.expect_cols and len(col_segs) != args.expect_cols:
        raise SystemExit(
            f"Expected {args.expect_cols} columns, found {len(col_segs)}. "
            f"Try tuning --white-threshold/--min-band-thickness. "
            f"col_segs={col_segs} col_bands={[ (b.a0,b.a1) for b in col_bands ]}"
        )
    if args.expect_rows:
        expected_rows = args.expect_rows - (1 if header_dropped else 0) if not args.keep_header else args.expect_rows
        if len(row_segs) != expected_rows:
            raise SystemExit(
                f"Expected {expected_rows} rows (after header handling), found {len(row_segs)}. "
                f"Try tuning --white-threshold/--min-band-thickness. "
                f"row_segs={row_segs} row_bands={[ (b.a0,b.a1) for b in row_bands ]}"
            )
    if len(col_keys) != len(col_segs):
        raise SystemExit(f"Column key count ({len(col_keys)}) != detected columns ({len(col_segs)})")

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Input: {args.input}")
    print(f"Size: {w}x{h}")
    print(f"Detected cols: {len(col_segs)} segs={col_segs}")
    print(f"Detected rows: {len(row_segs)} segs={row_segs} (header_dropped={header_dropped})")

    trim = max(0, int(args.trim))
    wrote = 0
    for r, (y0, y1) in enumerate(row_segs, start=1):
        for c, (x0, x1) in enumerate(col_segs):
            # Inclusive -> PIL crop uses (left, top, right_exclusive, bottom_exclusive)
            left = x0 + trim
            top = y0 + trim
            right = (x1 + 1) - trim
            bottom = (y1 + 1) - trim
            if right <= left or bottom <= top:
                raise SystemExit(
                    f"Invalid crop after trim={trim}: r={r} c={c} box=({left},{top},{right},{bottom})"
                )

            cell = img.crop((left, top, right, bottom))
            key = col_keys[c]
            out_path = os.path.join(args.out_dir, f"sample_{r:02d}_{key}.png")
            cell.save(out_path, optimize=True)
            wrote += 1

    print(f"Wrote {wrote} images to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


