#!/usr/bin/env python3
"""
Crop the "first column" (left side) out of already-split sample PNGs.

Example:
  python3 scripts/split_samples_first_column.py \
    --input-dir /Users/idave/Downloads/creativevr-webpage/assets/comparisons_samples \
    --out-dir /Users/idave/Downloads/creativevr-webpage/assets/comparisons_samples_firstcol \
    --first-col-width 2378

Optionally also write the remaining columns:
  --also-rest --rest-out-dir /path/to/rest_out
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from PIL import Image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Directory containing sample_*.png")
    ap.add_argument("--out-dir", required=True, help="Output directory for first-column crops")
    ap.add_argument(
        "--first-col-width",
        type=int,
        required=True,
        help="Width in pixels of the first column (crop x in [0, width)).",
    )
    ap.add_argument(
        "--glob",
        default="sample_*.png",
        help="Glob pattern to select inputs within input-dir (default: sample_*.png).",
    )
    ap.add_argument(
        "--also-rest",
        action="store_true",
        help="Also export the remaining columns (crop x in [width, W)).",
    )
    ap.add_argument(
        "--rest-out-dir",
        default="",
        help="Output dir for the remaining-columns crops (required if --also-rest).",
    )
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rest_dir: Path | None = None
    if args.also_rest:
        if not args.rest_out_dir:
            raise SystemExit("--rest-out-dir is required when --also-rest is set")
        rest_dir = Path(args.rest_out_dir)
        rest_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(in_dir.glob(args.glob))
    if not paths:
        raise SystemExit(f"No inputs found in {in_dir} matching {args.glob}")

    for p in paths:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        x = int(max(1, min(args.first_col_width, w - 1)))

        first = img.crop((0, 0, x, h))
        out_path = out_dir / p.name
        first.save(out_path, optimize=True)

        if rest_dir is not None:
            rest = img.crop((x, 0, w, h))
            rest_path = rest_dir / p.name
            rest.save(rest_path, optimize=True)

        print(f"Wrote first col: {out_path}  (x=0:{x}, size={x}x{h})")
        if rest_dir is not None:
            print(f"Wrote rest:      {rest_path}  (x={x}:{w}, size={w-x}x{h})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


