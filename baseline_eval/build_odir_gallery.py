"""
Build the ODIR-5K retrieval eval gallery: symlink one split's images (from
odir/build_odir_splits.py, typically odir_test.csv) into a flat folder keyed
by the same filename stem eval_model.py uses to align embeddings back to
CSV labels.

Usage:
    python baseline_eval/build_odir_gallery.py \
        --csv odir_data/odir_test.csv \
        --image_dir odir/odir/preprocessed_images \
        --output_dir ./eval_outputs/odir_exp1
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def build_gallery(df: pd.DataFrame, image_dir: Path, gallery_dir: Path) -> int:
    gallery_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    skipped = 0
    for filename in df["filename"]:
        src = image_dir / filename
        link = gallery_dir / filename
        if link.is_symlink() and not link.exists():
            link.unlink()  # broken symlink from a stale run
        if not link.is_symlink():
            if not src.exists():
                skipped += 1
                continue
            link.symlink_to(src)
        count += 1
    if skipped:
        log.warning(f"Could not find {skipped} images under {image_dir}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Build ODIR-5K retrieval eval gallery")
    parser.add_argument("--csv", required=True, help="Path to odir_data/odir_test.csv (or odir_val.csv)")
    parser.add_argument("--image_dir", default="odir/odir/preprocessed_images")
    parser.add_argument("--output_dir", required=True, help="Where to create the paired_data/ gallery")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit to a random N images — useful for a quick smoke test")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, usecols=["filename"])
    if args.max_samples:
        df = df.sample(args.max_samples, random_state=42)
        log.info(f"Limiting to {args.max_samples:,} samples")

    gallery_dir = Path(args.output_dir) / "paired_data"
    # Resolve to an absolute path: symlink targets are stored as-given, and a
    # relative image_dir would resolve relative to gallery_dir, not cwd.
    n = build_gallery(df, Path(args.image_dir).resolve(), gallery_dir)
    log.info(f"Created {n:,} image symlinks → {gallery_dir}")
    log.info(
        f"Run eval with: python baseline_eval/run_odir_evals.py "
        f"--paired_dir {gallery_dir} --csv {args.csv}"
    )


if __name__ == "__main__":
    main()
