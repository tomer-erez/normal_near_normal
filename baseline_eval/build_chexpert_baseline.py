"""
Build a CheXpert external-validation gallery for baseline_eval/run_all_evals.py.

CheXpert ships its own train.csv / valid.csv with the same pathology label
columns (and {1, 0, -1, NaN} encoding — see the Label Encoding table in
CLAUDE.md) as constants.CHEXPERT_LABELS. This script resolves CheXpert's
image paths, optionally keeps frontal views only, and produces:

  - paired_dir: a flat folder of {id}.jpg symlinks, same layout eval_model.py
    already reads via `paired_dir.glob("*.jpg")` (built by build_baseline.py
    for MIMIC-CXR)
  - a labels CSV with the metadata_dicom_id + chexpert_<Label> schema
    eval_model.py loads via constants.LABEL_COLS

No changes to eval_model.py / run_all_evals.py are needed — they already
take --paired_dir/--csv as plain arguments, so this is a drop-in external
(out-of-distribution) validation set alongside the existing MIMIC-CXR one.

Usage
-----
    # Official CheXpert validation set (234 images, expert-verified, no NaN/-1)
    python baseline_eval/build_chexpert_baseline.py \
        --chexpert_dir chexpert/chexpert \
        --split valid \
        --output_dir ./eval_outputs/chexpert_valid

    # Larger, noisier sample from the auto-labeled train split
    python baseline_eval/build_chexpert_baseline.py \
        --chexpert_dir chexpert/chexpert \
        --split train --max_samples 5000 \
        --output_dir ./eval_outputs/chexpert_train_5k

    # Then evaluate exactly like the MIMIC-CXR pipeline:
    python baseline_eval/run_all_evals.py \
        --paired_dir ./eval_outputs/chexpert_valid/paired_data \
        --csv        ./eval_outputs/chexpert_valid/chexpert_valid_labels.csv \
        --output_dir ./experiments/chexpert_valid_eval
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from constants import CHEXPERT_LABELS  # noqa: E402


def resolve_image_path(path_str: str, chexpert_dir: Path) -> Path:
    """
    CSV 'Path' column looks like:
        CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg
    Actual files live at:
        chexpert_dir / valid/patient64541/study1/view1_frontal.jpg
    """
    parts = Path(path_str).parts[1:]  # drop the "CheXpert-v1.0-small" prefix
    return chexpert_dir.joinpath(*parts)


def make_id(path_str: str) -> str:
    """.../patient64541/study1/view1_frontal.jpg -> patient64541_study1_view1_frontal"""
    patient, study, view = Path(path_str).parts[-3:]
    return f"{patient}_{study}_{Path(view).stem}"


def prepare_paired_folder(df: pd.DataFrame, chexpert_dir: Path, paired_dir: Path) -> int:
    paired_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    skipped = 0
    for _, row in df.iterrows():
        img_path = resolve_image_path(row["Path"], chexpert_dir)
        if not img_path.exists():
            skipped += 1
            continue

        link = paired_dir / f"{row['_id']}.jpg"
        if link.is_symlink() and link.resolve() != img_path.resolve():
            link.unlink()
        if not link.is_symlink():
            link.symlink_to(img_path)
        count += 1

    if skipped:
        log.warning(f"Missing {skipped:,} image files on disk (of {len(df):,} rows)")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Build a CheXpert retrieval eval gallery for run_all_evals.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--chexpert_dir", default="chexpert/chexpert",
                        help="Root folder containing train.csv/valid.csv and train/valid image dirs")
    parser.add_argument("--split", default="valid", choices=["train", "valid"],
                        help="'valid' (234 images, expert-verified, no NaN/uncertain labels) is the "
                             "recommended external-validation set. 'train' (~223k images, noisy "
                             "auto-labels) is available for larger-scale checks — use with --max_samples.")
    parser.add_argument("--output_dir", required=True, help="Where to write paired_data/ and the labels CSV")
    parser.add_argument("--frontal_only", action="store_true", default=True,
                        help="Keep only frontal-view images (default: True)")
    parser.add_argument("--include_lateral", dest="frontal_only", action="store_false",
                        help="Include lateral views too")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Random subsample size — useful for --split train")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    chexpert_dir = Path(args.chexpert_dir).resolve()  # absolute: symlink targets must not be cwd-relative
    csv_path = chexpert_dir / f"{args.split}.csv"
    log.info(f"Loading {csv_path}")
    df = pd.read_csv(csv_path)
    log.info(f"Loaded {len(df):,} rows")

    if args.frontal_only:
        before = len(df)
        df = df[df["Frontal/Lateral"] == "Frontal"].reset_index(drop=True)
        log.info(f"Frontal-only filter: {before:,} -> {len(df):,} rows")

    if args.max_samples and args.max_samples < len(df):
        df = df.sample(args.max_samples, random_state=args.seed).reset_index(drop=True)
        log.info(f"Subsampled to {len(df):,} rows")

    df["_id"] = df["Path"].apply(make_id)
    dupes = int(df["_id"].duplicated().sum())
    if dupes:
        raise ValueError(f"{dupes} duplicate ids generated by make_id() — check CheXpert Path format")

    output_dir = Path(args.output_dir)
    paired_dir = output_dir / "paired_data"
    log.info(f"Building paired folder at {paired_dir} …")
    n = prepare_paired_folder(df, chexpert_dir, paired_dir)
    log.info(f"Linked {n:,} images")
    if n == 0:
        log.error("No images linked — check --chexpert_dir")
        sys.exit(1)

    # Labels CSV in the schema eval_model.py expects: metadata_dicom_id +
    # chexpert_<Label>, same {1, 0, -1, NaN} encoding as MIMIC-CXR.
    out_cols = {"metadata_dicom_id": df["_id"]}
    for label in CHEXPERT_LABELS:
        out_cols[f"chexpert_{label}"] = df[label]
    labels_df = pd.DataFrame(out_cols)

    labels_csv_path = output_dir / f"chexpert_{args.split}_labels.csv"
    labels_df.to_csv(labels_csv_path, index=False)
    log.info(f"Saved labels CSV → {labels_csv_path}  ({len(labels_df):,} rows)")

    log.info(
        "Done! Evaluate with:\n"
        f"  python baseline_eval/run_all_evals.py \\\n"
        f"      --paired_dir {paired_dir} \\\n"
        f"      --csv {labels_csv_path} \\\n"
        f"      --output_dir ./experiments/chexpert_{args.split}_eval"
    )


if __name__ == "__main__":
    main()
