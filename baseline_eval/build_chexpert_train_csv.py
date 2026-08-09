"""
Convert CheXpert's raw train.csv/valid.csv into a training CSV that
train/chexpert_label_dataset.py (ChexpertLabelDataset) can read directly.

CheXpert's raw CSV has a 'Path' column like:
    CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg
and label columns named exactly (e.g. "Atelectasis", "Pleural Effusion").

This script drops the "CheXpert-v1.0-small" prefix (the remainder resolves
directly under --chexpert_dir, e.g. chexpert/chexpert/train/patient00001/...)
and renames the label columns from constants.CHEXPERT_LABELS to
"chexpert_<Label>", matching the schema CXRLabelDataset/ChexpertLabelDataset
and baseline_eval/eval_model.py already use for MIMIC-CXR.

This is a training-data converter only — for building an eval gallery
(paired_dir of symlinks + labels CSV), use build_chexpert_baseline.py instead.

Usage
-----
    python baseline_eval/build_chexpert_train_csv.py \
        --chexpert_dir chexpert/chexpert --split train \
        --output_csv chexpert_data/chexpert_train.csv
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


def _relative_path(path_str: str) -> str:
    """'CheXpert-v1.0-small/train/patient00001/...' -> 'train/patient00001/...'"""
    return "/".join(Path(path_str).parts[1:])


def main():
    parser = argparse.ArgumentParser(
        description="Build a CheXpert training CSV in ChexpertLabelDataset's schema",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--chexpert_dir", default="chexpert/chexpert",
                        help="Root folder containing train.csv/valid.csv and train/valid image dirs")
    parser.add_argument("--split", default="train", choices=["train", "valid"])
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--frontal_only", action="store_true", default=True,
                        help="Keep only frontal-view images (default: True)")
    parser.add_argument("--include_lateral", dest="frontal_only", action="store_false",
                        help="Include lateral views too")
    args = parser.parse_args()

    chexpert_dir = Path(args.chexpert_dir).resolve()
    csv_path = chexpert_dir / f"{args.split}.csv"
    log.info(f"Loading {csv_path}")
    df = pd.read_csv(csv_path)
    log.info(f"Loaded {len(df):,} rows")

    if args.frontal_only:
        before = len(df)
        df = df[df["Frontal/Lateral"] == "Frontal"].reset_index(drop=True)
        log.info(f"Frontal-only filter: {before:,} -> {len(df):,} rows")

    out = pd.DataFrame({"path": df["Path"].apply(_relative_path)})
    for label in CHEXPERT_LABELS:
        out[f"chexpert_{label}"] = df[label]

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    log.info(f"Saved → {output_csv}  ({len(out):,} rows)")


if __name__ == "__main__":
    main()
