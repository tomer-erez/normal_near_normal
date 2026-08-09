"""
Split ODIR-5K's full_df.csv into train/val/test CSVs, keyed by patient ID so a
patient's left and right eye images never land in different splits (both rows
for one patient share the same N/D/G/C/A/H/M/O label vector, so splitting by
row would leak near-duplicate label signal across train and eval).

Usage:
    python baseline_eval/build_odir_splits.py \
        --csv odir/odir/full_df.csv \
        --output-dir odir_data \
        --val-frac 0.1 --test-frac 0.1
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description="Build patient-level train/val/test splits for ODIR-5K")
    p.add_argument("--csv", required=True, help="Path to odir/odir/full_df.csv")
    p.add_argument("--output-dir", required=True, help="Directory to write odir_train.csv / odir_val.csv / odir_test.csv")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    log.info(f"Loaded {len(df):,} rows ({df['ID'].nunique():,} patients) from {args.csv}")

    patient_ids = df["ID"].unique()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    n_val = int(n * args.val_frac)
    n_test = int(n * args.test_frac)
    val_ids = set(patient_ids[:n_val])
    test_ids = set(patient_ids[n_val:n_val + n_test])
    train_ids = set(patient_ids[n_val + n_test:])

    splits = {
        "train": df[df["ID"].isin(train_ids)].reset_index(drop=True),
        "val": df[df["ID"].isin(val_ids)].reset_index(drop=True),
        "test": df[df["ID"].isin(test_ids)].reset_index(drop=True),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, split_df in splits.items():
        out_path = output_dir / f"odir_{name}.csv"
        split_df.to_csv(out_path, index=False)
        log.info(f"{name}: {len(split_df):,} rows ({split_df['ID'].nunique():,} patients) → {out_path}")


if __name__ == "__main__":
    main()
