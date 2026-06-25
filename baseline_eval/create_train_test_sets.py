"""
Filter all_txt_data_and_labels.csv into official MIMIC-CXR train and test splits
using mimic-cxr-2.0.0-split.csv.gz.

Produces two CSVs (train and test) containing only rows that have image files on disk.

Usage
-----
    python baseline_eval/create_train_test_sets.py \
        --csv      cxr_data/all_txt_data_and_labels.csv \
        --split    /path/to/mimic-cxr-2.0.0-split.csv.gz \
        --out_train cxr_data/mimic_cxr_train.csv \
        --out_test  cxr_data/mimic_cxr_official_test.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--csv", required=True,
                        help="Path to all_txt_data_and_labels.csv")
    parser.add_argument("--split", required=True,
                        help="Path to mimic-cxr-2.0.0-split.csv.gz (from PhysioNet)")
    parser.add_argument("--out_train", default="cxr_data/mimic_cxr_train.csv",
                        help="Output path for the train CSV")
    parser.add_argument("--out_test", default="cxr_data/mimic_cxr_official_test.csv",
                        help="Output path for the test CSV")
    args = parser.parse_args()

    log.info(f"Loading split file: {args.split}")
    split_df = pd.read_csv(args.split, compression="gzip")

    test_dicom_ids = set(split_df[split_df["split"] == "test"]["dicom_id"].tolist())
    train_dicom_ids = set(split_df[split_df["split"] == "train"]["dicom_id"].tolist())
    log.info(f"Official test set:  {len(test_dicom_ids):,} dicom_ids")
    log.info(f"Official train set: {len(train_dicom_ids):,} dicom_ids")

    log.info(f"Loading: {args.csv}")
    df = pd.read_csv(args.csv)
    log.info(f"Total rows: {len(df):,}")

    test_df = df[df["metadata_dicom_id"].isin(test_dicom_ids)].reset_index(drop=True)
    train_df = df[df["metadata_dicom_id"].isin(train_dicom_ids)].reset_index(drop=True)
    log.info(f"Rows in official test split:  {len(test_df):,}")
    log.info(f"Rows in official train split: {len(train_df):,}")

    for out_path, split_df_out, label in [
        (args.out_test, test_df, "test"),
        (args.out_train, train_df, "train"),
    ]:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        split_df_out.to_csv(path, index=False)
        log.info(f"Saved {label} → {path}  ({split_df_out.shape[1]} columns)")


if __name__ == "__main__":
    main()
