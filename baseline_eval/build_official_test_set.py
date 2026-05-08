"""
Filter all_txt_data_and_labels.csv to the official MIMIC-CXR test split
using mimic-cxr-2.0.0-split.csv.gz.

Produces a CSV of only test-set rows that also have image files on disk,
ready to be passed to build_baseline.py (with --max_samples omitted to use
all test images, or set to a smaller number for faster iteration).

Usage
-----
    python baseline/build_official_test_set.py \
        --csv      cxr_data/all_txt_data_and_labels.csv \
        --split    /path/to/mimic-cxr-2.0.0-split.csv.gz \
        --img_root /mnt/walkure_public/users/tomererez/mimic_cxr_jpg_images \
        --out      cxr_data/mimic_cxr_official_test.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SPLIT_FILE = (
    "/mnt/walkure_public/users/tomererez/mimic_cxr_jpg_images"
    "/mimic_cxr_jpg_images_from_google_cloud"
    "/mimic-cxr-jpg-2.1.0.physionet.org/mimic-cxr-2.0.0-split.csv.gz"
)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--csv", default="cxr_data/all_txt_data_and_labels.csv")
    parser.add_argument("--split", default=SPLIT_FILE)
    parser.add_argument("--out", default="cxr_data/mimic_cxr_official_test.csv")
    args = parser.parse_args()

    # Load official split
    log.info(f"Loading split file: {args.split}")
    split_df = pd.read_csv(args.split, compression="gzip")
    test_dicom_ids = set(split_df[split_df["split"] == "test"]["dicom_id"].tolist())
    log.info(f"Official test set: {len(test_dicom_ids):,} dicom_ids")

    # Load main CSV
    log.info(f"Loading: {args.csv}")
    df = pd.read_csv(args.csv)
    log.info(f"Total rows: {len(df):,}")

    # Filter to test split
    test_df = df[df["metadata_dicom_id"].isin(test_dicom_ids)].reset_index(drop=True)
    log.info(f"Rows in official test split: {len(test_df):,}")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(out_path, index=False)
    log.info(f"Saved → {out_path}")
    log.info(f"Label coverage: {test_df.shape[1]} columns")


if __name__ == "__main__":
    main()
