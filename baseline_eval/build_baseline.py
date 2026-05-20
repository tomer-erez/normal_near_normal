"""
Build a CLIP text-image retrieval baseline for MIMIC-CXR.

Steps:
  1. Read CSV, resolve image paths, create a paired folder
     (image symlinks + matching .txt report files)
  2. Run clip-retrieval inference → image & text embeddings
  3. Build FAISS index

Usage:
    python baseline/build_baseline.py \
        --csv cxr_data/all_txt_data_and_labels.csv \
        --image_dir /mnt/walkure_public/users/tomererez/mimic_cxr_jpg_images \
        --output_dir ./baseline_output

    # Quick test with 1k samples, skip prep if already done:
    python baseline/build_baseline.py \
        --csv cxr_data/all_txt_data_and_labels.csv \
        --image_dir /mnt/walkure_public/users/tomererez/mimic_cxr_jpg_images \
        --output_dir ./baseline_output \
        --max_samples 1000

    # Skip steps already completed:
    python baseline/build_baseline.py ... --skip_prepare --skip_inference
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def resolve_image_path(row, image_dir: Path) -> Path | None:
    """
    Build image path from a CSV row.

    txt_file_path format: p10\\p10000032\\s50414267.txt (Windows-style separators)
    Image path: image_dir / p10 / p10000032 / s50414267 / {dicom_id}.jpg
    """
    raw = str(row["txt_file_path"]).replace("\\", "/")
    parts = Path(raw).parts  # ('p10', 'p10000032', 's50414267.txt')
    if len(parts) != 3:
        return None
    study_dir = image_dir / parts[0] / parts[1] / parts[2].replace(".txt", "")
    img_path = study_dir / f"{row['metadata_dicom_id']}.jpg"
    return img_path  # skip exists() — slow on network mounts


def prepare_paired_folder(df: pd.DataFrame, image_dir: Path, paired_dir: Path) -> int:
    """
    Create a flat folder of symlinked images and matching .txt report files.

    clip-retrieval 'files' mode pairs files by stem:
        {dicom_id}.jpg  <->  {dicom_id}.txt

    Returns the number of valid pairs created.
    """
    paired_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    skipped = 0
    for _, row in df.iterrows():
        img_path = resolve_image_path(row, image_dir)
        if img_path is None:
            skipped += 1
            continue

        stem = str(row["metadata_dicom_id"])

        # Symlink image (replace broken symlinks)
        img_link = paired_dir / f"{stem}.jpg"
        if img_link.is_symlink() and not img_link.exists():
            img_link.unlink()
        if not img_link.is_symlink():
            img_link.symlink_to(img_path)

        # Write matching text file
        txt_file = paired_dir / f"{stem}.txt"
        if not txt_file.exists():
            txt_file.write_text(str(row["txt_content"]), encoding="utf-8")

        count += 1

    if skipped:
        log.warning(f"Could not find image files for {skipped} rows (missing from disk) ,count is {count}")
    return count


def run_inference(paired_dir: Path, embeddings_dir: Path, model: str, batch_size: int, workers: int):
    # import here so the script can be imported without torch installed
    from clip_retrieval.clip_inference.main import main as clip_inference  # noqa

    # clip_retrieval's "files" mode keys images by "stem.jpg" and texts by "stem.txt",
    # so enabling both at once causes a KeyError (the two key sets never intersect).
    # Fix: run image and text inference separately — they write to img_emb/ and text_emb/
    # respectively, and sorted order is identical since stems match.
    clip_inference(
        input_dataset=str(paired_dir),
        output_folder=str(embeddings_dir),
        input_format="files",
        clip_model=model,
        batch_size=batch_size,
        num_prepro_workers=workers,
        enable_text=False,
        enable_image=True,
        enable_metadata=False,
    )
    clip_inference(
        input_dataset=str(paired_dir),
        output_folder=str(embeddings_dir),
        input_format="files",
        clip_model=model,
        batch_size=batch_size,
        num_prepro_workers=workers,
        enable_text=True,
        enable_image=False,
        enable_metadata=False,
    )


def run_index(embeddings_dir: Path, index_dir: Path):
    from clip_retrieval.clip_index import clip_index  # noqa

    clip_index(
        embeddings_folder=str(embeddings_dir),
        index_folder=str(index_dir),
    )


def main():
    parser = argparse.ArgumentParser(description="Build CLIP retrieval baseline for MIMIC-CXR")
    parser.add_argument("--csv", required=True, help="Path to all_txt_data_and_labels.csv")
    parser.add_argument("--image_dir", default=r"/home/tomererez/normal_near_normal/cxr_data/images/mimic_cxr_jpg_images_from_google_cloud/mimic-cxr-jpg-2.1.0.physionet.org/files"
                        , help="Root MIMIC-CXR image directory")
    parser.add_argument("--output_dir", required=True, help="Where to store all artifacts")
    parser.add_argument("--model", default="ViT-B/32", help="CLIP model name (default: ViT-B/32)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit to first N rows — useful for a quick smoke test",
    )

    parser.add_argument("--skip_prepare", action="store_true", help="Skip step 1 (paired folder already exists)")
    parser.add_argument("--skip_inference", action="store_true", default=True, help="Skip step 2 (default: True)")
    parser.add_argument("--run_inference", dest="skip_inference", action="store_false", help="Enable step 2 CLIP inference")
    parser.add_argument("--skip_index", action="store_true", default=True, help="Skip step 3 (default: True)")
    parser.add_argument("--run_index", dest="skip_index", action="store_false", help="Enable step 3 FAISS index")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    paired_dir = output_dir / "paired_data"
    embeddings_dir = output_dir / "embeddings"
    index_dir = output_dir / "index"

    # ── Step 1: Prepare paired data folder ───────────────────────────────────
    if not args.skip_prepare:
        log.info("Loading CSV …")
        df = pd.read_csv(args.csv)
        log.info(f"Loaded {len(df):,} rows")

        if args.max_samples:
            df = df.sample(args.max_samples)
            log.info(f"Limiting to {args.max_samples:,} samples")

        log.info(f"Preparing paired folder at {paired_dir} …")
        n = prepare_paired_folder(df, Path(args.image_dir), paired_dir)
        log.info(f"Created {n:,} image/text pairs")
        if n == 0:
            log.error("No pairs created — check --image_dir and CSV paths")
            sys.exit(1)
    else:
        log.info("Skipping data preparation")

    # ── Step 2: CLIP inference ────────────────────────────────────────────────
    if not args.skip_inference:
        log.info(f"Running CLIP inference with model={args.model} …")
        run_inference(paired_dir, embeddings_dir, args.model, args.batch_size, args.workers)
        log.info("Inference complete")
    else:
        log.info("Skipping inference")

    # ── Step 3: Build FAISS index ─────────────────────────────────────────────
    if not args.skip_index:
        log.info("Building FAISS index …")
        run_index(embeddings_dir, index_dir)
        log.info(f"Index built → {index_dir}")
    else:
        log.info("Skipping indexing")

    log.info(
        f"Done! Paired data at: {paired_dir}\n"
        f"Run eval with: python baseline_eval/run_all_evals.py --paired_dir {paired_dir} --csv {args.csv}"
    )



if __name__ == "__main__":
    main()
