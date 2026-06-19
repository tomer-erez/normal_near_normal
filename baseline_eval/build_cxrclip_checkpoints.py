"""
Pack CXR-CLIP directory checkpoints into .pt zip files that torch.load() accepts.

CXR-CLIP checkpoints are distributed as extracted PyTorch zip archives
(directories containing archive/data/, archive/data.pkl, archive/version).
torch.load() needs a single zip file, not an extracted directory.
This script re-zips them.

Usage
-----
    python baseline_eval/build_cxrclip_checkpoints.py
    # or for specific models only:
    python baseline_eval/build_cxrclip_checkpoints.py --models r50_m swint_m
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

CHECKPOINTS_DIR = Path(__file__).parent.parent / "valid_pretrained_models_to_try"

# Models to convert: directory name → output .pt filename
DEFAULT_MODELS = {
    "r50_m":    "r50_m.pt",
    "swint_m":  "swint_m.pt",
    "r50_mc":   "r50_mc.pt",
    "swint_mc": "swint_mc.pt",
}


def pack(src_dir: Path, dst_pt: Path, overwrite: bool = False) -> None:
    if dst_pt.exists() and not overwrite:
        print(f"  skip  {dst_pt.name}  (already exists, use --overwrite to replace)")
        return
    archive_dir = src_dir / "archive"
    if not archive_dir.is_dir():
        print(f"  ERROR {src_dir}: no archive/ subdirectory found, skipping")
        return

    all_files = sorted(archive_dir.rglob("*"))
    members = [f for f in all_files if f.is_file()]
    total = len(members)
    print(f"  packing {src_dir.name}/ → {dst_pt.name}  ({total} files) ...", end="", flush=True)

    tmp = dst_pt.with_suffix(".pt.tmp")
    _EPOCH_1980 = (1980, 1, 1, 0, 0, 0)   # minimum timestamp ZIP supports
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for fpath in members:
            arcname = str(fpath.relative_to(src_dir))  # keeps archive/data/0 etc.
            zi = zipfile.ZipInfo(arcname, date_time=_EPOCH_1980)
            zi.compress_type = zipfile.ZIP_STORED
            zf.writestr(zi, fpath.read_bytes())

    tmp.rename(dst_pt)
    size_gb = dst_pt.stat().st_size / 1e9
    print(f"  done  ({size_gb:.2f} GB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack CXR-CLIP directories into .pt files")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS.keys()),
                        help="Directory names to pack (default: all)")
    parser.add_argument("--dir", type=Path, default=CHECKPOINTS_DIR,
                        help="Directory containing the model folders")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-pack even if the .pt file already exists")
    args = parser.parse_args()

    for model_name in args.models:
        src = args.dir / model_name
        if not src.is_dir():
            print(f"  skip  {model_name}  (directory not found at {src})")
            continue
        dst = args.dir / DEFAULT_MODELS.get(model_name, f"{model_name}.pt")
        pack(src, dst, overwrite=args.overwrite)

    print("All done.")


if __name__ == "__main__":
    main()
