from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(r"/mnt/walkure_public/users/tomererez/mimic_cxr_jpg_images/mimic_cxr_jpg_images_from_google_cloud/mimic-cxr-jpg-2.1.0.physionet.org/files")
MAX_WORKERS = 10
EXTS = {".jpg", ".jpeg"}

def count_jpgs_in_dir(d: Path) -> int:
    # Keep it simple + fast: avoid extra stat calls where possible
    cnt = 0
    for p in d.rglob("*"):
        if p.suffix.lower() in EXTS:
            # In MIMIC these should be files, but keep the guard anyway:
            if p.is_file():
                cnt += 1
    return cnt

def main():
    # Collect folders like p10, p11, ...
    p_dirs = sorted([d for d in ROOT.iterdir() if d.is_dir() and d.name.startswith("p")])

    total = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(count_jpgs_in_dir, d): d for d in p_dirs}

        for fut in as_completed(futures):
            d = futures[fut]
            c = fut.result()
            print(f"{d.name}: {c}")
            total += c

    print("TOTAL:", total)

if __name__ == "__main__":
    main()
