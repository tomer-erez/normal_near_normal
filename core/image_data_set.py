import torch
from torch.utils.data import Dataset
from pathlib import Path
import pydicom
import re
import numpy as np
import pandas as pd
import os
import warnings

def ds_to_windowed_array(ds, center=40, width=80):
    arr = ds.pixel_array.astype(np.int16)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = arr * slope + intercept
    lo, hi = center - width/2.0, center + width/2.0
    clipped = np.clip(hu, lo, hi)
    norm = (clipped - lo) / (hi - lo + 1e-6)   # float32 in [0,1]
    return norm.astype(np.float32)


def find_dicom_folder(series_root: Path) -> Path or None:
    """Find the folder under series_root that contains DICOMs.
       Priority:
         1) folder named 'reconstructed image' or 'reconstructed_image' (case-insensitive)
         2) any subfolder whose name contains 'reconstruct'
         3) series_root itself if it contains .dcm files
       Returns Path or None 
       if nothing found.
    """
    if not series_root.exists():
        return None
    # candidate exact names
    for name in ("reconstructed image", "reconstructed_image"):
        cand = series_root / name
        if cand.is_dir() and any(p.suffix.lower() == ".dcm" for p in cand.iterdir()):
            return cand
    # any subfolder containing 'reconstruct' in name
    for p in series_root.iterdir():
        if p.is_dir() and "reconstruct" in p.name.lower() and any(q.suffix.lower() == ".dcm" for q in p.iterdir()):
            return p
    # fallback: series_root itself if it contains DICOMs
    if any(p.suffix.lower() == ".dcm" for p in series_root.iterdir()):
        return series_root
    return None

def read_sorted_dcm_paths(folder: Path):
    """Return sorted list of dcm file paths (strings)."""
    files = [str(p) for p in folder.iterdir() if p.suffix.lower() == ".dcm"]
    if not files:
        return []
    def key_fn(p):
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True)
            if hasattr(ds, "ImagePositionPatient"):
                return float(ds.ImagePositionPatient[2])
            if hasattr(ds, "SliceLocation"):
                return float(ds.SliceLocation)
            if hasattr(ds, "InstanceNumber"):
                return int(ds.InstanceNumber)
        except Exception:
            pass
        m = re.search(r"(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else 0
    files.sort(key=key_fn)
    return files


class DummyCTDataset(Dataset):
    def __init__(self, length=100):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = torch.randn(1, 224, 224, 64)
        label = torch.randint(0, 3, (1,)).item()
        return image, label
    

class SinoCTDataset(Dataset):
    def __init__(self, args):
        """
        args.sinoCT_dataset_path: path to folder holding batch_0, batch_1, ...
        args.sinoCT_csv_path: path to CSV file with patient/series info and labels
        args.d_target: number of slices to pad/truncate to
        args.brain_window_function_center, args.brain_window_function_width: windowing parameters
        """
        self.dataset_path = Path(args.sinoCT_dataset_path)
        self.d_target = int(args.d_target)
        self.center = args.brain_window_function_center
        self.width = args.brain_window_function_width

        # ---------------- Scan all series ----------------
        all_series_dirs = sorted(
            [p for batch in self.dataset_path.iterdir() if batch.is_dir()
             for p in batch.iterdir()
             if p.is_dir() and re.match(r"series_\d+", p.name)]
        )
        print(f"Found {len(all_series_dirs)} series directories under {self.dataset_path}")

        series_to_folder = {}
        for sdir in all_series_dirs:
            dicom_folder = find_dicom_folder(sdir)
            series_to_folder[sdir.name] = str(dicom_folder) if dicom_folder else None

        # ---------------- Load CSV and filter ----------------
        df = pd.read_csv(args.sinoCT_csv_path, sep=None, engine="python")
        # Normalize series column
        if "patient_id" not in df.columns:
            for alt in ("path","patient","series"):
                if alt in df.columns:
                    df = df.rename(columns={alt:"patient_id"})
                    break
        assert "patient_id" in df.columns, "CSV missing patient_id column"

        df["series_folder"] = df["patient_id"].astype(str).map(series_to_folder)
        self.df = df[df["series_folder"].notnull()].reset_index(drop=True)
        missing = df[df["series_folder"].isnull()]
        print(f"CSV rows: {len(df)}, available on disk: {len(self.df)}, missing: {len(missing)}")
        if len(missing):
            warnings.warn(f"{len(missing)} entries in CSV have no corresponding dicom folder on disk "
                          f"(first 10 shown). Examples: {missing['patient_id'].tolist()[:10]}")

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _label_to_index(label_val):
        if isinstance(label_val, str) and "," in label_val:
            parts = [int(x) for x in label_val.split(",")]
            return int(np.argmax(parts))
        try:
            return int(label_val)
        except Exception:
            raise ValueError("Can't parse label: " + str(label_val))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        series_id = str(row["patient_id"])
        label_idx = self._label_to_index(row.iloc[1])  # assumes label in 2nd column
        folder = Path(row["series_folder"])
        dcm_paths = read_sorted_dcm_paths(folder)
        if len(dcm_paths) == 0:
            raise RuntimeError(f"No DICOMs found for {series_id} in {folder}")

        slices = [ds_to_windowed_array(pydicom.dcmread(p), center=self.center, width=self.width)
                  for p in dcm_paths]
        vol = np.stack(slices, axis=0)  # (D_orig, H, W)

        # pad or truncate
        d_orig, h, w = vol.shape
        if d_orig < self.d_target:
            pad = np.zeros((self.d_target - d_orig, h, w), dtype=np.float32)
            vol = np.concatenate([vol, pad], axis=0)
        elif d_orig > self.d_target:
            vol = vol[:self.d_target, :, :]

        vol_t = torch.from_numpy(vol).unsqueeze(0)  # (1, D, H, W)
        label_t = torch.tensor(label_idx, dtype=torch.long)
        return vol_t, label_t
