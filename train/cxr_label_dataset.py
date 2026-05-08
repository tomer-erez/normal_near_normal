"""
Dataset that maps MIMIC-CXR rows to (image_tensor, caption) pairs where
captions are built from positive CheXpert labels, matching the text queries
used in baseline_eval/eval_model.py.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

CHEXPERT_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
]
LABEL_COLS = [f"chexpert_{l}" for l in CHEXPERT_LABELS]


def _resolve_image_path(row, image_dir: Path) -> Path:
    raw = str(row["txt_file_path"]).replace("\\", "/")
    parts = Path(raw).parts  # ('p10', 'p10000032', 's50414267.txt')
    if len(parts) != 3:
        return None
    study_dir = image_dir / parts[0] / parts[1] / parts[2].replace(".txt", "")
    return study_dir / f"{row['metadata_dicom_id']}.jpg"


def _build_caption(labels: list[str]) -> str:
    return " and ".join(l.lower() for l in labels)


class CXRLabelDataset(Dataset):
    """
    Args:
        caption_mode: "single" (1 label), "pair" (2 labels), or "both" (random).
        max_samples: cap for debugging.
        seed: used only for shuffling when max_samples is set.
    """

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        transform,
        tokenizer,
        caption_mode: str = "both",
        max_samples: int | None = None,
        seed: int = 42,
    ):
        assert caption_mode in ("single", "pair", "both")
        self.transform = transform
        self.tokenizer = tokenizer
        self.caption_mode = caption_mode
        self.image_dir = Path(image_dir)

        df = pd.read_csv(
            csv_path,
            usecols=["txt_file_path", "metadata_dicom_id"] + LABEL_COLS,
            low_memory=False,
        )

        # Vectorized label filtering
        label_values = df[LABEL_COLS].fillna(0).values  # (n, 13) numpy array
        n_pos = (label_values > 0).sum(axis=1)
        min_pos = 2 if caption_mode == "pair" else 1
        df = df[n_pos >= min_pos].reset_index(drop=True)
        label_values = label_values[n_pos >= min_pos]

        # Vectorized path validation — drop rows with malformed txt_file_path
        txt_paths = df["txt_file_path"].str.replace("\\", "/", regex=False)
        parts_series = txt_paths.str.split("/")
        valid = parts_series.str.len() == 3
        df = df[valid].reset_index(drop=True)
        label_values = label_values[valid.values]
        parts_series = parts_series[valid].reset_index(drop=True)

        # Optional subsample before building Path objects
        if max_samples is not None and max_samples < len(df):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(df), size=max_samples, replace=False)
            idx.sort()
            df = df.iloc[idx].reset_index(drop=True)
            label_values = label_values[idx]
            parts_series = parts_series.iloc[idx].reset_index(drop=True)

        dicom_ids = df["metadata_dicom_id"].values
        records = []
        for i in range(len(df)):
            p = parts_series.iloc[i]
            img_path = self.image_dir / p[0] / p[1] / p[2].replace(".txt", "") / f"{dicom_ids[i]}.jpg"
            pos_indices = np.where(label_values[i] > 0)[0]
            pos_labels = [CHEXPERT_LABELS[j] for j in pos_indices]
            records.append((img_path, pos_labels))

        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx):
        # print('get item called')
        for _ in range(20):
            img_path, pos_labels = self.records[idx]
            try:
                img = Image.open(img_path).convert("RGB")
                break
            except (FileNotFoundError, OSError):
                idx = random.randrange(len(self.records))
        else:
            raise RuntimeError(f"Could not find a valid image after 20 retries (last: {img_path})")

        img = self.transform(img)

        # Build caption
        if self.caption_mode == "single":
            chosen = [random.choice(pos_labels)]
        elif self.caption_mode == "pair":
            chosen = random.sample(pos_labels, 2) if len(pos_labels) >= 2 else pos_labels[:1]
        else:  # both
            if len(pos_labels) >= 2 and random.random() < 0.5:
                chosen = random.sample(pos_labels, 2)
            else:
                chosen = [random.choice(pos_labels)]

        caption = _build_caption(chosen)
        # print(f"example for cxr data:\n{caption}\n image shape: img.shape")
        tokens = self.tokenizer([caption])[0]  # shape (context_len,)
        return img, tokens
