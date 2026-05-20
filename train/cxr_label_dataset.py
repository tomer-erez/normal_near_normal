"""
Dataset that maps MIMIC-CXR rows to (image_tensor, caption) pairs where
captions are built from positive CheXpert labels, matching the text queries
used in baseline_eval/eval_model.py.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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

        # Keep raw float array (NaN = not mentioned, 0 = explicit negative)
        # so we can later distinguish "negatively mentioned" from "not mentioned".
        raw = df[LABEL_COLS].values.astype(float)  # (n, 13); NaN preserved
        n_pos = (raw == 1).sum(axis=1)
        min_pos = 2 if caption_mode == "pair" else 1
        mask_pos = n_pos >= min_pos
        df = df[mask_pos].reset_index(drop=True) # drop rows with no positive labels, since they can't contribute to training (no positive labels → empty caption → no text features → no learning signal). This also speeds up the rest of the processing and reduces noise from "normal" cases that might be labeled as "No Finding" but are otherwise uninformative.
        raw = raw[mask_pos]# keep in sync with df after filtering out rows with no positive labels

        # Vectorized path validation — drop rows with malformed txt_file_path
        txt_paths = df["txt_file_path"].str.replace("\\", "/", regex=False)
        parts_series = txt_paths.str.split("/")
        valid = parts_series.str.len() == 3
        df = df[valid].reset_index(drop=True)
        raw = raw[valid.values]
        parts_series = parts_series[valid].reset_index(drop=True)

        # Optional subsample before building Path objects
        if max_samples is not None and max_samples < len(df):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(df), size=max_samples, replace=False)
            idx.sort()
            df = df.iloc[idx].reset_index(drop=True)
            raw = raw[idx]
            parts_series = parts_series.iloc[idx].reset_index(drop=True)

        # Encode labels: 1.0=positive, -1.0=negative, 0.0=ignore
        # NaN (not mentioned) and CSV 0 (explicitly ruled out) both → -1.0 (negative).
        # Only CSV -1 (uncertain/ambiguous) → 0.0 (ignore).
        label_encoded = np.where(raw == 1, 1.0, np.where(raw == -1, 0.0, -1.0)).astype(np.float32)

        dicom_ids = df["metadata_dicom_id"].values
        records = []
        for i in range(len(df)):
            p = parts_series.iloc[i]
            img_path = self.image_dir / p[0] / p[1] / p[2].replace(".txt", "") / f"{dicom_ids[i]}.jpg"
            pos_indices = np.where(raw[i] == 1)[0]
            pos_labels = [CHEXPERT_LABELS[j] for j in pos_indices]
            records.append((img_path, pos_labels, label_encoded[i].copy()))

        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx):
        for _ in range(20):
            img_path, pos_labels, label_vec = self.records[idx]
            try:
                img = Image.open(img_path).convert("RGB")
                break
            except (FileNotFoundError, OSError):
                idx = random.randrange(len(self.records))
        else:
            raise RuntimeError(f"Could not find a valid image after 20 retries (last: {img_path})")

        img = self.transform(img)

        # Build caption from positive labels
        if self.caption_mode == "single":
            chosen = [random.choice(pos_labels)]
        elif self.caption_mode == "pair":
            chosen = random.sample(pos_labels, 2) if len(pos_labels) >= 2 else pos_labels[:1]
        else:  # both
            if len(pos_labels) >= 2 and random.random() < 0.25:  # 25% pairs, 75% single
                chosen = random.sample(pos_labels, 2)
            else:
                chosen = [random.choice(pos_labels)]

        caption = _build_caption(chosen)
        tokens = self.tokenizer([caption])[0]  # (context_len,)
        label_tensor = torch.from_numpy(label_vec)  # (13,) float32

        return img, tokens, label_tensor
