"""
Dataset that maps ODIR-5K rows to (image_tensor, caption) pairs where captions
are built from positive disease labels, mirroring train/cxr_label_dataset.py's
CXRLabelDataset so both datasets can share train_lora.py and the label-aware
losses in train/label_aware_loss.py unchanged.

Label source: odir/odir/full_df.csv (or an odir_data/odir_{split}.csv produced
by odir/build_odir_splits.py) already has one row per eye image (see the
'filename' column), with N/D/G/C/A/H/M/O one-hot disease flags. Unlike MIMIC-CXR,
these flags are a clean, exhaustive 0/1 encoding — there is no NaN ("not
mentioned") or -1 ("uncertain") state, so every flag is either an explicit
positive or an explicit negative.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from constants_odir import ODIR_LABELS, ODIR_LABEL_COLS


def _build_caption(labels: list[str]) -> str:
    return " and ".join(l.lower() for l in labels)


class ODIRLabelDataset(Dataset):
    """
    Args:
        csv_path: path to a full_df.csv-schema CSV (e.g. odir_data/odir_train.csv).
        image_dir: directory containing the images named by the CSV's 'filename' column
                   (odir/odir/preprocessed_images by default).
        caption_mode: "single" (1 pos label), "pair" (2 pos labels),
                      "negative" (1 pos + 1 neg -> "cataract and no glaucoma"),
                      "both" (75% single / 25% pair),
                      "all"  (default caption_probs: 50% single / 25% pair / 25% negative).
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
        caption_weights: list[float] | None = None,
        max_samples: int | None = None,
        seed: int = 42,
    ):
        assert caption_mode in ("single", "pair", "both", "negative", "all"), caption_mode
        self.transform = transform
        self.tokenizer = tokenizer
        self.caption_mode = caption_mode

        if caption_weights is not None:
            total = sum(caption_weights)
            if total <= 0:
                raise ValueError("--caption-weights must not all be zero")
            self.caption_probs = tuple(w / total for w in caption_weights)
        else:
            self.caption_probs = (0.50, 0.25, 0.25)  # default "all" split
        self.image_dir = Path(image_dir)

        df = pd.read_csv(csv_path, usecols=["filename"] + ODIR_LABEL_COLS, low_memory=False)

        raw = df[ODIR_LABEL_COLS].values.astype(float)  # (n, 8); 0/1 only, no NaN
        n_pos = (raw == 1).sum(axis=1)
        min_pos = 2 if caption_mode == "pair" else 1
        mask_pos = n_pos >= min_pos
        df = df[mask_pos].reset_index(drop=True)
        raw = raw[mask_pos]

        if max_samples is not None and max_samples < len(df):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(df), size=max_samples, replace=False)
            idx.sort()
            df = df.iloc[idx].reset_index(drop=True)
            raw = raw[idx]

        # 1 -> positive (1.0), 0 -> explicit negative (-1.0). No uncertain/not-mentioned
        # state exists in ODIR, so this is a plain sign flip — no nan_mode needed.
        label_encoded = np.where(raw == 1, 1.0, -1.0).astype(np.float32)

        filenames = df["filename"].values
        records = []
        for i in range(len(df)):
            img_path = self.image_dir / filenames[i]
            pos_indices = np.where(raw[i] == 1)[0]
            neg_indices = np.where(raw[i] == 0)[0]
            pos_labels = [ODIR_LABELS[j] for j in pos_indices]
            neg_labels = [ODIR_LABELS[j] for j in neg_indices]
            records.append((img_path, pos_labels, neg_labels, label_encoded[i].copy()))

        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx):
        for _ in range(20):
            img_path, pos_labels, neg_labels, label_vec = self.records[idx]
            try:
                img = Image.open(img_path).convert("RGB")
                break
            except (FileNotFoundError, OSError):
                idx = random.randrange(len(self.records))
        else:
            raise RuntimeError(f"Could not find a valid image after 20 retries (last: {img_path})")

        img = self.transform(img)

        if self.caption_mode == "single":
            caption = random.choice(pos_labels).lower()
        elif self.caption_mode == "pair":
            caption = _build_caption(random.sample(pos_labels, 2))
        elif self.caption_mode == "negative":
            pos = random.choice(pos_labels)
            if neg_labels:
                neg = random.choice(neg_labels)
                caption = f"{pos.lower()} and no {neg.lower()}"
            else:
                caption = pos.lower()
        elif self.caption_mode == "both":
            if len(pos_labels) >= 2 and random.random() < 0.25:
                caption = _build_caption(random.sample(pos_labels, 2))
            else:
                caption = random.choice(pos_labels).lower()
        else:  # all: use self.caption_probs = (p_single, p_pair, p_neg)
            p_single, p_pair, p_neg = self.caption_probs
            can_pair = len(pos_labels) >= 2
            can_neg = bool(neg_labels)
            r = random.random()
            if r < p_single:
                caption = random.choice(pos_labels).lower()
            elif r < p_single + p_pair:
                if can_pair:
                    caption = _build_caption(random.sample(pos_labels, 2))
                else:
                    caption = random.choice(pos_labels).lower()
            else:
                if can_neg:
                    neg = random.choice(neg_labels)
                    caption = f"{random.choice(pos_labels).lower()} and no {neg.lower()}"
                else:
                    caption = random.choice(pos_labels).lower()

        tokens = self.tokenizer([caption])[0]  # (context_len,)
        label_tensor = torch.from_numpy(label_vec)  # (8,) float32

        return img, tokens, label_tensor
