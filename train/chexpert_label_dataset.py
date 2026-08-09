"""
Dataset that maps CheXpert rows to (image_tensor, caption) pairs where
captions are built from positive CheXpert labels, mirroring
train/cxr_label_dataset.py's CXRLabelDataset so both datasets can share
train_lora.py and the label-aware losses in train/label_aware_loss.py
unchanged.

Label source: a CSV produced by baseline_eval/build_chexpert_train_csv.py,
with a 'path' column (relative to --image-dir) and chexpert_<Label> columns
in the same {1, 0, -1, NaN} encoding as MIMIC-CXR (see the Label Encoding
table in CLAUDE.md).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from constants import CHEXPERT_LABELS, LABEL_COLS


def _build_caption(labels: list[str]) -> str:
    return " and ".join(l.lower() for l in labels)


class ChexpertLabelDataset(Dataset):
    """
    Args:
        csv_path: path to a build_chexpert_train_csv.py-schema CSV
                  (e.g. chexpert_data/chexpert_train.csv).
        image_dir: root directory the CSV's 'path' column is relative to
                   (chexpert/chexpert by default).
        caption_mode: "single" (1 pos label), "pair" (2 pos labels),
                      "negative" (1 pos + 1 neg -> "atelectasis and no edema"),
                      "both" (75% single / 25% pair),
                      "all"  (default caption_probs: 50% single / 25% pair / 25% negative).
        nan_mode: "negative" (default) — NaN (not mentioned) is treated the same
                  as CSV 0 (explicitly ruled out), both encoded as -1.0 (negative).
                  "ignore" — NaN is encoded as 0.0 (ignored), so only CSV 0 counts
                  as a negative label.
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
        nan_mode: str = "negative",
        max_samples: int | None = None,
        seed: int = 42,
    ):
        _ALIASES = {"single_only": "single", "pair_only": "pair", "neg_only": "negative"}
        caption_mode = _ALIASES.get(caption_mode, caption_mode)
        assert caption_mode in ("single", "pair", "both", "negative", "all")
        assert nan_mode in ("negative", "ignore")
        self.transform = transform
        self.tokenizer = tokenizer
        self.caption_mode = caption_mode
        self.nan_mode = nan_mode

        if caption_weights is not None:
            total = sum(caption_weights)
            assert total > 0, "--caption-weights must not all be zero"
            self.caption_probs = tuple(w / total for w in caption_weights)
        else:
            self.caption_probs = (0.50, 0.25, 0.25)  # default "all" split
        self.image_dir = Path(image_dir)

        df = pd.read_csv(csv_path, usecols=["path"] + LABEL_COLS, low_memory=False)

        # Keep raw float array (NaN = not mentioned, 0 = explicit negative)
        raw = df[LABEL_COLS].values.astype(float)  # (n, num_labels)
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

        # Encode labels: 1.0=positive, -1.0=negative, 0.0=ignore (uncertain).
        if nan_mode == "negative":
            label_encoded = np.where(raw == 1, 1.0, np.where(raw == -1, 0.0, -1.0)).astype(np.float32)
        else:  # ignore
            label_encoded = np.where(raw == 1, 1.0, np.where(raw == 0, -1.0, 0.0)).astype(np.float32)

        paths = df["path"].values
        records = []
        for i in range(len(df)):
            img_path = self.image_dir / paths[i]
            pos_indices = np.where(raw[i] == 1)[0]
            pos_labels = [CHEXPERT_LABELS[j] for j in pos_indices]
            if nan_mode == "negative":
                neg_indices = np.where(label_encoded[i] == -1.0)[0]
            else:
                neg_indices = np.where(raw[i] == 0)[0]
            neg_labels = [CHEXPERT_LABELS[j] for j in neg_indices]
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
        label_tensor = torch.from_numpy(label_vec)

        return img, tokens, label_tensor
