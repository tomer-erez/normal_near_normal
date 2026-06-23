#!/usr/bin/env python3
"""
Single-column comparison figure for WACV two-column paper (~3.5" wide).
Shows top-1 retrieval per model (CXR-CLIP vs. Ours) for 4 queries,
with a short violation annotation overlaid on each CXR-CLIP failure.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib import gridspec
from PIL import Image

REPO_ROOT     = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cxr-clip"))

from constants import LABEL_COLS

PAIRED_DIR    = REPO_ROOT / "eval_outputs/baseline_output_official_test/paired_data"
CACHE_DIR     = REPO_ROOT / "eval_outputs/baseline_output_official_test"
CSV_PATH      = REPO_ROOT / "cxr_data/mimic_cxr_official_test.csv"
CXRCLIP_CKPT  = REPO_ROOT / "valid_pretrained_models_to_try/swint_m.pt"
OURS_IMG_CKPT = REPO_ROOT / "valid_pretrained_models_to_try/swint_mc.pt"
OURS_MERGED   = REPO_ROOT / "experiments/labeldot_hnm_swint_hnm03/final_merged.pt"
OUT_PDF       = REPO_ROOT / "figs/comparison_collage.pdf"
OUT_PNG       = REPO_ROOT / "figs/comparison_collage.png"

TOP_K = 2

CORRECT_CLR = "#27ae60"
WRONG_CLR   = "#c0392b"
BORDER_LW   = 4.5

# Short display names for labels used in annotations
LABEL_SHORT = {
    "chexpert_Atelectasis":    "atelectasis",
    "chexpert_Pleural Effusion": "effusion",
    "chexpert_Pneumothorax":   "pneumothorax",
    "chexpert_Cardiomegaly":   "cardiomegaly",
    "chexpert_Edema":          "edema",
}

# (display_text, type, pos_label_cols, neg_label_cols)
# display_text uses \n for the second line
QUERIES = [
    (
        "atelectasis and\npleural effusion",
        "pair",
        ["chexpert_Atelectasis", "chexpert_Pleural Effusion"],
        [],
    ),
    (
        "pleural effusion and\npneumothorax",
        "pair",
        ["chexpert_Pleural Effusion", "chexpert_Pneumothorax"],
        [],
    ),
    (
        "pleural effusion\nand no cardiomegaly",
        "negation",
        ["chexpert_Pleural Effusion"],
        ["chexpert_Cardiomegaly"],
    ),
    (
        "edema and\nno cardiomegaly",
        "negation",
        ["chexpert_Edema"],
        ["chexpert_Cardiomegaly"],
    ),
]


def normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms == 0, 1.0, norms)


def is_relevant(row: dict, pos_cols: list, neg_cols: list) -> bool:
    if not all(row.get(c) == 1.0 for c in pos_cols):
        return False
    for c in neg_cols:
        if row.get(c, float("nan")) == 1.0:
            return False
    return True


def violation_text(row: dict, pos_cols: list, neg_cols: list, qtype: str) -> str:
    """Return a short (≤3 words) annotation explaining why the image is wrong."""
    if qtype == "pair":
        missing = [c for c in pos_cols if row.get(c, float("nan")) != 1.0]
        if len(missing) == len(pos_cols):
            return "neither found"
        if missing:
            return f"no {LABEL_SHORT.get(missing[0], missing[0])}"
        return "wrong"  # shouldn't happen
    else:  # negation
        # Check positive condition first
        for c in pos_cols:
            if row.get(c, float("nan")) != 1.0:
                return f"no {LABEL_SHORT.get(c, c)}"
        # Then negation violation
        for c in neg_cols:
            if row.get(c, float("nan")) == 1.0:
                return f"has {LABEL_SHORT.get(c, c)}"
        return "wrong"


def load_square_img(path: Path, size: int = 256) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    img = img.resize((size, size), Image.LANCZOS)
    return np.array(img)


def annotate_image(ax, symbol: str, correct: bool, viol_text: str = ""):
    """
    Overlay the ✓/✗ symbol in the top-right corner of the image,
    and the violation reason at the bottom (for incorrect images only).
    """
    clr = CORRECT_CLR if correct else WRONG_CLR
    # Symbol — top-right corner, inside the image
    ax.text(
        0.97, 0.97, symbol,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=11, fontweight="bold", color="white",
        bbox=dict(
            boxstyle="round,pad=0.15",
            facecolor=clr,
            alpha=0.88,
            edgecolor="none",
        ),
    )
    # Violation reason — bottom centre, inside the image
    if not correct and viol_text:
        ax.text(
            0.5, 0.04, viol_text,
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=6.5, fontweight="bold", color="white",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor=WRONG_CLR,
                alpha=0.82,
                edgecolor="none",
            ),
        )


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    image_paths = sorted(PAIRED_DIR.glob("*.jpg"))
    dicom_ids   = [p.stem for p in image_paths]
    n_gallery   = len(image_paths)
    log.info(f"Gallery: {n_gallery} images")

    df = pd.read_csv(CSV_PATH, usecols=["metadata_dicom_id"] + LABEL_COLS)
    df = df.set_index("metadata_dicom_id")
    label_records = df.reindex(dicom_ids)[LABEL_COLS].to_dict(orient="records")

    log.info("Loading cached image embeddings …")
    cxr_img_emb  = np.load(CACHE_DIR / "img_emb_cxrclip_swint_m.npy")
    ours_img_emb = np.load(
        CACHE_DIR / "img_emb_cxrclip_finetune_labeldot_hnm_swint_hnm03_final_merged.npy"
    )

    from baseline_eval.eval_model import CXRClipBackend, CXRClipFinetuneBackend
    log.info("Loading CXR-CLIP …")
    cxr_backend  = CXRClipBackend(str(CXRCLIP_CKPT), device)
    log.info("Loading Ours …")
    ours_backend = CXRClipFinetuneBackend(str(OURS_IMG_CKPT), str(OURS_MERGED), device)

    raw_queries = [q[0].replace("\n", " ") for q in QUERIES]
    cxr_q_emb   = normalize(cxr_backend.encode_texts(raw_queries))
    ours_q_emb  = normalize(ours_backend.encode_texts(raw_queries))

    results = []
    for i, (qtext, qtype, pos_cols, neg_cols) in enumerate(QUERIES):
        rel_mask = np.array([
            is_relevant(label_records[j], pos_cols, neg_cols)
            for j in range(n_gallery)
        ])
        sim_cxr  = cxr_q_emb[i]  @ cxr_img_emb.T
        sim_ours = ours_q_emb[i] @ ours_img_emb.T
        idxs_cxr  = np.argsort(-sim_cxr)[:TOP_K].tolist()
        idxs_ours = np.argsort(-sim_ours)[:TOP_K].tolist()

        cxr_correct  = [bool(rel_mask[j]) for j in idxs_cxr]
        ours_correct = [bool(rel_mask[j]) for j in idxs_ours]
        cxr_viol     = [violation_text(label_records[j], pos_cols, neg_cols, qtype)
                        for j in idxs_cxr]

        log.info(
            f"[{qtype}] '{raw_queries[i]}'  "
            f"CXR={cxr_correct}  Ours={ours_correct}"
        )
        results.append(dict(
            qtext=qtext, qtype=qtype,
            cxr_paths=[image_paths[j] for j in idxs_cxr],
            ours_paths=[image_paths[j] for j in idxs_ours],
            cxr_correct=cxr_correct,
            ours_correct=ours_correct,
            cxr_viol=cxr_viol,
        ))

    # ── Figure ───────────────────────────────────────────────────────────────
    # Single-column WACV width ~3.5 inches
    # Columns: [query label] [cxr image] [ours image]
    # Rows:    [header] [img+caption]×4 [legend]

    N_Q    = len(QUERIES)
    N_COLS = 1 + TOP_K + TOP_K   # label + cxr×K + ours×K
    w_lbl  = 1.05
    w_img  = 1.12
    h_img  = 1.10
    h_cap  = 0.20
    h_hdr  = 0.28
    h_leg  = 0.30

    fig_w  = w_lbl + TOP_K * w_img + TOP_K * w_img + 0.08
    fig_h  = h_hdr + N_Q * (h_img + h_cap) + h_leg + 0.05

    width_ratios  = [w_lbl] + [w_img] * TOP_K + [w_img] * TOP_K
    height_ratios = [h_hdr] + [h_img, h_cap] * N_Q + [h_leg]

    n_gs_rows = 1 + N_Q * 2 + 1
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        n_gs_rows, N_COLS,
        figure=fig,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        hspace=0.0,
        wspace=0.05,
        left=0.01, right=0.99, top=0.995, bottom=0.005,
    )

    # ── Header ───────────────────────────────────────────────────────────────
    # CXR-CLIP header spans columns 1..TOP_K
    ax_hdr_cxr = fig.add_subplot(gs[0, 1 : 1 + TOP_K])
    ax_hdr_cxr.axis("off")
    ax_hdr_cxr.text(
        0.5, 0.35, "CXR-CLIP",
        ha="center", va="center",
        fontsize=8, fontweight="bold", color=WRONG_CLR,
        transform=ax_hdr_cxr.transAxes,
    )

    # Ours header spans columns 1+TOP_K..end
    ax_hdr_ours = fig.add_subplot(gs[0, 1 + TOP_K : 1 + 2 * TOP_K])
    ax_hdr_ours.axis("off")
    ax_hdr_ours.text(
        0.5, 0.35, "Ours",
        ha="center", va="center",
        fontsize=8, fontweight="bold", color=CORRECT_CLR,
        transform=ax_hdr_ours.transAxes,
    )

    # Thin rule under header
    line = plt.Line2D(
        [0.01, 0.99], [1.0 - h_hdr / fig_h, 1.0 - h_hdr / fig_h],
        color="#cccccc", linewidth=0.7,
        transform=fig.transFigure, clip_on=False,
    )
    fig.add_artist(line)

    # ── Query rows ────────────────────────────────────────────────────────────
    for q_i, res in enumerate(results):
        img_row = 1 + q_i * 2
        cap_row = 1 + q_i * 2 + 1

        # ── Query label ──────────────────────────────────────────────────────
        ax_lbl = fig.add_subplot(gs[img_row : cap_row + 1, 0])
        ax_lbl.axis("off")

        tag_color = "#7d3c98" if res["qtype"] == "negation" else "#2471a3"
        # Show query text
        ax_lbl.text(
            0.96, 0.68,
            f'"{res["qtext"]}"',
            ha="right", va="center",
            fontsize=6.8, fontstyle="italic",
            fontweight="bold" if res["qtype"] == "negation" else "normal",
            transform=ax_lbl.transAxes,
            color="#1a1a1a",
        )
        # Query type badge
        ax_lbl.text(
            0.96, 0.22,
            f"[{res['qtype']}]",
            ha="right", va="center",
            fontsize=5.8, color=tag_color,
            transform=ax_lbl.transAxes,
        )

        # ── CXR-CLIP images ──────────────────────────────────────────────────
        for k in range(TOP_K):
            ax_cxr = fig.add_subplot(gs[img_row, 1 + k])
            arr_cxr = load_square_img(res["cxr_paths"][k])
            ax_cxr.imshow(arr_cxr, aspect="auto")
            ax_cxr.set_xticks([]); ax_cxr.set_yticks([])
            correct = res["cxr_correct"][k]
            bc = CORRECT_CLR if correct else WRONG_CLR
            for sp in ax_cxr.spines.values():
                sp.set_edgecolor(bc); sp.set_linewidth(BORDER_LW)
            sym = "✓" if correct else "✗"
            annotate_image(ax_cxr, sym, correct, res["cxr_viol"][k])

        # ── Ours images ──────────────────────────────────────────────────────
        for k in range(TOP_K):
            ax_ours = fig.add_subplot(gs[img_row, 1 + TOP_K + k])
            arr_ours = load_square_img(res["ours_paths"][k])
            ax_ours.imshow(arr_ours, aspect="auto")
            ax_ours.set_xticks([]); ax_ours.set_yticks([])
            correct2 = res["ours_correct"][k]
            bc2 = CORRECT_CLR if correct2 else WRONG_CLR
            for sp in ax_ours.spines.values():
                sp.set_edgecolor(bc2); sp.set_linewidth(BORDER_LW)
            sym2 = "✓" if correct2 else "✗"
            annotate_image(ax_ours, sym2, correct2)

        # ── Caption row (separator line) ─────────────────────────────────────
        ax_cap = fig.add_subplot(gs[cap_row, :])
        ax_cap.axis("off")
        # thin separator between query rows (not after last)
        if q_i < N_Q - 1:
            ax_cap.axhline(0.1, color="#e0e0e0", linewidth=0.6)

    # ── Legend ────────────────────────────────────────────────────────────────
    ax_leg = fig.add_subplot(gs[-1, :])
    ax_leg.axis("off")
    legend_patches = [
        mpatches.Patch(color=CORRECT_CLR, label="Relevant"),
        mpatches.Patch(color=WRONG_CLR,   label="Irrelevant"),
    ]
    ax_leg.legend(
        handles=legend_patches,
        loc="center",
        ncol=2,
        fontsize=7,
        frameon=True,
        edgecolor="#cccccc",
        handlelength=1.0,
        columnspacing=0.8,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT_PDF), bbox_inches="tight", facecolor="white")
    fig.savefig(str(OUT_PNG), bbox_inches="tight", dpi=250, facecolor="white")
    log.info(f"Saved → {OUT_PDF}")
    log.info(f"Saved → {OUT_PNG}")
    plt.close(fig)


if __name__ == "__main__":
    main()
