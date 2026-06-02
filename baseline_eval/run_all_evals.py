"""
Run retrieval evaluation for all models sequentially, then summarize results
into comparison tables and plots.

Query modes (--query_mode, default: all)
  single    13 queries — "atelectasis"         relevant: label==1
  pair      78 queries — "atelectasis and edema" relevant: both labels==1
  negative 156 queries — "atelectasis and no cardiomegaly"
                          relevant: pos label==1 AND neg label==0
  all       all 247 queries

Output files (in --output_dir):
  summary_single.csv    P@k / R@k per model, single-label queries
  summary_pair.csv      P@k / R@k per model, pair queries
  summary_negative.csv  P@k / R@k per model, negative queries (if run)
  summary_macro.csv     macro-averaged metrics per model (one row per model)
  plots/                comparison charts + per-model 13×13 heatmaps

Usage
-----
    python baseline_eval/run_all_evals.py \
        --paired_dir ./baseline_output/paired_data \
        --csv        cxr_data/all_txt_data_and_labels.csv \
        --output_dir ./experiments/my_exp1

    # Evaluate only negative-mode queries:
    python baseline_eval/run_all_evals.py ... --query_mode negative

    # Skip models already evaluated:
    python baseline_eval/run_all_evals.py ... --skip_existing
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import wandb
except ImportError:
    wandb = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
CHECKPOINTS_DIR = REPO_ROOT / "valid_pretrained_models_to_try"

CHEXPERT_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax",
]
LABEL_COLS = [f"chexpert_{l}" for l in CHEXPERT_LABELS]
LABEL_TO_IDX = {col: i for i, col in enumerate(LABEL_COLS)}
SHORT_LABELS = [l.replace("Enlarged Cardiomediastinum", "Enlarged CM") for l in CHEXPERT_LABELS]

MODELS = [
    {"name": "vanilla_clip_not_trained",  "model_type": "vanilla_clip",  "checkpoint": None},
    # {"name": "biomedclip",    "model_type": "biomedclip",    "checkpoint": None},
    # {"name": "cxrclip_r50_m",   "model_type": "cxrclip", "checkpoint": "r50_m.pt"},
    {"name": "cxrclip_r50_mc",  "model_type": "cxrclip", "checkpoint": "r50_mc.pt"},
    # {"name": "cxrclip_r50_mcc", "model_type": "cxrclip", "checkpoint": "r50_mcc.pt"},
    # {"name": "cxrclip_swint_m",   "model_type": "cxrclip", "checkpoint": "swint_m.pt"},
    # {"name": "cxrclip_swint_mc",  "model_type": "cxrclip", "checkpoint": "swint_mc.pt"},
    # {"name": "cxrclip_swint_mcc", "model_type": "cxrclip", "checkpoint": "swint_mcc.pt"},
    # Fine-tuned models — add entries here after training:
    # {
    #     "name": "lora_vitb32_vanilla_clip",
    #     "model_type": "finetuned",
    #     "checkpoint": None,
    #     "finetuned_base_model": "ViT-B-32",
    #     "finetuned_pretrained": "openai",
    #     "finetuned_checkpoint": "experiments/lora_vitb32_with_5_ep/final_merged.pt",
    # },
    # {
    #     "name": "lora_biomedclip",
    #     "model_type": "finetuned",
    #     "checkpoint": None,
    #     "finetuned_base_model": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    #     "finetuned_pretrained": "",
    #     "finetuned_checkpoint": "experiments/lora_biomedclip_with_5_ep/final_merged.pt",
    # },
    {
        "name": "vanilla_clip_loss",
        "model_type": "finetuned",
        "checkpoint": None,
        "finetuned_base_model": "ViT-B-32",
        "finetuned_pretrained": "",
        "finetuned_checkpoint": "experiments/_vanilla_bsz16x8/final_merged.pt",
    },
    #     {
    #     "name": "vanilla_siglip_loss",
    #     "model_type": "finetuned",
    #     "checkpoint": None,
    #     "finetuned_base_model": "ViT-B-32",
    #     "finetuned_pretrained": "",
    #     "finetuned_checkpoint": "experiments/_vanilla_bsz16x8_with_siglip/final_merged.pt",
    # },
    #         {
    #     "name": "ft_swint_siglip_loss",
    #     "model_type": "finetuned",
    #     "checkpoint": None,
    #     "finetuned_base_model": "ViT-B-32",
    #     "finetuned_pretrained": "",
    #     "finetuned_checkpoint": "experiments/fine_tuned_cxr_and_siglip/final_merged.pt",
    # },
    #             {
    #     "name": "ft_swint_clip_loss",
    #     "model_type": "finetuned",
    #     "checkpoint": None,
    #     "finetuned_base_model": "ViT-B-32",
    #     "finetuned_pretrained": "",
    #     "finetuned_checkpoint": "experiments/fine_tuned_cxr_and_cliploss/final_merged.pt",
    # },

    #     {
    #     "name": "negawareclip_01",
    #     "model_type": "finetuned",
    #     "checkpoint": None,
    #     "finetuned_base_model": "ViT-B-32",
    #     "finetuned_pretrained": "",
    #     "finetuned_checkpoint": "experiments/negawareclip_01/final_merged.pt",
    # },

    #         {
    #     "name": "negawareclip_05",
    #     "model_type": "finetuned",
    #     "checkpoint": None,
    #     "finetuned_base_model": "ViT-B-32",
    #     "finetuned_pretrained": "",
    #     "finetuned_checkpoint": "experiments/negawareclip_05/final_merged.pt",
    # },

    #         {
    #     "name": "negawaresiglip_01",
    #     "model_type": "finetuned",
    #     "checkpoint": None,
    #     "finetuned_base_model": "ViT-B-32",
    #     "finetuned_pretrained": "",
    #     "finetuned_checkpoint": "experiments/negawaresiglip_01/final_merged.pt",
    # },

    #         {
    #     "name": "negawaresiglip_05",
    #     "model_type": "finetuned",
    #     "checkpoint": None,
    #     "finetuned_base_model": "ViT-B-32",
    #     "finetuned_pretrained": "",
    #     "finetuned_checkpoint": "experiments/negawaresiglip_05/final_merged.pt",
    # },

    #         {
    #     "name": "r32a32",
    #     "model_type": "finetuned",
    #     "checkpoint": None,
    #     "finetuned_base_model": "ViT-B-32",
    #     "finetuned_pretrained": "",
    #     "finetuned_checkpoint": "experiments/r32a32/final_merged.pt",
    # },

    #             {
    #     "name": "r32a64",
    #     "model_type": "finetuned",
    #     "checkpoint": None,
    #     "finetuned_base_model": "ViT-B-32",
    #     "finetuned_pretrained": "",
    #     "finetuned_checkpoint": "experiments/r32a64/final_merged.pt",
    # },

    #                 {
    #     "name": "r64a64",
    #     "model_type": "finetuned",
    #     "checkpoint": None,
    #     "finetuned_base_model": "ViT-B-32",
    #     "finetuned_pretrained": "",
    #     "finetuned_checkpoint": "experiments/r64a64/final_merged.pt",
    # },

    #                     {
    #     "name": "r100a100",
    #     "model_type": "finetuned",
    #     "checkpoint": None,
    #     "finetuned_base_model": "ViT-B-32",
    #     "finetuned_pretrained": "",
    #     "finetuned_checkpoint": "experiments/r100a100/final_merged.pt",
    # },

                        {
        "name": "vit_l_14",
        "model_type": "finetuned",
        "checkpoint": None,
        "finetuned_base_model": "ViT-L-14",
        "finetuned_pretrained": "",
        "finetuned_checkpoint": "experiments/vit_l_14/final_merged.pt",
    },

    #                         {
    #     "name": "vitb32_cliploss_negaware_bs32_no_conflict_zeriong_version",
    #     "model_type": "finetuned",
    #     "checkpoint": None,
    #     "finetuned_base_model": "ViT-B-32",
    #     "finetuned_pretrained": "",
    #     "finetuned_checkpoint": "experiments/vitb32_cliploss_negaware_bs32_no_conflict_zeriong_version/final_merged.pt",
    # },

    #                         {
    #     "name": "vitb32_sigliploss_negaware_bs32_no_conflict_zeriong_version",
    #     "model_type": "finetuned",
    #     "checkpoint": None,
    #     "finetuned_base_model": "ViT-B-32",
    #     "finetuned_pretrained": "",
    #     "finetuned_checkpoint": "experiments/vitb32_sigliploss_negaware_bs32_no_conflict_zeriong_version/final_merged.pt",
    # },

    #                             {
    #     "name": "mode_ignore_nan_negaware",
    #     "model_type": "finetuned",
    #     "checkpoint": None,
    #     "finetuned_base_model": "ViT-B-32",
    #     "finetuned_pretrained": "",
    #     "finetuned_checkpoint": "experiments/vitb32_cliploss_negaware_nanmode_ignore/final_merged.pt",
    # },

                                    {
        "name": "fixed_negaware_005",
        "model_type": "finetuned",
        "checkpoint": None,
        "finetuned_base_model": "ViT-B-32",
        "finetuned_pretrained": "",
        "finetuned_checkpoint": "experiments/fixed_negaware_005/final_merged.pt",
    },
    
    {
        "name": "fixed_negaware_015",
        "model_type": "finetuned",
        "checkpoint": None,
        "finetuned_base_model": "ViT-B-32",
        "finetuned_pretrained": "",
        "finetuned_checkpoint": "experiments/fixed_negaware_015/final_merged.pt",    
        },
    
    {
        "name": "fixed_negaware_035",
        "model_type": "finetuned",
        "checkpoint": None,
        "finetuned_base_model": "ViT-B-32",
        "finetuned_pretrained": "",
        "finetuned_checkpoint": "experiments/fixed_negaware_035/final_merged.pt",
    },

        {
        "name": "fixed_negaware_055",
        "model_type": "finetuned",
        "checkpoint": None,
        "finetuned_base_model": "ViT-B-32",
        "finetuned_pretrained": "",
        "finetuned_checkpoint": "experiments/fixed_negaware_055/final_merged.pt",
    },

        {
        "name": "fixed_negaware_075",
        "model_type": "finetuned",
        "checkpoint": None,
        "finetuned_base_model": "ViT-B-32",
        "finetuned_pretrained": "",
        "finetuned_checkpoint": "experiments/fixed_negaware_075/final_merged.pt",
    },

        {
        "name": "fixed_negaware_105",
        "model_type": "finetuned",
        "checkpoint": None,
        "finetuned_base_model": "ViT-B-32",
        "finetuned_pretrained": "",
        "finetuned_checkpoint": "experiments/fixed_negaware_105/final_merged.pt",
    },
]

KS = [1, 5, 10]
METRIC_COLS = [f"P@{k}" for k in KS] + [f"R@{k}" for k in KS]


# ── Helpers ───────────────────────────────────────────────────────────────────

def results_path(model: dict) -> Path:
    if model["model_type"] in ("finetuned", "cxrclip_hybrid"):
        return REPO_ROOT / f"results_{model['name']}.csv"
    if model.get("checkpoint"):
        stem = Path(model["checkpoint"]).stem
        return REPO_ROOT / f"results_cxrclip_{stem}.csv"
    return REPO_ROOT / f"results_{model['model_type']}.csv"


def run_eval(model: dict, paired_dir: str, csv: str,
             query_mode: str = "all", batch_size: int = 64,
             max_samples: int | None = None, nan_mode: str = "negative") -> Path:
    """Run eval_model.py for one model. Returns path to results CSV."""
    cmd = [
        sys.executable, str(REPO_ROOT / "baseline_eval" / "eval_model.py"),
        "--model_type", model["model_type"],
        "--name", model["name"],
        "--paired_dir", paired_dir,
        "--csv", csv,
        "--query_mode", query_mode,
        "--batch_size", str(batch_size),
        "--nan-mode", nan_mode,
    ]
    if max_samples is not None:
        cmd += ["--max_samples", str(max_samples)]
    if model.get("checkpoint"):
        cmd += ["--cxrclip_checkpoint", str(CHECKPOINTS_DIR / model["checkpoint"])]
    if model["model_type"] == "finetuned":
        cmd += [
            "--finetuned_base_model", model["finetuned_base_model"],
            "--finetuned_pretrained", model.get("finetuned_pretrained", ""),
            "--finetuned_checkpoint", str(REPO_ROOT / model["finetuned_checkpoint"]),
        ]
    if model["model_type"] == "cxrclip_hybrid":
        cmd += [
            "--cxrclip_image_checkpoint", str(REPO_ROOT / model["cxrclip_image_checkpoint"]),
            "--hybrid_merged_checkpoint", str(REPO_ROOT / model["hybrid_merged_checkpoint"]),
            "--hybrid_text_model", model.get("hybrid_text_model", "ViT-B-32"),
            "--hybrid_text_pretrained", model.get("hybrid_text_pretrained", "openai"),
        ]
    log.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return results_path(model)


def build_summary(all_results: dict[str, pd.DataFrame], qtype: str) -> pd.DataFrame:
    """
    Wide summary DataFrame for one query type.
    Rows = queries, columns = (model, metric).
    """
    model_names = list(all_results.keys())
    first_df = next(
        df for df in all_results.values() if qtype in df["type"].values
    )
    subset = first_df[first_df["type"] == qtype][["query", "n_relevant"]].reset_index(drop=True)

    for model_name, df in all_results.items():
        if qtype not in df["type"].values:
            continue
        model_subset = (
            df[df["type"] == qtype][["query"] + METRIC_COLS]
            .rename(columns={c: f"{model_name}_{c}" for c in METRIC_COLS})
        )
        subset = subset.merge(model_subset, on="query", how="left")

    return subset


def build_macro_summary(all_results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per model, macro-averaged metrics for each query type present."""
    rows = []
    for model_name, df in all_results.items():
        row = {"model": model_name}
        for qtype in ["single", "pair", "negative"]:
            if qtype not in df["type"].values:
                continue
            subset = df[(df["type"] == qtype) & (df["n_relevant"] > 0)]
            for col in METRIC_COLS:
                row[f"{qtype}_{col}"] = subset[col].mean()
        rows.append(row)
    return pd.DataFrame(rows)


# ── Plots ─────────────────────────────────────────────────────────────────────

def make_plots(all_results: dict, macro_summary: pd.DataFrame, plots_dir: Path):
    plots_dir.mkdir(parents=True, exist_ok=True)
    model_names = list(all_results.keys())
    short_names = model_names
    qtypes_present = {
        qt for df in all_results.values() for qt in df["type"].unique()
    }

    # ── 1. Heatmap: P@10 and R@10 per label, per model (single queries) ───────
    if "single" in qtypes_present:
        for metric in ["P@10", "R@10"]:
            label_names = all_results[model_names[0]]
            label_names = label_names[label_names["type"] == "single"]["query"].tolist()

            matrix = np.array([
                all_results[m][all_results[m]["type"] == "single"][metric].values
                for m in model_names
            ])
            fig, ax = plt.subplots(figsize=(max(14, len(label_names) * 0.9), max(7, len(model_names) * 0.8)))
            sns.heatmap(matrix, xticklabels=label_names, yticklabels=short_names,
                        annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1,
                        linewidths=0.5, ax=ax)
            ax.set_title(f"{metric} per label — single label queries", fontsize=13, pad=12)
            ax.set_xlabel("CheXpert label")
            ax.set_ylabel("Model")
            plt.xticks(rotation=35, ha="right", fontsize=9)
            plt.tight_layout()
            path = plots_dir / f"heatmap_single_{metric.replace('@', 'at')}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            log.info(f"Saved → {path}")

    # ── 2. Macro bar chart: single and pair queries ────────────────────────────
    for qtype in ["single", "pair"]:
        if qtype not in qtypes_present:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Macro-averaged metrics — {qtype} label queries", fontsize=13)
        x = np.arange(len(model_names))
        width = 0.25
        for ax, prefix, metrics in [
            (axes[0], "P", ["P@1", "P@5", "P@10"]),
            (axes[1], "R", ["R@1", "R@5", "R@10"]),
        ]:
            # Collect all values first so we can set ylim before drawing labels
            all_metric_vals = []
            per_metric = []
            for metric in metrics:
                col = f"{qtype}_{metric}"
                vals = [
                    macro_summary.loc[macro_summary["model"] == m, col].values[0]
                    if col in macro_summary.columns else float("nan")
                    for m in model_names
                ]
                per_metric.append((metric, vals))
                all_metric_vals.extend(vals)
            finite_vals = [v for v in all_metric_vals if not np.isnan(v)]
            hi = max(finite_vals) if finite_vals else 1.0
            ylim_top = min(1.0, hi * 1.4)  # 40% headroom for bar labels
            ax.set_ylim(0, ylim_top)
            label_pad = ylim_top * 0.02  # 2% of axis height
            for i, (metric, vals) in enumerate(per_metric):
                bars = ax.bar(x + i * width, vals, width, label=metric)
                for bar, val in zip(bars, vals):
                    if np.isnan(val):
                        continue
                    text_y = min(bar.get_height() + label_pad, ylim_top * 0.97)
                    ax.text(bar.get_x() + bar.get_width() / 2, text_y,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=7)
            ax.set_xticks(x + width)
            ax.set_xticklabels(short_names, rotation=30, ha="right", fontsize=9)
            ax.set_ylabel(f"{'Precision' if prefix == 'P' else 'Recall'} @ k")
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune="both"))
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        path = plots_dir / f"macro_bar_{qtype}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        log.info(f"Saved → {path}")

    # ── 3. Per-label grouped bar: P@10 / R@10 for each label, bars = models ────
    if "single" in qtypes_present:
        for metric in ["P@10", "R@10"]:
            label_names = all_results[model_names[0]]
            label_names = label_names[label_names["type"] == "single"]["query"].tolist()
            n_labels = len(label_names)
            n_models = len(model_names)
            width = 0.8 / n_models
            x = np.arange(n_labels)
            fig, ax = plt.subplots(figsize=(max(14, n_labels * 1.0), 5))
            all_vals = []
            for i, (m, short) in enumerate(zip(model_names, short_names)):
                vals = all_results[m][all_results[m]["type"] == "single"][metric].values
                all_vals.extend(vals)
                ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=short)
            hi = max(all_vals) if all_vals else 1.0
            ax.set_ylim(0, min(1.0, hi * 1.3))
            ax.set_xticks(x)
            ax.set_xticklabels(label_names, rotation=35, ha="right", fontsize=9)
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} per label — all models", fontsize=13)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune="both"))
            ax.legend(fontsize=8, ncol=min(n_models, 4))
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            path = plots_dir / f"per_label_{metric.replace('@', 'at')}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            log.info(f"Saved → {path}")

    # ── 4. P@k / R@k curves (macro, single labels) ────────────────────────────
    if "single" in qtypes_present:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Macro P@k and R@k curves — single label queries", fontsize=13)
        for ax, prefix in [(axes[0], "P"), (axes[1], "R")]:
            for m, short in zip(model_names, short_names):
                col_base = f"single_{prefix}"
                vals = [
                    macro_summary.loc[macro_summary["model"] == m, f"{col_base}@{k}"].values[0]
                    if f"{col_base}@{k}" in macro_summary.columns else 0.0
                    for k in KS
                ]
                ax.plot(KS, vals, marker="o", label=short)
            ax.set_xlabel("k")
            ax.set_ylabel(f"{'Precision' if prefix == 'P' else 'Recall'} @ k")
            ax.set_xticks(KS)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        plt.tight_layout()
        path = plots_dir / "pk_rk_curves_single.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        log.info(f"Saved → {path}")

    # ── 5. Negative mode: macro bar chart ─────────────────────────────────────
    if "negative" in qtypes_present:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Macro-averaged metrics — 'yes A and no B' queries", fontsize=13)
        x = np.arange(len(model_names))
        width = 0.25
        for ax, prefix, metrics in [
            (axes[0], "P", ["P@1", "P@5", "P@10"]),
            (axes[1], "R", ["R@1", "R@5", "R@10"]),
        ]:
            all_metric_vals = []
            per_metric = []
            for metric in metrics:
                col = f"negative_{metric}"
                vals = [
                    macro_summary.loc[macro_summary["model"] == m, col].values[0]
                    if col in macro_summary.columns else 0.0
                    for m in model_names
                ]
                per_metric.append((metric, vals))
                all_metric_vals.extend(vals)
            hi = max(all_metric_vals) if all_metric_vals else 1.0
            ylim_top = min(1.0, hi * 1.4)
            ax.set_ylim(0, ylim_top)
            label_pad = ylim_top * 0.02
            for i, (metric, vals) in enumerate(per_metric):
                bars = ax.bar(x + i * width, vals, width, label=metric)
                for bar, val in zip(bars, vals):
                    text_y = min(bar.get_height() + label_pad, ylim_top * 0.97)
                    ax.text(bar.get_x() + bar.get_width() / 2, text_y,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=7)
            ax.set_xticks(x + width)
            ax.set_xticklabels(short_names, rotation=30, ha="right", fontsize=9)
            ax.set_ylabel(f"{'Precision' if prefix == 'P' else 'Recall'} @ k")
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune="both"))
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        path = plots_dir / "macro_bar_negative.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        log.info(f"Saved → {path}")

    # ── 6. Negative mode: 13×13 heatmap per model ─────────────────────────────
    # Rows = positive label required, columns = label that must be absent.
    # Cell value = P@10 (NaN if query had n_relevant=0 or doesn't exist).
    if "negative" in qtypes_present:
        n = len(CHEXPERT_LABELS)
        for m, short in zip(model_names, short_names):
            df = all_results[m]
            neg_df = df[(df["type"] == "negative") & df["pos_labels"].notna() & df["neg_labels"].notna()]
            if neg_df.empty:
                continue

            matrix = np.full((n, n), np.nan)
            for _, row in neg_df.iterrows():
                pos_col = row["pos_labels"]   # e.g. "chexpert_Atelectasis"
                neg_col = row["neg_labels"]   # e.g. "chexpert_Cardiomegaly"
                i = LABEL_TO_IDX.get(pos_col)
                j = LABEL_TO_IDX.get(neg_col)
                if i is not None and j is not None and row["n_relevant"] > 0:
                    matrix[i, j] = row["P@10"]

            fig, ax = plt.subplots(figsize=(14, 11))
            sns.heatmap(
                matrix,
                xticklabels=SHORT_LABELS,
                yticklabels=SHORT_LABELS,
                annot=True, fmt=".2f",
                cmap="YlOrRd",
                vmin=0, vmax=1,
                linewidths=0.3,
                ax=ax,
            )
            ax.set_title(
                f"P@10 — 'yes [row] and no [col]' queries\nModel: {short}",
                fontsize=12, pad=12,
            )
            ax.set_xlabel("Negated label  (must NOT be present / label=0)", fontsize=10)
            ax.set_ylabel("Positive label  (must be present / label=1)", fontsize=10)
            plt.xticks(rotation=40, ha="right", fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            plt.tight_layout()
            path = plots_dir / f"neg_heatmap_P10_{short}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            log.info(f"Saved → {path}")


# ── Publication-style table plots ────────────────────────────────────────────

def make_table_plots(macro_summary: pd.DataFrame, plots_dir: Path):
    """
    Render one PNG table per query type (single / pair / negative).
    Layout mirrors a LaTeX booktabs table:
      - Dark header row with white bold text
      - Alternating white / light-grey row shading
      - Best value per column highlighted green and bolded
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    # colours
    HDR_BG   = "#2c3e50"   # dark slate header background
    HDR_FG   = "white"
    ROW_ODD  = "white"
    ROW_EVEN = "#f2f2f2"   # very light grey
    BEST_BG  = "#c8f0d0"   # soft green for best-in-column
    ROW_LBL  = "#dce3ea"   # muted blue-grey for row-label column

    for qtype in ["single", "pair", "negative"]:
        cols = [f"{qtype}_{m}" for m in METRIC_COLS]
        available = [c for c in cols if c in macro_summary.columns]
        if not available:
            continue

        models  = macro_summary["model"].tolist()
        n_rows  = len(models)
        n_cols  = len(available)
        col_headers = [c.replace(f"{qtype}_", "") for c in available]

        data = macro_summary[available].values.astype(float)  # (n_rows, n_cols)
        best_row = np.nanargmax(data, axis=0)                 # index of best per column

        # ── build cell text & colours ─────────────────────────────────────────
        cell_text   = []
        cell_colors = []
        for i, _ in enumerate(models):
            row_txt = []
            row_clr = []
            base_bg = ROW_ODD if i % 2 == 0 else ROW_EVEN
            for j in range(n_cols):
                val = data[i, j]
                row_txt.append(f"{val:.4f}" if not np.isnan(val) else "—")
                row_clr.append(BEST_BG if best_row[j] == i else base_bg)
            cell_text.append(row_txt)
            cell_colors.append(row_clr)

        # ── figure ────────────────────────────────────────────────────────────
        col_w   = 1.1
        row_h   = 0.45
        lbl_w   = max(len(m) for m in models) * 0.11  # proportional to longest name
        fig_w   = lbl_w + n_cols * col_w + 0.4
        fig_h   = 0.7 + n_rows * row_h + 0.5
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")

        tbl = ax.table(
            cellText=cell_text,
            rowLabels=models,
            colLabels=col_headers,
            cellColours=cell_colors,
            rowColours=[ROW_LBL] * n_rows,
            colColours=[HDR_BG]  * n_cols,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.6)

        # style header cells
        for j in range(n_cols):
            cell = tbl[0, j]
            cell.set_text_props(color=HDR_FG, fontweight="bold")
            cell.set_edgecolor("#444444")

        # bold best-in-column values
        for j, best_i in enumerate(best_row):
            tbl[best_i + 1, j].set_text_props(fontweight="bold")

        # thicker top/bottom borders (booktabs feel)
        for j in range(n_cols):
            tbl[0,          j].set_linewidth(1.5)
            tbl[n_rows,     j].set_linewidth(1.5)

        ax.set_title(
            f"Macro retrieval metrics — {qtype} label queries\n"
            r"(green = best per column)",
            fontsize=11, fontweight="bold", pad=10,
        )

        plt.tight_layout()
        path = plots_dir / f"table_{qtype}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        log.info(f"Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run all model evaluations and produce summary CSVs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--paired_dir", required=True)
    parser.add_argument("--csv", required=False,default="cxr_data/all_txt_data_and_labels.csv",)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of dataset items to evaluate(default: all, which is 5159 for cxr dataset)")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to store all results CSVs and plots for this experiment")
    parser.add_argument("--query_mode", default="all",
                        choices=["single", "pair", "negative", "all"],
                        help="Which query types to evaluate (passed to eval_model.py). Default: all")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip models whose results CSV already exists")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for image and text encoding (default: 64)")
    parser.add_argument("--nan-mode", default="negative", choices=["negative", "ignore"],
                        help="How to treat NaN labels in negative-query relevance. "
                             "'negative' (default): NaN counts as absent (same as CSV 0). "
                             "'ignore': only CSV 0 counts as absent; NaN images are excluded.")
    parser.add_argument("--wandb-project", default=None,
                        help="W&B project name. Omit to disable W&B logging.")
    parser.add_argument("--wandb-run-name", default=None,
                        help="W&B run name (auto-generated if omitted).")
    parser.add_argument("--wandb-entity", default=None,
                        help="W&B entity (team/user). Uses your default entity if omitted.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Run evaluations ───────────────────────────────────────────────────────
    all_results = {}
    for i, model in enumerate(MODELS):
        print("\n" + "=" * 80)
        log.info(f"Evaluating model {i+1}/{len(MODELS)}: {model['name']}")
        out_path = output_dir / results_path(model).name

        if args.skip_existing and out_path.exists():
            log.info(f"Skipping {model['name']} — results already exist at {out_path}")
        else:
            checkpoint_file = CHECKPOINTS_DIR / model["checkpoint"] if model["checkpoint"] else None
            if model["checkpoint"] and not checkpoint_file.exists():
                log.warning(f"Checkpoint not found, skipping {model['name']}: {checkpoint_file}")
                continue
            if model["model_type"] == "finetuned":
                ft_ckpt = REPO_ROOT / model["finetuned_checkpoint"]
                if not ft_ckpt.exists():
                    log.warning(f"Finetuned checkpoint not found, skipping {model['name']}: {ft_ckpt}")
                    continue
            if model["model_type"] == "cxrclip_hybrid":
                merged_ckpt = REPO_ROOT / model["hybrid_merged_checkpoint"]
                if not merged_ckpt.exists():
                    log.warning(f"Hybrid checkpoint not found, skipping {model['name']}: {merged_ckpt}")
                    continue
            try:
                raw_path = run_eval(
                    model, args.paired_dir, args.csv,
                    query_mode=args.query_mode,
                    batch_size=args.batch_size,
                    max_samples=args.max_samples,
                    nan_mode=args.nan_mode,
                )
                if raw_path.exists() and raw_path != out_path:
                    shutil.move(str(raw_path), out_path)  # rename fails cross-filesystem
            except Exception as e:
                log.error(f"Failed to evaluate {model['name']}: {e}")
                continue

        if out_path.exists():
            all_results[model["name"]] = pd.read_csv(out_path)
        else:
            log.warning(f"No results file found for {model['name']}, skipping from summary")

    if not all_results:
        log.error("No results to summarize.")
        return

    macro_summary = build_macro_summary(all_results)

    # ── Build and save per-type summaries ────────────────────────────────────
    qtypes_present = {qt for df in all_results.values() for qt in df["type"].unique()}

    summary_paths = {}
    for qtype in ["single", "pair", "negative"]:
        if qtype not in qtypes_present:
            continue
        try:
            summary_df = build_summary(all_results, qtype)
            path = output_dir / f"summary_{qtype}.csv"
            summary_df.to_csv(path, index=False)
            summary_paths[qtype] = path
            log.info(f"Saved → {path}")
        except Exception as e:
            log.warning(f"Could not build {qtype} summary: {e}")

    macro_path = output_dir / "summary_macro.csv"
    macro_summary.to_csv(macro_path, index=False)
    log.info(f"Saved → {macro_path}")

    # ── Generate plots ────────────────────────────────────────────────────────
    plots_dir = output_dir / "plots"
    make_plots(all_results, macro_summary, plots_dir)
    make_table_plots(macro_summary, plots_dir)

    # ── W&B logging ───────────────────────────────────────────────────────────
    if args.wandb_project:
        if wandb is None:
            raise ImportError("wandb is not installed. Run: pip install wandb")
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            job_type="eval",
            config=vars(args),
            dir=str(output_dir),
        )
        # One summary table with all models × all metrics
        wandb.log({"eval/macro_summary": wandb.Table(dataframe=macro_summary)})
        # Flat scalars: eval/<model>/<qtype>_<metric>  (e.g. eval/lora_vitb32/single_P@10)
        for _, row in macro_summary.iterrows():
            metrics = {
                f"eval/{row['model']}/{col}": row[col]
                for col in macro_summary.columns
                if col != "model" and not (isinstance(row[col], float) and np.isnan(row[col]))
            }
            wandb.log(metrics)
        # Upload every plot image
        for plot_path in sorted(plots_dir.glob("*.png")):
            wandb.log({f"plots/{plot_path.stem}": wandb.Image(str(plot_path))})
        wandb.finish()
        log.info("W&B eval run finished.")

    # ── Print macro summary tables ────────────────────────────────────────────
    for qtype in ["single", "pair", "negative"]:
        cols = [f"{qtype}_{c}" for c in METRIC_COLS]
        available = [c for c in cols if c in macro_summary.columns]
        if not available:
            continue
        print("\n" + "=" * 80)
        print(f"MACRO SUMMARY — {qtype.upper()} label queries")
        print("=" * 80)
        print(macro_summary[["model"] + available].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        ))
    print()


if __name__ == "__main__":
    main()
