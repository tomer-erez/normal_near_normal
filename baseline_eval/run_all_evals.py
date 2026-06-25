"""
Run retrieval evaluation for all models sequentially, then summarize results
into comparison tables and plots.

Query types evaluated (counts for N=10 active CHEXPERT_LABELS):
  single                  10 queries — "atelectasis"           relevant: label==1
  pair                    45 queries — "atelectasis and edema"  relevant: both labels==1  [C(N,2)]
  negative                90 queries — "atelectasis and no X"  relevant: pos==1 AND neg∈{0,NaN}  (lenient)  [P(N,2)]
  negative_strict         90 queries — same queries, strict: neg must be CSV 0 (NaN not counted)
  negative_robust         90 queries — same queries, rephrased template, lenient
  negative_robust_strict  90 queries — rephrased template, strict

For each model four eval passes are run automatically:
  pass 1: single + pair + negative  (standard template, lenient: NaN=absent)
  pass 2: negative only             (robust/rephrased template, lenient)
  pass 3: negative only             (standard template, strict: only CSV 0 counts as absent)
  pass 4: negative only             (robust/rephrased template, strict)

Output files (in --output_dir):
  summary_single.csv                 P@k / R@k per model, single-label queries
  summary_pair.csv                   P@k / R@k per model, pair queries
  summary_negative.csv               P@k / R@k per model, negative queries (lenient)
  summary_negative_strict.csv        P@k / R@k per model, negative queries (strict)
  summary_negative_robust.csv        P@k / R@k per model, negative queries rephrased (lenient)
  summary_negative_robust_strict.csv P@k / R@k per model, negative queries rephrased (strict)
  summary_macro.csv                  macro-averaged metrics per model (one row per model)
  plots/                             comparison charts + per-model NxN heatmaps (N = active labels)

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
sys.path.insert(0, str(REPO_ROOT))

from constants import CHEXPERT_LABELS, LABEL_COLS
LABEL_TO_IDX = {col: i for i, col in enumerate(LABEL_COLS)}
SHORT_LABELS = CHEXPERT_LABELS

MODELS = [
    # Baselines
    # {"name": "vanilla_clip", "model_type": "vanilla_clip", "checkpoint": None},
    # {"name": "biomedclip",   "model_type": "biomedclip",   "checkpoint": None},
    {"name": "cxr-clip", "model_type": "cxrclip", "checkpoint": "swint_m.pt"},
    # Fine-tuned models — add entries here after training, e.g.:
    # {
    #     "name": "lora_vitb32",
    #     "model_type": "finetuned",
    #     "checkpoint": None,
    #     "finetuned_base_model": "ViT-B-32",
    #     "finetuned_pretrained": "openai",
    #     "finetuned_checkpoint": "experiments/lora_vitb32/final_merged.pt",
    # },
    {
        "name": "label_dot_hnm",
        "model_type": "cxrclip_finetune",
        "checkpoint": None,
        "cxrclip_finetune_image_checkpoint": "valid_pretrained_models_to_try/swint_mc.pt",
        "cxrclip_finetune_merged_checkpoint": "experiments/labeldot_hnm_swint_hnm03/final_merged.pt",
    },
    {
        "name": "label_dot_hnm_single75_neg25",
        "model_type": "cxrclip_finetune",
        "checkpoint": None,
        "cxrclip_finetune_image_checkpoint": "valid_pretrained_models_to_try/swint_mc.pt",
        "cxrclip_finetune_merged_checkpoint": "experiments/single75_neg25/final_merged.pt",
    },
]

KS = [1, 3, 5]
METRIC_COLS = [f"P@{k}" for k in KS] + [f"R@{k}" for k in KS] + [f"AP@{k}" for k in KS]
HNRR_COLS  = [f"HNRR@{k}" for k in KS]   # only meaningful for negative query types


# ── Helpers ───────────────────────────────────────────────────────────────────

def results_path(model: dict) -> Path:
    # eval_model.py always receives --name and saves to results_{name}.csv
    return REPO_ROOT / f"results_{model['name']}.csv"


def results_robust_path(model: dict) -> Path:
    """Repo-root path for linguistic robustness (rephrased template) results."""
    base = results_path(model)
    return base.with_name(base.stem + "_robust.csv")


def results_strict_path(model: dict) -> Path:
    """Repo-root path for strict-negation results (only CSV 0 counts as absent)."""
    base = results_path(model)
    return base.with_name(base.stem + "_strict.csv")


def results_robust_strict_path(model: dict) -> Path:
    """Repo-root path for rephrased template + strict negation results."""
    base = results_path(model)
    return base.with_name(base.stem + "_robust_strict.csv")


def _merge_model_results(model: dict, output_dir: Path) -> pd.DataFrame | None:
    """
    Load and merge all four result-variant CSVs for one model from output_dir.

    Returns a single DataFrame with a 'type' column distinguishing:
      negative              — standard template, lenient
      negative_robust       — rephrased template, lenient
      negative_strict       — standard template, strict
      negative_robust_strict — rephrased template, strict

    Returns None if the primary results file does not exist.
    """
    out_path = output_dir / results_path(model).name
    if not out_path.exists():
        return None
    frames = [pd.read_csv(out_path)]
    for path_fn, new_type in [
        (results_robust_path,       "negative_robust"),
        (results_strict_path,       "negative_strict"),
        (results_robust_strict_path,"negative_robust_strict"),
    ]:
        p = output_dir / path_fn(model).name
        if p.exists():
            df = pd.read_csv(p)
            df.loc[df["type"] == "negative", "type"] = new_type
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def run_eval(model: dict, paired_dir: str, csv: str,
             query_mode: str = "all", batch_size: int = 64,
             max_samples: int | None = None,
             neg_template: str | None = None,
             strict: bool = False) -> Path:
    """Run eval_model.py for one model. Returns path to results CSV."""
    name = model["name"] + ("_strict" if strict else "")
    cmd = [
        sys.executable, str(REPO_ROOT / "baseline_eval" / "eval_model.py"),
        "--model_type", model["model_type"],
        "--name", name,
        "--paired_dir", paired_dir,
        "--csv", csv,
        "--query_mode", query_mode,
        "--batch_size", str(batch_size),
        "--ks", *[str(k) for k in KS],
    ]
    if strict:
        cmd.append("--strict-negation")
    if neg_template is not None:
        cmd += ["--neg-template", neg_template]
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
    if model["model_type"] == "cxrclip_finetune":
        cmd += [
            "--cxrclip_finetune_image_checkpoint", str(REPO_ROOT / model["cxrclip_finetune_image_checkpoint"]),
            "--cxrclip_finetune_merged_checkpoint", str(REPO_ROOT / model["cxrclip_finetune_merged_checkpoint"]),
        ]
    log.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return results_strict_path(model) if strict else results_path(model)


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
        for qtype in ["single", "pair", "negative", "negative_strict", "negative_robust", "negative_robust_strict"]:
            if qtype not in df["type"].values:
                continue
            subset = df[(df["type"] == qtype) & (df["n_relevant"] > 0)]
            for col in METRIC_COLS:
                if col in subset.columns:
                    row[f"{qtype}_{col}"] = subset[col].mean()
            # HNRR: averaged over all negative queries (regardless of n_relevant)
            if qtype.startswith("negative"):
                neg_subset = df[df["type"] == qtype]
                for col in HNRR_COLS:
                    if col in neg_subset.columns:
                        row[f"{qtype}_{col}"] = neg_subset[col].mean()
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

    K_LAST = KS[-1]
    # ── 1. Heatmap: P@Klast and R@Klast per label, per model (single queries) ──
    if "single" in qtypes_present:
        for metric in [f"P@{K_LAST}", f"R@{K_LAST}"]:
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
            (axes[0], "P", [f"P@{k}" for k in KS]),
            (axes[1], "R", [f"R@{k}" for k in KS]),
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

    # ── 3. Per-label grouped bar: P@Klast / R@Klast for each label, bars = models ──
    if "single" in qtypes_present:
        for metric in [f"P@{K_LAST}", f"R@{K_LAST}"]:
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

    # ── 5. Negative mode: macro bar charts (one per negative variant) ──────────
    neg_variants = [
        ("negative",               "negation queries (lenient: NaN=absent)"),
        ("negative_strict",        "negation queries (strict: only CSV 0 counts)"),
        ("negative_robust",        "negation queries — rephrased template (lenient)"),
        ("negative_robust_strict", "negation queries — rephrased template (strict)"),
    ]
    for neg_qtype, neg_subtitle in neg_variants:
        if neg_qtype not in qtypes_present:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Macro-averaged metrics — {neg_subtitle}", fontsize=13)
        x = np.arange(len(model_names))
        width = 0.25
        for ax, prefix, metrics in [
            (axes[0], "P", [f"P@{k}" for k in KS]),
            (axes[1], "R", [f"R@{k}" for k in KS]),
        ]:
            all_metric_vals = []
            per_metric = []
            for metric in metrics:
                col = f"{neg_qtype}_{metric}"
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
        path = plots_dir / f"macro_bar_{neg_qtype}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        log.info(f"Saved → {path}")

    # ── 6. HNRR bar chart (one per negative variant) ──────────────────────────
    for neg_qtype, neg_subtitle in neg_variants:
        if neg_qtype not in qtypes_present:
            continue
        hnrr_cols = [f"{neg_qtype}_{c}" for c in HNRR_COLS]
        available = [c for c in hnrr_cols if c in macro_summary.columns]
        if not available:
            continue
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.suptitle(f"HNRR — hard negatives in top-k — {neg_subtitle}", fontsize=13)
        x = np.arange(len(model_names))
        width = 0.25
        all_vals = []
        per_metric = []
        for col in available:
            metric = col.replace(f"{neg_qtype}_", "")
            vals = [
                macro_summary.loc[macro_summary["model"] == m, col].values[0]
                if col in macro_summary.columns else float("nan")
                for m in model_names
            ]
            per_metric.append((metric, vals))
            all_vals.extend(vals)
        finite = [v for v in all_vals if not np.isnan(v)]
        hi = max(finite) if finite else 1.0
        ylim_top = min(1.0, hi * 1.4)
        ax.set_ylim(0, ylim_top)
        label_pad = ylim_top * 0.02
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
        ax.set_ylabel("HNRR @ k  (lower is better)")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune="both"))
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        path = plots_dir / f"hnrr_bar_{neg_qtype}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        log.info(f"Saved → {path}")

    # ── 7. Negative mode: NxN heatmap per model (both variants) ─────────────
    for neg_qtype, neg_subtitle in neg_variants:
        if neg_qtype not in qtypes_present:
            continue
        n = len(CHEXPERT_LABELS)
        for m, short in zip(model_names, short_names):
            df = all_results[m]
            neg_df = df[
                (df["type"] == neg_qtype)
                & df["pos_labels"].notna()
                & df["neg_labels"].notna()
            ]
            if neg_df.empty:
                continue

            matrix = np.full((n, n), np.nan)
            for _, row in neg_df.iterrows():
                pos_col = row["pos_labels"]
                neg_col = row["neg_labels"]
                i = LABEL_TO_IDX.get(pos_col)
                j = LABEL_TO_IDX.get(neg_col)
                if i is not None and j is not None and row["n_relevant"] > 0:
                    matrix[i, j] = row[f"P@{K_LAST}"]

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
                f"P@{K_LAST} — 'yes [row] and no [col]' queries\nModel: {short}  |  {neg_subtitle}",
                fontsize=12, pad=12,
            )
            ax.set_xlabel("Negated label  (must NOT be present)", fontsize=10)
            ax.set_ylabel("Positive label  (must be present)", fontsize=10)
            plt.xticks(rotation=40, ha="right", fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            plt.tight_layout()
            path = plots_dir / f"neg_heatmap_P10_{neg_qtype}_{short}.png"
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

    for qtype in ["single", "pair", "negative", "negative_strict", "negative_robust", "negative_robust_strict"]:
        cols = [f"{qtype}_{m}" for m in METRIC_COLS]
        available = [c for c in cols if c in macro_summary.columns]
        if not available:
            continue

        models  = macro_summary["model"].tolist()
        n_rows  = len(models)
        n_cols  = len(available)
        col_headers = [c.replace(f"{qtype}_", "") for c in available]

        data = macro_summary[available].values.astype(float)  # (n_rows, n_cols)
        best_val = np.nanmax(data, axis=0)
        best_mask = data == best_val[np.newaxis, :]
        is_unique_best = best_mask.sum(axis=0) == 1          # suppress green on ties

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
                row_clr.append(BEST_BG if (best_mask[i, j] and is_unique_best[j]) else base_bg)
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

        # bold best-in-column values (only when unique — no bold on ties)
        for j in range(n_cols):
            if is_unique_best[j]:
                best_i = int(np.where(best_mask[:, j])[0][0])
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


# ── Parallel coordinates plot ─────────────────────────────────────────────────

def make_parallel_coordinates_plot(macro_summary: pd.DataFrame, plots_dir: Path):
    """
    One line per model across axes: Single P@5, Pair P@5, Negation P@5,
    Neg-Robust P@5, HNRR@5 (inverted so higher = better).
    All axes are independently normalized to [0, 1] for comparability.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    # (summary_column, display_label, invert_axis)
    AXES_DEF = [
        ("single_P@5",                "Single\nP@5",          False),
        ("pair_P@5",                  "Pair\nP@5",            False),
        ("negative_P@5",              "Neg.\nP@5",            False),
        ("negative_strict_P@5",       "Neg.\nStrict P@5",     False),
        ("negative_robust_P@5",       "Neg-Rob.\nP@5",        False),
        ("negative_robust_strict_P@5","Neg-Rob.\nStr. P@5",   False),
        ("negative_HNRR@5",            "HNRR@5\n(lower=better)", True),
    ]
    avail = [(col, lbl, inv) for col, lbl, inv in AXES_DEF if col in macro_summary.columns]
    if len(avail) < 2:
        log.warning("Parallel coordinates: not enough metrics available, skipping.")
        return

    model_names = macro_summary["model"].tolist()
    n_models = len(model_names)
    n_ax = len(avail)

    raw = {col: macro_summary[col].values.astype(float) for col, _, _ in avail}

    def _norm(vals: np.ndarray, inverted: bool) -> np.ndarray:
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        if hi == lo:
            return np.full_like(vals, 0.5)
        n = (vals - lo) / (hi - lo)
        return 1.0 - n if inverted else n

    normed = {col: _norm(raw[col], inv) for col, _, inv in avail}

    colors = [plt.cm.tab10(i % 10) for i in range(n_models)]

    fig, ax = plt.subplots(figsize=(max(11, n_ax * 2.6), 6))
    # extra room: top for title, bottom for legend + axis labels
    fig.subplots_adjust(top=0.88, bottom=0.22)
    ax.set_xlim(-0.5, n_ax - 0.5)
    ax.set_ylim(-0.05, 1.05)   # tight — labels go outside via offset
    ax.axis("off")

    x_pos = list(range(n_ax))

    _lbl_kw = dict(fontsize=7, color="#444444",
                   bbox=dict(facecolor="white", edgecolor="#cccccc", pad=2.0,
                             boxstyle="round,pad=0.2"), zorder=5)

    # Vertical axis lines + min/max tick labels placed OUTSIDE data range
    for i, (col, lbl, inv) in enumerate(avail):
        ax.axvline(x=i, color="#aaaaaa", linewidth=1.0, zorder=1)
        lo, hi = float(np.nanmin(raw[col])), float(np.nanmax(raw[col]))
        bottom_val = hi if inv else lo   # value shown at y=0 end
        top_val    = lo if inv else hi   # value shown at y=1 end
        # tick nubs
        ax.plot([i - 0.03, i + 0.03], [0.0, 0.0], color="#999999", linewidth=0.8, zorder=2)
        ax.plot([i - 0.03, i + 0.03], [1.0, 1.0], color="#999999", linewidth=0.8, zorder=2)
        # labels sit just outside [0, 1] so data lines never cover them
        ax.text(i, -0.04, f"{bottom_val:.3f}", ha="center", va="top",    **_lbl_kw)
        ax.text(i,  1.04, f"{top_val:.3f}",    ha="center", va="bottom", **_lbl_kw)
        # axis name below the plot (in figure space, not data space)
        ax.text(i, -0.13, lbl, ha="center", va="top",
                fontsize=8.5, fontweight="bold", transform=ax.transData)
        if inv:
            ax.text(i, 1.13, "▲ better", ha="center", va="bottom",
                    fontsize=6.5, color="#888888", style="italic")

    # One line per model
    for m_idx, model_name in enumerate(model_names):
        ys = [float(normed[col][m_idx]) for col, _, _ in avail]
        if any(np.isnan(v) for v in ys):
            continue
        ax.plot(x_pos, ys, marker="o", markersize=5, linewidth=1.8,
                color=colors[m_idx], alpha=0.85, label=model_name, zorder=3)

    # Legend at the bottom, below the axis labels
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=min(n_models, 4), fontsize=8, frameon=True,
        fancybox=False, edgecolor="#cccccc",
    )
    ax.set_title(
        "Model comparison — parallel coordinates\n"
        "(axes independently normalized; HNRR inverted so ↑ = better)",
        fontsize=10, pad=10,
    )

    plt.tight_layout()
    base = plots_dir / "parallel_coordinates"
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base) + ".png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {base}.pdf / .png")


# ── Parallel coordinates — vertical layout (single-column paper) ──────────────

def make_parallel_coordinates_plot_vertical(macro_summary: pd.DataFrame, plots_dir: Path):
    """
    Vertical parallel coordinates — metrics stacked top-to-bottom, model lines
    run horizontally across the page. Fits in a single column (~3.5 in wide).
    Saves both .pdf and .png.
    """
    import math

    plots_dir.mkdir(parents=True, exist_ok=True)

    AXES_DEF = [
        ("single_P@5",                "Single P@5",        False),
        ("pair_P@5",                  "Pair P@5",          False),
        ("negative_P@5",              "Neg. P@5",          False),
        ("negative_strict_P@5",       "Neg. Strict P@5",   False),
        ("negative_robust_P@5",       "Neg-Rob. P@5",      False),
        ("negative_robust_strict_P@5","Neg-Rob. Str. P@5", False),
        ("negative_HNRR@5",            "HNRR@5 (↓)",          True),
    ]
    avail = [(col, lbl, inv) for col, lbl, inv in AXES_DEF if col in macro_summary.columns]
    if len(avail) < 2:
        log.warning("Vertical parallel coordinates: not enough metrics available, skipping.")
        return

    model_names = macro_summary["model"].tolist()
    n_models    = len(model_names)
    n_ax        = len(avail)

    raw = {col: macro_summary[col].values.astype(float) for col, _, _ in avail}

    def _norm(vals: np.ndarray, inv: bool) -> np.ndarray:
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        if hi == lo:
            return np.full_like(vals, 0.5)
        n = (vals - lo) / (hi - lo)
        return 1.0 - n if inv else n

    normed = {col: _norm(raw[col], inv) for col, _, inv in avail}
    colors = [plt.cm.tab10(i % 10) for i in range(n_models)]
    y_pos  = list(range(n_ax))

    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, ax = plt.subplots(figsize=(3.8, 5.8))
    fig.subplots_adjust(left=0.28, right=0.97, top=0.88, bottom=0.20)

    # y=0 at top, y=n_ax-1 at bottom (inverted)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(n_ax - 0.5, -0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks(y_pos)
    ax.set_yticklabels([lbl for _, lbl, _ in avail],
                       fontsize=8.5, fontweight="bold", ha="right")
    ax.tick_params(axis="y", length=0, pad=5)

    _val_kw = dict(fontsize=6.5, color="#555555",
                   bbox=dict(facecolor="white", edgecolor="#cccccc",
                             pad=1.5, boxstyle="round,pad=0.15"),
                   zorder=5)

    for i, (col, lbl, inv) in enumerate(avail):
        # Horizontal axis line
        ax.axhline(y=i, color="#aaaaaa", linewidth=1.0, zorder=1)
        # Tick nubs at x=0 (worst) and x=1 (best)
        ax.plot([0, 0], [i - 0.07, i + 0.07], color="#999999", linewidth=0.8, zorder=2)
        ax.plot([1, 1], [i - 0.07, i + 0.07], color="#999999", linewidth=0.8, zorder=2)

        lo, hi = float(np.nanmin(raw[col])), float(np.nanmax(raw[col]))
        worst = hi if inv else lo   # value at x=0 end
        best  = lo if inv else hi   # value at x=1 end
        # Value labels sit just below each axis line (i+0.24 = visually lower)
        ax.text(0.0, i + 0.24, f"{worst:.3f}", ha="center", va="top", **_val_kw)
        ax.text(1.0, i + 0.24, f"{best:.3f}",  ha="center", va="top", **_val_kw)
        # Direction hint above the line
        # if inv:
        #     ax.text(0.02, i - 0.17, "← better", ha="left",  va="bottom",
        #             fontsize=5.5, color="#888888", style="italic")
        # else:
        ax.text(0.98, i - 0.17, "better →", ha="right", va="bottom",
                fontsize=5.5, color="#888888", style="italic")

    # One line per model
    for m_idx, mname in enumerate(model_names):
        xs = [float(normed[col][m_idx]) for col, _, _ in avail]
        if any(math.isnan(x) for x in xs):
            continue
        ax.plot(xs, y_pos, marker="o", markersize=5, linewidth=1.8,
                color=colors[m_idx], alpha=0.85, label=mname, zorder=3)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=min(n_models, 3), fontsize=7.5, frameon=True,
        fancybox=False, edgecolor="#cccccc",
    )
    ax.set_title(
        "Model comparison — parallel coordinates\n",
        fontsize=8.5, pad=8,
    )

    plt.rcParams.update({"font.family": "sans-serif"})
    base = plots_dir / "parallel_coordinates_vertical"
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base) + ".png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {base}.pdf / .png")


# ── Radar / spider chart ──────────────────────────────────────────────────────

def make_radar_plot(macro_summary: pd.DataFrame, plots_dir: Path):
    """
    Saves two versions:
      radar.pdf / .png            — radar + summary table (main paper figure)
      radar_standalone.pdf / .png — compact radar only (column inset)
    """
    import math

    plots_dir.mkdir(parents=True, exist_ok=True)

    SPOKES = [
        ("single_AP@5",                "Single\nMAP@5",       False),
        ("pair_AP@5",                  "Pair\nMAP@5",         False),
        ("negative_AP@5",              "Neg.\nMAP@5",         False),
        ("negative_strict_AP@5",       "Neg.\nStr. MAP@5",    False),
        ("negative_robust_AP@5",       "Neg-Rob.\nMAP@5",     False),
        ("negative_robust_strict_AP@5","Neg-Rob.\nStr. MAP@5",False),
        ("negative_HNRR@5",            "HNRR@5\n(↓)",         True),
    ]
    DISP = {
        "vanilla_clip_not_trained": "CLIP",
        "cxrclip_r50_mc":           "CXR-CLIP",
        "labeldot_hnm_swint_hnm03": "Ours",
    }
    # Accessible, print-friendly palette — stable per model name
    _FIXED = {
        "vanilla_clip_not_trained": "#4878d0",   # blue
        "cxrclip_r50_mc":           "#ee854a",   # orange
        "labeldot_hnm_swint_hnm03": "#6acc65",   # green
    }
    _EXTRA = ["#d65f5f", "#956cb4", "#857c77", "#8c613c"]

    avail = [(c, lbl, inv) for c, lbl, inv in SPOKES if c in macro_summary.columns]
    N = len(avail)
    if N < 3:
        return

    model_names = macro_summary["model"].tolist()
    n_models    = len(model_names)

    raw    = {c: macro_summary[c].values.astype(float) for c, _, _ in avail}
    lo_raw = {c: float(np.nanmin(v)) for c, v in raw.items()}
    hi_raw = {c: float(np.nanmax(v)) for c, v in raw.items()}

    def _norm(col, inv):
        span = hi_raw[col] - lo_raw[col]
        n = np.zeros(n_models) if span == 0 else (raw[col] - lo_raw[col]) / span
        return 1.0 - n if inv else n

    normed = {c: _norm(c, inv) for c, _, inv in avail}
    angles = [2 * math.pi * i / N for i in range(N)]
    ang_cl = angles + [angles[0]]

    _extra_it = iter(_EXTRA)
    colors = [_FIXED.get(m) or next(_extra_it, "#777777") for m in model_names]

    def _ha(a):
        s = math.sin(a)
        return "center" if abs(s) < 0.18 else ("left" if s > 0 else "right")

    def _va(a):
        c = math.cos(a)
        return "bottom" if c > 0.55 else ("top" if c < -0.55 else "center")

    # ── shared helpers ─────────────────────────────────────────────────────────
    def _setup_polar(ax):
        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels([])
        ax.yaxis.grid(True, color="#e0e0e0", linestyle="dotted", linewidth=0.6)
        ax.spines["polar"].set_visible(False)
        ring_a = np.linspace(0, 2 * math.pi, 300)
        ax.plot(ring_a, np.ones(300), color="#bbbbbb", linewidth=0.8,
                linestyle="--", zorder=1)
        ax.set_xticks(angles)
        ax.set_xticklabels([])
        for a in angles:
            ax.plot([a, a], [0, 1], color="#d8d8d8", linewidth=0.7, zorder=1)

    def _draw_models(ax, legend=True):
        for m_idx, mname in enumerate(model_names):
            vals = [float(normed[c][m_idx]) for c, _, _ in avail]
            if any(math.isnan(v) for v in vals):
                continue
            vc = vals + [vals[0]]
            ax.fill(ang_cl, vc, color=colors[m_idx], alpha=0.13, zorder=2)
            ax.plot(ang_cl, vc, color=colors[m_idx], linewidth=2.0, zorder=3,
                    label=DISP.get(mname, mname) if legend else None)
            ax.scatter(angles, vals, color=colors[m_idx], s=36, zorder=4,
                       clip_on=False, edgecolors="white", linewidths=0.6)

    def _draw_spoke_labels(ax, r=1.22, fs=8.5):
        for i, (_, lbl, _) in enumerate(avail):
            a = angles[i]
            ax.text(a, r, lbl, ha=_ha(a), va=_va(a),
                    fontsize=fs, fontweight="bold", transform=ax.transData)

    # ── table data (shared between both versions) ──────────────────────────────
    HDR_BG    = "#2c3e50"
    HDR_FG    = "white"
    BEST_BG   = "#c6efce"
    col_hdrs  = [lbl.replace("\n", " ") for _, lbl, _ in avail]
    row_lbls  = [DISP.get(m, m) for m in model_names]
    cell_text = [[f"{raw[c][m_idx]:.3f}" for c, _, _ in avail]
                 for m_idx in range(n_models)]

    cell_colors_tbl = [["white"] * N for _ in range(n_models)]
    for col_idx, (c, _, inv) in enumerate(avail):
        valid = [(m, raw[c][m]) for m in range(n_models) if not math.isnan(raw[c][m])]
        if not valid:
            continue
        best_m = min(valid, key=lambda x: x[1])[0] if inv else max(valid, key=lambda x: x[1])[0]
        cell_colors_tbl[best_m][col_idx] = BEST_BG

    def _draw_table(ax_t):
        ax_t.axis("off")
        tbl = ax_t.table(
            cellText=cell_text,
            rowLabels=row_lbls,
            colLabels=col_hdrs,
            cellLoc="center",
            rowLoc="right",
            loc="center",
            cellColours=cell_colors_tbl,
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.0, 1.55)
        for (row, col), cell in tbl.get_celld().items():
            cell.set_linewidth(0.5)
            if col == -1 and row > 0:                       # row label
                m_idx = row - 1
                cell.set_text_props(color=colors[m_idx], fontweight="bold")
                cell.set_facecolor("#f8f8f8")
                cell.set_edgecolor("#cccccc")
            elif row == 0:                                  # column header
                cell.set_text_props(color=HDR_FG, fontweight="bold")
                cell.set_facecolor(HDR_BG)
                cell.set_edgecolor("#444444")
            if row > 0 and col >= 0:                        # bold best value
                m_idx  = row - 1
                c_name = avail[col][0]
                inv_c  = avail[col][2]
                valid  = [raw[c_name][m] for m in range(n_models)
                          if not math.isnan(raw[c_name][m])]
                if valid:
                    best_val = min(valid) if inv_c else max(valid)
                    if not math.isnan(raw[c_name][m_idx]) and raw[c_name][m_idx] == best_val:
                        cell.set_text_props(fontweight="bold")

    plt.rcParams.update({"font.family": "serif", "font.size": 9})

    # ════════════════════════════════════════════════════════════════════════════
    # Version 1: radar + table
    # ════════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(5.5, 7.0))
    ax  = fig.add_axes([0.10, 0.33, 0.80, 0.59], polar=True)
    _setup_polar(ax)
    _draw_models(ax, legend=True)
    _draw_spoke_labels(ax, r=1.22, fs=8.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.30),
              ncol=n_models, fontsize=8.5, frameon=True,
              fancybox=False, edgecolor="#cccccc",
              handlelength=1.6, columnspacing=1.0)
    _draw_table(fig.add_axes([0.02, 0.02, 0.96, 0.27]))

    plt.rcParams.update({"font.family": "sans-serif"})
    base = plots_dir / "radar"
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {base}.pdf / .png")

    # ════════════════════════════════════════════════════════════════════════════
    # Version 2: standalone radar (compact, no table — for column inset)
    # ════════════════════════════════════════════════════════════════════════════
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig2 = plt.figure(figsize=(3.8, 4.2))
    ax2  = fig2.add_axes([0.06, 0.13, 0.88, 0.80], polar=True)
    _setup_polar(ax2)
    _draw_models(ax2, legend=True)
    _draw_spoke_labels(ax2, r=1.25, fs=7.5)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08),
               ncol=n_models, fontsize=7.5, frameon=True,
               fancybox=False, edgecolor="#cccccc",
               handlelength=1.4, columnspacing=0.9)

    plt.rcParams.update({"font.family": "sans-serif"})
    base2 = plots_dir / "radar_standalone"
    fig2.savefig(str(base2) + ".pdf", bbox_inches="tight")
    fig2.savefig(str(base2) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig2)
    log.info(f"Saved → {base2}.pdf / .png")


# ── Rankings ──────────────────────────────────────────────────────────────────

def build_rankings(macro_summary: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Rank models on every metric (1=best). Saves rankings.csv to output_dir."""
    df = macro_summary.set_index("model")
    metric_cols = list(df.columns)
    hnrr_cols = [c for c in metric_cols if "HNRR" in c]

    rank_df = pd.DataFrame(index=df.index)
    for col in metric_cols:
        ascending = col in hnrr_cols  # HNRR: lower is better
        rank_df[col] = df[col].rank(method="min", ascending=ascending).astype(int)

    pr_cols = [c for c in metric_cols if "HNRR" not in c]
    rank_df["avg_rank_PR"] = rank_df[pr_cols].mean(axis=1)
    rank_df["overall_rank"] = rank_df["avg_rank_PR"].rank(method="min").astype(int)

    path = output_dir / "rankings.csv"
    rank_df.to_csv(path)
    log.info(f"Saved → {path}")
    return rank_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run all model evaluations and produce summary CSVs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--paired_dir", default=None,
                        help="Directory with paired image/text data (required unless --plots_only)")
    parser.add_argument("--csv", default="cxr_data/mimic_cxr_official_test.csv",)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of dataset items to evaluate(default: all, which is 5159 for cxr dataset)")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to store all results CSVs and plots for this experiment")
    parser.add_argument("--query_mode", default="all",
                        choices=["single", "pair", "negative", "all"],
                        help="Which query types to evaluate (passed to eval_model.py). Default: all")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip models whose results CSV already exists")
    parser.add_argument("--plots_only", action="store_true",
                        help="Skip evaluations; load existing result CSVs and regenerate all plots")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for image and text encoding (default: 128)")
    parser.add_argument("--wandb-project", default=None,
                        help="W&B project name. Omit to disable W&B logging.")
    parser.add_argument("--wandb-run-name", default=None,
                        help="W&B run name (auto-generated if omitted).")
    parser.add_argument("--wandb-entity", default=None,
                        help="W&B entity (team/user). Uses your default entity if omitted.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.plots_only and not args.paired_dir:
        parser.error("--paired_dir is required unless --plots_only is set")

    # ── plots_only: load existing CSVs, skip evaluations ─────────────────────
    if args.plots_only:
        log.info("--plots_only: loading existing result CSVs from %s", output_dir)
        all_results = {}
        for model in MODELS:
            out_path        = output_dir / results_path(model).name
            out_path_rob    = output_dir / results_robust_path(model).name
            out_path_str    = output_dir / results_strict_path(model).name
            out_path_rob_str = output_dir / results_robust_strict_path(model).name
            if not out_path.exists():
                log.info("  no CSV for %s, skipping", model["name"])
                continue
            df_main = pd.read_csv(out_path)
            frames  = [df_main]
            if out_path_rob.exists():
                df_rob = pd.read_csv(out_path_rob)
                df_rob.loc[df_rob["type"] == "negative", "type"] = "negative_robust"
                frames.append(df_rob)
            if out_path_str.exists():
                df_str = pd.read_csv(out_path_str)
                df_str.loc[df_str["type"] == "negative", "type"] = "negative_strict"
                frames.append(df_str)
            if out_path_rob_str.exists():
                df_rob_str = pd.read_csv(out_path_rob_str)
                df_rob_str.loc[df_rob_str["type"] == "negative", "type"] = "negative_robust_strict"
                frames.append(df_rob_str)
            all_results[model["name"]] = pd.concat(frames, ignore_index=True)
            log.info("  loaded %s", model["name"])
        # also pick up any radar-specific CSVs not in the active MODELS list
        RADAR_MODELS_STEMS = {
            "vanilla_clip_not_trained": "results_vanilla_clip_not_trained",
            "cxrclip_swint_m":          "results_cxrclip_swint_m",
            "labeldot_hnm_swint_hnm03": "results_labeldot_hnm_swint_hnm03",
        }
        for mname, stem in RADAR_MODELS_STEMS.items():
            if mname in all_results:
                continue
            p = output_dir / f"{stem}.csv"
            if not p.exists():
                continue
            dfs = [pd.read_csv(p)]
            for suffix, new_type in [
                ("_robust",       "negative_robust"),
                ("_strict",       "negative_strict"),
                ("_robust_strict","negative_robust_strict"),
            ]:
                ps = output_dir / f"{stem}{suffix}.csv"
                if ps.exists():
                    d = pd.read_csv(ps)
                    d.loc[d["type"] == "negative", "type"] = new_type
                    dfs.append(d)
            all_results[mname] = pd.concat(dfs, ignore_index=True)
            log.info("  loaded radar model %s", mname)
        if not all_results:
            log.error("No result CSVs found in %s", output_dir)
            return
        macro_summary = build_macro_summary(all_results)
        plots_dir = output_dir / "plots"
        make_plots(all_results, macro_summary, plots_dir)
        make_table_plots(macro_summary, plots_dir)
        make_parallel_coordinates_plot(macro_summary, plots_dir)
        make_parallel_coordinates_plot_vertical(macro_summary, plots_dir)
        # radar from the 3 paper models specifically
        radar_results = {m: df for m, df in all_results.items()
                         if m in RADAR_MODELS_STEMS}
        make_radar_plot(
            build_macro_summary(radar_results) if len(radar_results) >= 2 else macro_summary,
            plots_dir,
        )
        build_rankings(macro_summary, output_dir)
        log.info("plots_only done.")
        return

    # ── Run evaluations ───────────────────────────────────────────────────────
    # Four passes per model:
    #   pass 1: single + pair + negative  (standard template, lenient: NaN=absent)
    #   pass 2: negative only             (robust/rephrased template, lenient)
    #   pass 3: negative only             (standard template, strict: only CSV 0 counts)
    #   pass 4: negative only             (robust/rephrased template, strict)
    all_results = {}
    for i, model in enumerate(MODELS):
        print("\n" + "=" * 80)
        log.info(f"Evaluating model {i+1}/{len(MODELS)}: {model['name']}")

        out_path        = output_dir / results_path(model).name
        out_path_rob    = output_dir / results_robust_path(model).name
        out_path_str    = output_dir / results_strict_path(model).name
        out_path_rob_str = output_dir / results_robust_strict_path(model).name

        # ── guard: skip missing checkpoints ───────────────────────────────────
        checkpoint_file = CHECKPOINTS_DIR / model["checkpoint"] if model["checkpoint"] else None
        if model["checkpoint"] and not checkpoint_file.exists():
            log.warning(f"Checkpoint not found, skipping {model['name']}: {checkpoint_file}")
            continue
        if model["model_type"] == "finetuned":
            ft_ckpt = REPO_ROOT / model["finetuned_checkpoint"]
            if not ft_ckpt.exists():
                log.warning(f"Finetuned checkpoint not found, skipping {model['name']}: {ft_ckpt}")
                continue
        if model["model_type"] == "cxrclip_finetune":
            ft_ckpt = REPO_ROOT / model["cxrclip_finetune_merged_checkpoint"]
            if not ft_ckpt.exists():
                log.warning(f"CXRClip finetune checkpoint not found, skipping {model['name']}: {ft_ckpt}")
                continue
        if model["model_type"] == "cxrclip_hybrid":
            merged_ckpt = REPO_ROOT / model["hybrid_merged_checkpoint"]
            if not merged_ckpt.exists():
                log.warning(f"Hybrid checkpoint not found, skipping {model['name']}: {merged_ckpt}")
                continue

        from baseline_eval.eval_model import NEG_TEMPLATE_ROBUST

        # ── pass 1: all queries, standard template, lenient ───────────────────
        if args.skip_existing and out_path.exists():
            log.info(f"  [pass 1] Skipping — {out_path.name} already exists")
        else:
            try:
                raw_path = run_eval(
                    model, args.paired_dir, args.csv,
                    query_mode=args.query_mode,
                    batch_size=args.batch_size,
                    max_samples=args.max_samples,
                )
                if raw_path.exists() and raw_path != out_path:
                    shutil.move(str(raw_path), out_path)
            except Exception as e:
                log.error(f"  [pass 1] Failed for {model['name']}: {e}")
                continue

        # ── pass 2: negative queries, rephrased template, lenient ─────────────
        if args.query_mode not in ("all", "negative"):
            log.info(f"  [pass 2] Skipping robust (query_mode={args.query_mode})")
        elif args.skip_existing and out_path_rob.exists():
            log.info(f"  [pass 2] Skipping — {out_path_rob.name} already exists")
        else:
            try:
                raw_rob = run_eval(
                    model, args.paired_dir, args.csv,
                    query_mode="negative",
                    batch_size=args.batch_size,
                    max_samples=args.max_samples,
                    neg_template=NEG_TEMPLATE_ROBUST,
                )
                if raw_rob.exists() and raw_rob != out_path_rob:
                    shutil.move(str(raw_rob), out_path_rob)
            except Exception as e:
                log.error(f"  [pass 2] Failed for {model['name']}: {e}")

        # ── pass 3: negative queries, standard template, strict ───────────────
        if args.query_mode not in ("all", "negative"):
            log.info(f"  [pass 3] Skipping strict (query_mode={args.query_mode})")
        elif args.skip_existing and out_path_str.exists():
            log.info(f"  [pass 3] Skipping — {out_path_str.name} already exists")
        else:
            try:
                raw_str = run_eval(
                    model, args.paired_dir, args.csv,
                    query_mode="negative",
                    batch_size=args.batch_size,
                    max_samples=args.max_samples,
                    strict=True,
                )
                if raw_str.exists() and raw_str != out_path_str:
                    shutil.move(str(raw_str), out_path_str)
            except Exception as e:
                log.error(f"  [pass 3] Failed for {model['name']}: {e}")

        # ── pass 4: negative queries, rephrased template, strict ──────────────
        if args.query_mode not in ("all", "negative"):
            log.info(f"  [pass 4] Skipping robust+strict (query_mode={args.query_mode})")
        elif args.skip_existing and out_path_rob_str.exists():
            log.info(f"  [pass 4] Skipping — {out_path_rob_str.name} already exists")
        else:
            try:
                raw_rob_str = run_eval(
                    model, args.paired_dir, args.csv,
                    query_mode="negative",
                    batch_size=args.batch_size,
                    max_samples=args.max_samples,
                    neg_template=NEG_TEMPLATE_ROBUST,
                    strict=True,
                )
                if raw_rob_str.exists() and raw_rob_str != out_path_rob_str:
                    shutil.move(str(raw_rob_str), out_path_rob_str)
            except Exception as e:
                log.error(f"  [pass 4] Failed for {model['name']}: {e}")

        # ── merge results ──────────────────────────────────────────────────────
        if not out_path.exists():
            log.warning(f"No results file for {model['name']}, skipping from summary")
            continue

        df_main = pd.read_csv(out_path)
        frames = [df_main]
        if out_path_rob.exists():
            df_rob = pd.read_csv(out_path_rob)
            df_rob.loc[df_rob["type"] == "negative", "type"] = "negative_robust"
            frames.append(df_rob)
        if out_path_str.exists():
            df_str = pd.read_csv(out_path_str)
            df_str.loc[df_str["type"] == "negative", "type"] = "negative_strict"
            frames.append(df_str)
        if out_path_rob_str.exists():
            df_rob_str = pd.read_csv(out_path_rob_str)
            df_rob_str.loc[df_rob_str["type"] == "negative", "type"] = "negative_robust_strict"
            frames.append(df_rob_str)

        all_results[model["name"]] = pd.concat(frames, ignore_index=True)

    if not all_results:
        log.error("No results to summarize.")
        return

    macro_summary = build_macro_summary(all_results)

    # ── Build and save per-type summaries ────────────────────────────────────
    qtypes_present = {qt for df in all_results.values() for qt in df["type"].unique()}

    ALL_QTYPES = ["single", "pair", "negative", "negative_strict", "negative_robust", "negative_robust_strict"]

    summary_paths = {}
    for qtype in ALL_QTYPES:
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
    make_parallel_coordinates_plot(macro_summary, plots_dir)
    make_parallel_coordinates_plot_vertical(macro_summary, plots_dir)

    # Radar always uses these 3 specific models; load from individual CSVs
    # so it works even when the MODELS list is different from a prior run.
    RADAR_MODELS = {
        "vanilla_clip_not_trained": "results_vanilla_clip_not_trained",
        "cxrclip_swint_m":          "results_cxrclip_swint_m",
        "labeldot_hnm_swint_hnm03": "results_labeldot_hnm_swint_hnm03",
    }
    radar_results = {}
    for mname, stem in RADAR_MODELS.items():
        dfs = []
        for suffix, new_type in [
            ("",              None),
            ("_robust",       "negative_robust"),
            ("_strict",       "negative_strict"),
            ("_robust_strict","negative_robust_strict"),
        ]:
            p = output_dir / f"{stem}{suffix}.csv"
            if p.exists():
                df = pd.read_csv(p)
                if new_type:
                    df.loc[df["type"] == "negative", "type"] = new_type
                dfs.append(df)
        if dfs:
            radar_results[mname] = pd.concat(dfs, ignore_index=True)
    if len(radar_results) >= 2:
        radar_macro = build_macro_summary(radar_results)
        make_radar_plot(radar_macro, plots_dir)
    else:
        make_radar_plot(macro_summary, plots_dir)

    # ── Rankings ──────────────────────────────────────────────────────────────
    build_rankings(macro_summary, output_dir)

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
    for qtype in ALL_QTYPES:
        extra = HNRR_COLS if qtype.startswith("negative") else []
        cols = [f"{qtype}_{c}" for c in METRIC_COLS + extra]
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
