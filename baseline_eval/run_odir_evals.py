"""
Run retrieval evaluation for all ODIR models sequentially, then summarize
results into comparison tables and plots.

This is a leaner ODIR counterpart to run_all_evals.py. ODIR's N/D/G/C/A/H/M/O
labels are a clean 0/1 encoding with no NaN ("not mentioned") or -1
("uncertain") state, so — unlike MIMIC-CXR — there is no lenient/strict
negation ambiguity to evaluate separately: a single pass covers single, pair,
and negative queries. We additionally run a rephrased-template pass to check
robustness to query phrasing, matching the CXR pipeline's intent.

Query types evaluated
  single    8 queries  "cataract"                  relevant: label==1
  pair      28 queries "cataract and glaucoma"      relevant: both labels==1
  negative  56 queries "cataract and no glaucoma"   relevant: pos==1 AND neg==0
  negative_robust  56 queries — rephrased template, same relevance rule

Output files (in --output_dir):
  summary_single.csv / summary_pair.csv / summary_negative.csv / summary_negative_robust.csv
  summary_macro.csv          macro-averaged metrics per model (one row per model)
  plots/                     heatmap, macro bar charts, HNRR bar chart

Usage
-----
    python baseline_eval/run_odir_evals.py \
        --paired_dir ./eval_outputs/odir_exp1/paired_data \
        --csv        odir_data/odir_test.csv \
        --output_dir ./experiments/odir_exp1
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Fine-tuned ODIR models — add entries here after training with
# `train_lora.py --dataset odir`. Mirrors run_all_evals.py's MODELS convention.
MODELS = [
    # {
    #     "name": "lora_vitb32_odir",
    #     "model_type": "finetuned",
    #     "finetuned_base_model": "ViT-B-32",
    #     "finetuned_pretrained": "openai",
    #     "finetuned_checkpoint": "experiments/lora_vitb32_odir/final_merged.pt",
    # },
]

KS = [1, 3, 5]
METRIC_COLS = [f"P@{k}" for k in KS] + [f"R@{k}" for k in KS] + [f"AP@{k}" for k in KS]
HNRR_COLS = [f"HNRR@{k}" for k in KS]
NEG_TEMPLATE_ROBUST = "an image with {pos} but without {neg}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def results_path(model: dict, suffix: str = "") -> Path:
    return REPO_ROOT / f"results_{model['name']}{suffix}.csv"


def run_eval(model: dict, paired_dir: str, csv: str, batch_size: int,
             max_samples: int | None, robust: bool) -> Path:
    """Run eval_model.py --dataset odir for one model. Returns path to results CSV."""
    name = model["name"] + ("_robust" if robust else "")
    cmd = [
        sys.executable, str(REPO_ROOT / "baseline_eval" / "eval_model.py"),
        "--model_type", model["model_type"],
        "--dataset", "odir",
        "--name", name,
        "--paired_dir", paired_dir,
        "--csv", csv,
        "--query_mode", "all",
        "--batch_size", str(batch_size),
        "--ks", *[str(k) for k in KS],
    ]
    if robust:
        cmd += ["--neg-template", NEG_TEMPLATE_ROBUST]
    if max_samples is not None:
        cmd += ["--max_samples", str(max_samples)]
    if model["model_type"] == "finetuned":
        cmd += [
            "--finetuned_base_model", model["finetuned_base_model"],
            "--finetuned_pretrained", model.get("finetuned_pretrained", ""),
            "--finetuned_checkpoint", str(REPO_ROOT / model["finetuned_checkpoint"]),
        ]
    elif model["model_type"] == "cxrclip_finetune":
        cmd += [
            "--cxrclip_finetune_image_checkpoint", str(REPO_ROOT / model["cxrclip_finetune_image_checkpoint"]),
            "--cxrclip_finetune_merged_checkpoint", str(REPO_ROOT / model["cxrclip_finetune_merged_checkpoint"]),
        ]
    log.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return results_path(model, "_robust" if robust else "")


def build_summary(all_results: dict[str, pd.DataFrame], qtype: str) -> pd.DataFrame:
    model_names = list(all_results.keys())
    first_df = next(df for df in all_results.values() if qtype in df["type"].values)
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
    rows = []
    for model_name, df in all_results.items():
        row = {"model": model_name}
        for qtype in ["single", "pair", "negative", "negative_robust"]:
            if qtype not in df["type"].values:
                continue
            subset = df[(df["type"] == qtype) & (df["n_relevant"] > 0)]
            for col in METRIC_COLS:
                if col in subset.columns:
                    row[f"{qtype}_{col}"] = subset[col].mean()
            if qtype.startswith("negative"):
                neg_subset = df[df["type"] == qtype]
                for col in HNRR_COLS:
                    if col in neg_subset.columns:
                        row[f"{qtype}_{col}"] = neg_subset[col].mean()
        rows.append(row)
    return pd.DataFrame(rows)


# ── Plots (a lean subset of run_all_evals.py's — no WACV-paper-specific ─────────
# radar/parallel-coordinates/table plots, which hardcode CXR model names) ───────

def make_plots(all_results: dict, macro_summary: pd.DataFrame, plots_dir: Path):
    plots_dir.mkdir(parents=True, exist_ok=True)
    model_names = list(all_results.keys())
    qtypes_present = {qt for df in all_results.values() for qt in df["type"].unique()}
    K_LAST = KS[-1]

    # 1. Heatmap: P@Klast per label, per model (single queries)
    if "single" in qtypes_present:
        label_names = all_results[model_names[0]]
        label_names = label_names[label_names["type"] == "single"]["query"].tolist()
        metric = f"P@{K_LAST}"
        matrix = np.array([
            all_results[m][all_results[m]["type"] == "single"][metric].values
            for m in model_names
        ])
        fig, ax = plt.subplots(figsize=(max(10, len(label_names) * 0.9), max(4, len(model_names) * 0.8)))
        sns.heatmap(matrix, xticklabels=label_names, yticklabels=model_names,
                    annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1,
                    linewidths=0.5, ax=ax)
        ax.set_title(f"{metric} per label — single label queries", fontsize=13, pad=12)
        ax.set_xlabel("ODIR label")
        ax.set_ylabel("Model")
        plt.xticks(rotation=35, ha="right", fontsize=9)
        plt.tight_layout()
        path = plots_dir / f"heatmap_single_{metric.replace('@', 'at')}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        log.info(f"Saved → {path}")

    # 2. Macro bar chart: P@k / R@k per query type
    for qtype in ["single", "pair", "negative", "negative_robust"]:
        if qtype not in qtypes_present:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Macro-averaged metrics — {qtype} queries", fontsize=13)
        x = np.arange(len(model_names))
        width = 0.25
        for ax, prefix, metrics in [(axes[0], "P", [f"P@{k}" for k in KS]), (axes[1], "R", [f"R@{k}" for k in KS])]:
            for i, metric in enumerate(metrics):
                col = f"{qtype}_{metric}"
                vals = [
                    macro_summary.loc[macro_summary["model"] == m, col].values[0]
                    if col in macro_summary.columns else float("nan")
                    for m in model_names
                ]
                ax.bar(x + i * width, vals, width, label=metric)
            ax.set_xticks(x + width)
            ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=9)
            ax.set_ylabel(f"{'Precision' if prefix == 'P' else 'Recall'} @ k")
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        path = plots_dir / f"macro_bar_{qtype}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        log.info(f"Saved → {path}")

    # 3. HNRR bar chart for negative queries
    for qtype in ["negative", "negative_robust"]:
        cols = [f"{qtype}_{c}" for c in HNRR_COLS]
        available = [c for c in cols if c in macro_summary.columns]
        if not available:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle(f"HNRR — hard negatives in top-k — {qtype} queries", fontsize=13)
        x = np.arange(len(model_names))
        width = 0.25
        for i, col in enumerate(available):
            metric = col.replace(f"{qtype}_", "")
            vals = [
                macro_summary.loc[macro_summary["model"] == m, col].values[0]
                if col in macro_summary.columns else float("nan")
                for m in model_names
            ]
            ax.bar(x + i * width, vals, width, label=metric)
        ax.set_xticks(x + width)
        ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("HNRR @ k  (lower is better)")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        path = plots_dir / f"hnrr_bar_{qtype}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        log.info(f"Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run all ODIR model evaluations and summarize")
    parser.add_argument("--paired_dir", required=True, help="Gallery folder from build_odir_gallery.py")
    parser.add_argument("--csv", required=True, help="Path to odir_data/odir_test.csv")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    if not MODELS:
        log.error("MODELS list is empty — add entries at the top of run_odir_evals.py first.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, pd.DataFrame] = {}
    all_results_robust: dict[str, pd.DataFrame] = {}
    for model in MODELS:
        name = model["name"]
        rpath = results_path(model)
        if args.skip_existing and rpath.exists():
            log.info(f"Skipping {name} (results already exist at {rpath})")
        else:
            rpath = run_eval(model, args.paired_dir, args.csv, args.batch_size, args.max_samples, robust=False)
        all_results[name] = pd.read_csv(rpath)

        rpath_robust = results_path(model, "_robust")
        if args.skip_existing and rpath_robust.exists():
            log.info(f"Skipping {name} robust pass (results already exist at {rpath_robust})")
        else:
            rpath_robust = run_eval(model, args.paired_dir, args.csv, args.batch_size, args.max_samples, robust=True)
        neg_only = pd.read_csv(rpath_robust)
        neg_only["type"] = "negative_robust"
        all_results_robust[name] = neg_only

    merged_results = {
        name: pd.concat([all_results[name], all_results_robust[name]], ignore_index=True)
        for name in all_results
    }

    for qtype in ["single", "pair", "negative", "negative_robust"]:
        summary = build_summary(merged_results, qtype)
        out = output_dir / f"summary_{qtype}.csv"
        summary.to_csv(out, index=False)
        log.info(f"Saved → {out}")

    macro_summary = build_macro_summary(merged_results)
    macro_path = output_dir / "summary_macro.csv"
    macro_summary.to_csv(macro_path, index=False)
    log.info(f"Saved → {macro_path}")
    print("\n" + macro_summary.to_string(index=False))

    make_plots(merged_results, macro_summary, output_dir / "plots")


if __name__ == "__main__":
    main()
