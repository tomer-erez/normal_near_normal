"""
Rank models from a run_all_evals.py output folder.

Reads summary_macro.csv and produces:
  rankings.csv   — rows=models, cols=metrics, values=rank (1=best)

Usage:
    python baseline_eval/rank_models.py --output_dir ./eval_outputs/compare_find_best_model
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Rank models from run_all_evals output")
    parser.add_argument("--output_dir", required=True,
                        help="Folder produced by run_all_evals.py (contains summary_macro.csv)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    macro_path = output_dir / "summary_macro.csv"
    if not macro_path.exists():
        raise FileNotFoundError(f"summary_macro.csv not found in {output_dir}")

    df = pd.read_csv(macro_path).set_index("model")

    metric_cols = list(df.columns)
    hnrr_cols = [c for c in metric_cols if "HNRR" in c]

    rank_df = pd.DataFrame(index=df.index)
    for col in metric_cols:
        ascending = col in hnrr_cols  # HNRR: lower is better
        rank_df[col] = df[col].rank(method="min", ascending=ascending).astype(int)

    out_path = output_dir / "rankings.csv"
    rank_df.to_csv(out_path)
    print(f"Saved → {out_path}\n")

    # ── Print summary ──────────────────────────────────────────────────────────
    qtypes = ["single", "pair", "negative_nan", "negative_strict",
              "negative_robust_nan", "negative_robust_strict"]

    for qtype in qtypes:
        cols = [c for c in metric_cols if c.startswith(qtype + "_")]
        if not cols:
            continue
        print("=" * 80)
        print(f"RANKINGS — {qtype.upper()}")
        print("=" * 80)
        print(rank_df[cols].rename(columns=lambda c: c.replace(qtype + "_", "")).to_string())
        print()

    # ── Overall score: average rank across all P/R metrics (excluding HNRR) ───
    pr_cols = [c for c in metric_cols if "HNRR" not in c]
    rank_df["avg_rank_PR"] = rank_df[pr_cols].mean(axis=1)
    rank_df["overall_rank"] = rank_df["avg_rank_PR"].rank(method="min").astype(int)

    print("=" * 80)
    print("OVERALL RANKING  (avg rank across all P@k / R@k metrics, lower = better)")
    print("=" * 80)
    summary = rank_df[["overall_rank", "avg_rank_PR"]].sort_values("overall_rank")
    summary.columns = ["overall_rank", "avg_PR_rank"]
    print(summary.to_string(float_format=lambda x: f"{x:.2f}"))
    print()

    # ── Best model per metric ──────────────────────────────────────────────────
    print("=" * 80)
    print("BEST MODEL PER METRIC")
    print("=" * 80)
    for col in metric_cols:
        best_model = rank_df[col].idxmin()
        best_val = df.loc[best_model, col]
        direction = "(lower better)" if col in hnrr_cols else ""
        print(f"  {col:<45}  →  {best_model}  ({best_val:.4f}) {direction}")
    print()

    # Re-save with overall columns included
    rank_df.to_csv(out_path)
    print(f"Rankings (with overall) saved → {out_path}")


if __name__ == "__main__":
    main()
