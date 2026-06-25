"""
Bootstrap confidence intervals for macro-averaged retrieval metrics.

Resamples the per-query rows of results_*.csv files to estimate 95% CIs
around macro-averaged P@k and R@k for each (model, query_type) combination.

Usage
-----
    python baseline_eval/compute_bootstrap_ci.py \
        --results_dir eval_outputs/find_best_model_hnm_vs_labeldot_m_models \
        --models vanilla_clip biomedclip cxr-clip ours \
        --output bootstrap_ci.csv

    # Pretty-print to stdout (no --output):
    python baseline_eval/compute_bootstrap_ci.py \
        --results_dir eval_outputs/find_best_model_hnm_vs_labeldot_m_models
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["P@1", "P@3", "P@5", "R@3", "R@5"]
QTYPES  = ["single", "pair", "negative"]
QTYPE_DISPLAY = {"single": "Single (N=10)", "pair": "Pair (N=45)", "negative": "Negation (N=90)"}


def bootstrap_ci(values: np.ndarray, B: int = 2000, seed: int = 42) -> tuple[float, float]:
    """Bootstrap 95% CI for the mean of `values` via percentile method."""
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.array([
        values[rng.integers(0, n, n)].mean()
        for _ in range(B)
    ])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_results(results_dir: Path, model_name: str) -> pd.DataFrame | None:
    path = results_dir / f"results_{model_name}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True,
                        help="Directory containing results_*.csv files")
    parser.add_argument("--models", nargs="+",
                        default=["vanilla_clip", "biomedclip", "cxr-clip", "ours"],
                        help="Model names (must match results_{name}.csv filenames)")
    parser.add_argument("--B", type=int, default=2000, help="Bootstrap resamples")
    parser.add_argument("--output", default=None,
                        help="Save CI table to this CSV path (optional)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    rows = []
    for model in args.models:
        df = load_results(results_dir, model)
        if df is None:
            print(f"[WARN] No results file for '{model}' in {results_dir}")
            continue

        for qtype in QTYPES:
            subset = df[(df["type"] == qtype) & (df["n_relevant"] > 0)]
            if subset.empty:
                continue
            n_q = len(subset)
            row = {"model": model, "qtype": qtype, "n_queries": n_q}
            for m in METRICS:
                if m not in subset.columns:
                    continue
                vals = subset[m].values.astype(float)
                mean = vals.mean()
                lo, hi = bootstrap_ci(vals, B=args.B)
                row[f"{m}_mean"] = round(mean * 100, 1)
                row[f"{m}_ci_lo"] = round(lo * 100, 1)
                row[f"{m}_ci_hi"] = round(hi * 100, 1)
                row[f"{m}_ci_str"] = f"{mean*100:.1f} [{lo*100:.1f}, {hi*100:.1f}]"
            rows.append(row)

    if not rows:
        print("No results found.")
        return

    result_df = pd.DataFrame(rows)

    if args.output:
        result_df.to_csv(args.output, index=False)
        print(f"Saved → {args.output}")

    # Pretty-print table grouped by query type
    print()
    for qtype in QTYPES:
        sub = result_df[result_df["qtype"] == qtype]
        if sub.empty:
            continue
        n = int(sub["n_queries"].iloc[0])
        print(f"{'='*70}")
        print(f"  {QTYPE_DISPLAY[qtype]}   — mean [95% CI]  ({args.B} bootstrap resamples)")
        print(f"{'='*70}")
        header = f"{'Model':<20}  {'P@1':>22}  {'P@3':>22}  {'P@5':>22}"
        print(header)
        print("-" * len(header))
        for _, r in sub.iterrows():
            line = f"{r['model']:<20}  {r.get('P@1_ci_str',''):>22}  {r.get('P@3_ci_str',''):>22}  {r.get('P@5_ci_str',''):>22}"
            print(line)
        print()
        header2 = f"{'Model':<20}  {'R@3':>22}  {'R@5':>22}"
        print(header2)
        print("-" * len(header2))
        for _, r in sub.iterrows():
            line = f"{r['model']:<20}  {r.get('R@3_ci_str',''):>22}  {r.get('R@5_ci_str',''):>22}"
            print(line)
        print()

    # Highlight overlap analysis for single-label (the high-variance category)
    single = result_df[result_df["qtype"] == "single"]
    if len(single) >= 2:
        print(f"{'='*70}")
        print("  Single-label P@5 overlap analysis (are CIs non-overlapping?)")
        print(f"{'='*70}")
        for _, r in single.iterrows():
            lo = r.get("P@5_ci_lo", float("nan"))
            hi = r.get("P@5_ci_hi", float("nan"))
            print(f"  {r['model']:<20}  [{lo:.1f}, {hi:.1f}]")
        print()


if __name__ == "__main__":
    main()
