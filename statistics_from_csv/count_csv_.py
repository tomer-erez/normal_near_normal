import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from constants import LABEL_COLS

print("Loading CSV...")
df = pd.read_csv(
    r"/home/tomererez/normal_near_normal/cxr_data/all_txt_data_and_labels.csv"
)
print(f"dataset shape: {df.shape}")

chexpert_cols = LABEL_COLS

summary = []

print("Counting distributions for CheXpert labels:")

for col in chexpert_cols:
    # Raw counts including NaN
    counts = df[col].value_counts(dropna=False)

    total = len(df)

    num_pos = counts.get(1, 0)
    num_neg = counts.get(0, 0)
    num_uncertain = counts.get(-1, 0)
    num_na = df[col].isna().sum()

    # Normalized
    norm_pos = num_pos / total
    norm_neg = num_neg / total
    norm_uncertain = num_uncertain / total
    norm_na = num_na / total

    pathology = col.split("chexpert_")[1]

    print(f"\n{pathology}:")
    print(f"Total samples: {total:,}")
    print(f"1   -> count={num_pos:6d}, ratio={norm_pos:.3f}")
    print(f"0   -> count={num_neg:6d}, ratio={norm_neg:.3f}")
    print(f"-1  -> count={num_uncertain:6d}, ratio={norm_uncertain:.3f}")
    print(f"NaN -> count={num_na:6d}, ratio={norm_na:.3f}")

    summary.append({
        "column": pathology,
        "1": num_pos,
        "0": num_neg,
        "-1": num_uncertain,
        "NaN": num_na,
        "1_ratio": round(norm_pos, 3),
        "0_ratio": round(norm_neg, 3),
        "-1_ratio": round(norm_uncertain, 3),
        "NaN_ratio": round(norm_na, 3)
    })

# Optional summary dataframe
summary_df = pd.DataFrame(summary)

# Make pandas display floats with 3 decimals everywhere
pd.options.display.float_format = "{:.3f}".format

print("\nSummary table:")
print(summary_df)

num_cols = summary_df.select_dtypes(include='number').columns
totals = summary_df[num_cols].sum()
sum_row   = pd.Series({'column': 'Total',       **totals.to_dict()})
ratio_row = pd.Series({'column': 'Total_ratio', **(totals / len(df)).to_dict()})

df_with_total = pd.concat(
    [
        summary_df,
        sum_row.to_frame().T,
        ratio_row.to_frame().T
    ],
    ignore_index=True
)

print(df_with_total)


# show how many rows have k positive labels for k=0,1,...,13
print("\nDistribution of number of positive labels per sample:")
num_pos_labels = (df[chexpert_cols] == 1).sum(axis=1)
pos_label_counts = num_pos_labels.value_counts().sort_index()
for k, count in pos_label_counts.items():
    print(f"{k} positive labels: {count:,} samples ({count/len(df):.3%})")
    
# now for negative labels
print("\nDistribution of number of negative labels per sample:")
num_neg_labels = (df[chexpert_cols] == 0).sum(axis=1)
neg_label_counts = num_neg_labels.value_counts().sort_index()
for k, count in neg_label_counts.items():
    print(f"{k} negative labels: {count:,} samples ({count/len(df):.3%})")

# now for uncertain labels
print("\nDistribution of number of uncertain labels per sample:")
num_uncertain_labels = (df[chexpert_cols] == -1).sum(axis=1)
uncertain_label_counts = num_uncertain_labels.value_counts().sort_index()
for k, count in uncertain_label_counts.items():
    print(f"{k} uncertain labels: {count:,} samples ({count/len(df):.3%})")

#now for NaN labels
print("\nDistribution of number of NaN labels per sample:")
num_nan_labels = df[chexpert_cols].isna().sum(axis=1)
nan_label_counts = num_nan_labels.value_counts().sort_index()
for k, count in nan_label_counts.items():
    print(f"{k} NaN labels: {count:,} samples ({count/len(df):.3%})")

# Overall label value distribution across all labels and all samples
print("\nOverall label distribution (across all labels × all samples):")
total_cells = len(df) * len(chexpert_cols)
n1   = (df[chexpert_cols] == 1).sum().sum()
n0   = (df[chexpert_cols] == 0).sum().sum()
nm1  = (df[chexpert_cols] == -1).sum().sum()
nnan = df[chexpert_cols].isna().sum().sum()
for label, count in [("1  (positive)", n1), ("0  (negative)", n0), ("-1 (uncertain)", nm1), ("NaN (missing)", nnan)]:
    print(f"  {label}: {count:7,}  ({count/total_cells:.3%})")

# Count breakdown table: exactly 0, exactly 1, more than 1 per row
print("\nPer-row count breakdown:")
counts = {
    "1 (pos)":  (df[chexpert_cols] == 1).sum(axis=1),
    "0 (neg)":  (df[chexpert_cols] == 0).sum(axis=1),
    "-1 (unc)": (df[chexpert_cols] == -1).sum(axis=1),
    "NaN":      df[chexpert_cols].isna().sum(axis=1),
}
n = len(df)
rows = []
for label, s in counts.items():
    rows.append({
        "value": label,
        "=0":    f"{(s==0).mean():.1%}",
        "=1":    f"{(s==1).mean():.1%}",
        ">1":    f"{(s >1).mean():.1%}",
        ">4":    f"{(s >4).mean():.1%}",

    })
print(pd.DataFrame(rows).to_string(index=False))


# how many items have at least 1 positive and at least 1 negative label?
at_least_1_pos = (df[chexpert_cols] == 1).any(axis=1)
at_least_1_neg = (df[chexpert_cols] == 0).any(axis=1)
both_pos_neg = (at_least_1_pos & at_least_1_neg).sum()
print(f"\nSamples with at least 1 positive and at least 1 negative label: {both_pos_neg:,} ({both_pos_neg/len(df):.3%})")

# how many items have at least 1 positive and at least 1 uncertain label?
at_least_1_unc = (df[chexpert_cols] == -1).any(axis=1)
both_pos_unc = (at_least_1_pos & at_least_1_unc).sum()
print(f"Samples with at least 1 positive and at least 1 uncertain label: {both_pos_unc:,} ({both_pos_unc/len(df):.3%})")    

# how many items have at least 1 positive and at least 1 missing?
at_least_1_nan = df[chexpert_cols].isna().any(axis=1)
both_pos_nan = (at_least_1_pos & at_least_1_nan).sum()
print(f"Samples with at least 1 positive and at least 1 NaN label: {both_pos_nan:,} ({both_pos_nan/len(df):.3%})")

# ── Co-occurrence matrix ───────────────────────────────────────────────────────
print("\nBuilding co-occurrence matrix...")

label_names = [c.replace("chexpert_", "") for c in chexpert_cols]

pos_df = (df[chexpert_cols] == 1)                              # (N, L) bool
co_occ = (pos_df.values.astype(np.int32).T
          @ pos_df.values.astype(np.int32))                    # (L, L) int
pos_counts = pos_df.sum().values                               # (L,)

# pct[i, j] = 100 * P(j=1 | i=1)  — diagonal is always 100
with np.errstate(divide="ignore", invalid="ignore"):
    pct = np.where(
        pos_counts[:, None] > 0,
        100.0 * co_occ / pos_counts[:, None],
        0.0,
    )
pct_int = np.round(pct).astype(int)

# ── Figure ────────────────────────────────────────────────────────────────────
L = len(label_names)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})

fig, ax = plt.subplots(figsize=(7, 5.8))

# Main heatmap (off-diagonal)
diag_mask = np.eye(L, dtype=bool)
sns.heatmap(
    np.where(diag_mask, np.nan, pct_int.astype(float)),
    annot=pct_int,
    fmt="d",
    cmap="Blues",
    vmin=0,
    vmax=100,
    linewidths=0.3,
    linecolor="#cccccc",
    ax=ax,
    square=True,
    cbar_kws={"label": r"$P(\mathrm{col}=1 \mid \mathrm{row}=1)$ (%)", "shrink": 0.8, "pad": 0.02},
    annot_kws={"size": 8.5, "color": "#1a1a1a"},
    xticklabels=label_names,
    yticklabels=label_names,
)

# Diagonal cells — dark slate so they read as "always 100"
for i in range(L):
    ax.add_patch(mpatches.Rectangle(
        (i, i), 1, 1,
        fill=True, color="#2c3e50", zorder=3, lw=0,
    ))
    ax.text(
        i + 0.5, i + 0.5, "100",
        ha="center", va="center",
        fontsize=8.5, fontweight="bold", color="white", zorder=4,
    )

ax.set_xticklabels(label_names, rotation=40, ha="right", fontsize=9)
ax.set_yticklabels(label_names, rotation=0, fontsize=9)
ax.set_xlabel(None)
ax.set_ylabel(None)
ax.set_title("CheXpert label co-occurrence", fontsize=11, pad=10)

plt.tight_layout()
base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cooccurrence_matrix")
fig.savefig(base_path + ".pdf", bbox_inches="tight")
fig.savefig(base_path + ".png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {base_path}.pdf / .png")

# ── Pair-query difficulty figure ──────────────────────────────────────────────
print("\nBuilding pair-query difficulty figure...")

TEST_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cxr_data", "mimic_cxr_official_test.csv")
df_test = pd.read_csv(TEST_CSV)

pairs = []
for i in range(len(chexpert_cols)):
    for j in range(i + 1, len(chexpert_cols)):
        ci, cj = chexpert_cols[i], chexpert_cols[j]
        count = int(((df_test[ci] == 1) & (df_test[cj] == 1)).sum())
        li = label_names[i]
        lj = label_names[j]
        pairs.append((f"{li} & {lj}", count))

pairs.sort(key=lambda x: x[1])
pair_labels = [p[0] for p in pairs]
pair_counts = [p[1] for p in pairs]

TIER_RED    = "#e74c3c"
TIER_ORANGE = "#e67e22"
TIER_BLUE   = "#2980b9"

def _tier_color(c):
    if c <= 10:
        return TIER_RED
    if c < 50:
        return TIER_ORANGE
    return TIER_BLUE

colors = [_tier_color(c) for c in pair_counts]

plt.rcParams.update({
    "font.family": "serif",
    "font.size":   9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})

fig, ax = plt.subplots(figsize=(7, 9))

y_pos = np.arange(len(pair_labels))
bars = ax.barh(y_pos, pair_counts, color=colors, height=0.7, edgecolor="none")

ax.set_yticks(y_pos)
ax.set_yticklabels(pair_labels, fontsize=8)
ax.set_xlabel("Number of relevant images in test gallery", fontsize=9)
ax.set_title("Pair query difficulty — co-occurrence counts\n(test gallery: 5,159 images)", fontsize=10, pad=8)
ax.axvline(x=50, color="#555555", linewidth=1.0, linestyle="--")
ax.text(52, len(pair_labels) - 0.5, "50 relevant", va="top", fontsize=8, color="#555555")
ax.set_xlim(0, max(pair_counts) * 1.12)
ax.xaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

import matplotlib.patches as mpatches
legend_patches = [
    mpatches.Patch(color=TIER_RED,    label=r"$\leq$10 relevant"),
    mpatches.Patch(color=TIER_ORANGE, label="11–49 relevant"),
    mpatches.Patch(color=TIER_BLUE,   label=r"$\geq$50 relevant"),
]
ax.legend(handles=legend_patches, fontsize=8, loc="lower right")

plt.tight_layout()
pair_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pair_query_difficulty")
fig.savefig(pair_path + ".pdf", bbox_inches="tight")
fig.savefig(pair_path + ".png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {pair_path}.pdf / .png")
