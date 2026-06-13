import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
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
        "=2":    f"{(s==2).mean():.1%}",

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

# how many have at least two positive labels?
at_least_2_pos = (df[chexpert_cols] == 1).sum(axis=1) >= 2
print(f"Samples with at least 2 positive labels: {at_least_2_pos.sum():,} ({at_least_2_pos.mean():.3%})")

# at least 1 positive label?
at_least_1_pos = (df[chexpert_cols] == 1).any(axis=1)
print(f"Samples with at least 1 positive label: {at_least_1_pos.sum():,} ({at_least_1_pos.mean():.3%})")    

#avg number of ones for a label in the dataset
avg_num_ones_per_label = (df[chexpert_cols] == 1).mean().mean()
print(f"Average number of positive labels per sample: {avg_num_ones_per_label:.3f} ({avg_num_ones_per_label*100:.1f}%)")

# Summary table: rows with 0 / 1 / >1 occurrences of each label value
print("\nPer-row count breakdown (counts and % of rows):")
value_defs = [
    ("1  (pos)", (df[chexpert_cols] == 1).sum(axis=1)),
    ("0  (neg)", (df[chexpert_cols] == 0).sum(axis=1)),
    ("-1 (unc)", (df[chexpert_cols] == -1).sum(axis=1)),
    ("NaN     ", df[chexpert_cols].isna().sum(axis=1)),
]
n = len(df)
rows = []
for label, s in value_defs:
    rows.append({
        "value":   label,
        "0 rows":  f"{(s == 0).sum():,} ({(s == 0).mean():.1%})",
        "1 rows":  f"{(s == 1).sum():,} ({(s == 1).mean():.1%})",
        ">1 rows": f"{(s > 1).sum():,} ({(s > 1).mean():.1%})",
    })
print(pd.DataFrame(rows).to_string(index=False))
