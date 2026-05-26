import pandas as pd

print("Loading CSV...")
df = pd.read_csv(
    r"/home/tomererez/normal_near_normal/cxr_data/all_txt_data_and_labels.csv"
)

chexpert_cols = [col for col in df.columns if col.startswith("chexpert_")]

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
