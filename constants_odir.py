"""
Shared label constants for the ODIR-5K (Ocular Disease Intelligent Recognition)
retinal fundus dataset, used across train/odir_label_dataset.py and the ODIR
eval scripts under baseline_eval/.

To exclude a label, comment it out here — the change propagates to every
consumer automatically, mirroring constants.py's CheXpert convention.
"""

ODIR_LABELS = [
    "Normal",
    "Diabetes",
    "Glaucoma",
    "Cataract",
    "Age-related Macular Degeneration",
    "Hypertension",
    "Myopia",
    "Other",
]

# Single-letter one-hot columns in odir/odir/full_df.csv, in the same order as ODIR_LABELS.
ODIR_LABEL_COLS = ["N", "D", "G", "C", "A", "H", "M", "O"]
