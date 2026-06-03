"""
Shared label constants used across training and evaluation.

To exclude a label, comment it out here — the change will propagate to
train/cxr_label_dataset.py, baseline_eval/eval_model.py,
baseline_eval/run_all_evals.py, and baseline_eval/explorer.py automatically.
"""

CHEXPERT_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    # "Enlarged Cardiomediastinum",  # excluded: low CheXpert F1, noisy label
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    # "No Finding", # low precision from chexpert
    "Pleural Effusion",
    # "Pleural Other", # low precision from chexpert
    "Pneumonia",
    "Pneumothorax",
    # "Support Devices",  # excluded: not a pathology, irrelevant for retrieval
]

LABEL_COLS = [f"chexpert_{l}" for l in CHEXPERT_LABELS]
