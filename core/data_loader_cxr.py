import pathlib
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import ast
from func_utils import timeit



class CXRJPGDataset(Dataset):
    """
    Chest X-ray dataset with two modes:
    - "image": each __getitem__ returns a single image + text
    - "subject": each __getitem__ returns all images for a subject + text
    """

    @timeit
    def __init__(self, df_path, images_root, text_col="text", transform=None, mode="image"):
        self.df_path = pathlib.Path(df_path)
        self.images_root = pathlib.Path(images_root)
        self.text_col = text_col
        self.transform = transform
        assert mode in ["image", "subject"], "mode must be 'image' or 'subject'"
        self.mode = mode

        # Load CSV
        self.df = pd.read_csv(self.df_path)

        # Build list of pairs
        if self.mode == "image":
            self.pairs = self._build_pairs_image_text()
        else:
            self.pairs = self._build_pairs_subject()

        print(f"[INFO] constructed dataset: {len(self.pairs)} pairs in mode='{self.mode}'")

    def _build_pairs_image_text(self):
        """Single-image mode: one entry per image."""

        pairs = []
        missed = 0
        for idx, row in self.df.iterrows():
            subj = str(row["subject_id"])
            text = str(row[self.text_col]).strip()
            try:
                image_list = ast.literal_eval(row["image"])
            except Exception as e:
                continue

            for img_rel in image_list:
                img_path = self.images_root / img_rel.replace("files/", "").lstrip("/")
                if not img_path.exists():
                    missed += 1
                    continue

                study_id = pathlib.Path(img_rel).parent.name.replace("s", "")

                pairs.append({
                    "subject_id": subj,
                    "study_id": study_id,
                    "image_path": str(img_path),
                    "text": text
                })
        print(f"[WARNING] Missed {missed} images during construction of the dataset")
        return pairs

    @timeit
    def _build_pairs_subject(self):
        """Subject-level mode: one entry per subject, returns list of images + text."""
        pairs = []
        missed = 0
        for idx, row in self.df.iterrows():
            subj = str(row["subject_id"])
            text = str(row[self.text_col]).strip()
            try:
                image_list = ast.literal_eval(row["image"])
            except Exception as e:
                continue

            img_paths = []
            for img_rel in image_list:
                img_path = self.images_root / img_rel.replace("files/", "").lstrip("/")
                if img_path.exists():
                    img_paths.append(str(img_path))
                else:
                    missed += 1

            pairs.append({
                "subject_id": subj,
                "study_id": None,  # optional if you don't want a single study_id
                "image_paths": img_paths,
                "text": text
            })
        print(f"[WARNING] Missed {missed} images during construction of the dataset")
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]

        if self.mode == "image":
            image = Image.open(item["image_path"]).convert("RGB")
            if self.transform:
                image = self.transform(image)

            ip = item["image_path"] if self.mode == "image" else item["image_paths"]
            return {
                "image": image,
                "text": item["text"],
                "image_path": ip,
                "subject_id": item["subject_id"],
            }

        elif self.mode == "subject":  # subject mode
            images = []
            for path in item["image_paths"]:
                img = Image.open(path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                images.append(img)

            return {
                "image": images,  # list of tensors
                "text": item["text"],
                "image_path": item["image_paths"],
                "subject_id": item["subject_id"]
            }


def collate_subject(batch):
    """
    Collate function for subject-mode dataset.
    Keeps images as lists of tensors without stacking.
    :type batch: object
    """
    return {
        "image": [item["image"] for item in batch],
        "text": [item["text"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
        "subject_id": [item["subject_id"] for item in batch]
    }


if __name__ == '__main__':
    print("started")
    
    text_data_path = pathlib.Path(
        r"C:\Users\tomer.erez\Desktop\code_projects\normal_near_normal\cxr_data\mimic_cxr_full_text\all_txt_data.csv"
    )
    image_paths_root = pathlib.Path(
        r"C:\Users\tomer.erez\Desktop\code_projects\normal_near_normal\cxr_data\mimic_cxr_jpg_images_from_google_cloud\mimic-cxr-jpg-2.1.0.physionet.org\files"
    )
    mode = "image"
    train_ds = CXRJPGDataset(
        text_data_path=text_data_path,
        images_root=image_paths_root,
        text_col="text",
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor()
        ]),
        mode=mode
    )

    if mode == "image":
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    elif mode == "subject":

        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_subject)
    else:
        raise ValueError("Unknown mode")

    for batch in train_loader:
        images, text, image_path, subject_id = (batch[k] for k in ["image", "text", "image_path", "subject_id"])
        break
