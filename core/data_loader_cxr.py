import ast
import os
import pathlib
import pandas as pd
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from func_utils import timeit


class CXRJPGDataset(Dataset):
    """
    Single-image chest X-ray dataset for MIMIC-CXR JPG files.

    Each __getitem__ returns:
        {
          "image": transformed tensor,
          "text": report string,
          "image_path": str,
          "subject_id": str,
          "study_id": str
        }
    """

    def __init__(self, df_path, images_root, text_col="text", transform=None):
        self.df_path = pathlib.Path(df_path)
        self.images_root = str(images_root)
        self.text_col = text_col
        self.transform = transform

        # Load CSV
        self.df = pd.read_csv(self.df_path)

        # Build list of image-text pairs
        self.pairs = self._build_pairs()
        print("constructed pairs")

    def process_text_report(self):
        """
        maybe we want to
        :return:
        """
        pass

    @timeit
    def _build_pairs(self):
        """
        Build list of (image, text, subject_id, study_id) pairs.

        Directory structure:
            images_root/
                pXX/
                    pSUBJECT/
                        sSTUDY/
                            *.jpg
        """

        pairs = []
        missed = 0

        for _, row in self.df.iterrows():
            subj = str(row["subject_id"])
            text = str(row[self.text_col]).strip()

            # /files/p12/p12345678
            subj_dir = os.path.join(
                self.images_root,
                f"p{subj[:2]}",
                f"p{subj}",
            )

            if not os.path.isdir(subj_dir):
                missed += 1
                continue

            # FAST: list all study dirs inside subject
            study_dirs = [
                d for d in glob(os.path.join(subj_dir, "s*"))
                if os.path.isdir(d)
            ]

            if not study_dirs:
                missed += 1
                continue

            for sdir in study_dirs:
                study_id = os.path.basename(sdir)[1:]  # remove leading 's'

                # JPGs inside this study
                image_files = glob(os.path.join(sdir, "*.jpg"))

                if not image_files:
                    continue

                for img in image_files:
                    if not os.path.exists(img):
                        print("[MISSING]", img)
                        continue

                    pairs.append({
                        "subject_id": subj,
                        "study_id": study_id,
                        "image": img,
                        "text": text
                    })

        print("[INFO] Built", len(pairs), "pairs.")
        print("[missed_subjects_or_studies]", missed)
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        image_path = item["image"]

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "text": item["text"],
            "image_path": image_path,
            "subject_id": item["subject_id"],
            "study_id": item["study_id"]
        }


if __name__ == '__main__':
    print("started")

    df_root_path = pathlib.Path(
        r'C:\Users\tomer.erez\Desktop\code_projects\normal_near_normal\cxr_data\official_data_iccv_final'
    )

    train_ds = CXRJPGDataset(
        df_path=df_root_path / "mimic_cxr_aug_train.csv",
        images_root=df_root_path / "files",
        text_col="text",
        transform=transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor()
        ])
    )

    print("Dataset size:", len(train_ds))

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

    for batch in train_loader:
        image, text, image_path, subject_id, study_id = (batch[k] for k in
                                                         ["image", "text", "image_path", "subject_id", "study_id"])
        print(image.shape)
        print(len(text))
        break
