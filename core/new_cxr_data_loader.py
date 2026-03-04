from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import random
import time
from torchvision import transforms

def change_text_path_to_img_path(txt_file_path):
    """
    Converts:
        files/mimic-cxr/2.1.0/files/p10/p10000032/s50414267.txt
    To:
        p10/p10000032/s50414267/<random_image>.jpg
    """

    txt_path = Path(txt_file_path)

    # Extract identifiers
    # .../p10/p10000032/s50414267.txt
    study_id = txt_path.stem            # s50414267
    patient_id = txt_path.parent.name   # p10000032
    subject_id = txt_path.parent.parent.name  # p10

    # Relative image folder (WITHOUT image_root_path)
    image_folder = Path(subject_id) / patient_id / study_id

    # This path is relative — Dataset will prepend image_root_path
    return image_folder

class MIMICCXRImageTextDataset(Dataset):
    def __init__(
        self,
        csv_path,
        image_root_path,
        transform=None,
        change_text_path_to_img_path=None,
    ):
        """
        Args:
            csv_path (str or Path): CSV with columns ['txt_file_path', 'txt_content']
            image_root_path (str or Path): Root directory containing pXX folders
            transform: torchvision transforms for images
            change_text_path_to_img_path (callable):
                function(txt_file_path) -> relative image path
        """
        self.df = pd.read_csv(csv_path)
        self.image_root_path = Path(image_root_path)
        self.transform = transform

        assert change_text_path_to_img_path is not None, \
            "You must provide change_text_path_to_img_path function"

        self.change_text_path_to_img_path = change_text_path_to_img_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        txt_content = row["txt_content"]
        if pd.isna(txt_content):
            txt_content = ""

        txt_file_path = row["txt_file_path"]

        rel_img_folder = self.change_text_path_to_img_path(txt_file_path)
        img_dir = self.image_root_path / rel_img_folder

        if not img_dir.exists():
            return self.__getitem__(random.randint(0, len(self) - 1))

        jpgs = list(img_dir.glob("*.jpg"))
        if len(jpgs) == 0:
            return self.__getitem__(random.randint(0, len(self) - 1))

        img_path = random.choice(jpgs)

        with Image.open(img_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "text": txt_content,
            "image_path": str(img_path),
            "text_path": txt_file_path,
        }


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])

    a = time.time()
    dataset = MIMICCXRImageTextDataset(
        csv_path=r"C:\Users\tomer.erez\Desktop\code_projects\normal_near_normal\cxr_data\mimic_cxr_full_text\all_txt_data.csv",
        image_root_path=r"C:\Users\tomer.erez\Desktop\code_projects\normal_near_normal\cxr_data\mimic_cxr_jpg_images_from_google_cloud\mimic-cxr-jpg-2.1.0.physionet.org\files",
        change_text_path_to_img_path=change_text_path_to_img_path,
        transform=transform
    )
    print(f"Dataset creation time: {time.time() - a:.2f} seconds")
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Text content: {sample['text']}")


