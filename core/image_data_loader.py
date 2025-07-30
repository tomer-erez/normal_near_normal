import torch
from torch.utils.data import Dataset

class DummyCTDataset(Dataset):
    def __init__(self, length=100):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = torch.randn(1, 224, 224, 64)
        label = torch.randint(0, 3, (1,)).item()
        return image, label
