import torch
from torch.utils.data import Dataset

class DummyCTTextDataset(Dataset):
    def __init__(self, tokenizer, length=100, max_len=64):
        self.tokenizer = tokenizer
        self.length = length
        self.max_len = max_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        dummy_text = "כאב ראש חמור וחולשה כללית"
        inputs = self.tokenizer(dummy_text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors="pt")
        image = torch.randn(1, 224, 224)
        label = torch.randint(0, 3, (1,)).item()
        return image, inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0), label
