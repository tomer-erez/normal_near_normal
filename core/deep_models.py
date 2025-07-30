import torch
import torch.nn as nn
from transformers import BertModel

class CTTextCrossAttentionModel(nn.Module):
    def __init__(self, text_model='bert-base-multilingual-cased', hidden_dim=512, num_classes=3):
        super().__init__()
        self.text_encoder = BertModel.from_pretrained(text_model)
        self.ct_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, hidden_dim)
        )
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, ct_image, input_ids, attention_mask):
        text_feat = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        ct_feat = self.ct_encoder(ct_image).unsqueeze(1)
        attn_output, _ = self.cross_attention(ct_feat, text_feat, text_feat)
        logits = self.classifier(attn_output.squeeze(1))
        return logits
