import torch
import torch.nn as nn
from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

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

class Tiny3DCNN(nn.Module):
    def __init__(self, args=None,num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool3d(1)
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)





class Tiny3DTransformer(nn.Module):
    def __init__(
        self,
        args=None,
        in_channels: int = 1,
        num_classes: int = 2,
        embed_dim: int = 256,
        patch_size: Tuple[int,int,int] = (4, 32, 32),  # (depth, height, width)
        depth: int = 40,  # expected input depth
        height: int = 512, # expected input height
        width: int = 512,  # expected input width
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        use_cls_token: bool = True,
    ):
        """
        3D ViT-style classifier.

        Inputs expected shape: (B, C=1, D=40, H=512, W=512)
        patch_size must divide (D,H,W) respectively.
        """
        super().__init__()
        assert depth % patch_size[0] == 0, "patch depth must divide input depth"
        assert height % patch_size[1] == 0, "patch height must divide input height"
        assert width  % patch_size[2] == 0, "patch width must divide input width"

        self.patch_size = patch_size
        self.use_cls_token = use_cls_token

        # number of patches along each axis
        self.tokens_d = depth // patch_size[0]
        self.tokens_h = height // patch_size[1]
        self.tokens_w = width // patch_size[2]
        self.num_patches = self.tokens_d * self.tokens_h * self.tokens_w

        # Patch embed: conv3d with kernel == patch size and stride == patch size
        # outputs shape (B, embed_dim, tokens_d, tokens_h, tokens_w)
        self.patch_embed = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True
        )

        # cls token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_cls_token else None
        # +1 for cls token if used
        pos_len = self.num_patches + (1 if use_cls_token else 0)
        self.pos_embed = nn.Parameter(torch.randn(1, pos_len, embed_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True  # so input shape is (B, seq, embed)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # init
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        # patch_embed bias already default-initialized; we can initialize weights small
        nn.init.kaiming_normal_(self.patch_embed.weight, mode='fan_out', nonlinearity='relu')
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

    def forward(self, x: torch.Tensor):
        """
        x: (B, C=1, D=40, H=512, W=512)
        returns logits: (B, num_classes)
        """
        B = x.shape[0]
        # patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, tokens_d, tokens_h, tokens_w)
        # flatten spatial tokens to sequence
        x = x.view(B, x.shape[1], -1)   # (B, embed_dim, num_patches)
        x = x.permute(0, 2, 1).contiguous()  # (B, num_patches, embed_dim)

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B,1,embed)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, 1 + num_patches, embed_dim)

        # add positional embedding
        x = x + self.pos_embed  # broadcast (1, seq_len, embed_dim)

        # transformer encoder
        x = self.transformer(x)  # (B, seq_len, embed_dim)

        # classification from cls token (or mean pooling if no cls_token)
        if self.use_cls_token:
            cls_repr = x[:, 0]  # (B, embed_dim)
            out = self.norm(cls_repr)
        else:
            out = self.norm(x.mean(dim=1))

        logits = self.head(out)  # (B, num_classes)
        return logits
