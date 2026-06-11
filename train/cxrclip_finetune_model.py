"""
Full CXR-CLIP fine-tuning wrapper.

Freezes the ResNet50 image encoder (strong medical image features, no LoRA
needed) while keeping the Bio_ClinicalBERT text encoder fully trainable via
LoRA.  Presents an OpenCLIP-compatible interface so the rest of train_lora.py
works unchanged.

Tokenizer wrapper returns padded input_ids as a plain (N, max_len) int64
tensor — the same shape contract as open_clip.get_tokenizer — so
CXRLabelDataset needs no changes.  encode_text reconstructs the attention_mask
from non-zero positions.
"""

import logging
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
MAX_TEXT_LEN = 128  # ClinicalBERT context length


class ClinicalBERTTokenizerWrapper:
    """
    Drop-in replacement for open_clip.get_tokenizer().

    Wraps a HuggingFace tokenizer to return a (N, MAX_TEXT_LEN) int64 tensor
    of padded input_ids, exactly as open_clip's SimpleTokenizer does.
    """

    def __init__(self, hf_tokenizer, max_length: int = MAX_TEXT_LEN):
        self.tokenizer = hf_tokenizer
        self.max_length = max_length

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        enc = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return enc["input_ids"]  # (N, max_length)


def _load_cxrclip(checkpoint_path: str):
    """Return the full CXRClip model + config + HF tokenizer from a checkpoint."""
    sys.path.insert(0, str(REPO_ROOT / "cxr-clip"))
    from cxrclip.model import build_model
    from transformers import AutoTokenizer

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    tokenizer_name = cfg["tokenizer"]["pretrained_model_name_or_path"]
    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    default_cache = str(Path.home() / ".cache" / "huggingface" / "hub")
    model_cfg = dict(cfg["model"])
    model_cfg["text_encoder"] = dict(model_cfg["text_encoder"])
    model_cfg["text_encoder"]["cache_dir"] = default_cache
    model_cfg["text_encoder"]["local_files_only"] = False
    model_cfg["text_encoder"]["pretrained"] = False  # arch only; weights loaded below

    cxr_model = build_model(
        model_config=model_cfg,
        loss_config=cfg["loss"],
        tokenizer=hf_tokenizer,
    )
    cxr_model.load_state_dict(ckpt["model"], strict=False)

    return cxr_model, cfg, hf_tokenizer


class CXRClipFinetuneModel(nn.Module):
    """
    Frozen CXR-CLIP image encoder + trainable CXR-CLIP text encoder.

    The image side (ResNet50 + image_projection) is frozen — it already
    contains strong medical-image features.  The text side
    (Bio_ClinicalBERT + text_projection) is kept fully trainable and is the
    target for LoRA in train_lora.py.

    Interface mirrors OpenCLIP so the training loop needs no changes:
      encode_image(images, normalize=False) → (N, D)
      encode_text(tokens, normalize=False)  → (N, D)   tokens: (N, max_len) int64
      logit_scale: nn.Parameter
    """

    def __init__(self, checkpoint_path: str):
        super().__init__()

        cxr_model, cfg, hf_tokenizer = _load_cxrclip(checkpoint_path)

        # ── Image side (frozen) ───────────────────────────────────────────────
        self.image_encoder = cxr_model.image_encoder
        self.image_projection = cxr_model.image_projection if cxr_model.projection else None
        self._image_encoder_name = cfg["model"]["image_encoder"]["name"]

        for p in self.image_encoder.parameters():
            p.requires_grad_(False)
        if self.image_projection is not None:
            for p in self.image_projection.parameters():
                p.requires_grad_(False)

        # ── Text side (trainable) ─────────────────────────────────────────────
        self.text_encoder = cxr_model.text_encoder   # HuggingfaceTextEncoder wrapper
        self.text_projection = cxr_model.text_projection if cxr_model.projection else None
        self._text_pooling = cfg["model"]["text_encoder"]["pooling"]

        # ── Shared ───────────────────────────────────────────────────────────
        if isinstance(cxr_model.logit_scale, nn.Parameter):
            self.logit_scale = nn.Parameter(cxr_model.logit_scale.data.clone())
        else:
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

        self._hf_tokenizer = hf_tokenizer
        self._checkpoint_path = checkpoint_path
        self._image_size = cfg.get("base", {}).get("image_size", 224)

        log.info(
            "CXRClipFinetuneModel ready — "
            f"image_encoder={self._image_encoder_name} (frozen), "
            f"text_encoder=Bio_ClinicalBERT (trainable), "
            f"pooling={self._text_pooling}"
        )

    @staticmethod
    def make_preprocess(checkpoint_path: str) -> transforms.Compose:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        enc_name = ckpt["config"]["model"]["image_encoder"]["name"]
        image_size = ckpt["config"].get("base", {}).get("image_size", 224)
        if enc_name == "resnet":
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        else:
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        return transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def make_tokenizer(self) -> ClinicalBERTTokenizerWrapper:
        return ClinicalBERTTokenizerWrapper(self._hf_tokenizer, MAX_TEXT_LEN)

    def encode_image(self, images, normalize: bool = False):
        feat = self.image_encoder(images)
        if self._image_encoder_name != "resnet":
            feat = feat[:, 0]
        if self.image_projection is not None:
            feat = self.image_projection(feat)
        if normalize:
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat

    def encode_text(self, tokens, normalize: bool = False):
        # tokens: (N, max_len) int64, padded with 0
        attention_mask = (tokens != 0).long()
        text_tokens = {"input_ids": tokens, "attention_mask": attention_mask}

        feat = self.text_encoder(text_tokens)  # (N, seq_len, hidden_size)

        if self._text_pooling == "eos":
            eos_idx = attention_mask.sum(dim=-1) - 1
            feat = feat[torch.arange(feat.shape[0]), eos_idx]
        elif self._text_pooling == "bos":
            feat = feat[:, 0]
        elif self._text_pooling == "mean":
            mask_exp = attention_mask.unsqueeze(-1).expand(feat.size()).float()
            feat = torch.sum(feat * mask_exp, dim=1) / mask_exp.sum(dim=1).clamp(min=1e-9)
        else:
            raise ValueError(f"Unknown text pooling: {self._text_pooling}")

        if self.text_projection is not None:
            feat = self.text_projection(feat)
        if normalize:
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat
