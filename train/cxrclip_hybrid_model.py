"""
CXR-CLIP image encoder + OpenCLIP text encoder hybrid model.

The image encoder and projection come from a CXR-CLIP checkpoint and are frozen.
The text encoder components come from an OpenCLIP model and are fully trainable.
Both sides produce 512-dim L2-normalized embeddings (proj_dim=512 in all CXR-CLIP
checkpoints we use; ViT-B-32 text encoder also outputs 512-dim).

Module naming follows the standard OpenCLIP layout so that --lora-target text
in train_lora.py targets the correct layers without any changes.
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

_CXRCLIP_PATH = str(REPO_ROOT / "cxr-clip")
if _CXRCLIP_PATH not in sys.path:
    sys.path.insert(0, _CXRCLIP_PATH)


def _make_cxrclip_preprocess(enc_name: str, image_size: int) -> transforms.Compose:
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


def _load_cxrclip_image_side(checkpoint_path: str):
    """
    Load (image_encoder, image_projection, encoder_name, image_size)
    from a CXR-CLIP checkpoint, with all parameters already frozen.
    """
    from cxrclip.model import build_model
    from transformers import AutoTokenizer

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    tokenizer_name = cfg["tokenizer"]["pretrained_model_name_or_path"]
    cxr_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    default_cache = str(Path.home() / ".cache" / "huggingface" / "hub")
    model_cfg = dict(cfg["model"])
    model_cfg["text_encoder"] = dict(model_cfg["text_encoder"])
    model_cfg["text_encoder"]["cache_dir"] = default_cache
    model_cfg["text_encoder"]["local_files_only"] = False
    # Architecture only — weights come from the checkpoint state dict below
    model_cfg["text_encoder"]["pretrained"] = False

    cxr_model = build_model(
        model_config=model_cfg,
        loss_config=cfg["loss"],
        tokenizer=cxr_tokenizer,
    )
    cxr_model.load_state_dict(ckpt["model"], strict=False)

    encoder_name = cfg["model"]["image_encoder"]["name"]
    image_size = cfg.get("base", {}).get("image_size", 224)

    for p in cxr_model.image_encoder.parameters():
        p.requires_grad_(False)
    for p in cxr_model.image_projection.parameters():
        p.requires_grad_(False)

    return cxr_model.image_encoder, cxr_model.image_projection, encoder_name, image_size


class CXRClipHybridModel(nn.Module):
    """
    Frozen CXR-CLIP image encoder + trainable OpenCLIP text encoder.

    The image encoder (ResNet50 or Swin-T) and its projection head come from a
    CXR-CLIP checkpoint and are frozen. The text encoder is initialized from an
    OpenCLIP model (e.g. ViT-B-32 "openai") and is trainable — either fully or
    via LoRA when used with train_lora.py.

    Text components are inlined as top-level sub-modules/parameters using the
    same names as in the original OpenCLIP model (transformer.*, ln_final.*,
    positional_embedding, text_projection). This lets --lora-target text work
    with the existing _get_target_modules logic in train_lora.py unchanged.
    """

    def __init__(self, cxrclip_checkpoint_path: str, openclip_model):
        super().__init__()

        image_enc, image_proj, enc_name, img_size = _load_cxrclip_image_side(
            cxrclip_checkpoint_path
        )
        self.image_encoder = image_enc
        self.image_projection = image_proj
        self._image_encoder_name = enc_name
        self._image_size = img_size

        # ── Text side from OpenCLIP (fully trainable) ─────────────────────────
        # Inlined at the top level so LoRA targeting via "transformer.*" etc.
        # works without changes to _get_target_modules.
        self.transformer = openclip_model.transformer
        self.token_embedding = openclip_model.token_embedding
        self.positional_embedding = nn.Parameter(
            openclip_model.positional_embedding.data.clone()
        )
        self.ln_final = openclip_model.ln_final
        self.text_projection = nn.Parameter(
            openclip_model.text_projection.data.clone()
        )
        if hasattr(openclip_model, "attn_mask") and openclip_model.attn_mask is not None:
            self.register_buffer("attn_mask", openclip_model.attn_mask.clone())
        else:
            self.attn_mask = None

        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

        log.info(
            "CXRClipHybridModel ready — "
            f"image_encoder={enc_name} (frozen), "
            f"text_encoder=OpenCLIP-{type(openclip_model).__name__} (trainable)"
        )

    def get_preprocess(self) -> transforms.Compose:
        """Return the image preprocessing pipeline using already-loaded config."""
        return _make_cxrclip_preprocess(self._image_encoder_name, self._image_size)

    @staticmethod
    def make_preprocess(cxrclip_checkpoint_path: str) -> transforms.Compose:
        """Return preprocessing from a checkpoint path (loads config from disk)."""
        ckpt = torch.load(cxrclip_checkpoint_path, map_location="cpu", weights_only=False)
        enc_name = ckpt["config"]["model"]["image_encoder"]["name"]
        image_size = ckpt["config"].get("base", {}).get("image_size", 224)
        return _make_cxrclip_preprocess(enc_name, image_size)

    def encode_image(self, images, normalize: bool = False):
        feat = self.image_encoder(images)
        if self._image_encoder_name != "resnet":
            feat = feat[:, 0]  # CLS / first token for ViT-like encoders
        feat = self.image_projection(feat)
        if normalize:
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat

    def encode_text(self, tokens, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(tokens).to(cast_dtype)  # (N, L, C)
        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)  # (N, L, C)
        x = self.ln_final(x)
        # EOT pooling: highest token id is the end-of-text token position
        x = x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)] @ self.text_projection
        if normalize:
            x = x / x.norm(dim=-1, keepdim=True)
        return x
