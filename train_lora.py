"""
LoRA fine-tuning for open_clip models on MIMIC-CXR CheXpert label alignment.

Supported base models:
  - vanilla CLIP:  --base-model ViT-B-32 --pretrained openai
  - BiomedCLIP:    --base-model hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224

Saves a merged (LoRA absorbed) checkpoint at the end: {output_dir}/final_merged.pt
This can be loaded by baseline_eval/eval_model.py with --model_type finetuned.

Usage
-----
python train_lora.py \
    --base-model ViT-B-32 --pretrained openai \
    --train-csv cxr_data/mimic_cxr_train.csv \
    --image-dir /mnt/walkure_public/users/tomererez/mimic_cxr_jpg_images/ \
    --output-dir ./experiments/lora_vitb32

# Quick smoke-test:
python train_lora.py \
    --base-model ViT-B-32 --pretrained openai \
    --train-csv cxr_data/mimic_cxr_train.csv \
    --image-dir /mnt/walkure_public/users/tomererez/mimic_cxr_jpg_images/ \
    --output-dir ./experiments/lora_smoke \
    --epochs 1 --batch-size 32 --max-samples 500
"""

import argparse
import logging
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent / "open_clip" / "src"))
import open_clip
from open_clip.loss import ClipLoss

from train.cxr_label_dataset import CXRLabelDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="LoRA fine-tune open_clip on MIMIC-CXR CheXpert labels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    p.add_argument("--base-model", required=True,
                   help="open_clip model name, e.g. ViT-B-32 or "
                        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    p.add_argument("--pretrained", default="",
                   help="open_clip pretrained tag, e.g. 'openai'. Leave blank for hf-hub models.")
    # Data
    p.add_argument("--train-csv", required=True)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--caption-mode", default="both", choices=["single", "pair", "both"])
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap dataset size (useful for debugging)")
    # LoRA
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target", default="both", choices=["image", "text", "both"],
                   help="Which encoder(s) to apply LoRA to")
    p.add_argument("--lora-modules", default="in_proj,out_proj,c_fc,c_proj",
                   help="Comma-separated module name substrings to target with LoRA")
    # Training
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=1,
                   help="Gradient accumulation steps (effective batch = batch-size * steps)")
    p.add_argument("--save-frequency", type=int, default=1,
                   help="Save adapter checkpoint every N epochs")
    p.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--resume", default=None, help="Path to adapter .pt to resume from")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── LoRA setup ────────────────────────────────────────────────────────────────

def _get_target_modules(model, lora_module_names: list[str], lora_target: str) -> list[str]:
    """
    Return the full dotted paths of nn.Linear layers whose name ends with
    one of lora_module_names and lives inside the requested encoder.
    """
    encoder_prefix = {
        "image": ("visual.",),
        "text": ("transformer.", "text_projection", "token_embedding", "positional_embedding",
                 "ln_final", "attn_mask"),
        "both": ("",),  # matches everything
    }[lora_target]

    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        # Check encoder scope
        in_scope = any(name.startswith(pfx) for pfx in encoder_prefix)
        if not in_scope:
            continue
        # Check module suffix
        leaf = name.split(".")[-1]
        if any(leaf == m or name.endswith(f".{m}") for m in lora_module_names):
            targets.append(name)

    return targets


def apply_lora(model, args):
    from peft import get_peft_model, LoraConfig, TaskType

    lora_module_names = [m.strip() for m in args.lora_modules.split(",")]
    target_modules = _get_target_modules(model, lora_module_names, args.lora_target)

    if not target_modules:
        raise ValueError(
            f"No target modules found for lora_modules={lora_module_names}, "
            f"lora_target={args.lora_target}. "
            "Try --lora-modules or check the model's layer names."
        )
    log.info(f"LoRA target modules ({len(target_modules)}): {target_modules[:6]} ...")

    # Build safe suffixes: use bare leaf name only when every module with
    # that leaf name in the model is nn.Linear; otherwise use parent.leaf
    # to avoid hitting Conv2d / Sequential modules with the same name.
    leaf_to_paths: dict[str, list[str]] = {}
    for path in target_modules:
        leaf_to_paths.setdefault(path.split(".")[-1], []).append(path)

    target_suffixes: set[str] = set()
    for leaf, paths in leaf_to_paths.items():
        all_linear = all(
            isinstance(m, torch.nn.Linear)
            for n, m in model.named_modules()
            if n.split(".")[-1] == leaf
        )
        if all_linear:
            target_suffixes.add(leaf)
        else:
            for path in paths:
                target_suffixes.add(".".join(path.split(".")[-2:]))

    log.info(f"peft target_modules suffixes: {sorted(target_suffixes)}")

    config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=list(target_suffixes),
        bias="none",
    )
    model = get_peft_model(model, config)

    # If targeting only one encoder, freeze LoRA params in the other
    if args.lora_target != "both":
        freeze_prefix = "base_model.model.visual." if args.lora_target == "text" else None
        unfreeze_prefix = "base_model.model.visual." if args.lora_target == "image" else None
        for name, param in model.named_parameters():
            if "lora_" in name:
                if args.lora_target == "image":
                    param.requires_grad_(name.startswith("base_model.model.visual."))
                else:  # text
                    param.requires_grad_(not name.startswith("base_model.model.visual."))

    model.print_trainable_parameters()
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, epoch, amp_dtype, grad_accum_steps=1, total_epochs=10):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for step, (images, texts) in enumerate(loader):
        # print(texts.shape)
        # print(images.shape)
        images = images.to(device)
        texts = texts.to(device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
            image_features = model.encode_image(images, normalize=True)
            text_features = model.encode_text(texts, normalize=True)
            logit_scale = model.logit_scale.exp()
            loss = loss_fn(image_features, text_features, logit_scale)
            loss = loss / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                model.logit_scale.clamp_(0, math.log(100))

        total_loss += loss.item() * grad_accum_steps
        if step % 250 == 0:
            log.info(f"  epoch {epoch}/{total_epochs} step {step}/{len(loader)}  loss={loss.item() * grad_accum_steps:.4f}")

    return total_loss / len(loader)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Precision ─────────────────────────────────────────────────────────────
    amp_dtype = None
    scaler = None
    if args.precision == "fp16":
        amp_dtype = torch.float16
        scaler = torch.amp.GradScaler("cuda")
    elif args.precision == "bf16":
        amp_dtype = torch.bfloat16

    # ── Base model ────────────────────────────────────────────────────────────
    log.info(f"Loading base model: {args.base_model} pretrained={args.pretrained!r}")
    pretrained = args.pretrained if args.pretrained else None
    model, preprocess_train, _ = open_clip.create_model_and_transforms(
        args.base_model, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(args.base_model)

    model.requires_grad_(False)

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    model = apply_lora(model, args)

    if args.resume:
        log.info(f"Resuming adapter weights from {args.resume}")
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state, strict=False)

    model.to(device)

    # ── Dataset ───────────────────────────────────────────────────────────────
    log.info("Building dataset …")
    dataset = CXRLabelDataset(
        csv_path=args.train_csv,
        image_dir=args.image_dir,
        transform=preprocess_train,
        tokenizer=tokenizer,
        caption_mode=args.caption_mode,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    log.info(f"Dataset size: {len(dataset):,}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    batch = next(iter(loader))
    # ── Optimizer + loss ──────────────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    # Cosine LR schedule (epoch-level; epoch index is 0-based)
    warmup_epochs = max(1, args.epochs // 10)

    def lr_lambda(epoch_idx):
        if epoch_idx < warmup_epochs:
            return (epoch_idx + 1) / warmup_epochs
        progress = (epoch_idx - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loss_fn = ClipLoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    log.info(f"Training for {args.epochs} epochs …")
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, epoch, amp_dtype, args.grad_accum_steps, args.epochs)
        scheduler.step()
        log.info(f"Epoch {epoch} — avg loss: {avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if epoch % args.save_frequency == 0:
            ckpt_path = output_dir / f"epoch_{epoch}_adapter.pt"
            # Save only adapter (LoRA) weights
            adapter_state = {k: v for k, v in model.state_dict().items() if "lora_" in k}
            torch.save(adapter_state, ckpt_path)
            log.info(f"Adapter checkpoint saved → {ckpt_path}")

    # ── Merge LoRA into base model and save ───────────────────────────────────
    log.info("Merging LoRA weights into base model …")
    merged_model = model.merge_and_unload()
    merged_path = output_dir / "final_merged.pt"
    torch.save(merged_model.state_dict(), merged_path)
    log.info(f"Merged model saved → {merged_path}")
    log.info("Done.")


if __name__ == "__main__":
    main()
