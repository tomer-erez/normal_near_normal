"""
LoRA fine-tuning for open_clip models on MIMIC-CXR CheXpert label alignment.

Supported base models:
  - vanilla CLIP:  --base-model ViT-B-32 --pretrained openai
  - BiomedCLIP:    --base-model hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224

Saves a merged (LoRA absorbed) checkpoint at the end: {output_dir}/final_merged.pt
This can be loaded by baseline_eval/eval_model.py with --model_type finetuned.

Single-GPU usage
----------------
python train_lora.py \
    --base-model ViT-B-32 --pretrained openai \
    --train-csv cxr_data/mimic_cxr_train.csv \
    --image-dir /mnt/walkure_public/users/tomererez/mimic_cxr_jpg_images/ \
    --output-dir ./experiments/lora_vitb32

Multi-GPU usage (torchrun)
--------------------------
torchrun --nproc_per_node=4 train_lora.py \
    --base-model ViT-B-32 --pretrained openai \
    --train-csv cxr_data/mimic_cxr_train.csv \
    --image-dir /mnt/walkure_public/users/tomererez/mimic_cxr_jpg_images/ \
    --output-dir ./experiments/lora_vitb32_4gpu

# Quick smoke-test:
python train_lora.py \
    --base-model ViT-B-32 --pretrained openai \
    --train-csv cxr_data/mimic_cxr_train.csv \
    --image-dir /mnt/walkure_public/users/tomererez/mimic_cxr_jpg_images/ \
    --output-dir ./experiments/lora_smoke \
    --epochs 1 --batch-size 32 --max-samples 500
"""

import argparse
import copy
import logging
import math
import os
import ssl
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Bypass SSL verification for environments with corporate certificate inspection
ssl._create_default_https_context = ssl._create_unverified_context
os.environ.setdefault("CURL_CA_BUNDLE", "")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "")

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

try:
    import wandb
except ImportError:
    wandb = None

sys.path.insert(0, str(Path(__file__).parent / "open_clip" / "src"))
import open_clip
from open_clip.loss import ClipLoss, SigLipLoss

from train.cxr_label_dataset import CXRLabelDataset
from train.label_aware_loss import LabelAwareClipLoss, LabelAwareSigLipLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Distributed helpers ───────────────────────────────────────────────────────

def setup_distributed():
    """
    Initialize DDP when launched via torchrun (LOCAL_RANK is set).
    Returns (local_rank, rank, world_size, is_main).
    When running single-GPU, all values are 0/1/True.
    """
    if "LOCAL_RANK" not in os.environ:
        return 0, 0, 1, True
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size, rank == 0


def _raw_model(model):
    """Unwrap DDP to get the underlying peft model."""
    return model.module if isinstance(model, DDP) else model


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="LoRA fine-tune open_clip on MIMIC-CXR CheXpert labels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    p.add_argument("--base-model", required=True,
                   help="open_clip model name, e.g. ViT-B-32 or "
                        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224. "
                        "When --cxrclip-checkpoint is set, this specifies the OpenCLIP text "
                        "encoder used in the hybrid model (e.g. ViT-B-32 --pretrained openai).")
    p.add_argument("--pretrained", default="",
                   help="open_clip pretrained tag, e.g. 'openai'. Leave blank for hf-hub models.")
    p.add_argument("--cxrclip-checkpoint", default=None,
                   help="Path to a CXR-CLIP checkpoint (.pt). When provided, creates a hybrid "
                        "model that uses the CXR-CLIP image encoder (frozen) paired with an "
                        "OpenCLIP text encoder specified by --base-model / --pretrained. "
                        "E.g. valid_pretrained_models_to_try/r50_mc.pt")
    # Data
    p.add_argument("--train-csv", required=False,default="cxr_data/mimic_cxr_train.csv")
    p.add_argument("--image-dir", required=False,default="cxr_data/images/mimic_cxr_jpg_images_from_google_cloud/mimic-cxr-jpg-2.1.0.physionet.org/files/")
    p.add_argument("--val-csv", default=None,
                   help="Separate validation CSV. If omitted, --val-split is used instead.")
    p.add_argument("--val-split", type=float, default=0.1,
                   help="Fraction of training data held out for validation when --val-csv is not given. "
                        "Set to 0 to disable validation entirely.")
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
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=8,
                   help="Per-GPU batch size. Effective batch = batch-size * grad-accum-steps * world-size")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-7,
                   help="Minimum LR floor for cosine schedule")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=1,
                   help="Gradient accumulation steps")
    p.add_argument("--save-frequency", type=int, default=1,
                   help="Save adapter checkpoint every N epochs")
    p.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--resume", default=None, help="Path to adapter .pt to resume from")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report-interval-steps", type=int, default=500,
                   help="Interval (in steps) for reporting training loss during epoch")
    # Scheduler
    p.add_argument("--scheduler", default="cosine", choices=["cosine", "plateau"],
                   help="LR scheduler. 'plateau' reduces LR when val loss stagnates.")
    # Early stopping
    p.add_argument("--patience", type=int, default=5,
                   help="Early stopping patience in epochs (0 to disable)")
    # Label-aware loss
    p.add_argument("--match-mode", default="standard",
                   choices=["standard", "single_label", "two_label", "negative_aware"],
                   help="How to define positive pairs in the batch. "
                        "'standard': diagonal (vanilla CLIP). "
                        "'single_label': share ≥1 positive label. "
                        "'two_label': share ≥2 positive labels. "
                        "'negative_aware': single_label + repulsion for conflicting pairs. "
                        "See training_guide.txt for details.")
    p.add_argument("--negative-weight", type=float, default=0.5,
                   help="Weight for the repulsion loss term (negative_aware mode only).")
    p.add_argument("--negative-margin", type=float, default=0.0,
                   help="Cosine-sim margin for repulsion: pairs with sim < margin are not penalised "
                        "(negative_aware mode only).")
    # Loss variant
    p.add_argument("--loss", default="clip", choices=["clip", "siglip"],
                   help="Loss function. "
                        "'clip': softmax multi-positive cross-entropy (LabelAwareClipLoss / ClipLoss). "
                        "'siglip': sigmoid binary cross-entropy per pair (LabelAwareSigLipLoss / SigLipLoss). "
                        "Compatible with all --match-mode options. "
                        "SigLIP models (ViT-B-16-SigLIP etc.) expose a logit_bias parameter that "
                        "is picked up automatically; vanilla CLIP models work fine without it.")
    # Weights & Biases
    p.add_argument("--wandb-project", default=None,
                   help="W&B project name. Omit to disable W&B logging.")
    p.add_argument("--wandb-run-name", default=None,
                   help="W&B run name (auto-generated if omitted).")
    p.add_argument("--wandb-entity", default=None,
                   help="W&B entity (team/user). Uses your default entity if omitted.")
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
        in_scope = any(name.startswith(pfx) for pfx in encoder_prefix)
        if not in_scope:
            continue
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

    if args.lora_target != "both":
        for name, param in model.named_parameters():
            if "lora_" in name:
                if args.lora_target == "image":
                    param.requires_grad_(name.startswith("base_model.model.visual."))
                else:  # text
                    param.requires_grad_(not name.startswith("base_model.model.visual."))

    model.print_trainable_parameters()
    return model


# ── Training / evaluation ─────────────────────────────────────────────────────

def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, epoch,
                    amp_dtype, total_epochs, report_interval_steps, grad_accum_steps=1,
                    is_main=True, is_siglip=False, global_step=0):
    model.train()
    raw = _raw_model(model)
    total_loss = 0.0
    optimizer.zero_grad()
    use_labels = isinstance(loss_fn, (LabelAwareClipLoss, LabelAwareSigLipLoss))

    for step, (images, texts, labels) in enumerate(loader):
        images = images.to(device)
        texts = texts.to(device)
        labels_dev = labels.to(device)

        # Log batch-level pairing stats once per epoch so training signal is visible
        if is_main and step == 0 and use_labels:
            stats = loss_fn.batch_stats(labels_dev)
            log.info(
                f"  [batch stats] avg_positives_per_sample={stats['avg_positives_per_sample']:.2f}"
                f"  diagonal_only={stats['pct_diagonal_only']:.1f}%"
                f"  conflict_pairs={stats['pct_conflict_pairs']:.1f}%"
            )

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
            image_features = raw.encode_image(images, normalize=True)
            text_features = raw.encode_text(texts, normalize=True)
            logit_scale = raw.logit_scale.exp()
            if is_siglip:
                # SigLIP models carry a learned logit_bias; vanilla CLIP models don't.
                logit_bias = getattr(raw, "logit_bias", None)
                if use_labels:
                    loss = loss_fn(image_features, text_features, logit_scale, labels_dev, logit_bias)
                else:
                    loss = loss_fn(image_features, text_features, logit_scale, logit_bias)
            else:
                if use_labels:
                    loss = loss_fn(image_features, text_features, logit_scale, labels_dev)
                else:
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
                raw.logit_scale.clamp_(0, math.log(100))

        step_loss = loss.item() * grad_accum_steps
        total_loss += step_loss
        if is_main and step % report_interval_steps == 0:
            log.info(
                f"  epoch {epoch}/{total_epochs} step {step}/{len(loader)}"
                f"  loss={step_loss:.4f}"
            )
            if wandb is not None and wandb.run is not None:
                wandb.log({"train/loss_step": step_loss}, step=global_step + step)

    return total_loss / len(loader)


@torch.no_grad()
def eval_one_epoch(model, loader, loss_fn, device, amp_dtype, is_siglip=False):
    model.eval()
    raw = _raw_model(model)
    total_loss = 0.0
    use_labels = isinstance(loss_fn, (LabelAwareClipLoss, LabelAwareSigLipLoss))
    logit_bias = getattr(raw, "logit_bias", None) if is_siglip else None
    for images, texts, labels in loader:
        images = images.to(device)
        texts = texts.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
            image_features = raw.encode_image(images, normalize=True)
            text_features = raw.encode_text(texts, normalize=True)
            logit_scale = raw.logit_scale.exp()
            if is_siglip:
                if use_labels:
                    loss = loss_fn(image_features, text_features, logit_scale, labels.to(device), logit_bias)
                else:
                    loss = loss_fn(image_features, text_features, logit_scale, logit_bias)
            else:
                if use_labels:
                    loss = loss_fn(image_features, text_features, logit_scale, labels.to(device))
                else:
                    loss = loss_fn(image_features, text_features, logit_scale)
        total_loss += loss.item()
    return total_loss / len(loader)


# ── Early stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        """Returns True when training should stop."""
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
            return False
        self.counter += 1
        log.info(f"  Early-stop counter: {self.counter}/{self.patience}")
        return self.counter >= self.patience


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    local_rank, rank, world_size, is_main = setup_distributed()
    # Different seed per rank so each GPU sees a different data shuffle
    torch.manual_seed(args.seed + rank)

    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    if is_main and args.wandb_project:
        if wandb is None:
            raise ImportError("wandb is not installed. Run: pip install wandb")
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=vars(args),
            dir=str(output_dir),
        )

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main:
        log.info(f"Device: {device}  world_size: {world_size}")
    # Print all arguments
    for arg, value in vars(args).items():
        log.info(f"{arg}: {value}")
    # ── Precision ─────────────────────────────────────────────────────────────
    amp_dtype = None
    scaler = None
    if args.precision == "fp16":
        amp_dtype = torch.float16
        scaler = torch.amp.GradScaler("cuda")
    elif args.precision == "bf16":
        amp_dtype = torch.bfloat16

    # ── Base model ────────────────────────────────────────────────────────────
    pretrained = args.pretrained if args.pretrained else None

    if args.cxrclip_checkpoint:
        # Hybrid mode: frozen CXR-CLIP image encoder + trainable OpenCLIP text encoder
        if is_main:
            log.info(
                f"Hybrid mode — CXR-CLIP image encoder from: {args.cxrclip_checkpoint}  "
                f"OpenCLIP text encoder: {args.base_model} pretrained={args.pretrained!r}"
            )
        from train.cxrclip_hybrid_model import CXRClipHybridModel
        openclip_model, _, _ = open_clip.create_model_and_transforms(
            args.base_model, pretrained=pretrained, device="cpu"
        )
        model = CXRClipHybridModel(args.cxrclip_checkpoint, openclip_model)
        del openclip_model
        preprocess_train = CXRClipHybridModel.make_preprocess(args.cxrclip_checkpoint)
        tokenizer = open_clip.get_tokenizer(args.base_model)
        # image_encoder and image_projection are already frozen inside CXRClipHybridModel;
        # text components are trainable. LoRA will be applied to text Linear layers.
    else:
        if is_main:
            log.info(f"Loading base model: {args.base_model} pretrained={args.pretrained!r}")
        model, preprocess_train, _ = open_clip.create_model_and_transforms(
            args.base_model, pretrained=pretrained, device=device
        )
        tokenizer = open_clip.get_tokenizer(args.base_model)
        model.requires_grad_(False)

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    model = apply_lora(model, args)

    if args.resume:
        if is_main:
            log.info(f"Resuming adapter weights from {args.resume}")
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state, strict=False)

    model.to(device)

    # ── Wrap with DDP ─────────────────────────────────────────────────────────
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ── Dataset ───────────────────────────────────────────────────────────────
    if is_main:
        log.info("Building dataset …")
    full_dataset = CXRLabelDataset(
        csv_path=args.train_csv,
        image_dir=args.image_dir,
        transform=preprocess_train,
        tokenizer=tokenizer,
        caption_mode=args.caption_mode,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    if is_main:
        log.info(f"Full dataset size: {len(full_dataset):,}")

    # Split into train / val
    val_dataset = None
    if args.val_csv:
        train_dataset = full_dataset
        val_dataset = CXRLabelDataset(
            csv_path=args.val_csv,
            image_dir=args.image_dir,
            transform=preprocess_train,
            tokenizer=tokenizer,
            caption_mode=args.caption_mode,
            seed=args.seed,
        )
        if is_main:
            log.info(f"Train: {len(train_dataset):,}  Val (from --val-csv): {len(val_dataset):,}")
    elif args.val_split > 0:
        val_size = max(1, int(len(full_dataset) * args.val_split))
        train_size = len(full_dataset) - val_size
        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=generator
        )
        if is_main:
            log.info(
                f"Train: {len(train_dataset):,}  "
                f"Val (random split {args.val_split:.0%}): {len(val_dataset):,}"
            )
    else:
        train_dataset = full_dataset
        if is_main:
            log.info("No validation set (--val-split 0 and no --val-csv). Early stopping disabled.")

    # DistributedSampler ensures each GPU sees a non-overlapping shard
    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank,
                           shuffle=True, seed=args.seed)
        if world_size > 1 else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if val_dataset is not None:
        # Val runs on all ranks; losses are averaged independently (they're identical since
        # val data isn't sharded, which is fine for monitoring purposes)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
        )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    # ── Scheduler ─────────────────────────────────────────────────────────────
    warmup_epochs = max(1, args.epochs // 10)

    if args.scheduler == "cosine":
        min_factor = args.min_lr / args.lr

        def lr_lambda(epoch_idx):
            if epoch_idx < warmup_epochs:
                return (epoch_idx + 1) / warmup_epochs
            progress = (epoch_idx - warmup_epochs) / max(1, args.epochs - warmup_epochs)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:  # plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, min_lr=args.min_lr
        )

    # Loss function.
    # standard match_mode: use open_clip's distributed-aware ClipLoss / SigLipLoss.
    # label-aware modes: use local-batch LabelAwareClipLoss / LabelAwareSigLipLoss;
    #   distributed runs still benefit from DDP gradient sync even without all-gather.
    is_siglip = args.loss == "siglip"
    if args.match_mode == "standard":
        if not is_siglip:
            loss_fn = ClipLoss(
                gather_with_grad=True,
                rank=rank,
                world_size=world_size,
            )
        else:
            loss_fn = SigLipLoss(
                rank=rank,
                world_size=world_size,
            )
    else:
        if not is_siglip:
            loss_fn = LabelAwareClipLoss(
                match_mode=args.match_mode,
                neg_weight=args.negative_weight,
                neg_margin=args.negative_margin,
            )
        else:
            loss_fn = LabelAwareSigLipLoss(
                match_mode=args.match_mode,
                neg_weight=args.negative_weight,
                neg_margin=args.negative_margin,
            )
    if is_main:
        log.info(f"Loss: {args.loss}/{args.match_mode}"
                 + (f"  neg_weight={args.negative_weight}  neg_margin={args.negative_margin}"
                    if args.match_mode == "negative_aware" else ""))

    # ── Early stopping ────────────────────────────────────────────────────────
    early_stopper = None
    if args.patience > 0 and val_loader is not None:
        early_stopper = EarlyStopping(patience=args.patience)

    best_val_loss = float("inf")
    best_epoch = 0
    global_step = 0
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []

    def save_loss_plot():
        epochs_so_far = list(range(1, len(train_loss_history) + 1))
        fig, ax = plt.subplots()
        ax.plot(epochs_so_far, train_loss_history, label="train")
        if val_loss_history:
            ax.plot(epochs_so_far, val_loss_history, label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training curve")
        ax.legend()
        fig.savefig(output_dir / "loss_curve.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    # ── Training loop ─────────────────────────────────────────────────────────
    if is_main:
        eff_batch = args.batch_size * args.grad_accum_steps * world_size
        log.info(
            f"Training for up to {args.epochs} epochs  "
            f"(effective batch size: {eff_batch}) …"
        )

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)  # ensures different shuffle each epoch

        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler,
            device, epoch, amp_dtype, args.epochs,
            args.report_interval_steps, args.grad_accum_steps,
            is_main=is_main, is_siglip=is_siglip,
            global_step=global_step,
        )
        global_step += len(train_loader)

        val_loss = None
        if val_loader is not None:
            val_loss = eval_one_epoch(model, val_loader, loss_fn, device, amp_dtype, is_siglip=is_siglip)

        # Scheduler step
        if args.scheduler == "cosine":
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            monitor = val_loss if val_loss is not None else train_loss
            scheduler.step(monitor)
            current_lr = optimizer.param_groups[0]["lr"]

        # All logging and checkpointing only on rank 0
        if is_main:
            train_loss_history.append(train_loss)
            if val_loss is not None:
                val_loss_history.append(val_loss)
            save_loss_plot()

            if val_loss is not None:
                log.info(
                    f"Epoch {epoch}/{args.epochs} — "
                    f"train={train_loss:.4f}  val={val_loss:.4f}  lr={current_lr:.2e}"
                )
            else:
                log.info(
                    f"Epoch {epoch}/{args.epochs} — train={train_loss:.4f}  lr={current_lr:.2e}"
                )

            if wandb is not None and wandb.run is not None:
                raw = _raw_model(model)
                epoch_metrics = {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/lr": current_lr,
                    "train/logit_scale": raw.logit_scale.exp().item(),
                }
                if val_loss is not None:
                    epoch_metrics["val/loss"] = val_loss
                wandb.log(epoch_metrics, step=global_step)

            monitor_loss = val_loss if val_loss is not None else train_loss
            if (monitor_loss < best_val_loss) or epoch == 1:
                best_val_loss = monitor_loss
                best_epoch = epoch
                best_path = output_dir / "best_adapter.pt"
                raw = _raw_model(model)
                adapter_state = {k: v for k, v in raw.state_dict().items() if "lora_" in k}
                torch.save(adapter_state, best_path)
                log.info(
                    f"  ↑ New best ({'val' if val_loss is not None else 'train'})"
                    f" loss={best_val_loss:.4f} → {best_path}"
                )
                tmp = copy.deepcopy(raw).cpu()
                merged = tmp.merge_and_unload()
                merged_path = output_dir / "final_merged.pt"
                torch.save(merged.state_dict(), merged_path)
                log.info(f"  Merged best-epoch model saved → {merged_path}")
                del tmp, merged
                #delete all adapter checkpoints except the best one to save space
                for ckpt in output_dir.glob("epoch_*_adapter.pt"):
                    if ckpt != best_path:
                        ckpt.unlink()
                log.info(f"  Deleted old adapter checkpoints, keeping only the best one: {best_path}")

            if epoch % args.save_frequency == 0:
                ckpt_path = output_dir / f"epoch_{epoch}_adapter.pt"
                raw = _raw_model(model)
                adapter_state = {k: v for k, v in raw.state_dict().items() if "lora_" in k}
                torch.save(adapter_state, ckpt_path)
                log.info(f"Adapter checkpoint saved → {ckpt_path}")

        # Early stopping: decide on rank 0, then broadcast so all ranks exit together
        should_stop = False
        if is_main and early_stopper is not None and val_loss is not None:
            should_stop = early_stopper.step(val_loss)
            if should_stop:
                log.info(f"Early stopping triggered at epoch {epoch} (best was epoch {best_epoch})")
        if world_size > 1:
            stop_tensor = torch.tensor(int(should_stop), device=device)
            dist.broadcast(stop_tensor, src=0)
            should_stop = bool(stop_tensor.item())
        if should_stop:
            break

        if world_size > 1:
            dist.barrier()

    if is_main:
        log.info(f"Training complete. Best epoch: {best_epoch}  best loss: {best_val_loss:.4f}")

    # ── Merge LoRA into base model and save (rank 0 only) ─────────────────────
    if is_main:
        merged_path = output_dir / "final_merged.pt"
        if merged_path.exists():
            log.info(f"final_merged.pt already up-to-date (best epoch {best_epoch}).")
        else:
            # Fallback: reached only when no best epoch was ever saved (e.g. no validation
            # and training was cut before any improvement over the initial inf sentinel).
            raw = _raw_model(model)
            log.info("Merging LoRA weights into base model …")
            merged_model = raw.merge_and_unload()
            torch.save(merged_model.state_dict(), merged_path)
            log.info(f"Merged model saved → {merged_path}")
        log.info("Done.")
        
    if is_main:
        for ckpt in output_dir.glob("epoch_*_adapter.pt"):
            ckpt.unlink()

    if is_main and wandb is not None and wandb.run is not None:
        wandb.finish()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
