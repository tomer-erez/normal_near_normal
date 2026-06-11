"""
Unified text→image retrieval evaluation for multiple CLIP model backends.

Encodes images directly from paired_dir (no pre-computed embeddings needed).
Evaluates CheXpert label queries in three modes:

  single    13 queries  "atelectasis"
              relevant = label == 1

  pair      78 queries  "atelectasis and edema"
              relevant = both labels == 1

  negative  156 queries "atelectasis and no cardiomegaly"
              relevant = pos label == 1  AND  neg label == 0 or NaN
              (label=0 or NaN both mean the pathology is absent/unconfirmed)

  all       (default) runs single + pair + negative

Supported models
----------------
  vanilla_clip  : ViT-B/32 from OpenAI (baseline)
  biomedclip    : microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
  cxrclip       : CXR-CLIP checkpoint (.tar) from cxr-clip repo
  finetuned     : our merged LoRA checkpoint (final_merged.pt)
"""

import argparse
import logging
import sys
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
CXRCLIP_DIR = REPO_ROOT / "cxr-clip"
sys.path.insert(0, str(REPO_ROOT))

from constants import CHEXPERT_LABELS, LABEL_COLS


# ── Dataset ───────────────────────────────────────────────────────────────────

class ImageFolderDataset(Dataset):
    def __init__(self, image_paths: list[Path], transform):
        self.paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


# ── Model backends ────────────────────────────────────────────────────────────

class OpenCLIPBackend:
    """Covers vanilla CLIP (openai weights) and BiomedCLIP (hf-hub weights)."""

    def __init__(self, model_name: str, pretrained: str, device: torch.device):
        import open_clip
        self.device = device
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        model.eval()
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def encode_images(self, image_paths: list[Path], batch_size: int = 32) -> np.ndarray:
        dataset = ImageFolderDataset(image_paths, self.preprocess)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
        all_embs = []
        with torch.no_grad():
            for batch in loader:
                emb = self.model.encode_image(batch.to(self.device)).float().cpu().numpy()
                all_embs.append(emb)
        return _normalize(np.concatenate(all_embs, axis=0))

    def encode_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                tokens = self.tokenizer(texts[i : i + batch_size]).to(self.device)
                emb = self.model.encode_text(tokens).float().cpu().numpy()
                all_embs.append(emb)
        return _normalize(np.concatenate(all_embs, axis=0))


class FinetunedOpenCLIPBackend(OpenCLIPBackend):
    """OpenCLIP model loaded from a merged LoRA checkpoint (final_merged.pt)."""

    def __init__(self, base_model: str, pretrained: str, checkpoint_path: str, device: torch.device):
        super().__init__(base_model, pretrained or None, device)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        log.info(f"Loaded fine-tuned weights from {checkpoint_path}")


class CXRHybridBackend:
    """
    Hybrid model: frozen CXR-CLIP image encoder + fine-tuned OpenCLIP text encoder.
    Loads a merged checkpoint produced by train_lora.py with --cxrclip-checkpoint.
    """

    def __init__(
        self,
        cxrclip_image_checkpoint: str,
        hybrid_merged_checkpoint: str,
        text_model_name: str,
        text_pretrained: str,
        device: torch.device,
    ):
        sys.path.insert(0, str(REPO_ROOT))
        import open_clip
        from train.cxrclip_hybrid_model import CXRClipHybridModel

        self.device = device

        pretrained = text_pretrained if text_pretrained else None
        openclip_model, _, _ = open_clip.create_model_and_transforms(
            text_model_name, pretrained=pretrained, device="cpu"
        )
        self.tokenizer = open_clip.get_tokenizer(text_model_name)

        model = CXRClipHybridModel(cxrclip_image_checkpoint, openclip_model)
        del openclip_model

        state = torch.load(hybrid_merged_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        self.model = model

        self.preprocess = CXRClipHybridModel.make_preprocess(cxrclip_image_checkpoint)
        log.info(f"Loaded hybrid model from {hybrid_merged_checkpoint}")

    def encode_images(self, image_paths: list[Path], batch_size: int = 64) -> np.ndarray:
        dataset = ImageFolderDataset(image_paths, self.preprocess)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
        all_embs = []
        with torch.no_grad():
            for batch in loader:
                emb = self.model.encode_image(batch.to(self.device), normalize=False).float().cpu().numpy()
                all_embs.append(emb)
        return _normalize(np.concatenate(all_embs, axis=0))

    def encode_texts(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                tokens = self.tokenizer(texts[i : i + batch_size]).to(self.device)
                emb = self.model.encode_text(tokens, normalize=False).float().cpu().numpy()
                all_embs.append(emb)
        return _normalize(np.concatenate(all_embs, axis=0))


class CXRClipBackend:
    """Loads a CXR-CLIP checkpoint (.pt) from the cxr-clip repo."""

    def __init__(self, checkpoint_path: str, device: torch.device):
        sys.path.insert(0, str(CXRCLIP_DIR))
        from cxrclip.data.data_utils import load_tokenizer
        from cxrclip.model import build_model

        self.device = device
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]

        self.text_max_length = cfg.get("base", {}).get("text_max_length", 256)

        from transformers import AutoTokenizer
        tokenizer_name = cfg["tokenizer"]["pretrained_model_name_or_path"]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        default_cache = str(Path.home() / ".cache" / "huggingface" / "hub")
        model_cfg = dict(cfg["model"])
        model_cfg["text_encoder"] = dict(model_cfg["text_encoder"])
        model_cfg["text_encoder"]["cache_dir"] = default_cache
        model_cfg["text_encoder"]["local_files_only"] = False
        # Don't load pretrained HuggingFace weights — the CXR-CLIP checkpoint
        # contains all the weights; from_pretrained is only needed for the arch.
        model_cfg["text_encoder"]["pretrained"] = False

        self.model = build_model(
            model_config=model_cfg,
            loss_config=cfg["loss"],
            tokenizer=self.tokenizer,
        )
        self.model.load_state_dict(ckpt["model"], strict=False)
        self.model.to(device).eval()

        encoder_name = cfg["model"]["image_encoder"]["name"]
        if encoder_name == "resnet":
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        else:
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def encode_images(self, image_paths: list[Path], batch_size: int = 64) -> np.ndarray:
        dataset = ImageFolderDataset(image_paths, self.preprocess)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
        all_embs = []
        with torch.no_grad():
            for batch in loader:
                feat = self.model.encode_image(batch.to(self.device))
                if self.model.projection:
                    feat = self.model.image_projection(feat)
                all_embs.append(feat.float().cpu().numpy())
        return _normalize(np.concatenate(all_embs, axis=0))

    def encode_texts(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                tokens = self.tokenizer(
                    texts[i : i + batch_size],
                    padding="longest",
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.text_max_length,
                ).to(self.device)
                feat = self.model.encode_text(tokens)
                if self.model.projection:
                    feat = self.model.text_projection(feat)
                all_embs.append(feat.float().cpu().numpy())
        return _normalize(np.concatenate(all_embs, axis=0))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms == 0, 1.0, norms)


NEG_TEMPLATE_DEFAULT = "{pos} and no {neg}"
NEG_TEMPLATE_ROBUST  = "an image with {pos} but without {neg}"


def build_queries(query_mode: str = "all",
                  neg_template: str = NEG_TEMPLATE_DEFAULT) -> list[dict]:
    """
    Build the list of retrieval queries for evaluation.

    Each query dict has:
      query          — text sent to the model encoder
      type           — "single" | "pair" | "negative"
      pos_label_cols — LABEL_COLS that must == 1 in relevant images
      neg_label_cols — LABEL_COLS that must == 0 in relevant images (negative mode only)

    Modes:
      single   13 queries  "atelectasis"
      pair     78 queries  "atelectasis and edema"
      negative 156 queries — phrasing controlled by neg_template
      all      all of the above (247 total)

    neg_template: Python format string with {pos} and {neg} placeholders.
      Default:  "{pos} and no {neg}"           e.g. "atelectasis and no edema"
      Robust:   "an image with {pos} but without {neg}"
    """
    queries = []

    if query_mode in ("single", "all"):
        for label in CHEXPERT_LABELS:
            queries.append({
                "query": label.lower(),
                "type": "single",
                "pos_label_cols": [f"chexpert_{label}"],
                "neg_label_cols": [],
            })

    if query_mode in ("pair", "all"):
        for l1, l2 in combinations(CHEXPERT_LABELS, 2):
            queries.append({
                "query": f"{l1.lower()} and {l2.lower()}",
                "type": "pair",
                "pos_label_cols": [f"chexpert_{l1}", f"chexpert_{l2}"],
                "neg_label_cols": [],
            })

    if query_mode in ("negative", "all"):
        for l_pos, l_neg in permutations(CHEXPERT_LABELS, 2):
            queries.append({
                "query": neg_template.format(pos=l_pos.lower(), neg=l_neg.lower()),
                "type": "negative",
                "pos_label_cols": [f"chexpert_{l_pos}"],
                "neg_label_cols": [f"chexpert_{l_neg}"],
            })

    return queries


def retrieve_topk(query_emb: np.ndarray, gallery_emb: np.ndarray, k: int) -> np.ndarray:
    sim = query_emb @ gallery_emb.T
    n_gallery = sim.shape[1]
    if k >= n_gallery:
        # Gallery smaller than k — return all indices sorted by similarity
        return np.argsort(-sim, axis=1)
    part = np.argpartition(-sim, k, axis=1)[:, :k]
    rows = np.arange(part.shape[0])[:, None]
    order = np.argsort(-sim[rows, part], axis=1)
    return part[rows, order]


def precision_at_k(relevant: np.ndarray, k: int, n_relevant: int) -> float:
    if n_relevant == 0:
        return float("nan")
    return float(relevant[:k].mean())


def recall_at_k(relevant: np.ndarray, k: int, n_relevant: int) -> float:
    if n_relevant == 0:
        return float("nan")
    return float(relevant[:k].sum() / n_relevant)


def hnrr_at_k(hard_neg_mask: np.ndarray, k: int) -> float:
    """Fraction of top-k retrieved images that are hard negatives (A=1 AND B=1)."""
    return float(hard_neg_mask[:k].mean())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP model variants on CheXpert label retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model_type", required=True,
                        choices=["vanilla_clip", "biomedclip", "cxrclip", "finetuned",
                                 "cxrclip_hybrid"],
                        help="Which model backend to use")
    parser.add_argument("--paired_dir", required=True,
                        help="Folder with *.jpg symlinks (output of build_baseline.py)")
    parser.add_argument("--csv", required=True,
                        help="Path to all_txt_data_and_labels.csv")
    parser.add_argument("--query_mode", default="all",
                        choices=["single", "pair", "negative", "all"],
                        help="Which query types to evaluate. "
                             "'single': 13 single-label queries. "
                             "'pair': 78 two-label AND queries. "
                             "'negative': 156 'yes A and no B' queries (uses label=0 in CSV). "
                             "'all': all 247 queries (default).")
    parser.add_argument("--cxrclip_checkpoint", default=None,
                        help="Path to CXR-CLIP .tar checkpoint (required for --model_type cxrclip)")
    parser.add_argument("--finetuned_base_model", default=None,
                        help="open_clip model name for --model_type finetuned, e.g. ViT-B-32")
    parser.add_argument("--finetuned_pretrained", default="",
                        help="open_clip pretrained tag for --model_type finetuned, e.g. openai")
    parser.add_argument("--finetuned_checkpoint", default=None,
                        help="Path to final_merged.pt (required for --model_type finetuned)")
    parser.add_argument("--cxrclip_image_checkpoint", default=None,
                        help="Path to original CXR-CLIP checkpoint (.pt) used as image encoder "
                             "(required for --model_type cxrclip_hybrid)")
    parser.add_argument("--hybrid_merged_checkpoint", default=None,
                        help="Path to final_merged.pt from train_lora.py with --cxrclip-checkpoint "
                             "(required for --model_type cxrclip_hybrid)")
    parser.add_argument("--hybrid_text_model", default="ViT-B-32",
                        help="OpenCLIP model name used as text encoder in the hybrid model")
    parser.add_argument("--hybrid_text_pretrained", default="openai",
                        help="OpenCLIP pretrained tag for the hybrid text encoder")
    parser.add_argument("--name", default=None,
                        help="Override output filename stem, e.g. 'lora_vitb32'. "
                             "Saves to results_{name}.csv. Defaults to model_type.")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap the gallery to the first N images (useful for quick smoke tests)")
    parser.add_argument("--nan-mode", default="negative", choices=["negative", "ignore"],
                        help="How to treat NaN (not-mentioned) labels when deciding relevance for "
                             "negative queries. "
                             "'negative' (default): NaN and CSV 0 are both considered absent — "
                             "an image is relevant to 'X and no Y' if Y==0 OR Y is NaN. "
                             "'ignore': only CSV 0 counts as absent — NaN images are excluded "
                             "from both relevant and irrelevant sets for that label.")
    parser.add_argument("--neg-template", default=NEG_TEMPLATE_DEFAULT,
                        help="Format string for negative query text. "
                             "Placeholders: {pos} = positive label, {neg} = negated label. "
                             f"Default: '{NEG_TEMPLATE_DEFAULT}'. "
                             f"Robustness template: '{NEG_TEMPLATE_ROBUST}'.")
    args = parser.parse_args()

    if args.model_type == "cxrclip" and not args.cxrclip_checkpoint:
        parser.error("--cxrclip_checkpoint is required when --model_type is cxrclip")
    if args.model_type == "finetuned" and not (args.finetuned_base_model and args.finetuned_checkpoint):
        parser.error("--finetuned_base_model and --finetuned_checkpoint are required for --model_type finetuned")
    if args.model_type == "cxrclip_hybrid" and not (args.cxrclip_image_checkpoint and args.hybrid_merged_checkpoint):
        parser.error("--cxrclip_image_checkpoint and --hybrid_merged_checkpoint are required for --model_type cxrclip_hybrid")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Load model backend ────────────────────────────────────────────────────
    log.info(f"Loading model: {args.model_type} …")
    if args.model_type == "vanilla_clip":
        backend = OpenCLIPBackend("ViT-B-32", "openai", device)
    elif args.model_type == "biomedclip":
        backend = OpenCLIPBackend(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            pretrained="",
            device=device,
        )
    elif args.model_type == "finetuned":
        backend = FinetunedOpenCLIPBackend(
            args.finetuned_base_model,
            args.finetuned_pretrained,
            args.finetuned_checkpoint,
            device,
        )
    elif args.model_type == "cxrclip_hybrid":
        backend = CXRHybridBackend(
            args.cxrclip_image_checkpoint,
            args.hybrid_merged_checkpoint,
            args.hybrid_text_model,
            args.hybrid_text_pretrained,
            device,
        )
    else:
        backend = CXRClipBackend(args.cxrclip_checkpoint, device)

    # ── Load image paths from paired_dir ──────────────────────────────────────
    paired_dir = Path(args.paired_dir)
    image_paths = sorted(paired_dir.glob("*.jpg"))
    if args.max_samples is not None:
        image_paths = image_paths[: args.max_samples]
    dicom_ids = [p.stem for p in image_paths]
    n_total = len(image_paths)
    log.info(f"Found {n_total:,} images in paired_dir (max_samples={args.max_samples})")

    # ── Encode images (with disk cache) ──────────────────────────────────────
    ms_tag = f"_n{args.max_samples}" if args.max_samples is not None else ""
    if args.cxrclip_checkpoint:
        cache_name = f"img_emb_{args.model_type}_{Path(args.cxrclip_checkpoint).stem}{ms_tag}.npy"
    elif args.model_type == "cxrclip_hybrid":
        ckpt = Path(args.hybrid_merged_checkpoint)
        img_stem = Path(args.cxrclip_image_checkpoint).stem
        cache_name = f"img_emb_hybrid_{img_stem}_{ckpt.parent.name}_{ckpt.stem}{ms_tag}.npy"
    elif args.model_type == "finetuned":
        ckpt = Path(args.finetuned_checkpoint)
        # Use parent dir name + stem so different experiments don't collide on "final_merged"
        cache_name = f"img_emb_finetuned_{ckpt.parent.name}_{ckpt.stem}{ms_tag}.npy"
    else:
        cache_name = f"img_emb_{args.model_type}{ms_tag}.npy"
    cache_path = paired_dir.parent / cache_name

    if cache_path.exists():
        log.info(f"Loading cached image embeddings from {cache_path}")
        img_emb = np.load(cache_path)
    else:
        log.info("Encoding images …")
        img_emb = backend.encode_images(image_paths, batch_size=args.batch_size)
        np.save(cache_path, img_emb)
        log.info(f"Saved image embeddings → {cache_path}")
    log.info(f"Image embeddings: {img_emb.shape}")

    # ── Load CheXpert labels ──────────────────────────────────────────────────
    # Keep raw values: 1=positive, 0=explicit negative, -1=uncertain, NaN=not mentioned.
    # We need the 0 vs NaN distinction for "negative" query mode.
    log.info("Loading CheXpert labels …")
    df = pd.read_csv(args.csv, usecols=["metadata_dicom_id"] + LABEL_COLS)
    df = df.set_index("metadata_dicom_id")
    label_matrix = df.reindex(dicom_ids)[LABEL_COLS].values.astype(float)
    # label_matrix[i, j]: 1.0=positive, 0.0=explicit neg, NaN=not mentioned, -1.0=uncertain
    log.info(f"Label matrix shape: {label_matrix.shape}")

    # ── Build and encode queries ──────────────────────────────────────────────
    queries = build_queries(args.query_mode, neg_template=args.neg_template)
    log.info(f"Query mode: '{args.query_mode}'  →  {len(queries)} queries")
    query_strings = [q["query"] for q in queries]
    query_emb = backend.encode_texts(query_strings)

    # ── Retrieve and evaluate ─────────────────────────────────────────────────
    max_k = max(args.ks)
    ks = sorted(args.ks)
    log.info(f"Retrieving top-{max_k} for each query …")
    top_k = retrieve_topk(query_emb, img_emb, k=max_k)

    rows = []
    for i, qdef in enumerate(queries):
        pos_indices = [LABEL_COLS.index(c) for c in qdef["pos_label_cols"]]
        neg_indices = [LABEL_COLS.index(c) for c in qdef["neg_label_cols"]]

        # Positive condition: all required positive labels must equal 1
        pos_ok = (label_matrix[:, pos_indices] == 1.0).all(axis=1)

        if neg_indices:
            neg_cols = label_matrix[:, neg_indices]
            if args.nan_mode == "negative":
                # NaN (not mentioned) and CSV 0 (explicitly ruled out) both count as absent.
                # CSV -1 (uncertain) does NOT count as absent.
                neg_ok = ((neg_cols == 0.0) | np.isnan(neg_cols)).all(axis=1)
            else:
                # Only CSV 0 (explicitly ruled out) counts as absent.
                # NaN images are not considered relevant for this negative label.
                neg_ok = (neg_cols == 0.0).all(axis=1)
            relevant_mask = pos_ok & neg_ok
        else:
            relevant_mask = pos_ok

        n_relevant = int(relevant_mask.sum())
        retrieved_relevant = relevant_mask[top_k[i]]

        # Hard negative: pos label present (A=1) AND neg label also present (B=1)
        if neg_indices:
            hard_neg_mask = (
                pos_ok & (label_matrix[:, neg_indices] == 1.0).all(axis=1)
            )[top_k[i]]
        else:
            hard_neg_mask = np.zeros(len(top_k[i]), dtype=bool)

        row = {
            "query": qdef["query"],
            "type": qdef["type"],
            "n_relevant": n_relevant,
            "pos_labels": ",".join(qdef["pos_label_cols"]),
            "neg_labels": ",".join(qdef["neg_label_cols"]),
        }
        for k in ks:
            row[f"P@{k}"] = precision_at_k(retrieved_relevant, k, n_relevant)
            row[f"R@{k}"] = recall_at_k(retrieved_relevant, k, n_relevant)
            row[f"HNRR@{k}"] = hnrr_at_k(hard_neg_mask, k)
        rows.append(row)

    results_df = pd.DataFrame(rows)

    # ── Print results ─────────────────────────────────────────────────────────
    metric_cols = [f"P@{k}" for k in ks] + [f"R@{k}" for k in ks]
    print("\n" + "=" * 90)
    print(f"MODEL: {args.model_type}  |  query_mode: {args.query_mode}  |  gallery: {n_total:,} images")
    print("=" * 90)

    for qtype in ["single", "pair", "negative"]:
        subset = results_df[results_df["type"] == qtype]
        if subset.empty:
            continue
        print(f"\n{'─'*90}")
        print(f"  {qtype.upper()} LABEL QUERIES  ({len(subset)})")
        print(f"{'─'*90}")
        print(subset[["query", "n_relevant"] + metric_cols].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        ))
        non_empty = subset[subset["n_relevant"] > 0]
        print(f"\n  Macro averages (queries with n_relevant > 0, n={len(non_empty)}):")
        for k in ks:
            avg_p = non_empty[f"P@{k}"].mean()
            avg_r = non_empty[f"R@{k}"].mean()
            print(f"    P@{k}: {avg_p:.4f}   R@{k}: {avg_r:.4f}")

    print("\n" + "=" * 90 + "\n")

    # ── Save ──────────────────────────────────────────────────────────────────
    if args.name:
        out_path = REPO_ROOT / f"results_{args.name}.csv"
    elif args.cxrclip_checkpoint:
        ckpt_stem = Path(args.cxrclip_checkpoint).stem
        out_path = REPO_ROOT / f"results_cxrclip_{ckpt_stem}.csv"
    else:
        out_path = REPO_ROOT / f"results_{args.model_type}.csv"
    results_df.to_csv(out_path, index=False)
    log.info(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
