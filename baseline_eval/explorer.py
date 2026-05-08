"""
Interactive CXR retrieval explorer.

Type any free-form query and see what each model returns, with CheXpert labels
shown on every retrieved image so you can judge relevance at a glance.

Usage
-----
    python baseline_eval/explorer.py \
        --paired_dir eval_outputs/exp1/paired_data \
        --emb_dir    eval_outputs/exp1 \
        --csv        cxr_data/all_txt_data_and_labels.csv \
        --port       5050
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, request, send_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
CXRCLIP_DIR = REPO_ROOT / "cxr-clip"
CHECKPOINTS_DIR = REPO_ROOT / "more_models_to_try"

sys.path.insert(0, str(REPO_ROOT / "baseline_eval"))
from eval_model import (  # noqa: E402
    OpenCLIPBackend,
    FinetunedOpenCLIPBackend,
    CXRClipBackend,
    _normalize,
)

CHEXPERT_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax",
]
LABEL_COLS = [f"chexpert_{l}" for l in CHEXPERT_LABELS]

# Each entry maps a .npy filename prefix to how to build its text encoder.
# "emb_key" must match the stem of the .npy file in emb_dir.
MODEL_CONFIGS = [
    {
        "name": "vanilla_clip",
        "emb_key": "img_emb_vanilla_clip",
        "loader": lambda dev: OpenCLIPBackend("ViT-B-32", "openai", dev),
    },
    {
        "name": "biomedclip",
        "emb_key": "img_emb_biomedclip",
        "loader": lambda dev: OpenCLIPBackend(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", "", dev
        ),
    },
    {
        "name": "cxrclip_r50_mc",
        "emb_key": "img_emb_cxrclip_r50_mc",
        "loader": lambda dev: CXRClipBackend(str(CHECKPOINTS_DIR / "r50_mc.pt"), dev),
    },
    {
        "name": "cxrclip_swint_mc",
        "emb_key": "img_emb_cxrclip_swint_mc",
        "loader": lambda dev: CXRClipBackend(str(CHECKPOINTS_DIR / "swint_mc.pt"), dev),
    },
    {
        "name": "lora_vitb32_smoke",
        "emb_key": "img_emb_finetuned_final_merged",
        "loader": lambda dev: FinetunedOpenCLIPBackend(
            "ViT-B-32", "openai",
            str(REPO_ROOT / "experiments/lora_vitb32_smoke/final_merged.pt"),
            dev,
        ),
    },
]

# ── App state (filled at startup) ─────────────────────────────────────────────

app = Flask(__name__)
state: dict = {}   # populated in main()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    model_names = [m["name"] for m in state["models"]]
    return HTML_PAGE.replace("__MODEL_NAMES__", json.dumps(model_names))


@app.get("/image/<dicom_id>")
def serve_image(dicom_id):
    path = state["paired_dir"] / f"{dicom_id}.jpg"
    if not path.exists():
        return "not found", 404
    return send_file(path, mimetype="image/jpeg")


@app.get("/report/<dicom_id>")
def serve_report(dicom_id):
    path = state["paired_dir"] / f"{dicom_id}.txt"
    if not path.exists():
        return "", 204
    return path.read_text(errors="replace"), 200, {"Content-Type": "text/plain; charset=utf-8"}


@app.post("/search")
def search():
    body = request.get_json(force=True)
    query: str = body.get("query", "").strip()
    top_k: int = min(int(body.get("top_k", 10)), 50)

    if not query:
        return jsonify({"error": "empty query"}), 400

    results = {}
    for m in state["models"]:
        name = m["name"]
        backend = m["backend"]
        img_emb = m["img_emb"]   # (N, D) normalised

        try:
            q_emb = _normalize(backend.encode_texts([query]))   # (1, D)
            sims = (q_emb @ img_emb.T)[0]                       # (N,)
            top_idx = np.argsort(-sims)[:top_k]

            hits = []
            for rank, idx in enumerate(top_idx, 1):
                dicom_id = state["dicom_ids"][idx]
                score = float(sims[idx])
                raw_labels = state["label_matrix"][idx]          # float32 array
                # Keep labels with value > 0 (positive), -1 shown as uncertain
                label_dict = {}
                for label, val in zip(CHEXPERT_LABELS, raw_labels):
                    if val > 0:
                        label_dict[label] = "pos"
                    elif val < 0:
                        label_dict[label] = "unc"
                hits.append({"rank": rank, "dicom_id": dicom_id,
                             "score": score, "labels": label_dict})
            results[name] = hits
        except Exception as exc:
            log.exception(f"Search failed for {name}")
            results[name] = {"error": str(exc)}

    return jsonify({"results": results})


# ── HTML page ──────────────────────────────────────────────────────────────────

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CXR Retrieval Explorer</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; }

  header {
    position: sticky; top: 0; z-index: 100;
    background: #1a1d27; border-bottom: 1px solid #2d3148;
    padding: 14px 20px; display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
  }
  header h1 { font-size: 1rem; font-weight: 600; color: #7c84ff; white-space: nowrap; }
  #query { flex: 1; min-width: 200px; padding: 8px 12px; border-radius: 8px;
    border: 1px solid #3a3f5c; background: #252836; color: #e2e8f0; font-size: 0.95rem; }
  #query:focus { outline: none; border-color: #7c84ff; }
  select { padding: 8px 10px; border-radius: 8px; border: 1px solid #3a3f5c;
    background: #252836; color: #e2e8f0; font-size: 0.9rem; cursor: pointer; }
  button#searchBtn {
    padding: 8px 22px; border-radius: 8px; border: none;
    background: #7c84ff; color: #fff; font-size: 0.95rem; font-weight: 600;
    cursor: pointer; transition: background .15s;
  }
  button#searchBtn:hover { background: #9499ff; }
  button#searchBtn:disabled { background: #3a3f5c; cursor: default; }

  #status { padding: 20px; text-align: center; color: #64748b; font-size: 0.9rem; }

  .grid-wrap { overflow-x: auto; padding: 16px; }
  .grid { display: flex; gap: 12px; align-items: flex-start; min-width: max-content; }

  .model-col { width: 190px; }
  .model-header {
    background: #1e2136; border: 1px solid #2d3148; border-radius: 8px;
    padding: 8px 10px; text-align: center; font-size: 0.78rem; font-weight: 600;
    color: #a0a8ff; margin-bottom: 8px; word-break: break-all;
  }

  .hit {
    background: #1a1d27; border: 1px solid #2d3148; border-radius: 8px;
    margin-bottom: 8px; overflow: hidden; cursor: pointer;
    transition: border-color .15s, transform .1s;
  }
  .hit:hover { border-color: #7c84ff; transform: translateY(-1px); }

  .hit img { width: 100%; aspect-ratio: 1; object-fit: cover; display: block; }

  .hit-meta { padding: 6px 8px; }
  .hit-score { font-size: 0.72rem; color: #94a3b8; margin-bottom: 4px; }
  .labels { display: flex; flex-wrap: wrap; gap: 3px; }
  .label { font-size: 0.62rem; padding: 2px 5px; border-radius: 4px; font-weight: 500; }
  .label.pos { background: #14532d; color: #86efac; }
  .label.unc { background: #451a03; color: #fdba74; }

  /* Modal */
  #modal {
    display: none; position: fixed; inset: 0; z-index: 200;
    background: rgba(0,0,0,.85); align-items: center; justify-content: center;
  }
  #modal.open { display: flex; }
  #modal-inner {
    background: #1a1d27; border: 1px solid #3a3f5c; border-radius: 12px;
    max-width: 680px; width: 95%; max-height: 92vh; overflow-y: auto; padding: 16px;
    position: relative;
  }
  #modal-close {
    position: absolute; top: 10px; right: 12px;
    background: none; border: none; color: #94a3b8; font-size: 1.4rem; cursor: pointer;
  }
  #modal img { width: 100%; border-radius: 8px; margin-bottom: 12px; }
  #modal-labels { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 12px; }
  #modal-report {
    font-size: 0.8rem; color: #94a3b8; white-space: pre-wrap;
    background: #111420; border-radius: 6px; padding: 10px; max-height: 220px;
    overflow-y: auto; font-family: monospace;
  }
  #modal-title { font-size: 0.78rem; color: #64748b; margin-bottom: 10px; word-break: break-all; }
</style>
</head>
<body>
<header>
  <h1>CXR Explorer</h1>
  <input id="query" type="text" placeholder="e.g. pleural effusion, cardiomegaly and edema, …" autofocus>
  <label style="font-size:.85rem;color:#94a3b8">Top
    <select id="topk">
      <option value="5">5</option>
      <option value="10" selected>10</option>
      <option value="20">20</option>
    </select>
  </label>
  <button id="searchBtn">Search</button>
</header>

<div id="status">Enter a query above and press Search (or Enter).</div>
<div class="grid-wrap" id="gridWrap" style="display:none">
  <div class="grid" id="grid"></div>
</div>

<!-- Modal -->
<div id="modal">
  <div id="modal-inner">
    <button id="modal-close">&times;</button>
    <div id="modal-title"></div>
    <img id="modal-img" src="" alt="">
    <div id="modal-labels"></div>
    <pre id="modal-report">Loading report…</pre>
  </div>
</div>

<script>
const MODEL_NAMES = __MODEL_NAMES__;

const $ = id => document.getElementById(id);
const queryEl  = $("query");
const btn      = $("searchBtn");
const topkEl   = $("topk");
const statusEl = $("status");
const gridWrap = $("gridWrap");
const gridEl   = $("grid");

queryEl.addEventListener("keydown", e => { if (e.key === "Enter") doSearch(); });
btn.addEventListener("click", doSearch);
$("modal-close").addEventListener("click", closeModal);
$("modal").addEventListener("click", e => { if (e.target === $("modal")) closeModal(); });

async function doSearch() {
  const query = queryEl.value.trim();
  if (!query) return;
  btn.disabled = true;
  statusEl.textContent = "Searching…";
  gridWrap.style.display = "none";

  try {
    const resp = await fetch("/search", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({query, top_k: parseInt(topkEl.value)}),
    });
    const data = await resp.json();
    if (data.error) { statusEl.textContent = "Error: " + data.error; return; }
    renderResults(data.results, query);
  } catch(e) {
    statusEl.textContent = "Request failed: " + e;
  } finally {
    btn.disabled = false;
  }
}

function renderResults(results, query) {
  gridEl.innerHTML = "";
  const topk = parseInt(topkEl.value);

  for (const modelName of MODEL_NAMES) {
    const hits = results[modelName];
    const col = document.createElement("div");
    col.className = "model-col";

    const hdr = document.createElement("div");
    hdr.className = "model-header";
    hdr.textContent = modelName;
    col.appendChild(hdr);

    if (!hits || hits.error) {
      const err = document.createElement("div");
      err.style.cssText = "color:#f87171;font-size:.75rem;padding:8px";
      err.textContent = hits?.error ?? "no results";
      col.appendChild(err);
    } else {
      for (const hit of hits) {
        col.appendChild(buildHitCard(hit));
      }
    }
    gridEl.appendChild(col);
  }

  statusEl.textContent = `Results for "${query}"`;
  gridWrap.style.display = "";
}

function buildHitCard(hit) {
  const card = document.createElement("div");
  card.className = "hit";
  card.addEventListener("click", () => openModal(hit));

  const img = document.createElement("img");
  img.src = `/image/${hit.dicom_id}`;
  img.alt = hit.dicom_id;
  img.loading = "lazy";
  card.appendChild(img);

  const meta = document.createElement("div");
  meta.className = "hit-meta";

  const score = document.createElement("div");
  score.className = "hit-score";
  score.textContent = `#${hit.rank}  sim ${hit.score.toFixed(3)}`;
  meta.appendChild(score);

  const labelsDiv = document.createElement("div");
  labelsDiv.className = "labels";
  for (const [label, kind] of Object.entries(hit.labels)) {
    const pill = document.createElement("span");
    pill.className = `label ${kind}`;
    pill.textContent = label;
    labelsDiv.appendChild(pill);
  }
  meta.appendChild(labelsDiv);
  card.appendChild(meta);
  return card;
}

function openModal(hit) {
  $("modal-img").src = `/image/${hit.dicom_id}`;
  $("modal-title").textContent = hit.dicom_id + `  (sim ${hit.score.toFixed(4)})`;

  const labelsDiv = $("modal-labels");
  labelsDiv.innerHTML = "";
  for (const [label, kind] of Object.entries(hit.labels)) {
    const pill = document.createElement("span");
    pill.className = `label ${kind}`;
    pill.style.fontSize = ".8rem";
    pill.textContent = label;
    labelsDiv.appendChild(pill);
  }
  if (!Object.keys(hit.labels).length) {
    labelsDiv.innerHTML = '<span style="color:#64748b;font-size:.8rem">No positive labels</span>';
  }

  $("modal-report").textContent = "Loading report…";
  $("modal").classList.add("open");

  fetch(`/report/${hit.dicom_id}`)
    .then(r => r.text())
    .then(t => { $("modal-report").textContent = t || "(no report available)"; })
    .catch(() => { $("modal-report").textContent = "(failed to load report)"; });
}

function closeModal() {
  $("modal").classList.remove("open");
  $("modal-img").src = "";
}
</script>
</body>
</html>
"""


# ── Startup ────────────────────────────────────────────────────────────────────

def load_models(emb_dir: Path, device: torch.device) -> list[dict]:
    """Load embeddings + text encoders for all configured models."""
    loaded = []
    for cfg in MODEL_CONFIGS:
        emb_path = emb_dir / f"{cfg['emb_key']}.npy"
        if not emb_path.exists():
            log.warning(f"Embedding file not found, skipping {cfg['name']}: {emb_path}")
            continue

        log.info(f"Loading embeddings for {cfg['name']} …")
        img_emb = np.load(emb_path).astype(np.float32)

        log.info(f"Loading text encoder for {cfg['name']} …")
        try:
            backend = cfg["loader"](device)
        except Exception as exc:
            log.error(f"Failed to load backend for {cfg['name']}: {exc}")
            continue

        loaded.append({"name": cfg["name"], "img_emb": img_emb, "backend": backend})
        log.info(f"Ready: {cfg['name']}  embeddings={img_emb.shape}")

    return loaded


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paired_dir", required=True,
                        help="Folder with *.jpg images (output of build_baseline.py)")
    parser.add_argument("--emb_dir", required=True,
                        help="Directory containing img_emb_*.npy cache files")
    parser.add_argument("--csv", required=True,
                        help="Path to all_txt_data_and_labels.csv")
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()

    paired_dir = Path(args.paired_dir)
    emb_dir    = Path(args.emb_dir)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Image index
    image_paths = sorted(paired_dir.glob("*.jpg"))
    dicom_ids   = [p.stem for p in image_paths]
    log.info(f"Gallery: {len(dicom_ids):,} images")

    # Labels
    log.info("Loading label CSV …")
    df = pd.read_csv(args.csv, usecols=["metadata_dicom_id"] + LABEL_COLS)
    df = df.set_index("metadata_dicom_id")
    # Keep raw values (positive=1, uncertain=-1, negative=0/NaN)
    label_matrix = df.reindex(dicom_ids)[LABEL_COLS].fillna(0).values.astype(np.float32)

    # Models
    models = load_models(emb_dir, device)
    if not models:
        log.error("No models loaded — check your --emb_dir and checkpoints.")
        sys.exit(1)

    state["paired_dir"]   = paired_dir
    state["dicom_ids"]    = dicom_ids
    state["label_matrix"] = label_matrix
    state["models"]       = models

    log.info(f"Starting explorer on http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
