# Teeth-Similarity

Measuring visual similarity of **front-view teeth photos** as a foundation for IOTN
**Aesthetic Component (AC)** grading — *without* labeled data.

AC grading is framed as a **similarity / retrieval** problem, not classification. The IOTN AC
scale has 10 reference photos (grade 1 = most aligned -> grade 10 = most severe). A patient photo
is graded by finding the **most similar** reference. No model training required — just good
embeddings:

```
image -> ROI crop (teeth) -> pretrained encoder -> cosine similarity to 10 anchors -> grade (1-NN + weighted)
```

## Folder structure

```
Teeth-Similarity/
├── Front Teeth Google/            # 8 sample photos (Google)  -> used by the DINOv3 notebook
├── Front Teeth drg Laura/         # 18 clinical photos        -> used by the DINOv2 notebook
├── ac_references/                 # 10 IOTN AC reference anchors (ac_grade_01..10.png)
├── ac_reference_grid.png          # original AC reference grid
├── crop_references.py             # splits the grid into the 10 anchors
├── teeth_similarity_dinov2.ipynb  # main pipeline, DINOv2 encoder
├── teeth_similarity_dinov3.ipynb  # same pipeline, DINOv3 encoder (for comparison)
├── MODEL_SUMMARY.md               # objective / data / model / efficiency / benchmarking
└── README.md
```

## The two notebooks

| | `teeth_similarity_dinov2.ipynb` | `teeth_similarity_dinov3.ipynb` |
|---|---|---|
| Encoder | DINOv2 ViT-S/14 (`torch.hub`) | DINOv3 ViT-S/16 (HuggingFace `transformers`) |
| Test set | `Front Teeth drg Laura/` (18 clinical) | `Front Teeth Google/` (8 samples) |
| License | Apache-2.0 (open) | DINOv3 License (**gated** — accept terms + HF login) |

Both share an identical pipeline (ROI crop -> cosine -> `predict_ac` -> 3 probes), so results are
directly comparable. Run each **top-to-bottom (Run All)**.

## Setup

```bash
pip install -r requirements.txt
```

- `pillow-heif` is needed to read the `.heic` files in the Laura set.
- DINOv3 weights are gated: accept the license at its HuggingFace model page, then
  `huggingface-cli login` with a Read token.

Runs on Mac CPU/MPS (Apple Silicon). Encoder weights download once on first run.

## What to compare

- **Probe 3 (Spearman rho)** — does embedding distance grow with grade gap? Higher = more
  ordinal-aware = better grader.
- **Anchor separability** — the 10x10 anchor similarity heatmap.
- **Probe 1 & 2** — how shape-based / color-robust the encoder is.
- **`predict_ac`** — for the Laura set, filenames encode a weak label (normal / crowding /
  crossbite / protrusion...), a handy sanity check.

## Status & next steps

Enough to **explore and compare encoders**, not yet a validated grader. Still needed:

1. A small **dentist-graded validation set** (~30-50 photos) to compute **MAE**, **within-±1
   accuracy**, and **weighted Cohen's kappa** vs. clinician.
2. A **learned teeth detector** to replace the HSV ROI heuristic (the current fallback handles
   warm-lit photos but isn't a proper detector).
3. Optional: multi-exemplar **prototype** anchors, and metric-learning fine-tuning once labels exist.

## Model notes

- **DINOv2 ViT-S/14** — primary encoder; strong label-free k-NN features, Apache-2.0, MPS-friendly.
- **DINOv3 ViT-S/16** — larger pretraining (~1.7B images), sharper dense features; gated license.
- On-device (future): convert to **Core ML**, or swap to **MobileCLIP** / **MobileNetV3**. The
  grading logic is encoder-agnostic, so the backbone can be swapped freely.
