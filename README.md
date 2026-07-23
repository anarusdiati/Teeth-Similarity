# Teeth-Similarity

Measuring visual similarity of **front-view teeth photos** as a foundation for IOTN
**Aesthetic Component (AC)** grading — *without* labeled data.

The idea: AC grading is a **similarity** problem, not a classification one. The IOTN AC
scale has 10 reference photos (grade 1 = most aligned -> grade 10 = most severe). A
patient photo gets graded by finding the **most similar** reference. So we don't need to
train a model from scratch — we just need good image embeddings:

```
pretrained encoder -> embedding vector -> cosine similarity to 10 anchors -> 1-NN / ranking
```

## Contents

- `teeth_similarity_intro.ipynb` — learning notebook: load -> preprocess -> toy embedding
  (why it fails) -> **DINOv2** semantic embedding -> similarity matrix -> nearest-neighbor
  -> template AC grader -> limitations & next steps.
- `front_google_00*.png` — 8 sample front-teeth photos (from Google) for exploring the
  pipeline. *Not* the official AC reference set.

## Setup

```bash
pip install torch torchvision timm scikit-learn matplotlib pillow
```

Runs on Mac CPU/MPS (Apple Silicon). DINOv2 weights (~85 MB) download once via `torch.hub`.

## Status & next steps

Current sample data is enough to **learn the pipeline**, not to build a working grader.
Still needed:

1. The **10 official IOTN AC reference photos** (anchors).
2. Consistent **ROI crop** of the teeth region (removes lips/gums/watermarks).
3. A small **labeled validation set** (~30-50 photos) to measure MAE and within-±1 accuracy.

## Model notes

- **DINOv2 ViT-S/14** — primary encoder; strong label-free embeddings, MPS-friendly.
- On-device (future): DINOv2 -> Core ML, or swap to **MobileCLIP** / **MobileNetV3**.
  The grading logic is encoder-agnostic, so the backbone can be swapped later.
