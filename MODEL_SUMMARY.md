# DINOv2 Teeth-Similarity — Model Summary

Short technical summary of the front-view teeth similarity / IOTN Aesthetic Component (AC)
grading approach built in `teeth_similarity_intro.ipynb`.

---

## Objective

**ML task.** Label-free **image similarity / example-based retrieval**, framed as **ordinal
grading**. A patient photo is graded by finding the most similar of the 10 IOTN AC reference
photos (k-NN over embeddings). This is *not* supervised classification — the encoder is used
frozen (zero-shot); no class labels are trained on.

**Input.** One front-view intraoral RGB photo → auto ROI-crop to the teeth region → resized to
224×224 (multiple of 14 for the ViT-14 patch grid) → ImageNet-normalized.

**Output.** Predicted AC grade **1–10**, given two ways:
- *hard* — nearest reference (1-NN), and
- *soft* — similarity-weighted grade (ordinal), plus a ranked **top-k list of reference matches**
  each with a cosine-similarity score.

**Evaluation metric.**
- *Without labels:* Spearman monotonicity of anchor distances vs. grade gap, top1–top2 confidence
  margin, prediction stability under augmentation.
- *With dentist labels:* **MAE**, **within-±1 accuracy**, and **weighted Cohen's kappa** vs. clinician.

---

## Data

**Pre-processing / augmentation / feature engineering.**
- RGBA→RGB; **teeth ROI crop** via HSV heuristic (low saturation + high value = whitish teeth →
  bounding box + padding); resize 224; ImageNet mean/std normalize.
- Feature engineering = the **DINOv2 embedding itself** (CLS token, L2-normalized → cosine-ready).
- Augmentation is optional and inference-time only (e.g. horizontal-flip **TTA** averaging). No
  training augmentation, because the model is not trained.

**Train–test condition.**
- **No training / no split** — the encoder is frozen.
- **Anchors (reference set):** 10 official IOTN AC photos, 1 exemplar per grade, cropped from the
  SCAN grid (`ac_references/`).
- **Test set:** patient front-view photos (currently 8 sample images for learning the pipeline).
- **Validation (to add):** ~30–50 dentist-graded photos to unlock the labeled metrics above.

---

## Model training

**Architecture.** **DINOv2 ViT-S/14** — a small Vision Transformer, patch size 14, ~**21M**
parameters, **384-dim** embedding. Self-supervised pretrained by Meta AI on the curated LVD-142M
dataset. Used **frozen as a feature extractor** (no fine-tuning in the current pipeline).

**Tunable parameters** (hyperparameters, since weights are frozen):
- Encoder size: `vits14` → `vitb14` (86M/768-d) → `vitl14` (300M/1024-d).
- Pooling: CLS token vs. **mean of patch tokens** vs. concat.
- ROI: `s_thr`, `v_thr`, `pad`, `dens`.
- Input resolution (224 / 294 / 336…), weighted-grade temperature, top-k, TTA on/off.
- Number of exemplars per grade → **prototype** (mean) embeddings.
- *If fine-tuning later:* triplet / ArcFace metric-learning loss on labeled pairs.

**License.** **Apache License 2.0** (code and model weights) — permissive, allows commercial and
on-device use.

**Why this model works.** DINOv2 is trained self-supervised on a very large, diverse image corpus
and learns **general-purpose, structure-aware visual features** that transfer to new domains
**without labels**. Its features are strong specifically in **k-NN / retrieval** settings (a headline
result of the DINOv2 paper), which is exactly our setup — few reference images, no training data.
The features capture shape/part arrangement, which aligns with what AC grading depends on (tooth
crowding, spacing, protrusion) once the ROI crop removes irrelevant lip/skin/lighting variation.

---

## Model efficiency & performance

**Where it runs (Apple-first).**
- **Today:** Apple Silicon Mac via PyTorch **MPS** (works now); CPU fallback also fine.
- **On-device iOS / macOS:** convert to **Core ML** (`coremltools`) → runs on GPU / **Neural
  Engine**. ViT-S/14 is small enough to be feasible on-device.
- If ViT-S is still too heavy for a target device, swap to **MobileCLIP** (Apple, native Core ML
  export) or **MobileNetV3** — the grading logic is encoder-agnostic.

**Ways to make it more efficient.**
- **Cache anchor embeddings** — the 10 references are fixed, so only **one forward pass per query**
  is needed at inference.
- **Quantization** (fp16 / int8) and Core ML / ONNX compilation.
- **Smaller backbone** or **knowledge distillation** into a MobileViT / MobileNet student.
- Lower input resolution; batch queries; use DINOv2 "with registers" variants for cleaner features.

---

## Benchmarking result summary

**Task-specific benchmark: not yet available** — there is no labeled ground truth for these photos
(this is the current blocker). What we have so far:

- **Qualitative:** adding the **ROI crop** markedly improved match plausibility (predictions went
  from clearly wrong to sensible). This is the single biggest quality lever observed.
- **Pending quantitative:** label-free diagnostics (anchor-distance monotonicity, confidence margin)
  and dentist-validated **MAE / within-±1 / weighted-kappa** on a ~30–50 image validation set.

**Context (not our task):** on ImageNet-1k, DINOv2 features are strong under simple k-NN — ViT-S/14
reaches roughly **79% top-1** with a k-NN classifier — which is why the zero-shot retrieval setup is
reasonable here. Actual AC-grading accuracy must still be measured against clinician labels, and is
inherently bounded by **inter-examiner agreement** (AC grading is subjective even among dentists, so
within-±1 is the fair target).

---

*Sources: DINOv2 GitHub repository and model card (facebookresearch/dinov2), license Apache-2.0.*
