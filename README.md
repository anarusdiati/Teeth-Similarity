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
├── depth_anything_v2_inference.ipynb   # track B-1: is monocular depth usable here?
├── depth_features_fusion.ipynb         # track B-2: depth features, confounds, fusion
├── teeth_segmentation_sam2.ipynb       # track C: per-tooth segmentation -> geometry metrics
├── depth_out/                     # depth results (CSV/JSON/PNG; *.npy gitignored)
├── seg_out/                       # segmentation results (CSV/JSON; masks.npz gitignored)
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

## Three tracks

| Track | Notebook(s) | Question it answers |
|---|---|---|
| **A — embeddings** | `teeth_similarity_dinov2/dinov3` | Can pretrained features rank AC without labels? |
| **B — depth** | `depth_anything_v2_inference`, `depth_features_fusion` | Does monocular depth add anything on intraoral photos? |
| **C — geometry** | `teeth_segmentation_sam2` | Can per-tooth outlines give interpretable, unit-free alignment metrics? |

### Track B — findings so far

Depth Anything V2 (ViT-L) passes the basic sanity probes but only barely:

| Probe | Result |
|---|---|
| Luminance vs depth correlation | mean \|r\| = **0.549** (threshold 0.6) — passes, but 6/18 photos exceed 0.6 |
| Flip stability | corr = **0.996** — passes clearly |
| Specular pixels | **19.2%** of ROI on average |

Confound testing rejected 4 of 13 depth features (`rough_mean`, `rough_p90`, `arch_range`,
`arch_curvature`) as measuring photo properties rather than teeth. **Important caveat:** a
monocular depth map is a deterministic function of the RGB image, so it cannot add information
about AC — at best it re-encodes it in a form that is easier to exploit with little data.

### Track C — geometry metrics

All metrics are divided by `w_ref` (mean central-incisor width in the same photo), so they are
**unit-free and comparable across photos** — no mm calibration, no fiducial marker required.

Key metrics: `LII_norm` (normalized Little's Irregularity Index), `incisal_rms` (smile-arc
deviation), `tilt_std` / `tilt_asym` (axial inclination), `width_asym`, `midline_dev`,
`overbite_proxy`.

Verified with synthetic arches — a perfect arch gives `LII_norm = 0.000`, rising monotonically
with induced crowding.

## Setup

```bash
pip install -r requirements.txt
```

- `pillow-heif` is needed to read the `.heic` files in the Laura set.
- DINOv3 weights are gated: accept the license at its HuggingFace model page, then
  `huggingface-cli login` with a Read token.
- SAM 2 needs no GitHub clone — it ships with `transformers`
  (`pip install -U "transformers[torch]"`, model `facebook/sam2.1-hiera-large`, not gated).
- Depth Anything V2 uses `depth-anything/Depth-Anything-V2-Large-hf`, also not gated.

Runs on Mac CPU/MPS (Apple Silicon). Encoder weights download once on first run.

## What to compare

- **Probe 3 (Spearman rho)** — does embedding distance grow with grade gap? Higher = more
  ordinal-aware = better grader.
- **Anchor separability** — the 10x10 anchor similarity heatmap.
- **Probe 1 & 2** — how shape-based / color-robust the encoder is.
- **`predict_ac`** — for the Laura set, filenames encode a weak label (normal / crowding /
  crossbite / protrusion...), a handy sanity check.

## Status & next steps

Enough to **explore and compare approaches**, not yet a validated grader.

### Calibrate expectations first

A 2024 study ([Bioengineering 11(9):861](https://doi.org/10.3390/bioengineering11090861)) trained a
CNN on **1009 expert-graded frontal intraoral photos**, with overjet supplied as an extra input.
Exact AC 1–10 prediction was **not** achievable; only the binary split (AC 1–5 vs 6–10) worked well
(82% accuracy). Their grader's intra-rater reliability was κ = 0.84 (95% CI 0.76–0.93).

So the realistic target here is a **continuous score reported as bands**
(1–4 no need / 5–7 borderline / 8–10 definite need, per Richmond et al.), not 10-class
classification. Metrics must be MAE, within-±1, and quadratic weighted kappa — never plain accuracy.

### Still needed

1. **Pairwise comparison study** over 28 images (18 patients + 10 AC anchors, blinded), fitted with
   a Bradley–Terry model to get a latent score. Including the anchors gives calibration to the
   official AC scale for free, without anyone assigning absolute grades.
2. **ICC across raters** to establish whether a shared perceptual construct exists at all, and to
   set the ceiling for model performance (`r_max = sqrt(reliability)`).
3. **Fix FDI numbering** — currently positional (rank from midline), so a missing tooth or a
   diastema silently shifts every subsequent label. Planned fix: anchor the midline on the widest
   adjacent pair, then align against a width template via dynamic programming (Needleman–Wunsch),
   which handles gaps natively.
4. A **dentist-graded validation set** (~30–50 photos), ideally graded twice with a washout period
   so intra-rater reliability can be measured.

## Model notes

- **DINOv2 ViT-S/14** — primary encoder; strong label-free k-NN features, Apache-2.0, MPS-friendly.
- **DINOv3 ViT-S/16** — larger pretraining (~1.7B images), sharper dense features; gated license.
- On-device (future): convert to **Core ML**, or swap to **MobileCLIP** / **MobileNetV3**. The
  grading logic is encoder-agnostic, so the backbone can be swapped freely.
