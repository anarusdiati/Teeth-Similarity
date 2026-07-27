"""
Konversi DINOv2 ViT-S/14 -> Core ML (.mlpackage), PERSIS seperti pipeline
di teeth_similarity_dinov2.ipynb (pooling = CLS token, normalisasi ImageNet).

Output model "DINOv2Teeth.mlpackage":
  - Input  : "image"     -> RGB 224x224 (Core ML membagi 255 di dalam)
  - Output : "embedding" -> vektor 384-dim, SUDAH di-L2-normalize
             (cocok langsung untuk cosine similarity di iOS)

Jalankan sekali di Mac kamu (butuh internet untuk unduh bobot DINOv2):

    cd TeethAC_iOS/model
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements-coreml.txt
    python convert_dinov2_coreml.py

Hasil DINOv2Teeth.mlpackage otomatis tersimpan di folder ini.
"""

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct

# --- samakan dengan CONFIG notebook (baseline DINOv2) ---
ENCODER  = "dinov2_vits14"                 # vits14(baseline) | vitb14 | vitl14
IMG_SIZE = 224                             # kelipatan 14
EMBED_DIM = 384                            # vits14=384, vitb14=768, vitl14=1024
MEAN = [0.485, 0.456, 0.406]               # ImageNet
STD  = [0.229, 0.224, 0.225]
OUT_PATH = "DINOv2Teeth.mlpackage"


class DINOv2Embedder(nn.Module):
    """Normalisasi + backbone (CLS token) + L2-normalize, semua di dalam model."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x):                                  # x dalam [0,1]
        x = (x - self.mean) / self.std
        feats = self.backbone(x)                           # == x_norm_clstoken (pooling CLS)
        feats = nn.functional.normalize(feats, p=2, dim=1)
        return feats


def main():
    print(f"Memuat {ENCODER} dari torch.hub ...")
    backbone = torch.hub.load("facebookresearch/dinov2", ENCODER)
    backbone.eval()

    # Verifikasi: backbone(x) HARUS sama dengan forward_features(x)['x_norm_clstoken']
    x = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        a = backbone(x)
        b = backbone.forward_features(x)["x_norm_clstoken"]
    assert torch.allclose(a, b, atol=1e-5), "backbone() != CLS token — cek versi model"
    assert a.shape[1] == EMBED_DIM, f"dim {a.shape[1]} != {EMBED_DIM}"
    print("OK: pooling CLS terverifikasi, dim =", EMBED_DIM)

    model = DINOv2Embedder(backbone).eval()
    traced = torch.jit.trace(model, x)

    image_input = ct.ImageType(
        name="image",
        shape=(1, 3, IMG_SIZE, IMG_SIZE),
        scale=1.0 / 255.0,
        bias=[0.0, 0.0, 0.0],
        color_layout=ct.colorlayout.RGB,
    )

    print("Konversi ke Core ML ...")
    mlmodel = ct.convert(
        traced,
        inputs=[image_input],
        outputs=[ct.TensorType(name="embedding")],
        minimum_deployment_target=ct.target.iOS16,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
    )
    mlmodel.short_description = "DINOv2 ViT-S/14 teeth embedding (CLS, L2-normalized)"
    mlmodel.input_description["image"] = f"Foto gigi RGB {IMG_SIZE}x{IMG_SIZE} (sudah di-ROI-crop di app)"
    mlmodel.output_description["embedding"] = f"Vektor {EMBED_DIM}-dim ternormalisasi"
    mlmodel.save(OUT_PATH)
    print(f"Selesai -> {OUT_PATH}")
    print("Sekarang drag DINOv2Teeth.mlpackage ke project Xcode (atau taruh di folder ini bila pakai XcodeGen).")


if __name__ == "__main__":
    main()
