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
import torch.nn.functional as F
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


def _patch_attention_modules(backbone):
    """Replace traced-shape division in attention forward with a constant head_dim.

    PyTorch 2.9+ traces `C // num_heads` (where C comes from x.shape) as an `int`
    cast on a symbolic tensor, which coremltools 9 cannot lower.  Using the Python-int
    constant `self.dim // self.num_heads` avoids the dynamic cast entirely.
    """
    from dinov2.layers.attention import Attention

    def make_fixed_forward(attn_module):
        num_heads = attn_module.num_heads
        head_dim  = attn_module.dim // num_heads   # pure Python int — constant in graph

        def fixed_forward(x, is_causal=False):
            B, N, C = x.shape
            qkv = attn_module.qkv(x).reshape(B, N, 3, num_heads, head_dim)
            q, k, v = torch.unbind(qkv, 2)
            q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=attn_module.attn_drop if attn_module.training else 0,
                is_causal=is_causal,
            )
            x = x.transpose(1, 2).contiguous().view(B, N, C)
            x = attn_module.proj_drop(attn_module.proj(x))
            return x

        return fixed_forward

    for module in backbone.modules():
        if isinstance(module, Attention):
            module.forward = make_fixed_forward(module)


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

    # Patch attention modules: replace dynamic `C // num_heads` (traced int, unsupported
    # by coremltools 9 on PyTorch 2.9+) with the pre-computed constant head_dim.
    _patch_attention_modules(backbone)

    model = DINOv2Embedder(backbone).eval()

    # coremltools doesn't support upsample_bicubic2d; swap to bilinear during trace
    # (position-embedding quality difference is imperceptible)
    _orig_interpolate = F.interpolate
    def _bilinear_interpolate(input, size=None, scale_factor=None, mode="nearest",
                              align_corners=None, recompute_scale_factor=None, antialias=False):
        if mode == "bicubic":
            mode = "bilinear"
        return _orig_interpolate(input, size=size, scale_factor=scale_factor,
                                 mode=mode, align_corners=align_corners,
                                 recompute_scale_factor=recompute_scale_factor,
                                 antialias=antialias)
    F.interpolate = _bilinear_interpolate
    try:
        traced = torch.jit.trace(model, x)
    finally:
        F.interpolate = _orig_interpolate

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
