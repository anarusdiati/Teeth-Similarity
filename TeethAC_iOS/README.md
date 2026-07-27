# TeethAC — iOS app (on-device DINOv2)

Versi iOS dari `teeth_similarity_dinov2.ipynb`. Ambil **1 foto gigi depan** →
ROI-crop region gigi → embedding **DINOv2 ViT-S/14** (Core ML, on-device) →
cosine similarity ke **10 anchor IOTN AC** → **perkiraan grade 1–10**
(1-NN + tertimbang) beserta 3 referensi paling mirip.

Pipeline-nya dibuat **identik** dengan notebook: ROI crop (s_thr 0.30, v_thr 0.55,
pad 0.10), pooling CLS token, normalisasi ImageNet, temperature 0.1. Anchor
di-embed on-device lewat code path yang sama dengan query, jadi hasilnya konsisten.

```
TeethAC_iOS/
├── project.yml                       # XcodeGen (buat .xcodeproj 1 perintah)
├── model/
│   ├── convert_dinov2_coreml.py      # -> DINOv2Teeth.mlpackage
│   └── requirements-coreml.txt
├── Sources/
│   ├── TeethACApp.swift
│   ├── ContentView.swift             # UI: grade + top-3 referensi
│   ├── ROICropper.swift              # port teeth_roi_box (HSV heuristic)
│   ├── DINOv2FeatureExtractor.swift  # Core ML wrapper
│   ├── ACGrader.swift                # port predict_ac (hard + soft)
│   └── ReferenceStore.swift          # embed 10 anchor saat launch
├── Resources/Assets.xcassets/        # ac_grade_01..10 (sudah siap)
└── README.md
```

## Hanya ada 1 langkah yang wajib kamu jalankan sendiri: buat model

Bobot DINOv2 tak bisa aku unduh dari sini, jadi model dibuat sekali di Mac kamu.
Sisanya (kode Swift + 10 anchor + asset) sudah siap.

```bash
cd TeethAC_iOS/model
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-coreml.txt
python convert_dinov2_coreml.py          # -> model/DINOv2Teeth.mlpackage
```

## Lalu buka project (pilih salah satu)

### Cara A — XcodeGen (paling cepat, 1 perintah)

```bash
brew install xcodegen           # sekali saja, kalau belum ada
cd TeethAC_iOS
xcodegen generate
open TeethAC.xcodeproj
```

Tekan **Run** (⌘R). Selesai — `DINOv2Teeth.mlpackage` & anchor sudah otomatis ikut.

### Cara B — Xcode manual (tanpa install apa pun)

1. Xcode → **New Project → App** (Interface: SwiftUI, iOS 16+). Nama: `TeethAC`.
2. Hapus `ContentView.swift` bawaan. **Drag** semua file di `Sources/` ke project
   (centang *Copy items if needed* + target TeethAC).
3. **Drag** `Resources/Assets.xcassets` ke project (ganti/gabung dengan yang bawaan).
4. **Drag** `model/DINOv2Teeth.mlpackage` ke project (centang target TeethAC).
5. Tekan **Run**.

## Pakai

Buka app → **Pilih Foto Gigi** → app menampilkan perkiraan AC grade dan 3 anchor
paling mirip lengkap dengan skor cosine.

> Simulator bisa, tapi Neural Engine hanya aktif di **iPhone fisik** (lebih cepat).
> Untuk run di device, isi Team ID di Signing (Xcode) / `DEVELOPMENT_TEAM` di `project.yml`.

## Ganti / setel model

- **Encoder lebih akurat:** ubah `ENCODER` ke `dinov2_vitb14` di `convert_dinov2_coreml.py`
  + set `EMBED_DIM = 768` di script **dan** `embedDim = 768` di `DINOv2FeatureExtractor.swift`.
- **Parameter ROI / temperature:** ada di `ROICropper.swift` dan `ACGrader.swift`,
  namanya sama dengan CONFIG notebook.

## Catatan klinis

Alat bantu eksplorasi, **bukan diagnosis**. Grading AC subjektif antar-klinisi;
target wajar adalah selisih ±1. Belum divalidasi terhadap label dokter.
