# Catatan Progres — 27 Juli 2026

Ringkasan alur kerja, temuan, dan rencana jangka dekat untuk proyek penilaian
**IOTN Aesthetic Component (AC)** dari foto intraoral frontal.

---

## Alur besar hari ini

Satu pertanyaan berkembang jadi pertanyaan lain, dan tiap notebook lahir dari jawaban notebook
sebelumnya:

```
A. Embedding  →  bisakah fitur pretrained memeringkat AC tanpa label?
      ↓         ternyata bisa sebagian, tapi tidak ada variabel target untuk memvalidasi
B. Depth      →  apakah kedalaman monokuler menambah sesuatu?
      ↓         hampir tidak — dan secara teori memang TIDAK BISA menambah informasi
C. Geometri   →  kerapihan itu fenomena bidang gambar, bukan kedalaman
      ↓         metrik tanpa satuan berhasil dibuat, tapi segmentasinya belum andal
D. LLM judge  →  kita butuh variabel target; siapa yang menentukan "rapi"?
```

Benang merahnya: **kita terus mencari cara mengukur, lalu sadar bahwa yang belum ada justru
hal yang diukur.** AC bukan properti geometri gigi — ia properti *respons pengamat* terhadap
geometri itu. Tidak ada pengukuran pada gigi yang bisa mendefinisikannya; hanya memprediksinya.

---

## Isi tiap notebook

### A. `teeth_similarity_dinov2.ipynb` / `dinov3` (sudah ada sebelumnya)

**Tujuan.** Menilai AC sebagai persoalan *retrieval*, bukan klasifikasi — tanpa data berlabel.

**Alur.** `foto → ROI crop → embedding DINOv2 → cosine ke 10 anchor AC → grade (1-NN + tertimbang)`

**Batasnya.** Embedding adalah kotak hitam: kalau prediksinya salah, tidak ada cara menjelaskan
kenapa. Untuk skripsi yang harus dipertahankan di sidang, ini kelemahan nyata.

### B-1. `depth_anything_v2_inference.ipynb`

**Tujuan.** Uji kelayakan, **dirancang untuk gagal cepat**. Satu pertanyaan: apakah peta kedalaman
Depth Anything V2 pada foto intraoral membawa geometri nyata, atau cuma mengulang kecerahan flash?

**Empat probe.**

| Probe | Isi | Hasil |
|---|---|---|
| P1 | Korelasi luminance vs depth | \|r\| = **0.549** — lolos, tapi tipis (ambang 0.6); 6/18 foto di atas 0.6 |
| P2 | Profil kedalaman sepanjang lengkung | visual |
| P3 | Statistik depth per weak-label | arah tidak konsisten |
| P4 | Stabilitas terhadap flip horizontal | corr = **0.996** — lolos telak |
| — | Piksel spekular | **19.2%** ROI rata-rata |

### B-2. `depth_features_fusion.ipynb`

**Tujuan.** Kalau depth memang layak, ekstrak fiturnya dengan benar lalu uji apakah membantu.

**Yang dikerjakan.** Perbaikan bug notebook sebelumnya (grid tetap 384px, masking spekular,
koefisien bimodalitas Sarle, ambang tepi absolut), **uji konfound** tiap fitur terhadap properti
foto murni, profil per-gigi lewat deteksi puncak, dan fusi dengan DINOv2 ber-bootstrap CI.

**Hasil uji konfound — 4 dari 13 fitur dibuang:**

| Dibuang | Sebenarnya mengukur |
|---|---|
| `rough_mean` | resolusi file sumber (rho = −0.60) — **tetap terkonfound meski sudah di-resize** |
| `rough_p90`, `arch_range`, `arch_curvature` | aspect ratio ROI |

**Temuan teoretis yang mengubah kerangka berpikir.** Depth map adalah fungsi deterministik dari
foto: `D = f(RGB)`. Menurut *data processing inequality*:

```
I(D ; AC)  ≤  I(RGB ; AC)
```

Depth **tidak pernah menambah informasi**. Ia hanya menyusun ulang informasi yang sudah ada.
Nilainya bukan soal informasi, melainkan **efisiensi sampel** — dan itu berbanding terbalik dengan
ukuran dataset. Dengan 18 foto, prior buatan tangan masuk akal dicoba; dengan 10.000 foto berlabel,
fitur ini akan kalah telak dari CNN.

### C. `teeth_segmentation_sam2.ipynb`

**Tujuan.** Pindah ke geometri bidang gambar — tempat kerapihan sesungguhnya berada.

**Alur.** `foto → SAM 2 → mask per gigi → penomoran FDI → landmark → metrik ternormalisasi`

**Kunci desain: semua metrik dibagi `w_ref`** (lebar insisivus sentral pada foto yang sama). Hasilnya
tanpa satuan dan sebanding antar foto — **tanpa kalibrasi milimeter, tanpa marker**. Ini yang
menyelesaikan persoalan skala: kita tidak menaklukkan *scale ambiguity*, kita memutarinya dengan
hanya mengajukan pertanyaan yang tidak memerlukan skala.

**Metrik.** `LII_norm` (Little's Irregularity Index ternormalisasi), `incisal_rms` (deviasi smile
arc), `tilt_std` / `tilt_asym` (inklinasi aksial), `width_asym`, `solidity_min`, `midline_dev`,
`cant_deg`, `wh_central`, `overbite_proxy`.

**Terverifikasi** dengan lengkung sintetis: lengkung sempurna → `LII_norm = 0.000`, naik monoton
seiring keberjejalan yang disuntikkan (0 → 0.94 → 1.84 → 4.48).

**Status saat ini: BELUM ANDAL.** Dari 18 foto hanya **9 valid**, 5 di antaranya `fdi_suspect`.
Yang benar-benar bersih cuma 4 foto.

### D. `ac_llm_judge.ipynb`

**Tujuan.** Mendapatkan variabel target — penilaian AC yang sesungguhnya.

**Desain.**
- **Multi-model**, prioritas gratis: OpenRouter (`:free`), Gemini free tier, lokal open-weights;
  Claude/OpenAI opsional. Backend tanpa API key dilewati otomatis.
- **Tiga kondisi bertingkat:** A = foto saja (sebanding literatur), B = + overlay segmentasi/FDI/
  landmark, C = + tabel metrik geometri.
- **Keyakinan diukur, bukan ditanyakan.** `grade` = median dari k sampel independen;
  `confidence` = proporsi sampel dalam ±1 dari median. Keyakinan yang dilaporkan model tetap
  dicatat khusus untuk **menguji kalibrasinya**.
- **Kartu laporan per foto:** foto + overlay + grade + keyakinan + penalaran.

**Patokan dari literatur.**

| Studi | Data | Hasil |
|---|---|---|
| [Diagnostics 2025, 15(23):3048](https://www.mdpi.com/2075-4418/15/23/3048) | 150 foto, GPT-5 vs klinisi (κ=0.91, ICC=0.88) | **MAE 1.47**, akurasi **66.7%**, tanpa bias sistematis |
| [Bioengineering 2024, 11(9):861](https://doi.org/10.3390/bioengineering11090861) | 1009 foto + overjet, CNN khusus | AC 1–10 **tidak tercapai**; hanya biner (82%) |

Konsekuensinya untuk target proyek ini: **AC 1–10 sebagai klasifikasi bukan target yang bisa
divalidasi.** Keluarkan skor kontinu, laporkan pita (1–4 / 5–7 / 8–10 per Richmond dkk.), dan pakai
MAE, within-±1, serta quadratic weighted kappa — bukan akurasi polos.

### E. `labels/ac_llm_provisional.csv`

18 foto dinilai lewat perbandingan relatif (diurutkan dulu, baru diberi angka), dengan pita,
tingkat keyakinan, peringkat, dan alasan per foto. Distribusi: 9 foto di pita 1–4, 7 di 5–7,
2 di 8–10. **Label sementara yang bisa diganti**, bukan ground truth.

---

## Temuan lintas-notebook

**Geometri menentukan apa yang bisa diukur.** Overjet membentang sepanjang sumbu optik kamera —
arah dengan informasi paling sedikit pada foto frontal. Overbite dan keberjejalan berada di bidang
gambar. Karena itu **overjet tidak terukur dari foto frontal, overbite dan kerapihan terukur**.
Ini bukan keterbatasan model, melainkan konsekuensi proyeksi.

**Subjektivitas tidak dihilangkan, tapi dikarakterisasi.** Cara membuat AC objektif bukan mencari
besaran intrinsik pada gigi, melainkan **menjadikan pengamat bagian dari alat ukur** lalu mengukur
noise-nya: perbandingan berpasangan → Thurstone/Bradley-Terry → skor laten berskala interval; ICC
untuk menguji apakah konstruk bersamanya nyata; `√(reliabilitas)` sebagai plafon performa model.

**Metrik geometri belum berkorelasi dengan penilaian visual.** Pada 9 foto valid, `LII_norm` vs
penilaian AC: rho = +0.085 (p = 0.83). Tiga tafsir yang belum bisa dipisahkan: (a) segmentasinya
rusak sehingga metriknya mengukur gigi yang salah, (b) penilaiannya yang salah, (c) keduanya benar
tapi kerapihan kontak memang bukan penggerak utama AC. **Hipotesis (a) paling mungkin** dan harus
dieliminasi lebih dulu.

---

## Bug yang ditemukan & diperbaiki hari ini

| # | Bug | Dampak |
|---|---|---|
| 1 | `torchaudio` ABI mismatch | Semua import `transformers` gagal |
| 2 | `edge_density` pakai persentil atas datanya sendiri | Fitur **konstan 0.15** di semua foto, lolos uji konfound secara palsu |
| 3 | `bimodality` cuma menghitung riak histogram | Diganti koefisien bimodalitas Sarle (ambang 5/9) |
| 4 | `rough_mean` terkonfound resolusi sumber | **Tetap terkonfound meski di-resize** — memperbesar foto beresolusi rendah tidak menciptakan detail |
| 5 | `LII` euclidean punya lantai tak-nol | Lengkung sempurna memberi 0.536, bukan 0 → diganti komponen vertikal |
| 6 | `aspect_max = 3.2` terlalu ketat | Insisivus lateral (aspect 3.6) tertolak → kaninus dinomori sebagai lateral |
| 7 | Penomoran FDI bergeser diam-diam | Satu gigi hilang menggeser semua nomor **tanpa error** → ditambah `fdi_suspect` |
| 8 | Repo `facebookresearch/sam2` tidak bisa diakses | Diganti jalur `transformers` — tanpa clone GitHub |
| 9 | `.git/index.lock` basi + 36MB `.npy` ter-staged | Git terkunci; biner regenerable nyaris masuk riwayat selamanya |
| 10 | `sat_max = 0.45` salah untuk foto intraoral | Menolak **semua** mask di sebagian foto (lihat di bawah) |

Pola berulang yang layak diingat: **ambang absolut pada besaran yang skalanya berbeda tiap foto**
(bug 2, 4, 10). Perbaikannya selalu sama — buat ambangnya relatif terhadap distribusi foto itu
sendiri.

---

## Diagnosis segmentasi (hasil Section 10)

Foto `deep bite and protrusion 2`, 0 gigi atas terdeteksi:

```
SAM menghasilkan 6 mask MENTAH
Lolos filter: 0 dari 6   — semuanya tertolak "saturasi > 0.45"

Sapuan ambang:
  sat_max        0.45→0  0.55→3  0.65→4  0.75→4
  val_min        semua →0
  solidity_min   semua →0
  aspect_max     semua →0
  area_min_frac  semua →0
```

**Dua masalah, dan SAM adalah plafonnya.**

1. **Filter salah.** Mask 0 dan 1 punya `val` 0.94, `solidity` 1.00, aspect ~1.1 — hampir pasti
   gigi, tapi saturasinya 0.49 sehingga tertolak. Foto intraoral berselubung pink seluruhnya;
   asumsi "gigi putih, gusi merah" tidak berlaku.
2. **SAM hanya menghasilkan 6 mask untuk seluruh gambar.** Untuk automatic mask generation itu
   sangat sedikit. Sebersih apa pun filternya, tidak ada 12 gigi untuk ditemukan.

---

## Future work — berurutan, jangka dekat

### Prioritas 1 — benahi segmentasi (memblokir semua yang lain)

Selama hanya 4 dari 18 foto yang bersih, tidak ada angka hilir yang bermakna.

- [ ] **Longgarkan parameter SAM:** `points_per_crop=48`, `pred_iou_thresh=0.70`,
      `stability_score_thresh=0.80`, `crops_n_layers=1`
- [ ] **Ambang saturasi adaptif:** ganti konstanta global dengan persentil dari distribusi
      saturasi foto itu sendiri (gigi = objek paling tidak jenuh di dalam mulut)
- [ ] Kalau tetap gagal → **mode prompt-titik** (`SamModel` dengan `input_points`): ~12 klik ×
      18 foto ≈ satu sore. Hasil dijamin benar, sekaligus jadi data latih YOLOv8-seg
- [ ] **Perbaiki ROI crop HSV** — gagal parah pada foto #01 dan #05 (hasilnya potongan gigi atau
      mayoritas gusi). Notebook depth memakainya, jadi sebagian hasil depth dihitung pada crop salah

### Prioritas 2 — penomoran FDI yang tahan banting

- [ ] **Jangkar midline** pada pasangan gigi bertetangga terlebar & paling setara lebarnya
      (kebal terhadap lateral/kaninus yang hilang, kebal diastema)
- [ ] **Ganti pemeringkatan dengan DP alignment** (Needleman–Wunsch) terhadap template lebar;
      gap ditangani secara alami sebagai gigi hilang
- [ ] **Bedakan diastema vs gigi hilang** lewat lebar celah ternormalisasi (`G < 0.35` → diastema)
- [ ] **Bangun template lebar dari data sendiri**, bukan konstanta literatur — norma populasi
      Indonesia untuk gigi permanen anterior sangat tipis di literatur

### Prioritas 3 — dapatkan variabel target (bisa paralel dengan 1 & 2)

- [ ] **Studi perbandingan berpasangan**, 28 gambar (18 pasien + 10 anchor AC), penilai buta
      terhadap mana yang mana. 153+ pasang, ~15 menit per penilai
- [ ] Fit **Bradley–Terry** → skor laten kontinu. Anchor yang ikut dinilai memberi **kalibrasi ke
      skala AC resmi secara gratis**, tanpa siapa pun menetapkan grade absolut
- [ ] **3–5 penilai awam** → hitung **ICC**. Ini menguji apakah konstruk bersamanya nyata sekaligus
      menetapkan plafon performa model
- [ ] Sisipkan ~15 pasang duplikat untuk mengukur konsistensi intra-penilai

### Prioritas 4 — jalankan LLM judge (baseline dulu)

- [ ] **Kondisi A saja** (`CFG["conditions"]=["A"]`) — hemat kuota, dan langsung sebanding dengan
      MAE 1.47 dari Diagnostics 2025
- [ ] Uji dengan `CFG["subset"]=2` dulu untuk memastikan format respons tiap penyedia
- [ ] Buka kondisi B & C **hanya setelah** segmentasi bersih — kalau tidak, yang terukur adalah
      kualitas segmentasi, bukan nilai informasi geometri

### Prioritas 5 — sesi dengan drg. Laura

- [ ] Minta penilaian **dua kali dengan jeda** (min. seminggu), sisipkan foto duplikat →
      reliabilitas intra-rater = plafon performa model
- [ ] Tanyakan **foto mana yang framing-nya tidak layak dinilai**; kalau ada standar pemotretan
      intraoral yang beliau pakai, itu lebih berharga daripada perbaikan algoritma apa pun
- [ ] Konfirmasi **10 anchor AC** yang dipakai adalah versi resmi (hasil pindai bisa berbeda
      kontras/crop dari kartu IOTN asli, dan seluruh pipeline bergantung padanya)
- [ ] Bawa **kartu laporan per foto** dari notebook D — beliau bisa langsung menunjuk mana yang
      tidak disetujui, lengkap dengan alasan model yang tertulis

### Prioritas 6 — model & evaluasi (setelah target ada)

- [ ] Uji ulang korelasi metrik geometri terhadap skor laten sungguhan
- [ ] **Ridge regression** atau **ordinal proportional-odds** dari metrik → skor laten,
      λ dipilih lewat leave-one-out CV
- [ ] Kurangi ke 3–4 fitur — dengan n = 18 dan 12 fitur, jauh di bawah aturan 10 kejadian per
      prediktor
- [ ] Laporkan **MAE, within-±1, quadratic weighted kappa, Spearman**, semuanya dengan bootstrap CI
- [ ] Bandingkan terhadap **dua garis dasar**: selalu-tebak-median, dan plafon `√(reliabilitas)`

---

## Yang sebaiknya TIDAK dikerjakan dulu

- **Menyetel fitur depth lebih jauh.** Sudah terbukti kontributor minoritas, dan secara teori tidak
  bisa menambah informasi. Biarkan sebagai jalur yang sudah dieksplorasi dan didokumentasikan —
  itu sendiri hasil yang layak dilaporkan.
- **Mengejar klasifikasi AC 1–10.** Studi dengan 1009 foto berlabel ahli pun tidak mencapainya.
- **Menambah fitur baru** sebelum ada variabel target. Tanpa sesuatu untuk divalidasi, menambah
  fitur hanya menambah derajat kebebasan untuk menipu diri sendiri.

---

## Referensi kunci

- **Little RM (1975).** *The Irregularity Index: A quantitative score of mandibular anterior
  alignment.* AJODO 68(5):554–563
- **Evans R, Shaw W (1987).** Skala SCAN / Aesthetic Component — European Orthodontic Society
- **Brook PH, Shaw WC (1989).** Dental Health Component IOTN — *Eur. J. Orthod.*
- **Thurstone LL (1927).** *A law of comparative judgment.* Psychol. Rev.
- **Bradley RA, Terry ME (1952).** *Rank analysis of incomplete block designs.* Biometrika
- **Cohen J (1968).** Weighted kappa — Psychol. Bull.
- **Koo TK, Li MY (2016).** *A Guideline of Selecting and Reporting ICC for Reliability Research.*
  J. Chiropr. Med. 15:155–163
- **Diagnostics 2025, 15(23):3048** — LLM untuk IOTN-AC (MAE 1.47)
- **Bioengineering 2024, 11(9):861** — CNN untuk IOTN-AC (1009 foto)
