# Rubrik AC-Proxy 1–10 (Feature-Based Deterministic)
### Untuk anotasi foto intraoral frontal — pipeline YOLOv8-OBB

**Versi:** 1.0 · **Tanggal:** 28 Juli 2026

---

## 0. Peringatan metodologis — baca dulu

Ini **bukan IOTN Aesthetic Component resmi.** AC asli (Evans & Shaw, 1987) adalah *photo-matching scale*: 10 foto referensi berurutan dari paling menarik (1) sampai paling tidak menarik (10). Tidak ada satu pun definisi verbal atau ambang milimeter di instrumen aslinya — penilai mencocokkan kasus dengan foto yang paling mirip secara gestalt.

Rubrik ini adalah **proxy deterministik yang terkorelasi dengan AC**, dibangun dari fitur terukur agar reproducible dan bisa diaudit. Konsekuensinya untuk penulisan:

- Jangan tulis "IOTN AC" di paper. Tulis **"AC-proxy"** atau **"rubric-derived aesthetic score"**, dengan sitasi bahwa AC asli adalah photo-matching.
- Kalau kamu butuh klaim korelasi dengan AC asli, kamu perlu **validasi**: minta 2–3 ortodontis menilai subset (~50 foto) dengan foto referensi AC asli, lalu hitung Spearman ρ dan weighted kappa terhadap skor rubrik ini.
- Ambang mm di bawah diambil dari **IOTN DHC** (Brook & Shaw, 1989) karena itu satu-satunya bagian IOTN yang punya cut-off eksplisit. Ini pilihan desain, bukan properti AC.

**Banding rujukan AC resmi** (untuk interpretasi skor akhir): 1–4 = tidak butuh perawatan · 5–7 = borderline · 8–10 = butuh perawatan.

---

## 1. Prinsip desain

| Prinsip | Implikasi |
|---|---|
| **Deterministik** | Input yang sama → skor yang sama, selalu. Tidak ada "kesan keseluruhan". |
| **Worst-feature dominan, bukan murni aditif** | Meniru logika MOCDO: satu kelainan berat lebih menentukan daripada banyak kelainan ringan. Tapi berbeda dari DHC, akumulasi tetap diberi bobot kecil — karena secara *estetik* banyak masalah ringan memang terlihat lebih buruk. |
| **Normalisasi internal** | Semua ukuran mm diturunkan dari lebar mesiodistal insisivus sentral atas sebagai referensi skala (§4.1). Tidak butuh penggaris di foto. |
| **Bobot visibilitas** | Anterior bawah didiskon 1 tingkat keparahan — pada foto frontal, crowding bawah jauh kurang terlihat daripada crowding atas. |
| **Eksplisit soal yang tidak bisa diukur** | Fitur yang butuh foto lateral / model 3D / pemeriksaan klinis ditandai dan **dikeluarkan** dari skor, bukan ditebak (§5). |

---

## 2. Sembilan fitur & tabel keparahan

Setiap fitur menghasilkan **skor keparahan S ∈ {0, 1, 2, 3, 4}**.
Wilayah penilaian: **anterior saja, kaninus ke kaninus** (FDI 13–23 dan 33–43).

### F1 — Displacement titik kontak, anterior ATAS
Nilai terbesar dari 5 titik kontak (bukan jumlah). Ambang mengikuti DHC 1/2d/3d/4d.

| S | Kriteria |
|---|---|
| 0 | < 1,0 mm |
| 1 | 1,0 – 2,0 mm |
| 2 | > 2,0 – 4,0 mm |
| 3 | > 4,0 – 6,0 mm |
| 4 | > 6,0 mm |

### F2 — Displacement titik kontak, anterior BAWAH
Ambang identik F1, lalu **diskon visibilitas: S_efektif = max(0, S − 1)**.

### F3 — Rotasi (torsiversi + tipping)
Nilai terbesar dari selisih sumbu gigi terhadap tangen lengkung rahang. Lihat §4.3 — ada dua komponen yang harus digabung.

| S | Kriteria |
|---|---|
| 0 | < 10° |
| 1 | 10° – 20° |
| 2 | > 20° – 35° |
| 3 | > 35° – 55° |
| 4 | > 55° |

### F4 — Diastema / spacing
Celah tunggal terbesar antar gigi anterior atas.

| S | Kriteria |
|---|---|
| 0 | < 0,5 mm |
| 1 | 0,5 – 1,5 mm |
| 2 | > 1,5 – 3,0 mm |
| 3 | > 3,0 – 5,0 mm |
| 4 | > 5,0 mm |

### F5 — Overbite
Persentase mahkota klinis insisivus sentral bawah yang tertutup insisivus atas.

| S | Kriteria |
|---|---|
| 0 | 10 – 40 % (normal) |
| 1 | > 40 – 60 %, **atau** < 10 % (edge-to-edge / reduced) |
| 2 | > 60 – 80 % |
| 3 | > 80 – 100 % (complete, tanpa trauma terlihat) |
| 4 | 100 % + tanda trauma gingiva/palatal terlihat |

> Trauma gingival sulit dipastikan dari foto. Kalau ragu, pakai S=3 dan tandai `overbite_uncertain=True`.

### F6 — Open bite anterior
Celah vertikal saat oklusi. Ambang mengikuti DHC 2e/3e/4e.

| S | Kriteria |
|---|---|
| 0 | ≤ 1,0 mm (termasuk tidak ada) |
| 1 | > 1,0 – 2,0 mm |
| 2 | > 2,0 – 3,0 mm |
| 3 | > 3,0 – 4,0 mm |
| 4 | > 4,0 mm |

> Ambang bawah **1,0 mm, bukan 0** — DHC 2e berbunyi *"open bite greater than 1 mm but less than or equal to 2 mm"*. Ini sekaligus jadi *deadband* terhadap noise: selisih vertikal sub-milimeter yang diturunkan dari OBB pada foto tidak bermakna klinis.

### F7 — Crossbite anterior
Jumlah gigi anterior atas yang berada lingual terhadap gigi bawah.

| S | Kriteria |
|---|---|
| 0 | Tidak ada |
| 1 | 1 gigi |
| 2 | 2 gigi |
| 3 | 3 – 4 gigi |
| 4 | > 4 gigi (crossbite anterior penuh) |

### F8 — Gigi hilang / belum erupsi / ektopik (anterior)
Kategorikal — tetapkan langsung.

| S | Kriteria |
|---|---|
| 0 | Lengkap, erupsi normal |
| 1 | 1 gigi erupsi sebagian / sedikit ektopik |
| 2 | 1 gigi hilang atau jelas ektopik dengan celah |
| 3 | 2 gigi hilang |
| 4 | > 2 gigi hilang, atau impaksi dengan ruang terbuka lebar |

> **Perhatian dentisi campuran.** Pada anak, gigi yang "hilang" sering hanya belum erupsi secara fisiologis. Rubrik ini akan memberi skor tinggi yang menyesatkan. Filter dataset berdasarkan usia/tahap dentisi, atau tandai `mixed_dentition=True` dan kecualikan F8.

### F9 — Deviasi garis tengah (atas vs bawah)

| S | Kriteria |
|---|---|
| 0 | < 1,0 mm |
| 1 | 1,0 – 2,0 mm |
| 2 | > 2,0 – 3,0 mm |
| 3 | > 3,0 – 4,0 mm |
| 4 | > 4,0 mm |

---

## 3. Agregasi ke skor 1–10

```
S_max = max(S_F1 … S_F9)
S_sum = Σ(S_F1 … S_F9)

AC_raw = BASE[S_max] + 0.3 × (S_sum − S_max)

# Aturan sub-ambang (lihat catatan di bawah)
if S_max == 0 and ada_iregularitas_subambang:
    AC_raw = 2.0

AC = clamp(round_half_up(AC_raw), 1, 10)
```

**Aturan sub-ambang.** Tanpa ini, **AC = 2 mustahil tercapai**: kalau S_max = 0 maka semua S = 0, jadi S_sum = 0 dan AC_raw selalu tepat 1,0 → AC = 1; lompatan berikutnya BASE[1] = 3,0. Padahal pada skala AC asli, 1 vs 2 justru membedakan "sempurna" dari "hampir sempurna" — kategori yang umum di populasi. Kalau semua S = 0 tetapi ada iregularitas terukur di bawah ambang, tetapkan AC = 2:

| Fitur | Ambang sub-ambang |
|---|---|
| Displacement atas / bawah | ≥ 0,5 mm |
| Rotasi | ≥ 5° |
| Spacing | ≥ 0,25 mm |
| Garis tengah | ≥ 0,5 mm |

**Tabel BASE** (nonlinier — inilah yang membuat satu kelainan berat mendominasi):

| S_max | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| BASE | 1,0 | 3,0 | 4,5 | 6,5 | 8,5 |

**Kenapa nonlinier?** Dengan basis linier, satu kelainan berat tunggal (S_max=4, sisanya 0) hanya mencapai ~7 — padahal secara estetik kasus seperti itu jelas masuk band "butuh perawatan" (8–10). Sebaliknya basis linier yang cukup curam untuk mencapai 9 akan meledakkan kasus ringan-multipel ke 6–7. Kurva cembung menyelesaikan keduanya.

**Perilaku sistem (sanity check):**

| Kondisi | S_max | S_sum | AC |
|---|---|---|---|
| Oklusi ideal | 0 | 0 | 1 |
| Satu kelainan minor | 1 | 1 | 3 |
| Empat kelainan minor | 1 | 4 | 4 |
| Satu kelainan sedang | 2 | 2 | 5 |
| Sedang + beberapa ringan | 2 | 6 | 6 |
| Satu kelainan berat tunggal | 4 | 4 | 9 |
| Multipel berat | 4 | 12 | 10 |

---

## 4. Pemetaan ke output YOLOv8-OBB

Setiap deteksi: `(cls, cx, cy, w, h, θ)` dari `r.obb.xywhr`. Semua di bawah dihitung **post-processing**, bukan output jaringan.

### 4.1 Kalibrasi piksel → milimeter

Tanpa referensi fisik di foto, pakai **lebar mesiodistal insisivus sentral atas** sebagai skala anatomis.

```python
# Untuk insisivus, mahkota lebih tinggi daripada lebar → sisi pendek OBB = mesiodistal
md_px(t) = min(t.w, t.h)

MD_CENTRAL_MM = 8.5   # rerata populasi, SD ≈ 0.5
scale = MD_CENTRAL_MM / mean(md_px(FDI_11), md_px(FDI_21))   # mm per piksel
```

**Batasan yang wajib kamu dokumentasikan:**

1. **Variasi populasi** — SD ±0,5 mm pada 8,5 mm ≈ **±6 % galat skala sistematis** per subjek. Cukup untuk menggeser kasus tepat di ambang (mis. 3,9 vs 4,1 mm) melintasi batas S.
2. **Distorsi perspektif** — kamera bukan proyeksi ortografik. Gigi yang lebih jauh dari sumbu optik (kaninus, premolar) tampak *foreshortened*; estimasi mm-nya bias mengecil. Anterior atas paling akurat, kaninus paling tidak.
3. **Insisivus sentral rotasi** — kalau 11 atau 21 sendiri mengalami torsiversi, `md_px` mengecil dan **seluruh kalibrasi ikut salah** (semua ukuran jadi over-estimated). Mitigasi: kalau `|md_px(11) − md_px(21)| / mean > 0.15`, tandai `calibration_unreliable=True` dan gunakan gigi yang lebih lebar sebagai referensi, atau kecualikan dari dataset otomatis.

### 4.2 Displacement titik kontak (F1, F2)

Rekonstruksi titik kontak dari sudut OBB, lalu ukur tegak lurus lengkung.

```
1. Fit lengkung rahang: parabola kuadratik pada (cx, cy) semua gigi 13–23.
2. Untuk pasangan bersebelahan (i, i+1):
   - P_i     = titik tengah sisi DISTAL OBB gigi i
   - P_{i+1} = titik tengah sisi MESIAL OBB gigi i+1
   - d = |P_i − P_{i+1}| diproyeksikan ke NORMAL lengkung di titik tengahnya
3. F1 = max(d) × scale
```

Sudut OBB diperoleh dari rotasi standar:
`corner = (cx, cy) + R(θ) · (±w/2, ±h/2)`

> **Little's Irregularity Index vs DHC.** Little's II = *jumlah* 5 displacement (0 = ideal, 1–3 minimal, 4–6 sedang, 7–9 berat, ≥10 sangat berat). DHC pakai *nilai terbesar tunggal*. Rubrik ini pakai **nilai terbesar** agar sejalan dengan ambang DHC. Simpan Little's II sebagai fitur sekunder — berguna untuk analisis, dan reviewer ortodonti akan mengharapkannya.

### 4.3 Rotasi — θ saja tidak cukup

Ini bagian yang paling sering salah dipahami, dan relevan langsung dengan catatanmu bahwa θ belum terpakai.

**θ dari OBB pada foto frontal menangkap tipping dalam bidang (mesioversi/distoversi), BUKAN torsiversi.** Gigi yang berputar pada sumbu panjangnya tidak menghasilkan kotak yang miring — ia menghasilkan kotak yang **menyempit**.

Dua besaran terpisah, digabung di akhir:

```python
# (a) Tipping dalam bidang — dari θ
tangent_i = turunan lengkung rahang di cx_i
Δθ_tip    = sudut(sumbu_panjang_OBB_i, normal_lengkung_i)

# (b) Torsiversi — dari penyempitan lebar proyeksi
# Gigi berputar φ pada sumbu panjang → lebar terproyeksi = lebar_asli × cos(φ)
φ_torsi = arccos( clamp(md_px_observed / md_px_expected, 0, 1) )
#   md_px_expected = lebar MD normatif gigi tsb (per kelas FDI) ÷ scale

F3 = max over teeth of  sqrt(Δθ_tip² + φ_torsi²)
```

Gabungan kuadratik dipakai karena keduanya rotasi pada sumbu ortogonal; ini pendekatan, bukan turunan eksak dari geometri 3D.

**Catatan penting:** `φ_torsi` bergantung pada lebar normatif per gigi, yang punya variasi populasi sendiri — jadi lebih berisik daripada `Δθ_tip`. Untuk gigi dengan `md_px_observed > md_px_expected` (terjadi karena galat kalibrasi), clamp ke 0.

### 4.4 Overbite & open bite (F5, F6)

OBB gigi yang tertutup hanya membatasi **bagian yang terlihat** — di sini itu justru menguntungkan.

```python
h_visible = tinggi OBB insisivus bawah pada sumbu inciso-gingival
h_expected = tinggi mahkota normatif (insisivus sentral bawah ≈ 9.0 mm) / scale

overbite_pct = 100 × (1 − h_visible / h_expected)
```

Open bite: kalau tepi insisal atas berada **di atas** tepi insisal bawah (tidak ada tumpang tindih vertikal), jarak vertikal antar keduanya × `scale`.

> Asumsi tersembunyi: OBB bawah memang berhenti di batas oklusi, bukan menembus gigi atas. Detektor sering "menebak" bagian tertutup dan menghasilkan kotak yang terlalu tinggi. **Verifikasi ini pada anotasi ground-truth-mu** — kalau anotatormu melabeli seluruh gigi termasuk bagian tersembunyi, rumus ini rusak total dan F5 tidak bisa dipakai.

### 4.5 Garis tengah (F9)

```python
midline_atas  = (cx_11 + cx_21) / 2
midline_bawah = (cx_31 + cx_41) / 2
F9 = |midline_atas − midline_bawah| × scale
```

> Ini mengukur *dental midline discrepancy*, bukan deviasi terhadap garis tengah wajah (butuh landmark wajah — tidak ada di foto intraoral dengan retraktor).

---

## 5. Yang TIDAK bisa diskor dari foto frontal

Jangan tebak. Tandai sebagai missing dan keluarkan dari agregasi.

| Fitur | Kenapa tidak bisa | Butuh apa |
|---|---|---|
| **Overjet** (DHC 3a, 4a, 4b, 4m, 5a, 5m) | Dimensi antero-posterior — hilang total dalam proyeksi frontal | Foto lateral / model 3D / sefalometri |
| **Diskrepansi RCP–ICP** (DHC 2c, 3c, 4c) | Besaran fungsional, butuh manipulasi mandibula | Pemeriksaan klinis |
| **Kompetensi bibir** (memisahkan DHC 2a dari 3a) | Retraktor menghilangkan posisi bibir | Foto wajah istirahat |
| **Crossbite posterior** (DHC 4l) | Segmen bukal tertutup/terpotong retraktor | Foto oklusal / model |
| **Kesulitan bicara & mastikasi** (DHC 4m, 5m) | Laporan pasien | Anamnesis |
| **Trauma gingiva/palatal** (DHC 4f) | Palatal tidak terlihat dari frontal | Pemeriksaan klinis / foto oklusal |
| **Impaksi, supernumerary** (DHC 4x, 5i) | Butuh radiograf | Panoramik / CBCT |

**Konsekuensi ke desain sistemmu:** rubrik ini tidak akan pernah mereproduksi IOTN DHC penuh dari foto frontal saja. Kalau target akhirmu adalah DHC, kamu perlu input tambahan — atau membatasi klaim pada subset MOCDO yang memang frontal-computable (Crossbite anterior, Displacement, Overbite/openbite; sebagian Missing).

---

## 6. Protokol anotasi & QC

1. **Kriteria inklusi foto** — oklusi sentrik, retraktor terpasang, seluruh 13–23 dan 33–43 terlihat, tidak blur, tidak ada saliva/refleks yang menutupi tepi insisal. Buang yang tidak memenuhi *sebelum* skoring, bukan sesudah.
2. **Stratifikasi dentisi** — pisahkan permanen vs campuran. Untuk campuran, kecualikan F8 dan tandai datanya.
3. **Blind double-annotation** pada ≥ 20 % dataset (≈100 dari 500). Hitung **weighted kappa (linear)** dan **ICC** antar-anotator. Target κ_w ≥ 0,70; kalau di bawah itu, ambang di §2 belum cukup operasional — revisi, jangan dipaksakan.
4. **Kalibrasi berulang** — nilai ulang 30 foto acak setelah selesai semua 500, untuk mengukur *drift* intra-anotator. Ini yang paling sering dilupakan dan paling merusak label ML.
5. **Simpan skor fitur mentah (S_F1…S_F9), bukan hanya AC akhir.** Kalau nanti pembobotan direvisi, kamu bisa hitung ulang tanpa menganotasi ulang 500 foto. Ini menghemat berminggu-minggu.
6. **Urutan acak** — jangan skor foto berurutan per pasien atau per folder; urutan sistematis memicu anchoring.

---

## 7. Contoh terhitung

Foto yang sudah dinilai sebelumnya (crowding ringan–sedang, insisivus lateral atas sedikit rotasi, anterior bawah berjejal ringan, overbite sedang):

| Fitur | Nilai terukur | S |
|---|---|---|
| F1 displacement atas | 1,8 mm | 1 |
| F2 displacement bawah | 2,6 mm → band 2, diskon −1 | 1 |
| F3 rotasi | 15° | 1 |
| F4 spacing | 0,4 mm | 0 |
| F5 overbite | 35 % | 0 |
| F6 open bite | 0 | 0 |
| F7 crossbite | 0 | 0 |
| F8 missing/ektopik | tidak ada | 0 |
| F9 garis tengah | 1,2 mm | 1 |

```
S_max = 1 ; S_sum = 4
AC_raw = BASE[1] + 0.3 × (4 − 1) = 3.0 + 0.9 = 3.9
AC = 4        → band "tidak butuh perawatan"
```

Cocok dengan penilaian visual awal (4).

---

## 8. Kalibrasi ulang setelah 50 foto pertama — jangan lewati

Rubrik ini diuji lewat 48 unit test (`test_ac_scorer.py`): monotonisitas, keterjangkauan seluruh skor 1–10, batas ambang, determinisme, dan geometri OBB. Tapi **konsistensi internal ≠ kalibrasi yang benar.** Dua parameter di §3 dipilih berdasarkan penalaran, bukan data:

- `BASE` (kurva cembung)
- `0.3` (bobot ko-okurensi)

Pada simulasi populasi klinis plausibel, rubrik menghasilkan sebaran: **band 1–4 ≈ 34 %, band 5–7 ≈ 54 %, band 8–10 ≈ 12 %.** Studi epidemiologi AC umumnya melaporkan band borderline jauh lebih kecil (~15–20 %). Artinya rubrik ini kemungkinan **terlalu mudah mendorong kasus ke tengah** — tapi simulasi itu memakai parameter distribusi karanganku sendiri, jadi jangan diperlakukan sebagai bukti.

**Yang harus kamu lakukan:** setelah menganotasi 50 foto pertama, plot sebaran AC-nya. Kalau menumpuk di 5–7, turunkan bobot ko-okurensi (coba 0,2). Kalau terlalu banyak di 1–3, naikkan. Karena kamu menyimpan skor fitur mentah (§6 poin 5), kalibrasi ulang ini gratis — tidak perlu menganotasi ulang.

---

## 9. Sitasi

- Evans R, Shaw WC (1987) — *SCAN: Standardized Continuum of Aesthetic Need*, asal Aesthetic Component.
- Brook PH, Shaw WC (1989) — Dental Health Component & hierarki MOCDO; sumber semua ambang mm di §2.
- Little RM (1975) — *Irregularity Index*; kategori 0 / 1–3 / 4–6 / 7–9 / ≥10 mm.
