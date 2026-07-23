"""
Potong grid referensi IOTN Aesthetic Component (10 foto) menjadi 10 file:
  ac_grade_01.png ... ac_grade_10.png

Cara pakai:
  1. Simpan gambar grid ke folder ini sebagai: ac_reference_grid.png
  2. Jalankan:  python crop_references.py
     (atau:     python crop_references.py path/ke/grid.png )

Metode: proyeksi kecerahan. Latar tabel putih (terang), foto gelap.
Kolom/baris "gelap" = area foto -> dideteksi otomatis, jadi tahan terhadap
pergeseran piksel/ukuran gambar.
"""
import sys, numpy as np
from PIL import Image

SRC = sys.argv[1] if len(sys.argv) > 1 else "ac_reference_grid.png"
PAD_FRAC = 0.015   # buang sedikit tepi tiap sel

def runs(mask, min_len):
    """Kembalikan (start,end) untuk tiap run True yang panjang >= min_len."""
    out, s = [], None
    for i, v in enumerate(mask):
        if v and s is None: s = i
        elif not v and s is not None:
            if i - s >= min_len: out.append((s, i)); 
            s = None
    if s is not None and len(mask) - s >= min_len: out.append((s, len(mask)))
    return out

def main():
    im = Image.open(SRC).convert("RGB")
    a = np.asarray(im, dtype=np.float32).mean(2)   # grayscale HxW
    H, W = a.shape
    thr = 200.0                                    # < thr dianggap "gelap" (foto)

    col_dark = (a.mean(0) < thr)
    row_dark = (a.mean(1) < thr)

    # 2 kolom foto terlebar, urut kiri->kanan
    cols = sorted(runs(col_dark, int(W*0.05)), key=lambda r: r[1]-r[0], reverse=True)[:2]
    cols = sorted(cols, key=lambda r: r[0])
    # 5 baris foto terlebar, urut atas->bawah
    rows = sorted(runs(row_dark, int(H*0.05)), key=lambda r: r[1]-r[0], reverse=True)[:5]
    rows = sorted(rows, key=lambda r: r[0])

    if len(cols) != 2 or len(rows) != 5:
        print(f"[!] deteksi meleset (cols={len(cols)}, rows={len(rows)}). "
              f"Coba sesuaikan 'thr' di script. Menyimpan debug_projection.png")
        return

    px, py = int(W*PAD_FRAC), int(H*PAD_FRAC)
    saved = []
    # grade 1-5 = kolom kiri (top->bottom), 6-10 = kolom kanan (top->bottom)
    for ci, (x0, x1) in enumerate(cols):
        for ri, (y0, y1) in enumerate(rows):
            grade = ci*5 + ri + 1
            box = (x0+px, y0+py, x1-px, y1-py)
            crop = im.crop(box)
            name = f"ac_grade_{grade:02d}.png"
            crop.save(name)
            saved.append((name, crop.size))
    for n, s in sorted(saved):
        print(f"  saved {n}  {s}")
    print(f"[OK] {len(saved)} foto referensi tersimpan.")

if __name__ == "__main__":
    main()
