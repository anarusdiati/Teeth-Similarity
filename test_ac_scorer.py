"""Uji konsistensi internal rubrik AC-proxy. Jalankan: python test_ac_scorer.py"""

import math
import sys

from ac_scorer import (
    BASE, Case, calibrate_scale, combine_rotation, md_px, obb_corners,
    overbite_pct_from_obb, round_half_up, s_displacement_lower,
    s_displacement_upper, s_overbite, score, torsiversion_deg,
)

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILS.append(name)


print("\n[1] Reproduksi kasus tervalidasi (foto yang dinilai manual = 4)")
r = score(Case(disp_upper_mm=1.8, disp_lower_mm=2.6, rotation_deg=15.0,
               spacing_mm=0.4, overbite_pct=35.0, midline_mm=1.2))
check("AC == 4", r.ac == 4, f"dapat {r.ac} (raw {r.ac_raw})")
check("S_max == 1", r.s_max == 1, f"dapat {r.s_max}")
check("S_sum == 4", r.s_sum == 4, f"dapat {r.s_sum}")

print("\n[2] Titik ekstrem")
check("oklusi ideal -> 1", score(Case(overbite_pct=25.0)).ac == 1)
check("semua berat -> 10", score(Case(
    disp_upper_mm=8, disp_lower_mm=8, rotation_deg=70, spacing_mm=6,
    overbite_pct=100, overbite_trauma=True, openbite_mm=6,
    crossbite_n=6, missing_s=4, midline_mm=6)).ac == 10)
check("satu kelainan berat tunggal -> 9",
      score(Case(disp_upper_mm=8.0, overbite_pct=25.0)).ac == 9,
      f"dapat {score(Case(disp_upper_mm=8.0, overbite_pct=25.0)).ac}")

print("\n[3] Monotonisitas: keparahan naik tidak boleh menurunkan AC")
prev = 0
for mm in [x / 4 for x in range(0, 45)]:
    ac = score(Case(disp_upper_mm=mm, overbite_pct=25.0)).ac
    if ac < prev:
        check(f"monoton di disp={mm}", False, f"{prev} -> {ac}")
        break
    prev = ac
else:
    check("AC monoton naik terhadap displacement atas", True)

print("\n[4] Rentang keluaran selalu 1..10 (sweep acak)")
import random
random.seed(0)
ok = True
for _ in range(20000):
    c = Case(
        disp_upper_mm=random.uniform(0, 12), disp_lower_mm=random.uniform(0, 12),
        rotation_deg=random.uniform(0, 90), spacing_mm=random.uniform(0, 8),
        overbite_pct=random.uniform(0, 100), overbite_trauma=random.random() < .1,
        openbite_mm=random.uniform(0, 8), crossbite_n=random.randint(0, 6),
        missing_s=random.randint(0, 4), midline_mm=random.uniform(0, 6))
    res = score(c)
    if not (1 <= res.ac <= 10) or not (0 <= res.s_max <= 4):
        ok = False
        break
check("20k kasus acak tetap di rentang 1-10", ok)

print("\n[5] Diskon visibilitas anterior bawah")
check("bawah 1 tingkat lebih ringan daripada atas (3mm)",
      s_displacement_lower(3.0) == s_displacement_upper(3.0) - 1)
check("bawah tidak pernah negatif", s_displacement_lower(0.2) == 0)

print("\n[6] Overbite bidirectional (deep DAN reduced sama-sama dihukum)")
check("normal 25% -> 0", s_overbite(25.0) == 0)
check("edge-to-edge 5% -> 1", s_overbite(5.0) == 1)
check("deep 70% -> 2", s_overbite(70.0) == 2, f"dapat {s_overbite(70.0)}")
check("complete 95% -> 3", s_overbite(95.0) == 3, f"dapat {s_overbite(95.0)}")
check("trauma -> 4 apapun pct", s_overbite(25.0, trauma=True) == 4)
check("None -> 0 (tidak menghukum data hilang)", s_overbite(None) == 0)

print("\n[6b] Open bite: deadband 1mm (DHC 2e = '>1mm', bukan '>0')")
from ac_scorer import s_openbite
check("celah 0.4mm (noise) -> 0", s_openbite(0.4) == 0)
check("celah 1.0mm -> 0", s_openbite(1.0) == 0)
check("celah 1.5mm -> 1", s_openbite(1.5) == 1)
check("celah 4.5mm -> 4", s_openbite(4.5) == 4)

print("\n[7] Batas ambang tepat (nilai persis di edge masuk band bawah)")
check("disp 2.0mm -> S=1 (bukan 2)", s_displacement_upper(2.0) == 1)
check("disp 2.01mm -> S=2", s_displacement_upper(2.01) == 2)
check("disp 4.0mm -> S=2", s_displacement_upper(4.0) == 2)

print("\n[8] round_half_up, bukan banker's rounding")
check("2.5 -> 3", round_half_up(2.5) == 3)
check("3.5 -> 4", round_half_up(3.5) == 4)
check("berbeda dari round() bawaan", round(2.5) == 2 and round_half_up(2.5) == 3)

print("\n[9] BASE cembung (justifikasi nonlinieritas di rubrik §3)")
d = [BASE[i + 1] - BASE[i] for i in range(4)]
check("selisih BASE membesar (2.0, 1.5, 2.0, 2.0)", d[-1] >= d[1],
      f"selisih {d}")
check("S_max=4 sendirian mencapai band 'butuh perawatan'",
      score(Case(missing_s=4, overbite_pct=25.0)).ac >= 8)

print("\n[10] Dentisi campuran mengecualikan F8")
mixed = score(Case(missing_s=4, overbite_pct=25.0, mixed_dentition=True))
check("F8 dinolkan saat mixed_dentition", mixed.features["F8_missing"] == 0)
check("flag ikut tersimpan di output", mixed.flags["mixed_dentition"] is True)

print("\n[11] Kalibrasi px->mm")
scale, unreliable = calibrate_scale(100.0, 100.0)
check("100px = 8.5mm -> 0.085 mm/px", math.isclose(scale, 0.085))
check("simetris -> reliable", unreliable is False)
_, unreliable2 = calibrate_scale(100.0, 70.0)
check("asimetri 35% -> ditandai unreliable", unreliable2 is True)

print("\n[12] Torsiversi dari penyempitan lebar (bukan dari theta)")
mm_per_px = 0.085
full_px = 8.5 / mm_per_px                      # 11 tanpa rotasi
check("lebar penuh -> torsiversi 0 deg",
      abs(torsiversion_deg(full_px, 11, mm_per_px)) < 1e-6)
half_px = full_px * math.cos(math.radians(60))  # diputar 60 deg
check("lebar x cos(60) -> terdeteksi ~60 deg",
      abs(torsiversion_deg(half_px, 11, mm_per_px) - 60.0) < 0.01,
      f"dapat {torsiversion_deg(half_px, 11, mm_per_px):.2f}")
check("lebar > ekspektasi di-clamp ke 0 (bukan NaN/crash)",
      torsiversion_deg(full_px * 1.3, 11, mm_per_px) == 0.0)
check("gabungan tipping 30 + torsi 40 = 50",
      math.isclose(combine_rotation(30, 40), 50.0))

print("\n[13] Geometri OBB")
c = obb_corners(0, 0, 4, 2, 0.0)
check("OBB tanpa rotasi -> sudut benar", sorted(c) == sorted(
    [(-2.0, -1.0), (2.0, -1.0), (2.0, 1.0), (-2.0, 1.0)]))
c90 = obb_corners(0, 0, 4, 2, math.pi / 2)
check("rotasi 90 deg menukar rentang x/y",
      abs(max(p[1] for p in c90) - 2.0) < 1e-9)
check("md_px pakai sisi pendek", md_px(60.0, 100.0) == 60.0)

print("\n[14] Overbite dari OBB")
mm_per_px = 0.085
full_h = 9.0 / mm_per_px
check("mahkota bawah penuh terlihat -> 0% overbite",
      abs(overbite_pct_from_obb(full_h, mm_per_px)) < 1e-9)
check("separuh terlihat -> ~50%",
      abs(overbite_pct_from_obb(full_h * 0.5, mm_per_px) - 50.0) < 1e-9)
check("tidak pernah negatif", overbite_pct_from_obb(full_h * 1.5, mm_per_px) == 0.0)

print("\n[15] Reachability: setiap skor 1-10 harus bisa tercapai")
reachable = set()
random.seed(2)
# Sampling uniform lebar tidak pernah menghasilkan kasus ringan (9 fitur
# independen -> hampir pasti ada yang berat). Skala keparahan per-kasus dulu.
for _ in range(50000):
    k = random.choice([0.02, 0.1, 0.3, 0.6, 1.0])   # pengali keparahan
    reachable.add(score(Case(
        disp_upper_mm=random.uniform(0, 9) * k, disp_lower_mm=random.uniform(0, 9) * k,
        rotation_deg=random.uniform(0, 70) * k, spacing_mm=random.uniform(0, 6) * k,
        overbite_pct=25.0 + (random.uniform(0, 75) * k),
        openbite_mm=random.uniform(0, 6) * k,
        crossbite_n=int(random.randint(0, 5) * k), missing_s=int(random.randint(0, 4) * k),
        midline_mm=random.uniform(0, 5) * k)).ac)
missing = sorted(set(range(1, 11)) - reachable)
check("semua skor 1-10 tercapai", not missing, f"tidak tercapai: {missing}")
check("oklusi benar-benar sempurna tetap 1",
      score(Case(overbite_pct=25.0)).ac == 1)
check("iregularitas sub-ambang -> 2",
      score(Case(disp_upper_mm=0.7, rotation_deg=6.0, overbite_pct=25.0)).ac == 2,
      f"dapat {score(Case(disp_upper_mm=0.7, rotation_deg=6.0, overbite_pct=25.0)).ac}")
check("aturan sub-ambang tidak merusak kasus tervalidasi",
      score(Case(disp_upper_mm=1.8, disp_lower_mm=2.6, rotation_deg=15.0,
                 spacing_mm=0.4, overbite_pct=35.0, midline_mm=1.2)).ac == 4)

print("\n[16] Determinisme")
c = Case(disp_upper_mm=3.3, rotation_deg=22.0, overbite_pct=55.0, crossbite_n=1)
check("100 pemanggilan identik", len({score(c).ac for _ in range(100)}) == 1)

print("\n" + "=" * 60)
if FAILS:
    print(f"{len(FAILS)} GAGAL: {FAILS}")
    sys.exit(1)
print("Semua uji lolos.")

print("\n[Distribusi] sweep 20k kasus acak uniform:")
random.seed(1)
hist = {i: 0 for i in range(1, 11)}
for _ in range(20000):
    hist[score(Case(
        disp_upper_mm=random.uniform(0, 8), disp_lower_mm=random.uniform(0, 8),
        rotation_deg=random.uniform(0, 60), spacing_mm=random.uniform(0, 5),
        overbite_pct=random.uniform(0, 100), openbite_mm=random.uniform(0, 5),
        crossbite_n=random.randint(0, 4), missing_s=random.randint(0, 2),
        midline_mm=random.uniform(0, 4))).ac] += 1
for k, v in hist.items():
    print(f"  AC {k:2d} | {'#' * (v // 100):<40} {v * 100 / 20000:5.1f}%")
