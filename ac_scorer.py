"""
AC-Proxy Scorer 1-10 — implementasi rubrik feature-based deterministik.
Pendamping: rubrik_ac_1-10.md

CATATAN: ini BUKAN IOTN Aesthetic Component resmi (AC asli = photo-matching
terhadap 10 foto referensi Evans & Shaw 1987). Ini proxy terkorelasi yang
dibangun dari fitur terukur agar reproducible. Lihat §0 dokumen rubrik.

Ambang mm diturunkan dari IOTN DHC (Brook & Shaw 1989).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional

__version__ = "1.0"

# ---------------------------------------------------------------- konstanta

BASE = {0: 1.0, 1: 3.0, 2: 4.5, 3: 6.5, 4: 8.5}
CO_OCCURRENCE_W = 0.30

# Lebar mesiodistal normatif (mm) — untuk kalibrasi & deteksi torsiversi.
MD_NORM_MM = {
    11: 8.5, 21: 8.5,   # insisivus sentral atas
    12: 6.5, 22: 6.5,   # insisivus lateral atas
    13: 7.6, 23: 7.6,   # kaninus atas
    31: 5.3, 41: 5.3,   # insisivus sentral bawah
    32: 5.9, 42: 5.9,   # insisivus lateral bawah
    33: 6.8, 43: 6.8,   # kaninus bawah
}
CROWN_HEIGHT_LOWER_CENTRAL_MM = 9.0


# ---------------------------------------------------------------- utilitas

def round_half_up(x: float) -> int:
    """round() Python memakai banker's rounding (2.5 -> 2). Rubrik butuh 2.5 -> 3."""
    return math.floor(x + 0.5)


def _band(value: float, edges: tuple[float, ...]) -> int:
    """edges = batas ATAS inklusif untuk S=0,1,2,3. Di atas batas terakhir -> 4."""
    for s, edge in enumerate(edges):
        if value <= edge:
            return s
    return 4


# ---------------------------------------------------------------- fitur

def s_displacement_upper(mm: float) -> int:
    """F1 — displacement titik kontak anterior atas (terbesar tunggal). DHC 1/2d/3d/4d."""
    return _band(mm, (1.0, 2.0, 4.0, 6.0))


def s_displacement_lower(mm: float) -> int:
    """F2 — idem anterior bawah, dengan diskon visibilitas 1 tingkat."""
    return max(0, _band(mm, (1.0, 2.0, 4.0, 6.0)) - 1)


def s_rotation(deg: float) -> int:
    """F3 — rotasi gabungan (tipping in-plane + torsiversi). Lihat rubrik §4.3."""
    return _band(deg, (10.0, 20.0, 35.0, 55.0))


def s_spacing(mm: float) -> int:
    """F4 — diastema/celah tunggal terbesar, anterior atas."""
    return _band(mm, (0.5, 1.5, 3.0, 5.0))


def s_overbite(pct: Optional[float], trauma: bool = False) -> int:
    """F5 — % mahkota insisivus bawah yang tertutup. Bidirectional: deep DAN reduced buruk."""
    if trauma:
        return 4
    if pct is None:
        return 0
    if 10.0 <= pct <= 40.0:
        return 0
    if pct < 10.0:                      # edge-to-edge / reduced overbite
        return 1
    return min(4, _band(pct, (60.0, 80.0, 100.0)) + 1)


def s_openbite(mm: float) -> int:
    """F6 — open bite anterior. DHC 2e/3e/4e.

    Ambang bawah 1.0 mm, BUKAN 0. DHC 2e berbunyi "open bite greater than 1 mm
    but less than or equal to 2 mm" — jadi celah <=1 mm bukan temuan. Ini juga
    berfungsi sebagai deadband terhadap noise pengukuran: selisih sub-milimeter
    dari OBB pada foto tidak bermakna klinis.
    """
    return _band(mm, (1.0, 2.0, 3.0, 4.0))


def s_crossbite(n_teeth: int) -> int:
    """F7 — jumlah gigi anterior atas yang lingual terhadap bawah."""
    return _band(n_teeth, (0, 1, 2, 4))


def s_midline(mm: float) -> int:
    """F9 — deviasi garis tengah dental atas vs bawah."""
    return _band(mm, (1.0, 2.0, 3.0, 4.0))


# ---------------------------------------------------------------- kasus

@dataclass
class Case:
    """Fitur terukur satu foto. Semua mm sudah dikalibrasi (lihat calibrate_scale)."""
    disp_upper_mm: float = 0.0
    disp_lower_mm: float = 0.0
    rotation_deg: float = 0.0
    spacing_mm: float = 0.0
    overbite_pct: Optional[float] = None
    overbite_trauma: bool = False
    openbite_mm: float = 0.0
    crossbite_n: int = 0
    missing_s: int = 0          # F8 kategorikal langsung 0-4, lihat rubrik §2
    midline_mm: float = 0.0

    # flag kualitas — tidak memengaruhi skor, tapi wajib disimpan
    mixed_dentition: bool = False
    calibration_unreliable: bool = False
    overbite_uncertain: bool = False


@dataclass
class Result:
    ac: int
    ac_raw: float
    s_max: int
    s_sum: int
    features: dict
    band: str
    flags: dict

    def as_dict(self) -> dict:
        return asdict(self)


SUBTHRESHOLD = {
    "disp_upper_mm": 0.5,
    "disp_lower_mm": 0.5,
    "rotation_deg": 5.0,
    "spacing_mm": 0.25,
    "midline_mm": 0.5,
}


def _has_subthreshold(case: Case) -> bool:
    """Ada iregularitas terukur tapi belum mencapai S=1 pada fitur manapun."""
    return any(getattr(case, k) >= v for k, v in SUBTHRESHOLD.items())


def _band_label(ac: int) -> str:
    """Banding rujukan AC resmi (Evans & Shaw)."""
    if ac <= 4:
        return "tidak butuh perawatan"
    if ac <= 7:
        return "borderline"
    return "butuh perawatan"


def score(case: Case) -> Result:
    """Hitung AC-proxy 1-10 dari fitur terukur. Deterministik."""
    feats = {
        "F1_disp_upper": s_displacement_upper(case.disp_upper_mm),
        "F2_disp_lower": s_displacement_lower(case.disp_lower_mm),
        "F3_rotation": s_rotation(case.rotation_deg),
        "F4_spacing": s_spacing(case.spacing_mm),
        "F5_overbite": s_overbite(case.overbite_pct, case.overbite_trauma),
        "F6_openbite": s_openbite(case.openbite_mm),
        "F7_crossbite": s_crossbite(case.crossbite_n),
        "F8_missing": max(0, min(4, case.missing_s)),
        "F9_midline": s_midline(case.midline_mm),
    }

    # Dentisi campuran: gigi "hilang" sering hanya belum erupsi. Kecualikan F8.
    if case.mixed_dentition:
        feats["F8_missing"] = 0

    s_max = max(feats.values())
    s_sum = sum(feats.values())
    raw = BASE[s_max] + CO_OCCURRENCE_W * (s_sum - s_max)

    # Aturan sub-ambang: tanpa ini AC=2 mustahil tercapai. Kalau S_max=0 maka
    # semua S=0, jadi S_sum=0 dan raw selalu tepat 1.0 -> AC=1; lompatan
    # berikutnya BASE[1]=3.0. Padahal pada skala AC asli, 1 vs 2 justru
    # membedakan "sempurna" dari "hampir sempurna" — kasus yang umum.
    if s_max == 0 and _has_subthreshold(case):
        raw = 2.0

    ac = max(1, min(10, round_half_up(raw)))

    return Result(
        ac=ac,
        ac_raw=round(raw, 2),
        s_max=s_max,
        s_sum=s_sum,
        features=feats,
        band=_band_label(ac),
        flags={
            "mixed_dentition": case.mixed_dentition,
            "calibration_unreliable": case.calibration_unreliable,
            "overbite_uncertain": case.overbite_uncertain,
        },
    )


# ---------------------------------------------------------------- OBB helpers

def md_px(w: float, h: float) -> float:
    """Lebar mesiodistal dalam piksel. Mahkota anterior lebih tinggi daripada lebar,
    jadi sisi pendek OBB = mesiodistal."""
    return min(w, h)


def calibrate_scale(md_px_11: float, md_px_21: float) -> tuple[float, bool]:
    """px -> mm dari lebar insisivus sentral atas.

    Returns (mm_per_px, unreliable). Ditandai unreliable kalau 11 dan 21 berbeda
    >15% — indikasi salah satunya rotasi, yang membuat SELURUH kalibrasi bias.
    """
    mean_px = (md_px_11 + md_px_21) / 2.0
    if mean_px <= 0:
        raise ValueError("lebar insisivus sentral harus > 0")
    asymmetry = abs(md_px_11 - md_px_21) / mean_px
    return MD_NORM_MM[11] / mean_px, asymmetry > 0.15


def torsiversion_deg(md_px_obs: float, fdi: int, mm_per_px: float) -> float:
    """Torsiversi dari penyempitan lebar proyeksi: w_proj = w_true * cos(phi).

    PENTING: theta dari OBB TIDAK menangkap ini. Gigi yang berputar pada sumbu
    panjangnya menghasilkan kotak yang MENYEMPIT, bukan kotak yang miring.
    """
    expected_px = MD_NORM_MM[fdi] / mm_per_px
    ratio = max(0.0, min(1.0, md_px_obs / expected_px))
    return math.degrees(math.acos(ratio))


def combine_rotation(tip_deg: float, torsi_deg: float) -> float:
    """Gabungkan tipping in-plane (dari theta) dengan torsiversi. Aproksimasi:
    keduanya rotasi pada sumbu ortogonal, digabung kuadratik."""
    return math.hypot(tip_deg, torsi_deg)


def obb_corners(cx: float, cy: float, w: float, h: float, theta: float
                ) -> list[tuple[float, float]]:
    """4 sudut OBB. theta dalam radian (konvensi Ultralytics xywhr)."""
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    out = []
    for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        out.append((cx + dx * cos_t - dy * sin_t,
                    cy + dx * sin_t + dy * cos_t))
    return out


def overbite_pct_from_obb(h_visible_px: float, mm_per_px: float) -> float:
    """% mahkota insisivus sentral bawah yang tertutup.

    ASUMSI KRITIS: OBB gigi bawah hanya membatasi bagian yang TERLIHAT.
    Kalau anotatormu melabeli seluruh gigi termasuk bagian tersembunyi di balik
    gigi atas, rumus ini rusak dan F5 tidak boleh dipakai. Verifikasi dulu.
    """
    expected_px = CROWN_HEIGHT_LOWER_CENTRAL_MM / mm_per_px
    return max(0.0, 100.0 * (1.0 - h_visible_px / expected_px))


if __name__ == "__main__":
    demo = Case(disp_upper_mm=1.8, disp_lower_mm=2.6, rotation_deg=15.0,
                spacing_mm=0.4, overbite_pct=35.0, midline_mm=1.2)
    r = score(demo)
    print(f"AC = {r.ac}  (raw {r.ac_raw}, S_max {r.s_max}, S_sum {r.s_sum}) — {r.band}")
    for k, v in r.features.items():
        print(f"  {k:16s} S={v}")
