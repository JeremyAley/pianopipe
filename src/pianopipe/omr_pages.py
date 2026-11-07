"""The OMR benchmark's pages (docs/omr-engine.md §4): a score engraved as printed piano pages, then damaged the way
phone photos and flatbed scans damage them, the score itself kept as the correct answer.

Engravers: Verovio (LGPL-3.0, a library) in four SMuFL fonts, and LilyPond (GPL-3.0, run as a separate program through
its musicxml2ly) in its own. Pages are US Letter at 300 dpi, greyscale.

Damages (each alone, so the benchmark can rank them, plus two mixes): tilt, perspective, curl (a book's page bending
toward the spine), crop-brace (the left edge cut through the brace, as a tight scan does), crop-clef (cut through the
clefs), crop-tight (every margin cut to the ink), blur, lighting (uneven light, grey paper), jpeg (noise and heavy
compression), lowdpi (150 dpi), and phone and flatbed mixes. Each is seeded by the case's name, so a page comes out
the same every run.

Needs numpy and Pillow, and for Verovio pages verovio and resvg_py (`pip install pianopipe[bench]`).
"""
from __future__ import annotations

import io
import random
import shutil
import subprocess
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path

DPI = 300
PAGE_W, PAGE_H = 2550, 3300  # US Letter at 300 dpi
LILYPOND = Path("D:/Tools/lilypond-2.24.4/bin")


@dataclass(frozen=True)
class Engraver:
    name: str
    kind: str  # "verovio" or "lilypond"
    font: str = ""  # Verovio's music font
    space: float = 20.0  # staff space in pixels at 300 dpi (20: a 7 mm staff, a usual printed book)


ENGRAVERS = {e.name: e for e in [
    Engraver("verovio-leipzig", "verovio", "Leipzig", 20),
    Engraver("verovio-bravura", "verovio", "Bravura", 22),
    Engraver("verovio-leland", "verovio", "Leland", 18),
    Engraver("verovio-petaluma", "verovio", "Petaluma", 21),
    Engraver("lilypond", "lilypond", space=20),
]}

DAMAGES = ("clean", "tilt", "perspective", "curl", "crop-brace", "crop-clef", "crop-tight", "blur", "lighting", "jpeg",
           "lowdpi", "phone", "flatbed")


# ---------------------------------------------------------------- engraving

def engrave(xml: str, engraver: Engraver, lilypond: Path = LILYPOND):
    """MusicXML text to page images (PIL, mode L, 300 dpi)."""
    if engraver.kind == "verovio":
        return _verovio(xml, engraver)
    return _lilypond(xml, engraver, lilypond)


def _verovio(xml: str, e: Engraver):
    import resvg_py
    import verovio
    from PIL import Image

    from .engrave import FONTS, SERIF

    units = 18 / e.space  # Verovio draws a staff space 18 units high
    tk = verovio.toolkit()
    tk.setOptions({
        "pageWidth": round(PAGE_W * units), "pageHeight": round(PAGE_H * units), "scale": round(100 / units),
        "font": e.font, "adjustPageHeight": False, "breaks": "auto", "footer": "none", "xmlIdSeed": 1,
        "pageMarginLeft": round(190 * units), "pageMarginRight": round(170 * units), "pageMarginTop": round(150 * units),
        "pageMarginBottom": round(200 * units),
    })
    if not tk.loadData(xml):
        raise ValueError("Verovio couldn't read the score")
    fonts = [str(p) for p in sorted(FONTS.glob("*.ttf"))]
    pages = []
    for k in range(1, tk.getPageCount() + 1):
        raw = bytes(resvg_py.svg_to_bytes(svg_string=tk.renderToSVG(k), background="#ffffff", width=PAGE_W,
                                          height=PAGE_H, skip_system_fonts=True, font_files=fonts, font_family=SERIF,
                                          serif_family=SERIF))
        pages.append(Image.open(io.BytesIO(raw)).convert("L"))
    return pages


def _lilypond(xml: str, e: Engraver, bin_dir: Path):
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        (t / "score.musicxml").write_text(xml, encoding="utf-8")
        python = bin_dir / "python.exe" if (bin_dir / "python.exe").exists() else Path(shutil.which("python3") or "python")
        run = dict(capture_output=True, text=True, cwd=tmp, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        r = subprocess.run([str(python), str(bin_dir / "musicxml2ly.py"), "-o", "score.ly", "score.musicxml"], **run)
        if r.returncode or not (t / "score.ly").exists():
            raise ValueError(f"musicxml2ly failed: {r.stderr[-300:]}")
        ly = (t / "score.ly").read_text(encoding="utf-8")
        size = round(e.space * 4 * 72 / DPI)  # staff height in points
        head = (f'#(set-default-paper-size "letter")\n#(set-global-staff-size {size})\n'
                '\\header { tagline = ##f }\n')
        ly = ly.replace("\n", "\n" + head, 1) if ly.startswith("\\version") else head + ly
        (t / "score.ly").write_text(ly, encoding="utf-8")
        exe = bin_dir / "lilypond.exe" if (bin_dir / "lilypond.exe").exists() else bin_dir / "lilypond"
        r = subprocess.run([str(exe), "--png", f"-dresolution={DPI}", "-o", "out", "score.ly"], **run)
        files = sorted(t.glob("out*.png"), key=lambda p: int("".join(c for c in p.stem.rpartition("page")[2] if c.isdigit()) or 0))
        if r.returncode or not files:
            raise ValueError(f"lilypond failed: {r.stderr[-300:]}")
        return [_fit(Image.open(f).convert("L")) for f in files]


def _fit(img):
    """A page exactly PAGE_W x PAGE_H (LilyPond's can be a pixel off)."""
    from PIL import Image

    if img.size == (PAGE_W, PAGE_H):
        return img
    out = Image.new("L", (PAGE_W, PAGE_H), 255)
    out.paste(img.crop((0, 0, min(img.width, PAGE_W), min(img.height, PAGE_H))))
    return out


# ---------------------------------------------------------------- finding the staves (for the crops)

def staves(img) -> tuple[list[tuple[int, int]], float]:
    """Staff lines as (y, x where the line starts), and the staff space. Lines are rows dark across much of the page."""
    import numpy as np

    dark = np.asarray(img) < 128
    h, w = dark.shape
    rows = np.flatnonzero(dark.sum(axis=1) > 0.35 * w)
    groups, cur = [], []
    for y in rows:
        if cur and y - cur[-1] > 1:
            groups.append(cur)
            cur = []
        cur.append(int(y))
    if cur:
        groups.append(cur)
    ys = [round(sum(g) / len(g)) for g in groups]
    gaps = sorted(b - a for a, b in zip(ys, ys[1:]))
    space = float(gaps[len(gaps) // 3]) if gaps else 20.0  # most gaps are inside staves
    lines = []
    run = max(8, int(4 * space))
    for y in ys:
        row = dark[max(0, y - 1):y + 2].any(axis=0).astype(np.int32)
        sums = np.convolve(row, np.ones(run, dtype=np.int32), "valid")
        full = np.flatnonzero(sums == run)
        if len(full):
            lines.append((y, int(full[0])))
    return lines, space


def staff_left(img) -> tuple[int, float]:
    """Where most systems' staff lines start, and the staff space."""
    lines, space = staves(img)
    xs = sorted(x for _, x in lines)
    return (xs[len(xs) // 2] if xs else 0), space


# ---------------------------------------------------------------- damage

def damage(pages: list, kind: str, seed: str) -> tuple[list, int]:
    """Damaged copies of the pages and their dpi."""
    rnd = random.Random(zlib.crc32(seed.encode()))
    if kind == "flatbed":
        out = [_noise(_paper(_rotate(p, rnd.uniform(0.6, 1.0) * rnd.choice((-1, 1))), 0.93), 4, rnd) for p in pages]
        return [_scale(p, 200 / DPI) for p in out], 200
    if kind == "lowdpi":
        return [_scale(p, 150 / DPI) for p in pages], 150
    out = []
    for p in pages:
        if kind == "clean":
            q = p
        elif kind == "tilt":
            q = _rotate(p, rnd.uniform(2.5, 3.5) * rnd.choice((-1, 1)))
        elif kind == "perspective":
            q = _perspective(p, rnd, 0.06)
        elif kind == "curl":
            q = _curl(p, rnd, 1.0)
        elif kind == "crop-brace":
            x, space = staff_left(p)
            q = p.crop((max(0, x + 2), 0, p.width, p.height))
        elif kind == "crop-clef":
            x, space = staff_left(p)
            q = p.crop((max(0, round(x + 1.6 * space)), 0, p.width, p.height))
        elif kind == "crop-tight":
            q = _crop_tight(p)
        elif kind == "blur":
            q = _blur(p, 2.0)
        elif kind == "lighting":
            q = _lighting(p, rnd)
        elif kind == "jpeg":
            q = _jpeg(_noise(p, 10, rnd), 20)
        elif kind == "phone":
            q = _jpeg(_blur(_lighting(_curl(_perspective(p, rnd, 0.04), rnd, 0.6), rnd), 1.2), 40)
        else:
            raise ValueError(f"unknown damage {kind!r}")
        out.append(q)
    return out, DPI


def _rotate(img, degrees: float):
    from PIL import Image

    return img.rotate(degrees, resample=Image.BICUBIC, expand=True, fillcolor=255)


def _scale(img, f: float):
    from PIL import Image

    return img.resize((round(img.width * f), round(img.height * f)), Image.LANCZOS)


def _blur(img, radius: float):
    from PIL import ImageFilter

    return img.filter(ImageFilter.GaussianBlur(radius))


def _jpeg(img, quality: int):
    from PIL import Image

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("L")


def _noise(img, sigma: float, rnd: random.Random):
    import numpy as np
    from PIL import Image

    g = np.random.default_rng(rnd.randrange(1 << 30))
    a = np.asarray(img, dtype=np.float32) + g.normal(0, sigma, (img.height, img.width))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _paper(img, white: float):
    """Paper a little grey: white becomes `white` (0-1)."""
    import numpy as np
    from PIL import Image

    a = np.asarray(img, dtype=np.float32) * white
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _crop_tight(img):
    import numpy as np

    dark = np.asarray(img) < 160
    ys, xs = np.flatnonzero(dark.any(axis=1)), np.flatnonzero(dark.any(axis=0))
    if not len(ys):
        return img
    m = 4
    return img.crop((max(0, xs[0] - m), max(0, ys[0] - m), min(img.width, xs[-1] + m + 1), min(img.height, ys[-1] + m + 1)))


def _lighting(img, rnd: random.Random):
    """A light falling off across the page and a soft shadow (a hand or the phone), on greyish paper."""
    import numpy as np
    from PIL import Image

    h, w = img.height, img.width
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rnd.uniform(0, 2 * np.pi)
    ramp = (np.cos(ang) * (xx / w - 0.5) + np.sin(ang) * (yy / h - 0.5))
    light = 0.86 + 0.28 * ramp
    cx, cy, r = rnd.uniform(0.1, 0.9) * w, rnd.uniform(0.1, 0.9) * h, rnd.uniform(0.15, 0.3) * w
    light -= 0.32 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))
    a = np.asarray(img, dtype=np.float32) / 255.0
    out = (25 + a * 205) * np.clip(light, 0.35, 1.05)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _sample(a, sy, sx, fill: float = 255.0):
    """Bilinear sample of array a at float coordinates; outside the page is `fill`."""
    import numpy as np

    h, w = a.shape
    y0, x0 = np.floor(sy).astype(np.int64), np.floor(sx).astype(np.int64)
    fy, fx = sy - y0, sx - x0
    out = np.zeros(sy.shape, dtype=np.float32)
    for dy, wy in ((0, 1 - fy), (1, fy)):
        for dx, wx in ((0, 1 - fx), (1, fx)):
            yi, xi = y0 + dy, x0 + dx
            inside = (yi >= 0) & (yi < h) & (xi >= 0) & (xi < w)
            v = np.where(inside, a[np.clip(yi, 0, h - 1), np.clip(xi, 0, w - 1)], fill)
            out += wy * wx * v
    return out


def _curl(img, rnd: random.Random, strength: float):
    """The page bending toward a spine on one side: lines near it curve and bunch up, and it darkens there."""
    import numpy as np
    from PIL import Image

    a = np.asarray(img, dtype=np.float32)
    h, w = a.shape
    left = rnd.random() < 0.5
    w0 = 0.4 * w
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = xx if left else (w - 1 - xx)
    s = np.clip((w0 - d) / w0, 0, 1) ** 2  # 0 on the flat page, 1 at the spine
    bend = strength * 0.035 * h
    sy = yy + s * bend * (0.4 + 0.6 * (yy - h / 2) / (h / 2))  # lines curve, more so away from the middle
    sx = xx + (-1 if left else 1) * s * strength * 0.06 * w  # and the page is foreshortened toward the spine
    out = _sample(a, sy, sx, fill=60.0) * (1 - 0.5 * strength * s)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _perspective(img, rnd: random.Random, amount: float):
    """A phone held a little off square: the page a trapezoid, slightly turned."""
    import numpy as np
    from PIL import Image

    w, h = img.width, img.height
    k = amount * w
    j = lambda: rnd.uniform(-0.3, 0.3) * k  # noqa: E731
    # where each corner of the output comes from in the page (top-left, top-right, bottom-right, bottom-left)
    src = [(-k + j(), -0.5 * k + j()), (w + k + j(), -0.5 * k + j()), (w + 0.2 * k + j(), h + 0.3 * k + j()),
           (-0.2 * k + j(), h + 0.3 * k + j())]
    dst = [(0, 0), (w, 0), (w, h), (0, h)]
    rows, rhs = [], []
    for (x, y), (u, v) in zip(dst, src):
        rows += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        rhs += [u, v]
    coeffs = np.linalg.solve(np.array(rows, dtype=np.float64), np.array(rhs, dtype=np.float64))
    return img.transform((w, h), Image.PERSPECTIVE, tuple(coeffs), Image.BICUBIC, fillcolor=70)


def save_tiff(pages: list, path: str | Path, dpi: int) -> None:
    """One multi-page TIFF, as Audiveris takes a book's pages."""
    pages[0].save(path, save_all=True, append_images=pages[1:], compression="tiff_lzw", dpi=(dpi, dpi))
