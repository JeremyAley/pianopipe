"""Sheet music pages for a level: notation (notation.py) laid out by Verovio, drawn to greyscale PNG pages, with a map
of where each bar is on them (docs/sheet-music.md; the files are described in docs/formats.md).

Needs the optional extras (`pip install pianopipe[sheet]`): verovio (LGPL-3.0, the engraver, used as an unmodified
library), resvg_py (MIT, SVG to pixels) and Pillow. The rest of pianopipe works without them.

Pages are sized for the app's book (about 40 x 52 cm at a metre): 1200 x 1560 pixels with about 16 pixels between staff
lines, so a staff is about 1.2 degrees tall there. Verovio's ids are seeded, so the same level engraves to the same
files and the app doesn't download unchanged pages again.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from fractions import Fraction as Q
from pathlib import Path

from . import midi_io, notation

PAGE_W, PAGE_H = 1200, 1560  # pixels
SCALE = 90  # Verovio's zoom in percent: staff space = 18 units * SCALE / 100 = 16 px
MARGIN = 70  # Verovio units (at this scale, 63 px)
SHEET_VERSION = 1
MAX_PAGES = 64
SVG = "{http://www.w3.org/2000/svg}"
XLINK_HREF = "{http://www.w3.org/1999/xlink}href"


def fingering_for(level: int) -> bool:
    """Finger numbers in the notation for levels 1-3 (learning to read); level 4 is the plain score."""
    return level <= 3


def available() -> bool:
    try:
        import resvg_py  # noqa: F401
        import verovio  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass
class Sheet:
    json_path: Path
    sha256: str
    pages: int
    data: dict


def _toolkit():
    import verovio

    tk = verovio.toolkit()
    tk.setOptions({
        "pageWidth": round(PAGE_W * 100 / SCALE), "pageHeight": round(PAGE_H * 100 / SCALE), "scale": SCALE,
        "adjustPageHeight": False, "svgHtml5": True, "breaks": "auto", "header": "none", "footer": "none",
        "pageMarginTop": MARGIN, "pageMarginBottom": MARGIN, "pageMarginLeft": MARGIN, "pageMarginRight": MARGIN,
        "spacingSystem": 6, "spacingStaff": 8, "justifyVertically": True, "xmlIdSeed": 1, "mnumInterval": 0,
    })
    return tk


def _translate(transform: str) -> tuple[float, float]:
    m = re.search(r"translate\(\s*(-?[\d.]+)[ ,]+(-?[\d.]+)", transform or "")
    return (float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0)


def _classes(el) -> set[str]:
    return set((el.get("class") or "").split())


def page_layout(svg: str) -> tuple[list[dict], dict[str, float]]:
    """(systems, note x): each system's measures in order with their extent, and each note's head x, all as fractions
    of the page (x from the left, y from the top)."""
    root = ET.fromstring(svg)
    inner = next(e for e in root.iter(f"{SVG}svg") if "definition-scale" in _classes(e))
    _, _, vbw, vbh = (float(v) for v in inner.get("viewBox").split())
    margin = next(e for e in inner.iter(f"{SVG}g") if "page-margin" in _classes(e))
    mx, my = _translate(margin.get("transform"))
    systems = []
    notes: dict[str, float] = {}
    for sys_el in margin.iter(f"{SVG}g"):
        if "system" not in _classes(sys_el):
            continue
        measures = []
        for m_el in sys_el.iter(f"{SVG}g"):
            if "measure" not in _classes(m_el):
                continue
            xs, ys = [], []
            for st in m_el.iter(f"{SVG}g"):
                if "staff" not in _classes(st):
                    continue
                for path in st.findall(f"{SVG}path"):
                    nums = [float(v) for v in re.findall(r"-?[\d.]+", path.get("d", ""))]
                    if len(nums) == 4 and nums[1] == nums[3]:  # a staff line
                        xs += [nums[0], nums[2]]
                        ys.append(nums[1])
            if not xs:
                continue
            measures.append({"left": (min(xs) + mx) / vbw, "right": (max(xs) + mx) / vbw,
                             "top": (min(ys) + my) / vbh, "bottom": (max(ys) + my) / vbh})
            for n_el in m_el.iter(f"{SVG}g"):
                if "note" in _classes(n_el) and n_el.get("data-id"):
                    head = next((u for u in n_el.iter(f"{SVG}use")), None)
                    if head is not None:
                        notes[n_el.get("data-id")] = (_translate(head.get("transform"))[0] + mx) / vbw
        if measures:
            systems.append({"top": min(m["top"] for m in measures), "bottom": max(m["bottom"] for m in measures),
                            "measures": measures})
    return systems, notes


def _png(svg: str) -> bytes:
    import resvg_py
    from PIL import Image

    raw = bytes(resvg_py.svg_to_bytes(svg_string=svg, background="#ffffff", width=PAGE_W, height=PAGE_H))
    img = Image.open(io.BytesIO(raw)).convert("L")
    if img.size != (PAGE_W, PAGE_H):
        img = img.resize((PAGE_W, PAGE_H))
    # Black and white, 1 bit a pixel: a third the size of 16 greys (about 16 KB a page, 2026-10-26). The edges stay
    # smooth in the headset, because the book shows each page at under half its pixel width and the mipmaps average them.
    img = img.point(lambda v: 255 if v > 160 else 0).convert("1", dither=Image.Dither.NONE)
    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


def engrave(level_midi: str | Path, out_dir: str | Path, stem: str, level: int, title: str = "",
            fingering: bool | None = None, keep_musicxml: bool = False) -> Sheet:
    """Engraves an app-ready level file into out_dir/<stem>.sheet.json and out_dir/<stem>-p<N>.png. Returns the
    sheet.json's path, sha256 and page count."""
    song = midi_io.load(level_midi)
    score = notation.build(song, level, title)
    xml = notation.to_musicxml(score, fingering_for(level) if fingering is None else fingering)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if keep_musicxml:
        (out_dir / f"{stem}.musicxml").write_text(xml, encoding="utf-8")
    tk = _toolkit()
    if not tk.loadData(xml):
        raise ValueError(f"Verovio couldn't read the notation of {level_midi}")
    count = tk.getPageCount()
    if count > MAX_PAGES:
        raise ValueError(f"{count} pages is more than {MAX_PAGES}")
    for old in out_dir.glob(f"{stem}-p*.png"):
        old.unlink()
    pages, systems, measures = [], [], []
    note_x: dict[str, float] = {}
    for page in range(1, count + 1):
        svg = tk.renderToSVG(page)
        sys_list, xs = page_layout(svg)
        note_x.update(xs)
        for s in sys_list:
            si = len(systems)
            systems.append({"page": page, "top": round(s["top"], 4), "bottom": round(s["bottom"], 4)})
            for m in s["measures"]:
                measures.append({"bar": len(measures) + 1, "system": si, "left": round(m["left"], 4),
                                 "right": round(m["right"], 4), "onsets": []})
        png = _png(svg)
        name = f"{stem}-p{page}.png"
        (out_dir / name).write_bytes(png)
        pages.append({"file": name, "sha256": hashlib.sha256(png).hexdigest()})
    if len(measures) != len(score.bars):
        raise ValueError(f"engraved {len(measures)} bars, the level has {len(score.bars)}")
    # Where each start in a bar is drawn: (fraction of the bar's length, x), from Verovio's time map.
    starts = [Q(b.start - score.bars[0].start, score.tpb) for b in score.bars] + \
             [Q(score.bars[-1].end - score.bars[0].start, score.tpb)]
    onsets: dict[int, dict[float, float]] = {}
    bi = 0
    for entry in tk.renderToTimemap({"includeMeasures": False, "includeRests": False}):
        q = Q(entry.get("qstamp", 0)).limit_denominator(96)
        xs = [note_x[i] for i in entry.get("on", []) if i in note_x]
        if not xs:
            continue
        while bi + 1 < len(score.bars) and starts[bi + 1] <= q:
            bi += 1
        frac = float((q - starts[bi]) / (starts[bi + 1] - starts[bi]))
        cur = onsets.setdefault(bi, {})
        cur[round(frac, 4)] = round(min(xs + [cur.get(round(frac, 4), 2.0)]), 4)
    for i, m in enumerate(measures):
        m["onsets"] = [v for f, x in sorted(onsets.get(i, {}).items()) for v in (f, x)]  # flat: fraction, x, fraction, x
    data = {
        "version": SHEET_VERSION, "level": level, "bars": len(measures), "width": PAGE_W, "height": PAGE_H,
        "pages": pages, "systems": systems, "measures": measures,
    }
    path = out_dir / f"{stem}.sheet.json"
    raw = (json.dumps(data, separators=(",", ":")) + "\n").encode("utf-8")
    path.write_bytes(raw)  # bytes: no newline translation on Windows, so the sha256 is the file's
    return Sheet(path, hashlib.sha256(raw).hexdigest(), len(pages), data)
