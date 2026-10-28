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
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction as Q
from pathlib import Path

from . import midi_io, musicxml, notation

PAGE_W, PAGE_H = 1200, 1560  # pixels
SCALE = 90  # Verovio's zoom in percent: staff space = 18 units * SCALE / 100 = 16 px
MARGIN = 70  # Verovio units (at this scale, 63 px)
SHEET_VERSION = 1
SHEET_VERSION_REPEATS = 2  # measures written once, each listing the bars it is played as (`plays`)
MAX_PAGES = 64
SOURCE_MISSING = 0.01  # share of a level's notes a score's notation may leave without a head (see _engrave)
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
    source: bool = False  # engraved from the source score's own notation (not from the level file)
    note: str = ""  # why a source score's notation wasn't used


class SourceMismatch(ValueError):
    """The source score's notation can't stand for the level: its bars or measures don't line up with the app's."""


def _toolkit():
    import verovio

    tk = verovio.toolkit()
    tk.setOptions({
        "pageWidth": round(PAGE_W * 100 / SCALE), "pageHeight": round(PAGE_H * 100 / SCALE), "scale": SCALE,
        "adjustPageHeight": False, "svgHtml5": True, "breaks": "auto", "header": "none", "footer": "none",
        "pageMarginTop": MARGIN, "pageMarginBottom": MARGIN, "pageMarginLeft": MARGIN, "pageMarginRight": MARGIN,
        "spacingSystem": 6, "spacingStaff": 8, "justifyVertically": True, "xmlIdSeed": 1, "mnumInterval": 0,
        # The time map in written order: a score engraved from its source plays repeats the app has unrolled, and the
        # bar map lists each measure's played bars itself.
        "expandNever": True,
    })
    return tk


def _translate(transform: str) -> tuple[float, float]:
    m = re.search(r"translate\(\s*(-?[\d.]+)[ ,]+(-?[\d.]+)", transform or "")
    return (float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0)


def _classes(el) -> set[str]:
    return set((el.get("class") or "").split())


# Notehead widths in SMuFL glyph units (1000 an em, 4 staff spaces), for the head's centre: whole, half, black.
HEAD_WIDTH = {"E0A2": 405, "E0A3": 314, "E0A4": 314}
HEAD_WIDTH_DEFAULT = 314


@dataclass
class Head:
    left: float  # the head's left edge (where the bar's cursor sits at its start)
    x: float  # its centre (where the app marks the note hit or missed)
    y: float
    measure: int = 0  # which of the page's measures draws it (0-based, in order)


def page_layout(svg: str) -> tuple[list[dict], dict[str, Head]]:
    """(systems, noteheads): each system's measures in order with their extent, and each note's head (by Verovio id)
    and measure, all as fractions of the page (x from the left, y from the top)."""
    root = ET.fromstring(svg)
    inner = next(e for e in root.iter(f"{SVG}svg") if "definition-scale" in _classes(e))
    _, _, vbw, vbh = (float(v) for v in inner.get("viewBox").split())
    margin = next(e for e in inner.iter(f"{SVG}g") if "page-margin" in _classes(e))
    mx, my = _translate(margin.get("transform"))
    systems = []
    notes: dict[str, Head] = {}
    count = 0
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
                    group = next((g for g in n_el.iter(f"{SVG}g") if "notehead" in _classes(g)), n_el)
                    head = next((u for u in group.iter(f"{SVG}use")), None)
                    if head is not None:
                        hx, hy = _translate(head.get("transform"))
                        sc = re.search(r"scale\(\s*(-?[\d.]+)", head.get("transform") or "")
                        glyph = (head.get(XLINK_HREF) or head.get("href") or "#").lstrip("#").split("-")[0]
                        w = HEAD_WIDTH.get(glyph, HEAD_WIDTH_DEFAULT) * (float(sc.group(1)) if sc else 1.0)
                        notes[n_el.get("data-id")] = Head((hx + mx) / vbw, (hx + w / 2 + mx) / vbw, (hy + my) / vbh, count)
            count += 1
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
            fingering: bool | None = None, keep_musicxml: bool = False, source: str | Path | None = None) -> Sheet:
    """Engraves an app-ready level file into out_dir/<stem>.sheet.json and out_dir/<stem>-p<N>.png. Returns the
    sheet.json's path, sha256 and page count.

    With `source`, the MusicXML score the level is the full arrangement of, the pages show the score's own notation
    (its keyboard part as written: spelling, voices, slurs, dynamics, grace notes, repeats written once) and the bar
    map lists each measure's played bars (`plays`, docs/formats.md). If the score's measures don't line up with the
    level's bars, or more than SOURCE_MISSING of the level's notes would have no head (SourceMismatch), or the score
    has no single two-staff keyboard part, the level is engraved as usual and Sheet.note says why."""
    song = midi_io.load(level_midi)
    score = notation.build(song, level, title)
    app_starts = [Q(b.start, score.tpb) for b in score.bars] + [Q(score.bars[-1].end, score.tpb)]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    note = ""
    if source is not None:
        try:
            src = musicxml.load(source)
            if not src.keyboard:
                raise SourceMismatch("the score has no two-staff keyboard part the hands came from")
            xml = musicxml.keyboard_notation(source, src.keyboard)
            return _engrave(xml, out_dir, stem, level, keep_musicxml, app_starts, src, Counter(n.pitch for n in song.notes))
        except (SourceMismatch, musicxml.MusicXMLError) as exc:
            note = str(exc)
    xml = notation.to_musicxml(score, fingering_for(level) if fingering is None else fingering)
    sheet = _engrave(xml, out_dir, stem, level, keep_musicxml, app_starts)
    sheet.note = note
    return sheet


def _engrave(xml: str, out_dir: Path, stem: str, level: int, keep_musicxml: bool, app_starts: list[Q],
             src: musicxml.Score | None = None, expected: Counter | None = None) -> Sheet:
    """Lays out and draws `xml` and writes the pages and bar map. `app_starts`: where each of the level's bars starts
    and the last ends (quarters). Without `src` the notation is the level's own, a measure per bar; with it, the
    source's, a measure per written measure, played at the bars src.order says, which must head the `expected`
    pitches (the level's notes)."""
    tk = _toolkit()
    if not tk.loadData(xml):
        raise ValueError("Verovio couldn't read the notation")
    count = tk.getPageCount()
    if count > MAX_PAGES:
        raise ValueError(f"{count} pages is more than {MAX_PAGES}")
    svgs, systems, measures = [], [], []
    heads: dict[str, Head] = {}
    for page in range(1, count + 1):
        svgs.append(tk.renderToSVG(page))
        sys_list, hs = page_layout(svgs[-1])
        for h in hs.values():
            h.measure += len(measures)
        heads.update(hs)
        for s in sys_list:
            si = len(systems)
            systems.append({"page": page, "top": round(s["top"], 4), "bottom": round(s["bottom"], 4)})
            for m in s["measures"]:
                measures.append({"bar": len(measures) + 1, "system": si, "left": round(m["left"], 4),
                                 "right": round(m["right"], 4), "onsets": [], "notes": []})
    bars = len(app_starts) - 1
    timemap = tk.renderToTimemap({"includeMeasures": True, "includeRests": True})
    mei = tk.getMEI()
    if src is None:
        if len(measures) != bars:
            raise ValueError(f"engraved {len(measures)} bars, the level has {bars}")
        spans = list(zip(app_starts, app_starts[1:]))
        plays = None
    else:
        spans, plays = _source_bars(src, len(measures), app_starts, timemap, mei)
    # Where each start in a bar is drawn: (fraction of the bar's length, x), from Verovio's time map; and each note
    # struck there, for the app's hit and miss marks: not the far end of a tie (the MEI's tie endids), which is the same
    # key held.
    silent = set(re.findall(r'<tie\b[^>]*\bendid="#([^"]+)"', mei))
    onsets: dict[int, dict[float, float]] = {}
    struck: dict[int, list[tuple[float, int, float, float]]] = {}
    for entry in timemap:
        q = Q(entry.get("qstamp", 0)).limit_denominator(96)
        on = [i for i in entry.get("on", []) if i in heads]
        for mi in sorted({heads[i].measure for i in on}):
            here = [i for i in on if heads[i].measure == mi]
            a, b = spans[mi]
            frac = round(float(min(max((q - a) / (b - a), Q(0)), Q(1))), 4)
            cur = onsets.setdefault(mi, {})
            cur[frac] = round(min([heads[i].left for i in here] + [cur.get(frac, 2.0)]), 4)
            for i in here:
                if i not in silent:
                    h = heads[i]
                    struck.setdefault(mi, []).append((frac, int(tk.getMIDIValuesForElement(i)["pitch"]), round(h.x, 4), round(h.y, 4)))
    for i, m in enumerate(measures):
        m["onsets"] = [v for f, x in sorted(onsets.get(i, {}).items()) for v in (f, x)]  # flat: fraction, x, fraction, x
        notes = sorted(set(struck.get(i, [])))
        if src is not None:
            # A unison the score writes in two voices is one key (the level merged them): one head for it.
            notes = [n for k, n in enumerate(notes) if k == 0 or n[:2] != notes[k - 1][:2]]
        m["notes"] = [v for n in notes for v in n]  # flat: fraction, pitch, x, y
        if plays is not None:
            m["plays"] = plays[i]
    if expected is not None:
        # The score's notation must have a head for (nearly) every note the level strikes, once per playing: a tie the
        # score continues on one playing of a repeat but not the other, or ties our reader bridges and Verovio
        # doesn't, cost a few. More than SOURCE_MISSING of them and the level's own notation serves better.
        heads: Counter = Counter()
        for m in measures:
            for k in range(1, len(m["notes"]), 4):
                heads[int(m["notes"][k])] += len(m.get("plays", [m["bar"]]))
        missing = sum(max(0, c - heads[p]) for p, c in expected.items())
        if missing > SOURCE_MISSING * sum(expected.values()):
            raise SourceMismatch(f"{missing} of the level's {sum(expected.values())} notes have no notehead in the score")
    data = {
        "version": SHEET_VERSION if plays is None else SHEET_VERSION_REPEATS, "level": level, "bars": bars,
        "width": PAGE_W, "height": PAGE_H, "pages": [], "systems": systems, "measures": measures,
    }
    # Written only now, so a source that doesn't fit leaves nothing behind.
    if keep_musicxml:
        (out_dir / f"{stem}.musicxml").write_text(xml, encoding="utf-8")
    for old in out_dir.glob(f"{stem}-p*.png"):
        old.unlink()
    for page, svg in enumerate(svgs, 1):
        png = _png(svg)
        name = f"{stem}-p{page}.png"
        (out_dir / name).write_bytes(png)
        data["pages"].append({"file": name, "sha256": hashlib.sha256(png).hexdigest()})
    path = out_dir / f"{stem}.sheet.json"
    raw = (json.dumps(data, separators=(",", ":")) + "\n").encode("utf-8")
    path.write_bytes(raw)  # bytes: no newline translation on Windows, so the sha256 is the file's
    return Sheet(path, hashlib.sha256(raw).hexdigest(), len(data["pages"]), data, source=src is not None)


def _source_bars(src: musicxml.Score, written: int, app_starts: list[Q], timemap: list[dict],
                 mei: str) -> tuple[list[tuple[Q, Q]], list[list[int]] | None]:
    """For a score engraved from its source: where each written measure starts and ends on Verovio's time map
    (quarters), and each one's played bars (1-based app bars; None when every measure is played once, in order, as one
    bar). Raises SourceMismatch unless the played measures are the level's bars (same starts).

    The times are Verovio's: it times notes by their written values, so a cadenza of small notes squeezed into a
    measure in the score's durations lasts longer there, but a fraction of the measure is the same in both. Measures
    are matched to the time map by id, as the time map can put a stretch of them out of order (the Hungarian
    Rhapsody's last pages at negative times); a measure whose next one doesn't start after it ends a bar later."""
    tpb = musicxml.TPB
    played = [Q(t, tpb) for t in src.bars]
    bars = len(app_starts) - 1
    if len(played) - 1 < bars or played[:bars + 1] != app_starts:
        raise SourceMismatch("the score's measures aren't the level's bars")
    if not src.order or max(src.order) >= written:
        raise SourceMismatch(f"engraved {written} measures, the score has {max(src.order, default=-1) + 1}")
    at: dict[str, Q] = {}
    for e in timemap:
        if e.get("measureOn"):
            at.setdefault(e["measureOn"], Q(e["qstamp"]).limit_denominator(96))
    ids = re.findall(r'<measure\b[^>]*\bxml:id="([^"]+)"', mei)
    if len(ids) != written or any(i not in at for i in ids):
        raise SourceMismatch(f"the time map has {len(at)} measures, the score {written}")
    starts = [at[i] for i in ids]
    plays: list[list[int]] = [[] for _ in range(written)]
    for k, mi in enumerate(src.order[:bars]):
        plays[mi].append(k + 1)
    end = max((Q(e.get("qstamp", 0)).limit_denominator(96) for e in timemap), default=Q(0))
    spans = []
    for mi, a in enumerate(starts):
        b = starts[mi + 1] if mi + 1 < written else end
        if b <= a:
            k = plays[mi][0] if plays[mi] else 0
            b = a + (app_starts[k] - app_starts[k - 1] if k else 1)
        spans.append((a, b))
    if written == bars and all(p == [k + 1] for k, p in enumerate(plays)):
        return spans, None
    return spans, plays
