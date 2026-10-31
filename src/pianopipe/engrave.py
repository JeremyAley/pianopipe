"""Sheet music pages for a level: notation (notation.py) laid out by Verovio, drawn to greyscale PNG pages, with a map
of where each bar is on them (docs/sheet-music.md; the files are described in docs/formats.md).

Needs the optional extras (`pip install pianopipe[sheet]`): verovio (LGPL-3.0, the engraver, used as an unmodified
library), resvg_py (MIT, SVG to pixels) and Pillow. The rest of pianopipe works without them.

Pages are sized for the app's book (37 x 48 cm at Medium, about a metre away): 1200 x 1560 pixels with 14.4 pixels between
staff lines, so a staff is about 1.8 cm, 1 degree tall there (docs/sheet-music.md §5.3). Denser since 2026-10-28 (16
pixels, wider margins and gaps before: Für Elise's levels took 4, 7, 10 and 7 pages, now 3, 5, 6 and 5). Verovio's ids
are seeded, so the same level engraves to the same files and the app doesn't download unchanged pages again.
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
SCALE = 80  # Verovio's zoom in percent: staff space = 18 units * SCALE / 100 = 14.4 px
MARGIN = 65  # Verovio units (at this scale, 52 px: 1.6 cm at Medium; bar numbers clear the binding, gate r3)
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
        "pageMarginTop": MARGIN_TOP, "pageMarginBottom": MARGIN, "pageMarginLeft": MARGIN, "pageMarginRight": MARGIN,
        # Gaps beyond what the notes need (ledger lines, finger numbers): small, so more systems fit a page; what's left
        # over is shared out between them (justifyVertically).
        "spacingSystem": 2, "spacingStaff": 6, "justifyVertically": True, "xmlIdSeed": 1, "mnumInterval": 0,
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


@dataclass
class Header:
    """Page furniture (ui-visual-pass.md §4.2): page 1's title block and every page's number, printed in the score's
    serif. `level` and `tempo` are the subtitle and the tempo word ("Level 2 · Melody + bass", "Moderato")."""
    title: str = ""
    composer: str = ""
    level: str = ""
    tempo: str = ""  # the source's own marking ("Poco moto"); never a word guessed from the tempo (gate r2)
    beat: str = ""  # else a metronome mark: the beat's note ("quarter", "dotted quarter", "half") ...
    bpm: int = 0  # ... and how many a minute



# Page 1's header (2026-10-29, gate r2: bigger furniture). Verovio reserves the height of three rows of credits (top,
# middle, bottom) above the first system on page 1 only; their text is replaced by our own title block (_furniture), so
# the bar map and the printed block always agree. The top margin is wider than the others, for the page numbers.
_CREDITS = ''.join(f'<credit page="1"><credit-words justify="{j}" valign="{v}">.</credit-words></credit>'
                   for j, v in (("center", "top"), ("center", "middle"), ("right", "bottom")))
MARGIN_TOP = 110  # Verovio units: 88 px (the others are MARGIN, 40 px)
# Sizes in pixels (a staff space is 14.4): the title about 2.6 spaces tall, the rest 1.2 or more (gate r2: the old
# 19-21 px collapsed to a few headset pixels at the bench).
TITLE_PX, META_PX, NUMBER_PX = 58, 31, 25  # gate r3: the row under the title 31 px (caps 1.4 staff spaces), bold
MARK_SCALE = 1.1  # a metronome mark's note and number, a little larger to weigh as much as the bold words
TITLE_Y, META_Y, NUMBER_Y = 130, 168, 54  # baselines; Verovio's own bottom row sits at 166, clear of the first system

# Two tones (ui-visual-pass.md §4.2): the page is 2-bit, each level with one meaning, so the app can print staff lines
# lighter than the notes without dimming thin symbols (gate r2: stems at the staff's strength on Synthwave):
#   0 ink (notes, stems, everything that isn't a line), 1 staff (staff lines, ledger lines, barlines),
#   2 ink at half (the symbols' anti-aliased edges), 3 paper.
# The page is drawn twice, with and without the lines, to tell them apart. The app's GreyPng unpacks the levels into an
# ink and a staff channel.
INK, STAFF, INK_EDGE, PAPER = 0, 85, 170, 255
FONTS = Path(__file__).parent / "fonts"  # Liberation Serif (SIL OFL 1.1): Times' metrics, so the text looks as before
SERIF = "Liberation Serif"


def _no_lines_css(page_id: str) -> str:
    """Verovio's own style sheet strokes every path with `#<id> path`, so these rules carry the page's id too."""
    return ", ".join(f"#{page_id} {sel}" for sel in ("g.staff > path", "g.barLine path", "g.ledgerLines path")) + " { visibility: hidden; }"


# A metronome mark faster than this at its beat is more likely a file's tempo than the music's (a sonatina's MIDI at
# quarter = 203): the title block prints nothing instead of a wrong number (gate r3).
PLAUSIBLE_BPM = 176


def metronome_mark(quarter_bpm: float, num: int = 4, den: int = 4) -> tuple[str, int]:
    """The metronome mark for a tempo in quarters a minute, at the beat: the dotted quarter in compound time (6/8, 9/8,
    12/8), the half in 2/2 and 3/2, else the quarter (3/8 too: Für Elise's file is quarter = 69)."""
    if den == 8 and num >= 6 and num % 3 == 0:
        return "dotted quarter", round(quarter_bpm / 1.5)
    if den == 2:
        return "half", round(quarter_bpm / 2)
    return "quarter", round(quarter_bpm)


def _note(x: float, baseline: float, px: float, k: float, beat: str) -> tuple[str, float]:
    """A small note for a metronome mark (the text font has none), drawn at x on the baseline, px tall. Returns the
    SVG and its width (viewBox units)."""
    rx, ry = 0.36 * px * k, 0.25 * px * k
    cx, cy = x + rx, (baseline - 0.2 * px) * k
    hollow = beat == "half"
    head = (f'<ellipse cx="{cx:.0f}" cy="{cy:.0f}" rx="{rx:.0f}" ry="{ry:.0f}" transform="rotate(-22 {cx:.0f} {cy:.0f})" '
            + ('fill="none" stroke="#000000" stroke-width="%.0f"/>' % (0.08 * px * k) if hollow else 'fill="#000000"/>'))
    sx = cx + rx * 0.9
    stem = f'<rect x="{sx - 0.07 * px * k:.0f}" y="{cy - 1.05 * px * k:.0f}" width="{0.07 * px * k:.0f}" height="{1.05 * px * k:.0f}" fill="#000000"/>'
    width = 2 * rx + 0.12 * px * k
    dot = ""
    if beat.startswith("dotted"):
        dot = f'<circle cx="{x + width + 0.12 * px * k:.0f}" cy="{cy:.0f}" r="{0.09 * px * k:.0f}" fill="#000000"/>'
        width += 0.3 * px * k
    return head + stem + dot, width


def _with_credits(xml: str) -> str:
    """The notation with our page-1 credits instead of any it had (a source score's own title block would print twice)."""
    xml = re.sub(r"<credit\b.*?</credit>", "", xml, flags=re.S)
    return xml.replace("<part-list", _CREDITS + "<part-list", 1)


def _text(x: float, y: float, size: float, s: str, anchor: str = "start", style: str = "", weight: str = "") -> str:
    from xml.sax.saxutils import escape

    extra = (f' font-style="{style}"' if style else "") + (f' font-weight="{weight}"' if weight else "")
    return (f'<text x="{x:.0f}" y="{y:.0f}" font-family="{SERIF}" font-size="{size:.0f}" text-anchor="{anchor}"'
            f' fill="#000000"{extra}>{escape(s)}</text>')


def _furniture(svg: str, page: int, header: Header | None, lines: bool = True) -> str:
    """Replaces Verovio's page-1 header with our title block and prints the page number in the page's outer top corner
    (odd pages are the left of a spread in the app's book: top left; even pages top right; none on page 1, under the
    title). Without `lines`, staff lines, ledger lines and barlines are left out (the two-tone pass)."""
    svg = re.sub(r'<g\b[^>]*class="pgHead[^"]*"[^>]*>.*?</g>', "", svg, flags=re.S)
    page_id = re.search(r'<svg\b[^>]*\bid="([^"]+)"', svg)
    style_end = svg.find("</style>")
    if not lines and page_id and style_end >= 0:
        svg = svg[:style_end] + _no_lines_css(page_id.group(1)) + svg[style_end:]
    if header is None:
        return svg
    m = re.search(r'<svg\b[^>]*class="definition-scale"[^>]*viewBox="([^"]+)"', svg)
    if not m:
        return svg
    vbw = float(m.group(1).split()[2])
    k = vbw / PAGE_W  # viewBox units a pixel
    margin = MARGIN * SCALE / 100 * k  # the side margins in viewBox units (40 px)
    parts = []
    if page > 1:
        left = page % 2 == 1
        parts.append(_text(margin if left else vbw - margin, NUMBER_Y * k, NUMBER_PX * k, str(page), "start" if left else "end"))
    else:
        mid = vbw / 2
        if header.title:
            # A long title gets smaller to stay inside the margins (a Times letter is about half its size wide).
            size = min(TITLE_PX, (PAGE_W - 120) / (0.48 * len(header.title)))
            parts.append(_text(mid, TITLE_Y * k, size * k, header.title, "middle"))
        if header.level:
            parts.append(_text(mid, META_Y * k, META_PX * k * 0.92, header.level, "middle", style="italic", weight="bold"))
        if header.tempo:
            parts.append(_text(margin, META_Y * k, META_PX * k, header.tempo, weight="bold"))
        elif header.bpm:
            px = META_PX * MARK_SCALE
            svg_note, w = _note(margin, META_Y, px, k, header.beat)
            parts.append(svg_note)
            parts.append(_text(margin + w + 0.3 * px * k, META_Y * k, px * k, f"= {header.bpm}", weight="bold"))
        if header.composer:
            parts.append(_text(vbw - margin, META_Y * k, META_PX * k, header.composer, "end", weight="bold"))
    # Into the page's own coordinates (the definition-scale svg), before it closes.
    at = svg.find("</svg>")
    return svg[:at] + "".join(parts) + svg[at:]


def _render(svg: str):
    import resvg_py
    from PIL import Image

    fonts = [str(p) for p in sorted(FONTS.glob("*.ttf"))]
    raw = bytes(resvg_py.svg_to_bytes(svg_string=svg, background="#ffffff", width=PAGE_W, height=PAGE_H, skip_system_fonts=True,
                                      font_files=fonts, font_family=SERIF, serif_family=SERIF))
    img = Image.open(io.BytesIO(raw)).convert("L")
    return img if img.size == (PAGE_W, PAGE_H) else img.resize((PAGE_W, PAGE_H))


def _png(svg: str, page: int = 1, header: Header | None = None) -> bytes:
    from PIL import Image, ImageChops

    full = _render(_furniture(svg, page, header, lines=True))
    symbols = _render(_furniture(svg, page, header, lines=False))
    out = Image.new("L", full.size, PAPER)
    # Lines: dark in the full page, not in the symbols' (a third of the way to ink or more).
    out.paste(STAFF, mask=ImageChops.subtract(symbols, full).point(lambda v: 255 if v >= 90 else 0))
    out.paste(INK_EDGE, mask=symbols.point(lambda v: 255 if v < 217 else 0))
    out.paste(INK, mask=symbols.point(lambda v: 255 if v < 145 else 0))  # thin strokes (a sharp's uprights) count as ink
    return _grey2(out)


def _grey2(img) -> bytes:
    """A 2-bit greyscale PNG of a page already at the four levels (INK, STAFF, INK_EDGE, PAPER): about twice a 1-bit
    page. The app's GreyPng reads it (colour type 0); Pillow writes 2-bit PNGs only for palettes, so the file is put
    together here."""
    import struct
    import zlib

    packed = img.tobytes("raw", "L;2")
    row = (img.width * 2 + 7) // 8
    data = b"".join(b"\x00" + packed[y * row:(y + 1) * row] for y in range(img.height))

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", img.width, img.height, 2, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(data, 9)) + chunk(b"IEND", b"")


def engrave(level_midi: str | Path, out_dir: str | Path, stem: str, level: int, title: str = "",
            fingering: bool | None = None, keep_musicxml: bool = False, source: str | Path | None = None,
            composer: str = "", tempo: str = "") -> Sheet:
    """Engraves an app-ready level file into out_dir/<stem>.sheet.json and out_dir/<stem>-p<N>.png. Returns the
    sheet.json's path, sha256 and page count.

    With `source`, the MusicXML score the level is the full arrangement of, the pages show the score's own notation
    (its keyboard part as written: spelling, voices, slurs, dynamics, grace notes, repeats written once) and the bar
    map lists each measure's played bars (`plays`, docs/formats.md). If the score's measures don't line up with the
    level's bars, or more than SOURCE_MISSING of the level's notes would have no head (SourceMismatch), or the score
    has no single two-staff keyboard part, the level is engraved as usual and Sheet.note says why.

    Page 1 opens with a title block (title, the level, `composer`, and `tempo`: the source's own tempo marking, or else
    a metronome mark from the level's tempo) and every page after it has its number."""
    song = midi_io.load(level_midi)
    score = notation.build(song, level, title)
    header = page_header(song, level, title, composer, tempo)
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
            # Where the score prints words at its start (its tempo marking), the title block leaves the tempo out;
            # keyboard_notation drops metronome marks, so a score without words keeps the header's.
            opening = _opening_words(xml)
            return _engrave(xml, out_dir, stem, level, keep_musicxml, app_starts, src, Counter(n.pitch for n in song.notes),
                            header=Header(header.title, header.composer, header.level) if opening else header)
        except (SourceMismatch, musicxml.MusicXMLError) as exc:
            note = str(exc)
    xml = notation.to_musicxml(score, fingering_for(level) if fingering is None else fingering)
    sheet = _engrave(xml, out_dir, stem, level, keep_musicxml, app_starts, header=header)
    sheet.note = note
    return sheet


# Tempo markings a score prints at its start (tools/source_tempo.py reads sources with it too): Italian, French and
# German tempo words, capitalised as opening markings are ("dolce", "con pedale" are not tempo markings).
TEMPO_WORDS = (r"Grave|Largo|Larghetto|Lento|Adagio|Adagietto|Andante|Andantino|Moderato|Allegretto|Allegro|Vivace|"
               r"Vivo|Presto|Prestissimo|Poco moto|Tempo di [A-Za-z]+|Tempo giusto|Maestoso|Sostenuto|Comodo|"
               r"Alla [A-Za-z]+|Moderé|Modéré|Lent|Vif|Animé|Assez [a-z]+|Mässig|Langsam|Lebhaft|Rasch|Schnell|"
               r"Sehr [a-z]+|Mouvement de [a-z]+|Risoluto|Molto [A-Za-z]+")
TEMPO_START = re.compile(rf"^(?:{TEMPO_WORDS})\b")


def _opening_words(xml: str) -> bool:
    """Whether the notation's first measure prints a tempo marking (words in a direction that start with a tempo word)."""
    first = re.search(r"<measure\b.*?</measure>", xml, flags=re.S)
    return bool(first and any(TEMPO_START.match(w.strip()) for w in re.findall(r"<words[^>]*>([^<]+)</words>", first.group(0))))


def page_header(song, level: int, title: str, composer: str = "", tempo: str = "") -> Header:
    """The title block's words: the level as "Level 2 · Melody + bass", and the source's tempo marking or, without
    one, a metronome mark from the level's median tempo at its first time signature's beat."""
    from .difficulty import LEVELS
    from .features import tempo_bpm

    end = max((n.end for n in song.notes), default=1)
    sig = next((m for _, m in sorted(song.meta, key=lambda e: e[0]) if m.type == "time_signature"), None)
    num, den = (sig.numerator, sig.denominator) if sig is not None else (4, 4)
    label = LEVELS.get(level, "")
    # The time-weighted median tempo (features.tempo_bpm), not the file's first tempo event.
    beat, bpm = metronome_mark(tempo_bpm(song.meta, end, song.ticks_per_beat), num, den)
    if bpm > PLAUSIBLE_BPM or bpm < 20:
        beat, bpm = "", 0
    return Header(title=title, composer=composer, level=f"Level {level}" + (f" · {label}" if label else ""),
                  tempo=tempo, beat="" if tempo else beat, bpm=0 if tempo else bpm)


def _engrave(xml: str, out_dir: Path, stem: str, level: int, keep_musicxml: bool, app_starts: list[Q],
             src: musicxml.Score | None = None, expected: Counter | None = None, header: Header | None = None) -> Sheet:
    """Lays out and draws `xml` and writes the pages and bar map. `app_starts`: where each of the level's bars starts
    and the last ends (quarters). Without `src` the notation is the level's own, a measure per bar; with it, the
    source's, a measure per written measure, played at the bars src.order says, which must head the `expected`
    pitches (the level's notes)."""
    tk = _toolkit()
    if header is not None:
        xml = _with_credits(xml)
        tk.setOptions({"header": "auto"})
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
        png = _png(svg, page, header)
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
