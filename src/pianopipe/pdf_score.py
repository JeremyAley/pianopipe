"""Reading the piano notation in a publisher's vector PDF into the per-measure geometry that omr.py reads from an
Audiveris project, so that omr_repair's solver can time it.

Sheet Music Direct's PDFs are exports from Finale (Maestro font), Sibelius (Opus) and MuseScore (Leland). The
notation is real: staff lines, barlines, stems and beams are vector drawings, and note heads, clefs, accidentals,
flags, dots and rests are music-font glyphs at exact positions. So every pitch can be read without guessing:

- staves are runs of five evenly spaced long horizontal lines; the piano is the lowest pair of staves in a system
  joined by barlines (a barline runs from the treble's top line to the bass's bottom line, a stem never does);
- a head's pitch is its staff step under the clef in force (clefs change mid-bar, often as small clefs), with the
  key signature, and accidentals carried through the bar; a head between the staves belongs to the staff whose
  ledger lines reach it; notes under an 8va line sound an octave up;
- a chord is the heads on one stem; its value comes from the head, beams (filled shapes), flags, dots and tuplets;
- ties are curves between two heads of the same pitch;
- each bar also keeps the segno and coda signs and the words printed over, under or beside it (pdf_songs reads the
  jumps and the tempo from them).

pdf_check.py compares an OMR export with this bar by bar and rewrites bars from it; see docs/catalog-expansion.md
§3.2. Needs pymupdf (the [omr] extra).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from . import omr

# Fonts laid out like Sonata (one Mac Roman character per symbol). Sibelius's Opus puts them in the private use
# area at U+F000 + the Mac Roman code. Tests add a font of their own here.
SONATA_FONTS = ["Maestro", "Opus", "Petrucci", "Engraver", "Sonata", "Jazz", "Broadway", "Reprise", "Inkpen"]
SMUFL_FONTS = ["Leland", "Bravura", "Petaluma", "Finale Maestro", "Finale Broadway", "Finale Jazz", "Sebastian"]
NOT_MUSIC = ("Text", "Chord", "Times", "Plain", "Lyric")

# (kind, value): heads and rests by their value in quarter notes, clefs by kind, accidentals by alter, flags by count
SONATA = {
    "œ": ("head", Fraction(1)), "˙": ("head", Fraction(2)), "w": ("head", Fraction(4)), "W": ("head", Fraction(8)),
    "&": ("clef", "TREBLE"), "?": ("clef", "BASS"), "B": ("clef", "ALTO"),
    "#": ("acc", 1), "b": ("acc", -1), "n": ("acc", 0), "Ü": ("acc", 2), "º": ("acc", -2),
    "J": ("flag", 1), "j": ("flag", 1), "K": ("flag", 2), "k": ("flag", 2), "R": ("flag", 2), "r": ("flag", 2),
    "∑": ("rest", Fraction(4)), "Ó": ("rest", Fraction(2)), "Œ": ("rest", Fraction(1)), "‰": ("rest", Fraction(1, 2)),
    "≈": ("rest", Fraction(1, 4)), "®": ("rest", Fraction(1, 8)),
    "-": ("ledger", None),
    **{str(d): ("time", d) for d in range(10)}, "c": ("time", "common"), "C": ("time", "cut"),
    # the coda sign is byte 0xDE: Þ as Windows reads it, ﬁ as Mac Roman does (Opus's private use area)
    "%": ("sign", "segno"), "Þ": ("sign", "coda"), "ﬁ": ("sign", "coda"),
}
SONATA_SPECIAL = {".": ("dot", None), "™": ("dot", None),  # also read from the fonts' "Special" companions
                  "“": ("ottava", "8"), "”": ("ottava", "va"), "‘": ("ottava", "vb")}  # Opus Special's 8va / 8vb
SMUFL = {
    0xE0A4: ("head", Fraction(1)), 0xE0A3: ("head", Fraction(2)), 0xE0A2: ("head", Fraction(4)),
    0xE0A0: ("head", Fraction(8)),
    0xE050: ("clef", "TREBLE"), 0xE07A: ("clef", "TREBLE"), 0xE062: ("clef", "BASS"), 0xE07C: ("clef", "BASS"),
    0xE05C: ("clef", "ALTO"), 0xE07B: ("clef", "ALTO"),
    0xE262: ("acc", 1), 0xE260: ("acc", -1), 0xE261: ("acc", 0), 0xE263: ("acc", 2), 0xE264: ("acc", -2),
    0xE1E7: ("dot", None),
    0xE240: ("flag", 1), 0xE241: ("flag", 1), 0xE242: ("flag", 2), 0xE243: ("flag", 2), 0xE244: ("flag", 3),
    0xE245: ("flag", 3), 0xE246: ("flag", 4), 0xE247: ("flag", 4),
    0xE4E3: ("rest", Fraction(4)), 0xE4E4: ("rest", Fraction(2)), 0xE4E5: ("rest", Fraction(1)),
    0xE4E6: ("rest", Fraction(1, 2)), 0xE4E7: ("rest", Fraction(1, 4)), 0xE4E8: ("rest", Fraction(1, 8)),
    0xE022: ("ledger", None),
    0xE047: ("sign", "segno"), 0xE048: ("sign", "coda"), 0xE049: ("sign", "coda"),
    **{0xE880 + d: ("tuplet", d) for d in range(10)},
    **{0xE080 + d: ("time", d) for d in range(10)}, 0xE08A: ("time", "common"), 0xE08B: ("time", "cut"),
}
REST_SHAPES = {Fraction(4): "WHOLE_REST", Fraction(2): "HALF_REST", Fraction(1): "QUARTER_REST",
               Fraction(1, 2): "EIGHTH_REST", Fraction(1, 4): "ONE_16TH_REST", Fraction(1, 8): "ONE_32ND_REST"}
HEAD_SHAPES = {Fraction(1): "NOTEHEAD_BLACK", Fraction(2): "NOTEHEAD_VOID", Fraction(4): "WHOLE_NOTE",
               Fraction(8): "BREVE"}
# Staff steps below the top line of each key signature accidental, in order, on a treble staff (bass: 2 lower).
KEY_SHARPS = [0, 3, -1, 2, 5, 1, 4]
KEY_FLATS = [4, 1, 5, 2, 6, 3, 7]
OCTAVE_TEXT = re.compile(r"^\(?(8va|8vb|8va bassa|15ma|15mb)\)?$")
GRACE_SIZE = 0.8  # heads smaller than this share of the page's usual head are grace notes


@dataclass
class Glyph:
    kind: str  # head, rest, clef, acc, dot, flag, ledger, tuplet, time, sign (segno, coda)
    value: object
    x0: float
    x1: float
    y: float  # the glyph's origin: a head's or rest's staff position, the line a clef sits on
    size: float
    staff: Staff | None = None
    grace: bool = False
    source: str = ""  # a head's alteration from: "sign" (its own), "bar" (earlier in the bar), "key", "tie"
    octaves: int = 0  # a head under an 8va (1) or 8vb (-1) line

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2


@dataclass
class Staff:
    top: float
    bottom: float
    x0: float
    x1: float
    index: int = 0  # 1 treble, 2 bass for the piano's staves; 0 for any other staff

    @property
    def space(self) -> float:
        return (self.bottom - self.top) / 4

    def step(self, y: float) -> int:
        """Staff steps below the top line."""
        return round((y - self.top) / (self.space / 2))

    def distance(self, y: float) -> float:
        return 0.0 if self.top <= y <= self.bottom else min(abs(y - self.top), abs(y - self.bottom))


@dataclass
class Beam:
    x0: float
    x1: float
    y0: float  # centre line at x0 and x1
    y1: float
    thickness: float

    def y_at(self, x: float) -> float:
        if self.x1 - self.x0 < 1e-6:
            return self.y0
        return self.y0 + (self.y1 - self.y0) * (x - self.x0) / (self.x1 - self.x0)


@dataclass
class Stem:
    x: float
    y0: float
    y1: float


@dataclass
class Curve:
    """A tie or slur: its left and right ends."""
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class Grace:
    staff: int
    x: float
    pitches: list[tuple[str, int, int]]  # (step, alter, octave)
    midis: list[int]


@dataclass
class Chord(omr.Chord):
    grace: bool = False

    @property
    def extent(self) -> tuple[float, float]:
        """Left and right edge of its heads (or the rest)."""
        if self.heads:
            return min(h.box.x for h in self.heads), max(h.box.right for h in self.heads)
        return self.box.x, self.box.right


@dataclass
class Bar:
    """One bar of the piano part as printed: where it is and what it holds, as omr.py's Measure."""
    number: int  # 1-based, in print order
    page: int  # 1-based
    system: int  # 0-based over the whole score
    measure: omr.Measure
    graces: list[Grace] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)  # what makes its timing unsafe to read
    time: tuple[int, int] | None = None  # a time signature printed in it: (beats, beat type)
    signs: list[str] = field(default_factory=list)  # segno and coda signs over it, left to right
    words: list[str] = field(default_factory=list)  # text over, under or beside it (directions, lyrics), in order


@dataclass
class PdfScore:
    bars: list[Bar]
    systems: list[int]  # index of each system's first bar
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- the page's raw marks

@dataclass
class _Page:
    hlines: list[tuple[float, float, float]] = field(default_factory=list)  # (y, x0, x1)
    vlines: list[tuple[float, float, float]] = field(default_factory=list)  # (x, y0, y1)
    fills: list[list[tuple[float, float]]] = field(default_factory=list)  # filled polygons: beam candidates
    curves: list[Curve] = field(default_factory=list)
    glyphs: list[Glyph] = field(default_factory=list)
    words: list[tuple[str, float, float, float, float, bool]] = field(default_factory=list)  # text, box, italic


def _family(font: str) -> str | None:
    name = font.split("+")[-1]
    if any(k in name for k in NOT_MUSIC):
        return None
    if any(k in name for k in SMUFL_FONTS):
        return "smufl"
    if any(k in name for k in SONATA_FONTS):
        return "sonata-special" if "Special" in name else "sonata"
    return None


def symbol(family: str, ch: str) -> tuple[str, object] | None:
    """What a music-font character means: (kind, value), or None."""
    code = ord(ch)
    if family == "smufl" or 0xE000 <= code < 0xF000:
        return SMUFL.get(code)
    if 0xF000 <= code <= 0xF0FF:
        ch = bytes([code - 0xF000]).decode("mac_roman")
    if family == "sonata-special":
        return SONATA_SPECIAL.get(ch)
    return SONATA.get(ch) or SONATA_SPECIAL.get(ch)


def _chains(items: list) -> list[list[tuple[float, float]]]:
    """A path's straight-line items split into connected polygons (one list of points each)."""
    out: list[list[tuple[float, float]]] = []
    for it in items:
        if it[0] == "l":
            a, b = (it[1].x, it[1].y), (it[2].x, it[2].y)
            if out and abs(out[-1][-1][0] - a[0]) < 0.01 and abs(out[-1][-1][1] - a[1]) < 0.01:
                out[-1].append(b)
            else:
                out.append([a, b])
        elif it[0] == "qu":
            q = it[1]
            out.append([(q.ul.x, q.ul.y), (q.ur.x, q.ur.y), (q.lr.x, q.lr.y), (q.ll.x, q.ll.y)])
    return out


def _read_page(page) -> _Page:
    p = _Page()
    for d in page.get_drawings():
        items = d["items"]
        filled = d.get("fill") is not None
        if any(it[0] == "c" for it in items):
            pts = [q for it in items for q in it[1:] if hasattr(q, "x")]
            if pts:
                a, b = min(pts, key=lambda q: q.x), max(pts, key=lambda q: q.x)
                p.curves.append(Curve(a.x, a.y, b.x, b.y))
            continue
        for it in items:
            if it[0] == "re":
                r = it[1]
                if r.height < 1.2 and r.width > 2:
                    p.hlines.append(((r.y0 + r.y1) / 2, r.x0, r.x1))
                elif r.width < 4 and r.height > 3:
                    p.vlines.append(((r.x0 + r.x1) / 2, r.y0, r.y1))
                elif filled:
                    p.fills.append([(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)])
            elif it[0] == "l" and not filled:
                a, b = it[1], it[2]
                if abs(a.y - b.y) < 0.3 and abs(a.x - b.x) > 1:
                    p.hlines.append(((a.y + b.y) / 2, min(a.x, b.x), max(a.x, b.x)))
                elif abs(a.x - b.x) < 0.3 and abs(a.y - b.y) > 3:
                    p.vlines.append(((a.x + b.x) / 2, min(a.y, b.y), max(a.y, b.y)))
        if filled:
            for poly in _chains(items):
                if len(poly) >= 3:
                    p.fills.append(poly)
    raw = page.get_text("rawdict")
    for block in raw["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                fam = _family(span["font"])
                if fam is None:
                    text = "".join(c["c"] for c in span["chars"]).strip()
                    if text:
                        x0, y0, x1, y1 = span["bbox"]
                        italic = bool(span["flags"] & 2) or "Italic" in span["font"] or "Oblique" in span["font"]
                        p.words.append((text, x0, y0, x1, y1, italic))
                    continue
                for c in span["chars"]:
                    sym = symbol(fam, c["c"])
                    if sym is None:
                        continue
                    x, y = c["origin"]
                    p.glyphs.append(Glyph(sym[0], sym[1], c["bbox"][0], c["bbox"][2], y, span["size"]))
    # an octave sign drawn as glyphs (an "8" and a "va" or "vb" beside it) reads as the words "8va" / "8vb"
    marks = [g for g in p.glyphs if g.kind == "ottava"]
    for g in marks:
        if g.value != "8":
            continue
        suffix = next((h for h in marks if h.value != "8" and 0 <= h.x0 - g.x0 < g.size and abs(h.y - g.y) < g.size / 2),
                      None)
        if suffix is not None:
            p.words.append((f"8{suffix.value}", g.x0, g.y - 8, suffix.x1, g.y + 1, True))
    return p


# ---------------------------------------------------------------- staves, systems, barlines

def _staves(hlines: list[tuple[float, float, float]]) -> list[Staff]:
    """Runs of five evenly spaced long horizontal lines that overlap horizontally, by height. Two systems can share
    a height (a coda printed beside the end of a line): each line belongs to the staff it overlaps."""
    rows: list[list[float]] = []  # [y, x0, x1]: segments that touch (a staff line drawn bar by bar) joined
    for y, x0, x1 in sorted(hlines, key=lambda h: (round(h[0], 1), h[1])):
        same = [r for r in rows if abs(r[0] - y) < 0.3 and x0 <= r[2] + 2]
        if same:
            same[-1][2] = max(same[-1][2], x1)
        else:
            rows.append([y, x0, x1])
    long = sorted((r for r in rows if r[2] - r[1] >= 60), key=lambda r: (r[0], r[1]))
    used: set[int] = set()
    staves = []
    for i, top in enumerate(long):
        if i in used:
            continue
        below = [(j, r) for j, r in enumerate(long) if j not in used and r[0] > top[0] + 2.4
                 and min(r[2], top[2]) - max(r[1], top[1]) > 40]
        for j1, first in below[:3]:
            gap = first[0] - top[0]
            if gap > 15:
                break
            run, idx = [top, first], [i, j1]
            for _ in range(3):
                nxt = next(((j, r) for j, r in below if abs(r[0] - run[-1][0] - gap) < 0.12 * gap
                            and min(r[2], top[2]) - max(r[1], top[1]) > 40), None)
                if nxt is None:
                    break
                run.append(nxt[1])
                idx.append(nxt[0])
            if len(run) == 5:
                staves.append(Staff(run[0][0], run[4][0], min(r[1] for r in run), max(r[2] for r in run)))
                used.update(idx)
                break
    return sorted(staves, key=lambda s: (s.top, s.x0))


def _coverage(vlines: list[tuple[float, float, float]], x: float, y0: float, y1: float, tol: float = 0.6) -> float:
    """How much of [y0, y1] vertical lines at x cover."""
    segs = sorted((max(a, y0), min(b, y1)) for vx, a, b in vlines if abs(vx - x) <= tol and b > y0 and a < y1)
    total, end = 0.0, y0
    for a, b in segs:
        if b > end:
            total += b - max(a, end)
            end = b
    return total / max(y1 - y0, 1e-6)


def _barlines(vlines: list[tuple[float, float, float]], upper: Staff, lower: Staff) -> list[float]:
    """x of every line that runs from the upper staff's top line to the lower staff's bottom line."""
    xs = sorted({round(x, 1) for x, a, b in vlines
                 if upper.x0 - 2 <= x <= upper.x1 + 2 and b > upper.top - 1 and a < lower.bottom + 1})
    found = []
    sp = upper.space
    for x in xs:
        if found and x - found[-1] < 0.6:
            continue
        if _coverage(vlines, x, upper.top + 0.3, lower.bottom - 0.3) < 0.95:
            continue
        # a stem of a cross-staff run reaches from a bass note to a beam over the treble: it sticks out where a
        # barline doesn't (only the system's first line may join staves above or below)
        out = [(a, b) for vx, a, b in vlines if abs(vx - x) <= 0.6
               and (a < upper.top - sp < b - sp or a + sp < lower.bottom + sp < b)]
        if out and x - upper.x0 > 2:
            continue
        found.append(x)
    return found


def _piano_staves(staves: list[Staff], vlines) -> list[tuple[Staff, Staff, list[float]]]:
    """The piano's grand staves on a page: in each system (staves joined by the line at their left edge), the
    lowest pair of neighbouring staves that barlines join."""
    systems: list[list[Staff]] = []
    for s in staves:
        for group in reversed(systems):
            prev = group[-1]
            if min(prev.x1, s.x1) - max(prev.x0, s.x0) <= 40:
                continue
            left = min(prev.x0, s.x0)
            if any(_coverage(vlines, x, prev.top + 0.3, s.bottom - 0.3) >= 0.95
                   for x in {round(vx, 1) for vx, _, _ in vlines if abs(vx - left) < 4}):
                group.append(s)
                break
            systems.append([s])
            break
        else:
            systems.append([s])
    out = []
    for group in systems:
        for k in range(len(group) - 1, 0, -1):
            xs = _barlines(vlines, group[k - 1], group[k])
            if len(xs) >= 2:
                out.append((group[k - 1], group[k], xs))
                break
    # reading order: by height, and systems side by side left to right
    out.sort(key=lambda p: p[0].top)
    rows: list[list] = []
    for p in out:
        if rows and p[0].top < rows[-1][0][1].bottom:
            rows[-1].append(p)
        else:
            rows.append([p])
    return [p for row in rows for p in sorted(row, key=lambda p: p[0].x0)]


def _bar_ranges(xs: list[float], space: float) -> list[tuple[float, float]]:
    """Bars between barlines; a double or thin-thick barline (lines closer than 1.2 spaces) is one."""
    groups: list[list[float]] = []
    for x in xs:
        if groups and x - groups[-1][-1] < 1.2 * space:
            groups[-1].append(x)
        else:
            groups.append([x])
    return [(a[-1], b[0]) for a, b in zip(groups, groups[1:]) if b[0] - a[-1] > 2 * space]


# ---------------------------------------------------------------- reading a system

def _trim_heads(glyphs: list[Glyph], vlines) -> None:
    """A glyph's box can be wider than its ink (Opus heads carry side bearing), and an up-stem sits on the ink's right
    edge. Measure the ink from the stems (the usual distance from a head's left edge to a stem on its right) and
    trim the heads of that size to it."""
    by_size: dict[tuple[float, object], list[float]] = {}
    for g in glyphs:
        if g.kind == "head" and g.value in (Fraction(1), Fraction(2)):
            w = g.x1 - g.x0
            for x, a, b in vlines:
                if 0.6 * w <= x - g.x0 <= w + 0.5 and a - 1 <= g.y <= b + 1:
                    by_size.setdefault((round(g.size, 1), g.value), []).append(x - g.x0)
    ink = {k: sorted(v)[len(v) // 2] for k, v in by_size.items() if len(v) >= 3}
    bearing = {}  # box minus ink of a black head, by size: whole notes carry the same
    for g in glyphs:
        w = ink.get((round(g.size, 1), g.value))
        if w is not None and g.kind == "head":
            if g.value == Fraction(1):
                bearing[round(g.size, 1)] = max(0.0, g.x1 - g.x0 - w)
            g.x1 = min(g.x1, g.x0 + w + 0.3)
    for g in glyphs:
        if g.kind == "head" and g.value >= 4 and round(g.size, 1) in bearing:
            g.x1 -= bearing[round(g.size, 1)]


def _assign_staves(glyphs: list[Glyph], staves: list[Staff], ledgers: list[tuple[float, float, float]],
                   pairs: list[tuple[Staff, Staff]]) -> None:
    """Each glyph to the nearest staff; a head between the piano's staves to the one whose ledger lines reach it."""
    for g in glyphs:
        cands = [s for s in staves if s.x0 - 8 <= g.x0 <= s.x1 + 2]
        if not cands:
            continue
        g.staff = min(cands, key=lambda s: s.distance(g.y))
        if g.kind != "head":
            continue
        for up, low in pairs:
            if up.bottom < g.y < low.top and up.x0 - 8 <= g.x0 <= up.x1 + 2:
                sp = up.space
                ok_up = (g.y - up.bottom) / (sp / 2) < 1.6 or _has_ledger(ledgers, g, up.bottom + sp, sp)
                ok_low = (low.top - g.y) / (sp / 2) < 1.6 or _has_ledger(ledgers, g, low.top - sp, sp)
                if ok_up != ok_low:
                    g.staff = up if ok_up else low


def _has_ledger(ledgers: list[tuple[float, float, float]], g: Glyph, y: float, sp: float) -> bool:
    return any(abs(ly - y) < 0.25 * sp and lx0 <= g.x1 + 0.2 * sp and lx1 >= g.x0 - 0.2 * sp for ly, lx0, lx1 in ledgers)


def _key_runs(accs: list[Glyph], heads: list[Glyph], staff: Staff, clef_at) -> list[tuple[float, int, list[Glyph]]]:
    """Key signatures on one staff: (x where it starts, fifths, its glyphs). A run of accidentals at the standard
    places for the clef in force there (clef_at(x)), none of them right before a head at its height. Naturals before
    the sharps or flats cancel the old key."""
    sp = staff.space
    free = [a for a in sorted(accs, key=lambda a: a.x0)
            if not any(abs(h.y - a.y) < 0.3 * sp and -0.3 * sp <= h.x0 - a.x1 <= 1.0 * sp for h in heads)]
    runs: list[list[Glyph]] = []
    for a in free:
        if runs and a.x0 - runs[-1][-1].x1 < 1.0 * sp:
            runs[-1].append(a)
        else:
            runs.append([a])
    out = []
    for run in runs:
        shift = (_diatonic(staff, staff.top, clef_at(run[0].x0)) - 38) % 7  # steps lower than in the treble clef
        naturals = [a for a in run if a.value == 0]
        rest = run[len(naturals):]
        if any(a.value == 0 for a in rest):
            continue
        kinds = {a.value for a in rest}
        if len(kinds) > 1:
            continue
        if rest:
            order = KEY_SHARPS if rest[0].value == 1 else KEY_FLATS
            if [staff.step(a.y) - shift for a in rest] != order[:len(rest)]:
                continue
            out.append((run[0].x0, len(rest) * (1 if rest[0].value == 1 else -1), run))
        elif naturals:
            steps = [staff.step(a.y) - shift for a in naturals]
            if steps in (KEY_SHARPS[:len(steps)], KEY_FLATS[:len(steps)]):
                out.append((run[0].x0, 0, run))
    return out


def _key_alter(name: str, fifths: int) -> int:
    if fifths >= 0:
        return 1 if name in "FCGDAEB"[:fifths] else 0
    return -1 if name in "BEADGCF"[:-fifths] else 0


def _octave_lines(words, hlines, staves: list[Staff]) -> list[tuple[Staff, float, float, int]]:
    """8va/8vb/15ma lines: (staff, x from, x to, octaves)."""
    out = []
    for text, x0, y0, x1, y1, _ in words:
        m = OCTAVE_TEXT.match(text.strip())
        if not m:
            continue
        kind = m.group(1)
        octaves = (2 if kind.startswith("15") else 1) * (-1 if kind.endswith(("b", "bassa")) else 1)
        yc = (y0 + y1) / 2
        cands = [s for s in staves if s.x0 - 5 <= x0 <= s.x1]
        if not cands:
            continue
        if octaves > 0:
            below = [s for s in cands if s.top > yc - 2]
            staff = min(below or cands, key=lambda s: s.distance(yc))
        else:
            above = [s for s in cands if s.bottom < yc + 2]
            staff = min(above or cands, key=lambda s: s.distance(yc))
        sp = staff.space
        end = x1 + 0.5 * sp
        for y, a, b in sorted(hlines, key=lambda h: h[1]):
            if abs(y - yc) < 0.8 * sp and a >= x0 and a <= end + 2.5 * sp and b - a < 60:
                end = max(end, b)
        out.append((staff, x0 - 0.5 * sp, end + 0.5 * sp, octaves))
    return out


@dataclass
class _State:
    clef: dict[int, str] = field(default_factory=lambda: {1: "TREBLE", 2: "BASS"})
    key: dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0})
    ties: dict[tuple[int, int], int] = field(default_factory=dict)  # (staff, diatonic) -> alter, tied over a break


def read(path: str | Path) -> PdfScore:
    """Every bar of the piano part of a vector PDF, in order."""
    import pymupdf  # the [omr] extra

    with pymupdf.open(str(path)) as doc:
        pages = [_read_page(page) for page in doc]
    score = PdfScore([], [])
    state = _State()
    ids = iter(range(1, 10 ** 9))
    for pno, raw in enumerate(pages, 1):
        staves = _staves(raw.hlines)
        pairs = _piano_staves(staves, raw.vlines)
        if not pairs:
            continue
        ledgers = [(y, a, b) for y, a, b in raw.hlines if b - a < 25] + \
                  [(g.y, g.x0, g.x1) for g in raw.glyphs if g.kind == "ledger"]
        glyphs, seen = [], set()
        for g in raw.glyphs:  # a head two voices share can be drawn twice on the same spot: one head, two stems
            key = (g.kind, str(g.value), round(g.x0), round(g.y), round(g.size))
            if g.kind != "ledger" and (g.kind != "head" or key not in seen):
                glyphs.append(g)
                seen.add(key)
        _trim_heads(glyphs, raw.vlines)
        _assign_staves(glyphs, staves, ledgers, [(a, b) for a, b, _ in pairs])
        head_sizes = sorted(g.size for g in glyphs if g.kind == "head")
        usual = head_sizes[len(head_sizes) // 2] if head_sizes else 0
        for g in glyphs:
            g.grace = g.kind in ("head", "flag") and g.size < GRACE_SIZE * usual
        octaves = _octave_lines(raw.words, raw.hlines, staves)
        marks = _marks(glyphs, raw.words)
        for k, (upper, lower, xs) in enumerate(pairs):
            upper.index, lower.index = 1, 2
            score.systems.append(len(score.bars))
            # the system's marks: below the piano system above it (in the same column) down to just under its bass
            ceiling = max((lo.bottom + 3 * lo.space for up, lo, _ in pairs[:k]
                           if lo.bottom < upper.top and min(lo.x1, upper.x1) - max(lo.x0, upper.x0) > 40), default=0.0)
            _read_system(score, state, pno, upper, lower, xs, glyphs, raw, octaves, ids,
                         [mk for mk in marks if ceiling < mk[1] <= lower.bottom + 3 * lower.space])
    if not score.bars:
        score.warnings.append("no piano staves found: not a vector PDF of notation")
    return score


END_WORDS = re.compile(r"^(d\.?\s?[sc%]\.?|dal\s+segno|da\s+capo|to\s+coda|fine)(\W|$)", re.I)
SIGN_CHARS = {"": "segno", "": "coda", "": "coda"}  # SMuFL, also in text fonts (Bravura Text)


def _marks(glyphs: list[Glyph], words) -> list[tuple[str, float, float, float, bool]]:
    """Signs and words on a page as (text or sign, y, x0, x1, is a sign). A sign drawn inside a line of text is part
    of it: "D." and a segno is "D.%", "To" and a coda sign is "To Coda"; beside the heading "Coda" it stays a sign."""
    signs = [(g.value, g.x0, g.x1, g.y, g.size) for g in glyphs if g.kind == "sign"]
    texts: list[list] = []
    for text, x0, y0, x1, y1, _ in sorted(words, key=lambda w: w[1]):
        if text in SIGN_CHARS:  # a sign set in a text font
            signs.append((SIGN_CHARS[text], x0, x1, y1, y1 - y0))
            continue
        text = "".join({"segno": "%", "coda": " Coda"}.get(SIGN_CHARS.get(c), c) for c in text)
        h = y1 - y0
        prev = next((t for t in reversed(texts) if abs((t[2] + t[4]) / 2 - (y0 + y1) / 2) < 0.3 * h
                     and -0.1 * h <= x0 - t[3] < 0.6 * h), None)
        if prev is not None:  # one line of text set in several pieces
            prev[0] += (" " if x0 - prev[3] > 0.15 * h else "") + text
            prev[3], prev[2], prev[4] = x1, min(prev[2], y0), max(prev[4], y1)
        else:
            texts.append([text, x0, y0, x1, y1])
    out, used = [], set()
    for text, x0, y0, x1, y1 in texts:
        if re.search(r"(\bd\.|\bto|\bal|\bcoda)\W*$", text, re.I) and not re.fullmatch(r"\W*coda\W*", text, re.I):
            for k, (value, gx0, gx1, gy, size) in enumerate(signs):
                if k not in used and y0 - 0.3 * size <= gy <= y1 + 0.5 * size and -0.6 * size <= gx0 - x1 <= 0.6 * size:
                    text, x1 = text + ("%" if value == "segno" else " Coda"), gx1
                    used.add(k)
        out.append((text, (y0 + y1) / 2, x0, x1, False))
    out += [(value, gy - size / 2, gx0, gx1, True) for k, (value, gx0, gx1, gy, size) in enumerate(signs) if k not in used]
    return out


def _read_system(score: PdfScore, state: _State, pno: int, upper: Staff, lower: Staff, xs: list[float],
                 glyphs: list[Glyph], raw: _Page, octaves, ids, marks=()) -> None:
    sp = upper.space
    left, right = xs[0], xs[-1]
    mine = [g for g in glyphs if g.staff in (upper, lower) and left - 0.5 * sp <= g.x0 <= right + 0.5 * sp]
    staff_of = {id(upper): 1, id(lower): 2}
    heads = [g for g in mine if g.kind == "head"]
    stems = [Stem(x, a, b) for x, a, b in _joined(raw.vlines)
             if left < x < right and b - a > 1.5 * sp and b > upper.top - 8 * sp and a < lower.bottom + 8 * sp
             and not any(abs(x - bx) < 0.6 for bx in xs)]
    beams = _beams(raw.fills, left, right, upper.top - 8 * sp, lower.bottom + 8 * sp, sp)
    # pitches: clefs and keys in force, accidentals through the bar, 8va lines
    pitch: dict[int, tuple[int, int, bool]] = {}  # id(head glyph) -> (diatonic, alter, written accidental)
    bars = _bar_ranges(xs, sp)
    for staff in (upper, lower):
        idx = staff_of[id(staff)]
        events = sorted((g for g in mine if g.staff is staff and g.kind in ("clef", "acc", "head")), key=lambda g: g.x0)
        accs = [g for g in events if g.kind == "acc"]
        clefs = [(g.x0, g.value) for g in events if g.kind == "clef"]
        before = state.clef[idx]
        keys = _key_runs(accs, [g for g in events if g.kind == "head"], staff,
                         lambda x: next((v for cx, v in reversed(clefs) if cx < x), before))
        key_glyphs = {id(a) for _, _, run in keys for a in run}
        key_at = sorted((x, f) for x, f, _ in keys)
        altered: dict[int, int] = {}
        bar_k = -1
        notes = [g for g in events if g.kind == "head"]
        for g in events:
            k = next((i for i, (a, b) in enumerate(bars) if a - 0.5 * sp <= g.x0 < b), None)
            if k != bar_k:
                altered, bar_k = {}, k
            while key_at and key_at[0][0] <= g.x0:
                state.key[idx] = key_at.pop(0)[1]
            if g.kind == "clef":
                state.clef[idx] = g.value
            elif g.kind == "acc" and id(g) not in key_glyphs:
                target = [h for h in notes if abs(h.y - g.y) < 0.35 * sp and 0 <= h.x0 - g.x1 + 0.2 * sp < 4 * sp]
                if target:
                    h = min(target, key=lambda h: h.x0)
                    d = _diatonic(staff, h.y, state.clef[idx])
                    altered[d] = g.value
                    pitch[id(h)] = (d, g.value, True)
                    h.source = "sign"
            elif g.kind == "head":
                d = _diatonic(staff, g.y, state.clef[idx])
                if id(g) in pitch:
                    continue
                name = "CDEFGAB"[d % 7]
                pitch[id(g)] = (d, altered.get(d, _key_alter(name, state.key[idx])), False)
                g.source = "bar" if d in altered else "key"
        for h in notes:  # 8va lines move the written note
            shift = sum(o for s, a, b, o in octaves if s is staff and a <= h.x0 <= b)
            if shift:
                d, alter, written = pitch[id(h)]
                pitch[id(h)] = (d + 7 * shift, alter, written)
                h.octaves = shift
    chords = _chords(heads, stems, beams, mine, sp, staff_of, pitch, ids)
    _ties(chords, raw.curves, sp, state, right)
    _tuplets(chords, raw.words, glyphs, beams, sp, upper, lower)
    for k, (x0, x1) in enumerate(bars):
        inside = [c for c in chords if x0 <= c.x < x1]
        if k == 0 and not inside and len(bars) > 1:
            continue  # the header before a start-repeat barline
        m = omr.Measure(len(score.bars) + 1, pno, x0, x1, interline=sp)
        bar = Bar(m.number, pno, len(score.systems) - 1, m, time=_time_signature(
            [g for g in mine if g.kind == "time" and x0 <= g.x0 < x1], upper, lower))
        full = [c.anchor for c in inside if not c.grace]
        for c in inside:  # small notes in a column with normal ones take time: cue notes, not grace notes
            if c.grace and any(abs(a - c.anchor) < 0.8 * sp for a in full):
                c.grace = False
        for c in inside:
            if c.grace:
                bar.graces.append(Grace(c.staff, c.x, [_name(h) for h in c.heads], [h.midi for h in c.heads]))
            else:
                m.chords.append(c)
        for c in m.chords:
            if c.heads and c.stem is None and any(h.shape != "WHOLE_NOTE" and h.shape != "BREVE" for h in c.heads):
                bar.problems.append(f"a note at {_name(c.heads[0])} has no stem")
        for s in (1, 2):
            on = [c for c in m.chords if c.staff == s]
            if len(on) == 1 and on[0].rest == "WHOLE_REST":
                on[0].measure_rest = True
        m.chords.sort(key=lambda c: (c.x, c.staff))
        score.bars.append(bar)
    mine_bars = [b for b in score.bars if b.system == len(score.systems) - 1]
    for text, y, x0, x1, sign in sorted(marks, key=lambda mk: (mk[1] // (2 * sp), mk[2])):
        if not mine_bars or x1 < left - 6 * sp or x0 > right + sp:
            continue
        # jumps at a bar's end (D.S., To Coda, Fine) end at or just past its barline; the rest start in their bar
        at = (x0 + x1) / 2 if sign or END_WORDS.match(text) else x0
        bar = next((b for b in mine_bars if at < b.measure.right), mine_bars[-1])
        (bar.signs if sign else bar.words).append(text)


def _joined(vlines: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """Vertical segments at one x that overlap joined into one line (a stem can be drawn twice, in pieces)."""
    columns: list[list[tuple[float, float, float]]] = []
    for v in sorted(vlines):
        if columns and v[0] - columns[-1][0][0] < 0.3:
            columns[-1].append(v)
        else:
            columns.append([v])
    out = []
    for col in columns:
        merged: list[list[float]] = []
        for x, a, b in sorted(col, key=lambda v: v[1]):
            if merged and a < merged[-1][2] - 0.5:
                merged[-1][2] = max(merged[-1][2], b)
            else:
                merged.append([x, a, b])
        out += [tuple(m) for m in merged]
    return out


def _time_signature(digits: list[Glyph], upper: Staff, lower: Staff) -> tuple[int, int] | None:
    """A time signature from its glyphs on the treble staff (or the bass): the digits above the middle line over
    those below, or a common or cut time sign."""
    for staff in (upper, lower):
        on = sorted((g for g in digits if g.staff is staff), key=lambda g: g.x0)
        if not on:
            continue
        if on[0].value in ("common", "cut"):
            return (4, 4) if on[0].value == "common" else (2, 2)
        mid = (staff.top + staff.bottom) / 2
        top = "".join(str(g.value) for g in on if isinstance(g.value, int) and g.y < mid)
        bottom = "".join(str(g.value) for g in on if isinstance(g.value, int) and g.y > mid)
        if top and bottom and int(bottom) in (1, 2, 4, 8, 16, 32):
            return int(top), int(bottom)
    return None


def _diatonic(staff: Staff, y: float, clef: str) -> int:
    """C4 = 28; the top line is F5 under a treble clef, A3 under a bass clef, G4 under an alto clef."""
    top = {"TREBLE": 38, "BASS": 26, "ALTO": 32}[clef]
    return top - staff.step(y)


def _name(h: omr.Head) -> tuple[str, int, int]:
    octave, step = divmod(h.diatonic, 7)
    return "CDEFGAB"[step], h.alter, octave


def _beams(fills, left: float, right: float, top: float, bottom: float, sp: float) -> list[Beam]:
    out = []
    for poly in fills:
        poly = list(dict.fromkeys((round(x, 2), round(y, 2)) for x, y in poly))  # a closed path repeats its start
        xs = [p[0] for p in poly]
        x0, x1 = min(xs), max(xs)
        if x1 - x0 < 0.7 * sp or x1 < left or x0 > right:
            continue
        l = [p[1] for p in poly if abs(p[0] - x0) < 0.4]
        r = [p[1] for p in poly if abs(p[0] - x1) < 0.4]
        if len(l) < 2 or len(r) < 2 or len(l) + len(r) != len(poly):
            continue
        t = (max(l) - min(l) + max(r) - min(r)) / 2
        y0, y1 = (max(l) + min(l)) / 2, (max(r) + min(r)) / 2
        if not 0.25 * sp <= t <= 0.85 * sp or not top <= y0 <= bottom:
            continue
        if abs(y1 - y0) > (x1 - x0):  # steeper than 45 degrees: not a beam
            continue
        out.append(Beam(x0, x1, y0, y1, t))
    return out


def _chords(heads: list[Glyph], stems: list[Stem], beams: list[Beam], mine: list[Glyph], sp: float,
            staff_of: dict[int, int], pitch: dict[int, tuple[int, int]], ids) -> list[omr.Chord]:
    """Heads on one stem make a chord; whole notes in one column do too; rests are chords of their own."""
    chords: list[omr.Chord] = []
    used: set[int] = set()
    tol = 0.3 * sp

    def head(g: Glyph) -> omr.Head:
        st = g.staff
        d, alter, written = pitch[id(g)]
        octave, step = divmod(d, 7)
        h = omr.Head(next(ids), HEAD_SHAPES[g.value], st.step(g.y) - 4,
                     omr.Box(g.x0, g.y - sp / 2, g.x1 - g.x0, sp), d, alter,
                     12 * (octave + 1) + omr.NATURAL_PCS[step] + alter)
        h.glyph, h.written = g, written  # type: ignore[attr-defined]
        return h

    # each stem's own heads: those whose normal side touches it (left of an up-stem, right of a down-stem), plus
    # heads a second away pushed to the other side, unless that head is another stem's own
    found = []
    for s in sorted(stems, key=lambda s: s.x):
        touching = [g for g in heads if g.value in (Fraction(1), Fraction(2))
                    and (abs(g.x1 - s.x) <= tol or abs(g.x0 - s.x) <= tol)
                    and s.y0 - 0.6 * sp <= g.y <= s.y1 + 0.6 * sp]
        if not touching:
            continue
        top, bottom = min(g.y for g in touching), max(g.y for g in touching)
        up = (top - s.y0) > (s.y1 - bottom)
        normal = [g for g in touching if abs((g.x1 if up else g.x0) - s.x) <= tol]
        if normal:
            found.append((s, up, normal, [g for g in touching if g not in normal]))
    owned = {id(g) for _, _, normal, _ in found for g in normal}
    for s, up, normal, other in found:
        on = normal + [g for g in other if id(g) not in owned
                       and any(abs(pitch[id(g)][0] - pitch[id(n)][0]) == 1 for n in normal)]
        staves = [staff_of[id(g.staff)] for g in on]
        staff = max(set(staves), key=staves.count)
        tip = s.y0 if up else s.y1
        hits = [k for k, b in enumerate(beams) if b.x0 - 0.5 <= s.x <= b.x1 + 0.5
                and s.y0 - 0.6 * sp <= b.y_at(s.x) <= s.y1 + 0.6 * sp
                and abs(b.y_at(s.x) - tip) < 2.5 * sp]
        flags = [g for g in mine if g.kind == "flag" and s.x - 0.4 * sp <= g.x0 <= s.x + 0.8 * sp
                 and abs(g.y - tip) < 2.5 * sp]
        # black and void heads on one stem are two voices whose stems coincide: a chord each, the black one
        # taking the beams and flags
        kinds = sorted({g.value for g in on})
        for value in kinds:
            part = [g for g in on if g.value == value]
            black = value == kinds[0]
            hs = [head(g) for g in sorted(part, key=lambda g: -g.y)]
            c = Chord(next(ids), staff, _union([h.box for h in hs]), hs, stem=next(ids), stem_up=up, stem_x=s.x,
                          beams=len(hits) if black else 0, beam_ids=tuple(hits) if black else (),
                          flags=max((g.value for g in flags), default=0) if black else 0, voice=1)
            c.grace = all(g.grace for g in part)
            c.stem_box = omr.Box(s.x, s.y0, 0, s.y1 - s.y0)
            chords.append(c)
        used.update(id(g) for g in on)
    loose = sorted((g for g in heads if id(g) not in used), key=lambda g: (staff_of[id(g.staff)], g.x0))
    for g in loose:  # whole notes (and any head without a stem) by column
        st = staff_of[id(g.staff)]
        same = next((c for c in chords if c.stem is None and c.staff == st and not c.rest
                     and abs(c.heads[0].box.x - g.x0) < 1.2 * (g.x1 - g.x0)
                     and all(h.glyph.value == g.value for h in c.heads)), None)  # type: ignore[attr-defined]
        h = head(g)
        if same is not None and all(x.diatonic != h.diatonic for x in same.heads):
            same.heads.append(h)
            same.box = _union([x.box for x in same.heads])
            continue
        c = Chord(next(ids), st, h.box, [h], voice=1)
        c.grace = g.grace
        chords.append(c)
    for g in mine:
        if g.kind == "rest":
            c = Chord(next(ids), staff_of[id(g.staff)], omr.Box(g.x0, g.y - sp, g.x1 - g.x0, 2 * sp),
                          rest=REST_SHAPES[g.value], voice=1)
            chords.append(c)
    _dots(chords, [g for g in mine if g.kind == "dot"], sp)
    return chords


def _union(boxes: list[omr.Box]) -> omr.Box:
    x0, y0 = min(b.x for b in boxes), min(b.y for b in boxes)
    return omr.Box(x0, y0, max(b.right for b in boxes) - x0, max(b.bottom for b in boxes) - y0)


def _dots(chords: list[omr.Chord], dots: list[Glyph], sp: float) -> None:
    """Augmentation dots: right of a head or rest at about its height. A dot over or under a head is staccato."""
    targets = []  # (chord, x1, y)
    for c in chords:
        if c.heads:
            targets += [(c, h.box.right, h.box.cy) for h in c.heads]
        else:
            targets.append((c, c.box.right, c.box.cy))
    heads_x = [(h.box.x, h.box.right, h.box.cy) for c in chords for h in c.heads]
    count: dict[int, set[float]] = {}
    for d in sorted(dots, key=lambda d: d.x0):
        if any(a - 0.1 * sp <= d.cx <= b + 0.1 * sp and abs(y - d.y) < 2.5 * sp for a, b, y in heads_x):
            continue
        cands = [(d.x0 - x1 + 2 * abs(d.y - y), c) for c, x1, y in targets
                 if (d.staff is None or d.staff.index in (0, c.staff)) and 0 <= d.x0 - x1 <= 2.5 * sp and abs(d.y - y) <= 0.8 * sp]
        if not cands:
            continue
        _, c = min(cands, key=lambda t: t[0])
        count.setdefault(c.id, set()).add(round(d.x0, 0))
    for c in chords:
        c.dots = len(count.get(c.id, ()))


def _ties(chords: list[omr.Chord], curves: list[Curve], sp: float, state: _State, right: float) -> None:
    """A curve from just right of a head to just left of the next head of the same pitch on its staff is a tie.
    One that runs off the end of the system ties into the next system's first note of that pitch."""
    heads = [(c, h) for c in chords if not c.grace for h in c.heads]
    pending, state.ties = state.ties, {}
    firsts: dict[tuple[int, int], tuple[float, omr.Head]] = {}
    for c, h in heads:
        k = (c.staff, h.diatonic)
        if k not in firsts or h.box.x < firsts[k][0]:
            firsts[k] = (h.box.x, h)
    for (staff, d), alter in pending.items():
        first = firsts.get((staff, d))
        if first is not None:
            first[1].ties.add("stop")
            if not first[1].written:  # type: ignore[attr-defined]
                first[1].alter = alter
                first[1].glyph.source = "tie"  # type: ignore[attr-defined]
    for cv in sorted(curves, key=lambda cv: cv.x0):
        if abs(cv.y0 - cv.y1) > 0.8 * sp or cv.x1 - cv.x0 < 0.5 * sp:
            continue
        starts = [(c, h) for c, h in heads if h.box.right - 0.7 * sp <= cv.x0 <= h.box.right + 1.5 * sp
                  and abs(h.box.cy - cv.y0) <= 1.1 * sp]
        ends = [(c, h) for c, h in heads if h.box.x - 1.5 * sp <= cv.x1 <= h.box.x + 0.7 * sp
                and abs(h.box.cy - cv.y1) <= 1.1 * sp]
        pairs = [(a, b) for a in starts for b in ends if a[0].staff == b[0].staff and a[1].diatonic == b[1].diatonic
                 and b[1].box.x > a[1].box.x]
        if pairs:
            a, b = min(pairs, key=lambda p: abs(p[0][1].box.cy - cv.y0) + abs(p[1][1].box.cy - cv.y1))
            a[1].ties.add("start")
            b[1].ties.add("stop")
            if not b[1].written:  # type: ignore[attr-defined]
                b[1].alter = a[1].alter
                b[1].glyph.source = "tie"  # type: ignore[attr-defined]
        elif starts and cv.x1 >= right - 2 * sp:
            c, h = min(starts, key=lambda t: abs(t[1].box.cy - cv.y0))
            h.ties.add("start")
            state.ties[(c.staff, h.diatonic)] = h.alter
    for c, h in heads:  # a tie carries the alteration
        octave, step = divmod(h.diatonic, 7)
        h.midi = 12 * (octave + 1) + omr.NATURAL_PCS[step] + h.alter


def _tuplets(chords, words, glyphs, beams: list[Beam], sp: float, upper: Staff, lower: Staff) -> None:
    """A tuplet number (italic text or a SMuFL tuplet digit) over or under a beam group or bracket times the
    chords under it by 2/3 (for 3 and 6), 4/5..."""
    marks = [(int(t), (x0 + x1) / 2, (y0 + y1) / 2) for t, x0, y0, x1, y1, italic in words
             if italic and t in ("3", "5", "6", "7")]
    marks += [(g.value, g.cx, g.y) for g in glyphs if g.kind == "tuplet" and g.value in (3, 5, 6, 7)]
    for n, x, y in marks:
        if not (upper.top - 9 * sp <= y <= lower.bottom + 9 * sp) or not (upper.x0 <= x <= upper.x1):
            continue
        normal = 2 if n == 3 else 4
        ratio = Fraction(normal, n)
        group = [k for k, b in enumerate(beams) if b.x0 - 0.5 * sp <= x <= b.x1 + 0.5 * sp and abs(b.y_at(x) - y) < 3 * sp]
        members = [c for c in chords if c.heads and set(c.beam_ids) & set(group)] if group else []
        if not members:
            members = _bracketed(chords, n, x, y, sp, (upper, lower))
        for c in members:
            c.tuplet = ratio


def _bracketed(chords, n: int, x: float, y: float, sp: float, staves: tuple[Staff, Staff]) -> list:
    """The n chords under a bracketed tuplet number: in n neighbouring columns on the staff nearest the number, the
    run centred under it, in each column the chord nearest the number (the voice the bracket is on)."""
    near = [c for c in chords if not c.grace and abs(c.box.cy - y) < 8 * sp and abs(c.anchor - x) < 10 * sp]
    if not near:
        return []
    staff = 1 if staves[0].distance(y) <= staves[1].distance(y) else 2
    cols: list[list] = []
    for c in sorted((c for c in near if c.staff == staff), key=lambda c: c.anchor):
        if cols and c.anchor - cols[-1][0].anchor < 0.6 * sp:
            cols[-1].append(c)
        else:
            cols.append([c])
    picks = [min(col, key=lambda c: abs(c.box.cy - y)) for col in cols]
    best = None
    for start in range(len(picks) - n + 1):
        window = picks[start:start + n]
        if window[-1].anchor - window[0].anchor <= 14 * sp:
            off = abs(sum(c.anchor for c in window) / n - x)
            if best is None or off < best[0]:
                best = (off, window)
    return best[1] if best else []
