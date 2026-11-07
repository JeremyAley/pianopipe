"""Reading an Audiveris project (.omr) into per-measure geometry: chords, rests, stems, beams, flags, dots.

An .omr file is a zip holding book.xml and one sheet#N/sheet#N.xml per page. Each sheet keeps every symbol
Audiveris accepted ("inters": heads, stems, beams, rests...) with its bounding box, the links between them
("relations": head-stem, beam-stem, chord-stem, augmentation...), and its layout: pages, systems, parts, staves
and measures (with the head and rest chords of each). omr_repair uses this to re-time measures that Audiveris's
own rhythm step got wrong; see docs/catalog-expansion.md §3.2.

Standard library only, like the rest of the pipeline.
"""

from __future__ import annotations

import io
import re
import struct
import zipfile
import zlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

# Note head shapes and their value in quarter notes before beams, flags and dots.
HEAD_VALUES = {
    "NOTEHEAD_BLACK": Fraction(1), "NOTEHEAD_BLACK_SMALL": Fraction(1), "NOTEHEAD_VOID": Fraction(2),
    "NOTEHEAD_VOID_SMALL": Fraction(2), "WHOLE_NOTE": Fraction(4), "WHOLE_NOTE_SMALL": Fraction(4),
    "BREVE": Fraction(8),
}
REST_VALUES = {
    "BREVE_REST": Fraction(8), "WHOLE_REST": Fraction(4), "HALF_REST": Fraction(2), "QUARTER_REST": Fraction(1),
    "EIGHTH_REST": Fraction(1, 2), "ONE_16TH_REST": Fraction(1, 4), "ONE_32ND_REST": Fraction(1, 8),
    "ONE_64TH_REST": Fraction(1, 16), "ONE_128TH_REST": Fraction(1, 32),
}
CLEF_STEPS = {  # diatonic step (C4 = 28) of the middle staff line, by clef kind
    "TREBLE": 34, "BASS": 22, "ALTO": 28, "TENOR": 26, "TREBLE_8VB": 27, "TREBLE_8VA": 41, "BASS_8VB": 15,
}
OCTAVE_SHIFTS = {"OTTAVA": 1, "QUINDICESIMA": 2, "VENTIDUESIMA": 3}
SHARPS = "FCGDAEB"  # key signature order
NATURAL_PCS = [0, 2, 4, 5, 7, 9, 11]


class OmrError(ValueError):
    """The file isn't an Audiveris project we can read."""


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def bottom(self) -> float:
        return self.y + self.h


@dataclass
class Inter:
    id: int
    kind: str  # the XML tag: head, stem, beam, rest, flag...
    shape: str
    box: Box
    staff: int | None  # sheet-wide staff id
    attrs: dict[str, str]


@dataclass
class Head:
    id: int
    shape: str
    step: int  # Audiveris pitch: staff steps from the middle line, positive downwards
    box: Box
    diatonic: int | None = None  # filled in by _pitch: C4 = 28
    alter: int = 0
    midi: int | None = None
    ties: set[str] = field(default_factory=set)  # "start" (a tie leaves this head), "stop" (one arrives)


@dataclass
class Chord:
    """A head chord (one stem's heads) or a rest, with what its symbols say about its duration."""
    id: int
    staff: int  # 1 or 2 within the part (1 = top staff)
    box: Box
    heads: list[Head] = field(default_factory=list)
    rest: str = ""  # rest shape, "" for a head chord
    stem: int | None = None
    stem_up: bool | None = None
    stem_x: float | None = None
    beams: int = 0  # beams and beam hooks on the stem
    beam_ids: tuple[int, ...] = ()
    flags: int = 0
    seen_beams: int = 0  # beams found in the page image where Audiveris saw none (_beam_evidence)
    stem_box: Box | None = None
    dots: int = 0
    tuplet: Fraction | None = None  # e.g. 2/3 for a triplet
    measure_rest: bool = False
    voice: int | None = None  # Audiveris's voice, None if it left the chord out of every voice

    @property
    def x(self) -> float:
        """Left edge of the chord's heads (or the rest), for sorting and for matching MusicXML's default-x."""
        if self.heads:
            return min(h.box.x for h in self.heads)
        return self.box.x

    @property
    def anchor(self) -> float:
        """Where the chord sits in time: the centre of a head on the normal side of the stem (heads pushed to the
        other side by a second don't count), or the rest's centre."""
        if self.heads and self.stem_x is not None:
            w = sum(h.box.w for h in self.heads) / len(self.heads)
            return self.stem_x - w / 2 if self.stem_up else self.stem_x + w / 2
        if self.heads:
            centres = sorted(h.box.cx for h in self.heads)
            return centres[len(centres) // 2]
        return self.box.cx

    @property
    def value(self) -> Fraction | None:
        """Duration in quarter notes from the symbols alone (None for a measure rest)."""
        return self.value_with()

    def value_with(self, extra_beams: int = 0, extra_dots: int = 0, tuplet: Fraction | None = None) -> Fraction | None:
        """The duration with symbols Audiveris missed: beams (each halves a beamed chord's value), dots, a tuplet."""
        if self.rest:
            if self.measure_rest:
                return None
            base = REST_VALUES.get(self.rest, Fraction(1))
        else:
            base = max((HEAD_VALUES.get(h.shape, Fraction(1)) for h in self.heads), default=Fraction(1))
            beams = max(self.beams, self.flags, self.seen_beams) + extra_beams
            if base == 1 or (base == 2 and beams):  # a void head on a beam is a misread black one
                base = Fraction(1, 2 ** beams)
        value = base * (2 - Fraction(1, 2 ** (self.dots + extra_dots)))
        if self.tuplet or tuplet:
            value *= self.tuplet or tuplet
        return value


@dataclass
class Measure:
    """One measure of the piano part as Audiveris laid it out."""
    number: int  # 1-based, in the order of the score (matches the MusicXML export)
    sheet: int
    left: float
    right: float
    chords: list[Chord] = field(default_factory=list)
    time: Fraction | None = None  # time signature printed in this measure, in quarter notes
    abnormal: bool = False
    interline: float = 20.0
    absent: bool = False  # the part has no staves in this system (Audiveris split one system in two)
    alone: bool = False  # the part is the only one in its system although the score has others
    unread: list[tuple[int, float]] = field(default_factory=list)  # (staff, x) of head-like marks no symbol explains
    top: float = 0.0  # the piano's top and bottom staff lines on the page (for crops of the bar)
    bottom: float = 0.0


@dataclass
class Score:
    measures: list[Measure]


def _box(el: ET.Element | None) -> Box:
    if el is None:
        return Box(0, 0, 0, 0)
    return Box(float(el.get("x", 0)), float(el.get("y", 0)), float(el.get("w", 0)), float(el.get("h", 0)))


def _ints(text: str | None) -> list[int]:
    return [int(t) for t in (text or "").split()]


class Sheet:
    """One page of the project: its inters, relations and layout."""

    def __init__(self, root: ET.Element, number: int, image: bytes | None = None):
        self.number = number
        self.interline = float(root.find("scale/interline").get("main", 20)) if root.find("scale/interline") is not None else 20.0
        beam = root.find("scale/beam")
        self.beam_thickness = (float(beam.get("main-thickness", 0)) if beam is not None else 0.0) or self.interline / 2
        self._png = image
        self._image: Image | None = None
        head = root.find("scale/black-head")
        self.head_size = ((float(head.get("mean-width", 0)), float(head.get("mean-height", 0))) if head is not None
                          else (1.3 * self.interline, 1.1 * self.interline))
        self.glyphs: dict[int, Box] = {}
        for g in root.iter("glyph"):
            rt = g.find("run-table")
            if g.get("id") and rt is not None:
                self.glyphs[int(g.get("id"))] = Box(float(g.get("left", 0)), float(g.get("top", 0)),
                                                    float(rt.get("width", 0)), float(rt.get("height", 0)))
        self.inters: dict[int, Inter] = {}
        self.links: dict[int, list[tuple[str, int]]] = {}  # inter id -> [(relation kind, other inter id)]
        self.sides: dict[tuple[int, int], str] = {}  # (slur, head) -> "LEFT" or "RIGHT"
        for sig in root.iter("sig"):
            inters = sig.find("inters")
            for el in inters if inters is not None else []:
                i = int(el.get("id"))
                staff = el.get("staff")
                self.inters[i] = Inter(i, el.tag, el.get("shape", ""), _box(el.find("bounds")),
                                       int(staff) if staff else None, dict(el.attrib))
            rels = sig.find("relations")
            for rel in rels if rels is not None else []:
                kind = rel[0].tag if len(rel) else ""
                s, t = int(rel.get("source")), int(rel.get("target"))
                self.links.setdefault(s, []).append((kind, t))
                if kind == "slur-head":
                    self.sides[(s, t)] = rel[0].get("side", "")
                self.links.setdefault(t, []).append((kind, s))
        self.root = root

    @property
    def image(self) -> Image | None:
        """The binarised page Audiveris worked from, or None if the project doesn't keep one we can read."""
        if self._image is None and self._png:
            self._image = Image.from_png(self._png)
            self._png = None
        return self._image

    def linked(self, i: int, kind: str, tag: str | None = None) -> list[Inter]:
        out = []
        for k, other in self.links.get(i, []):
            if k == kind and other in self.inters and (tag is None or self.inters[other].kind == tag):
                out.append(self.inters[other])
        return out

    def chord(self, cid: int, staff_index: dict[int, int]) -> Chord | None:
        inter = self.inters.get(cid)
        if inter is None:
            return None
        staff = inter.staff
        if staff is None:  # Audiveris leaves some chords without a staff; their heads have one
            staff = next((m.staff for m in self.linked(cid, "containment") if m.staff is not None), None)
        c = Chord(cid, staff_index.get(staff, 1), inter.box)
        if inter.kind == "rest-chord":
            rests = self.linked(cid, "containment", "rest")
            if not rests:
                return None
            c.rest = rests[0].shape
            c.box = rests[0].box
            members = rests
        else:
            members = self.linked(cid, "containment", "head")
            c.heads = [Head(h.id, h.shape, round(float(h.attrs.get("pitch", 0))), h.box) for h in members]
            for h in c.heads:
                for slur in self.linked(h.id, "slur-head", "slur"):
                    if slur.attrs.get("tie") == "true":
                        side = self.sides.get((slur.id, h.id), "")
                        h.ties.add({"LEFT": "start", "RIGHT": "stop"}.get(side, ""))
                h.ties.discard("")
            stems = self.linked(cid, "chord-stem", "stem")
            if stems:
                stem = stems[0]
                c.stem = stem.id
                head_y = sum(h.box.y + h.box.h / 2 for h in c.heads) / max(1, len(c.heads))
                c.stem_up = stem.box.y + stem.box.h / 2 < head_y
                c.stem_x = stem.box.cx
                c.stem_box = stem.box
                beams = self.linked(stem.id, "beam-stem")
                c.beam_ids = tuple(sorted(b.id for b in beams if b.kind == "beam"))
                c.beams = len(beams)
                for f in self.linked(stem.id, "flag-stem"):
                    m = re.search(r"FLAG_(\d)", f.shape)
                    c.flags = max(c.flags, int(m.group(1)) if m else 1)
        for m in members:  # every head of a dotted chord has its own dot; a second dot hangs off the first
            dots = self.linked(m.id, "augmentation", "augmentation-dot")
            c.dots = max(c.dots, len(dots) + sum(1 for d in dots for _ in self.linked(d.id, "double-dot", "augmentation-dot")))
        for t in self.linked(cid, "chord-tuplet", "tuplet"):
            n = re.search(r"TUPLET_(\w+)", t.shape)
            c.tuplet = {"THREE": Fraction(2, 3), "SIX": Fraction(2, 3), "FIVE": Fraction(4, 5)}.get(n.group(1) if n else "", Fraction(2, 3))
        return c


def read_sheets(path: str | Path) -> list[Sheet]:
    data = Path(path).read_bytes()
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise OmrError(f"not an Audiveris project: {e}") from e
    names = [n for n in z.namelist() if re.fullmatch(r"sheet#(\d+)/sheet#\1\.xml", n)]
    if not names:
        raise OmrError("no sheets in the Audiveris project")
    names.sort(key=lambda n: int(re.match(r"sheet#(\d+)", n).group(1)))
    sheets = []
    for n in names:
        number = int(re.match(r"sheet#(\d+)", n).group(1))
        png = f"sheet#{number}/BINARY.png"
        sheets.append(Sheet(ET.fromstring(z.read(n)), number, z.read(png) if png in z.namelist() else None))
    return sheets


class Image:
    """An 8-bit greyscale PNG (what Audiveris writes for BINARY.png), decoded with the standard library."""

    def __init__(self, width: int, height: int, pixels: bytes):
        self.width, self.height, self.pixels = width, height, pixels

    def dark(self, x: float, y: float) -> bool:
        xi, yi = int(x), int(y)
        return 0 <= xi < self.width and 0 <= yi < self.height and self.pixels[yi * self.width + xi] < 128

    @classmethod
    def from_png(cls, data: bytes) -> Image | None:
        if data[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        pos, idat, header = 8, [], None
        while pos + 8 <= len(data):
            n, kind = struct.unpack(">I4s", data[pos:pos + 8])
            if kind == b"IHDR":
                header = struct.unpack(">IIBBBBB", data[pos + 8:pos + 21])
            elif kind == b"IDAT":
                idat.append(data[pos + 8:pos + 8 + n])
            pos += 12 + n
        if header is None or header[2:5] != (8, 0, 0) or header[6]:  # 8-bit grey, not interlaced
            return None
        w, h = header[0], header[1]
        raw = zlib.decompress(b"".join(idat))
        out = bytearray(w * h)
        prev = bytearray(w)
        for r in range(h):
            f, row = raw[r * (w + 1)], bytearray(raw[r * (w + 1) + 1:(r + 1) * (w + 1)])
            if f == 1:
                for i in range(1, w):
                    row[i] = (row[i] + row[i - 1]) & 255
            elif f == 2:
                row = bytearray((a + b) & 255 for a, b in zip(row, prev))
            elif f == 3:
                for i in range(w):
                    row[i] = (row[i] + ((row[i - 1] if i else 0) + prev[i]) // 2) & 255
            elif f == 4:
                for i in range(w):
                    a, b, c = (row[i - 1] if i else 0), prev[i], (prev[i - 1] if i else 0)
                    p = a + b - c
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                    row[i] = (row[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
            out[r * w:(r + 1) * w] = row
            prev = row
        return cls(w, h, bytes(out))


def _unread_heads(sheet: Sheet, free: list[Box], extents: dict[int, tuple[float, float]], left: float,
                  right: float) -> list[tuple[int, float]]:
    """Marks Audiveris kept as glyphs but read as nothing, that are the size and blackness of a note head and sit on
    one of the part's staves between two barlines: probably notes it missed."""
    image = sheet.image
    if image is None:
        return []
    w, h = sheet.head_size
    il = sheet.interline
    out = []
    for g in free:
        # (not just after the barline, where a time signature Audiveris missed leaves digits like these)
        if not (left + 2.2 * il < g.cx < right and 0.75 * w <= g.w <= 1.5 * w and 0.7 * h <= g.h <= 1.5 * h):
            continue
        staff = next((s for s, (top, bottom) in extents.items() if top - 0.5 * il <= g.cy <= bottom + 0.5 * il), None)
        if staff is None:
            continue
        inside = [(x, y) for x in range(int(g.x), int(g.right)) for y in range(int(g.y), int(g.bottom))]
        if inside and sum(image.dark(x, y) for x, y in inside) >= 0.6 * len(inside):
            out.append((staff, g.cx))
    return out


def _beam_count(image: Image, x: float, y_from: float, y_to: float, thickness: float) -> int:
    """Beams crossing the vertical line at x between two heights: dark runs a beam thick (two beams printed close
    together can merge into one run of about twice the thickness)."""
    step = 1 if y_to >= y_from else -1
    runs, run = [], 0
    for y in range(int(y_from), int(y_to) + step, step):
        if image.dark(x, y):
            run += 1
        elif run:
            runs.append(run)
            run = 0
    if run:
        runs.append(run)
    return sum(max(1, round(r / (thickness * 1.35))) for r in runs if r >= 0.5 * thickness)


def _beam_evidence(sheet: Sheet, chords: list[Chord]) -> None:
    """Look in the page image for beams between neighbouring stems that Audiveris linked to no beam or flag.

    For each pair of black-headed, unbeamed chords on one staff with stems the same way, beams are counted on
    three vertical lines between the stems, near their tips: the left chord takes the count near its stem, the
    right one near its own (a beam hook covers only one side), and only if the middle line crosses a beam too.
    """
    image = sheet.image
    if image is None:
        return
    il, t = sheet.interline, sheet.beam_thickness
    plain = [c for c in chords if c.heads and c.stem_box is not None and not c.beams and not c.flags
             and all(h.shape.startswith("NOTEHEAD_BLACK") for h in c.heads)]
    for staff in {c.staff for c in plain}:
        for up in (True, False):
            row = sorted((c for c in plain if c.staff == staff and c.stem_up is up), key=lambda c: c.stem_box.cx)
            for a, b in zip(row, row[1:]):
                xa, xb = a.stem_box.cx, b.stem_box.cx
                if not 0.8 * il < xb - xa < 6 * il:
                    continue
                ya, yb = (a.stem_box.y, b.stem_box.y) if up else (a.stem_box.bottom, b.stem_box.bottom)

                def count(f: float) -> int:
                    x, y = xa + f * (xb - xa), ya + f * (yb - ya)
                    if up:
                        return _beam_count(image, x, y - 0.4 * il, y + 2.5 * il, t)
                    return _beam_count(image, x, y + 0.4 * il, y - 2.5 * il, t)

                mid = count(0.5)
                if mid:
                    a.seen_beams = max(a.seen_beams, mid, min(count(0.2), 3))
                    b.seen_beams = max(b.seen_beams, mid, min(count(0.8), 3))



def read(path: str | Path, part_id: int | None = None) -> Score:
    """The measures of one part (by default the last two-staff part of each system: the piano)."""
    measures: list[Measure] = []
    clefs: dict[int, list[tuple[float, str]]] = {}  # staff (1, 2) -> [(x, clef kind)] in effect, carried across systems
    keys: dict[int, int] = {}
    for sheet in read_sheets(path):
        for system in sheet.root.iter("system"):
            parts = system.findall("part")
            if not parts:
                continue
            if part_id is not None:
                chosen = next((p for p in parts if p.get("id") == str(part_id)), None)
            else:
                chosen = next((p for p in reversed(parts) if len(p.findall("staff")) >= 2), None)
            if chosen is None:
                # The MusicXML export still has these measures, with only rests in this part.
                for m_el in parts[0].findall("measure"):
                    if not m_el.get("id", "").endswith("C"):
                        measures.append(Measure(len(measures) + 1, sheet.number, 0, 0, interline=sheet.interline, absent=True))
                continue
            staff_ids = [int(s.get("id")) for s in chosen.findall("staff")]
            staff_index = {sid: k + 1 for k, sid in enumerate(staff_ids)}
            stacks = system.findall("stack")
            free = [sheet.glyphs[g] for g in _ints(system.findtext("free-glyphs")) if g in sheet.glyphs]
            extents = {}
            for st in chosen.findall("staff"):
                ys = [float(pt.get("y")) for pt in st.iter("point")]
                if ys:
                    extents[staff_index[int(st.get("id"))]] = (min(ys), max(ys))
            for st in chosen.findall("staff"):
                idx = staff_index[int(st.get("id"))]
                header = st.find("header")
                if header is not None:
                    clef = sheet.inters.get(int(header.findtext("clef") or 0))
                    key = sheet.inters.get(int(header.findtext("key") or 0))
                    if clef is not None:
                        clefs[idx] = [(0.0, clef.attrs.get("kind", "TREBLE"))]
                    if key is not None:
                        keys[idx] = int(key.attrs.get("fifths", 0))
            for k, m_el in enumerate(chosen.findall("measure")):
                stack = stacks[k] if k < len(stacks) else None
                if m_el.get("id", "").endswith("C") and measures:
                    # a measure Audiveris cut in two (a courtesy time signature, a mid-bar barline): one measure
                    m = measures[-1]
                    m.right = float(stack.get("right", m.right)) if stack is not None else m.right
                else:
                    m = Measure(len(measures) + 1, sheet.number,
                                float(stack.get("left", 0)) if stack is not None else 0,
                                float(stack.get("right", 0)) if stack is not None else 0,
                                abnormal=m_el.get("abnormal") == "true", interline=sheet.interline,
                                alone=len(parts) == 1,
                                top=min((e[0] for e in extents.values()), default=0.0),
                                bottom=max((e[1] for e in extents.values()), default=0.0))
                    measures.append(m)
                voice_of: dict[int, int] = {}
                for v in m_el.findall("voice"):
                    vid = int(v.get("id"))
                    for val in v.iter("value"):
                        voice_of.setdefault(int(val.get("chord")), vid)
                    if v.get("measure-rest-chord"):
                        voice_of[int(v.get("measure-rest-chord"))] = vid
                measure_rests = {int(v.get("measure-rest-chord")) for v in m_el.findall("voice") if v.get("measure-rest-chord")}
                for cid in _ints(m_el.findtext("head-chords")) + _ints(m_el.findtext("rest-chords")):
                    c = sheet.chord(cid, staff_index)
                    if c is None:
                        continue
                    c.voice = voice_of.get(cid)
                    c.measure_rest = cid in measure_rests or (c.rest == "WHOLE_REST" and _alone(c, m_el, sheet, staff_index))
                    m.chords.append(c)
                _pitch(sheet, m_el, m, staff_index, clefs, keys)
                _beam_evidence(sheet, m.chords)
                if stack is not None:
                    m.unread.extend(_unread_heads(sheet, free, extents, float(stack.get("left", 0)),
                                                  float(stack.get("right", 0))))
                for tid in [] if m_el.get("id", "").endswith("C") else _ints(m_el.findtext("times")):
                    t = sheet.inters.get(tid)
                    if t is not None and t.attrs.get("time-rational"):
                        num, den = t.attrs["time-rational"].split("/")
                        m.time = Fraction(4 * int(num), int(den))
                m.chords.sort(key=lambda c: (c.x, c.staff))
    return Score(measures)


def _pitch(sheet: Sheet, m_el: ET.Element, m: Measure, staff_index: dict[int, int],
           clefs: dict[int, list[tuple[float, str]]], keys: dict[int, int]) -> None:
    """MIDI pitches for the measure's heads: clef, key signature, accidentals carried through the measure, 8va."""
    for cid in _ints(m_el.findtext("clefs")):
        clef = sheet.inters.get(cid)
        if clef is not None and clef.box.x > m.left + 1:
            clefs.setdefault(staff_index.get(clef.staff, 1), []).append((clef.box.x, clef.attrs.get("kind", "TREBLE")))
    for kid in _ints(m_el.findtext("keys")):
        key = sheet.inters.get(kid)
        if key is not None:
            keys[staff_index.get(key.staff, 1)] = int(key.attrs.get("fifths", 0))
    shifts = []
    for i in sheet.inters.values():
        if i.kind == "octave-shift" and i.staff in staff_index:
            el = next(e for e in sheet.root.iter("octave-shift") if e.get("id") == str(i.id))
            line = el.find("line")
            x1 = float(line.find("p2").get("x")) if line is not None and line.find("p2") is not None else i.box.right
            octaves = OCTAVE_SHIFTS.get(i.shape, 1) * (-1 if i.attrs.get("kind") == "BASSA" else 1)
            shifts.append((staff_index[i.staff], i.box.x, x1, octaves))
    altered: dict[tuple[int, int], int] = {}
    heads = sorted(((c.staff, h) for c in m.chords for h in c.heads), key=lambda sh: sh[1].box.x)
    for staff, h in heads:
        kind = next((k for x, k in reversed(clefs.get(staff, [(0.0, "TREBLE")])) if x <= h.box.x), "TREBLE")
        diatonic = CLEF_STEPS.get(kind, 34) - h.step
        octave, step = divmod(diatonic, 7)
        alter = None
        for a in sheet.linked(h.id, "alter-head", "alter"):
            alter = {"SHARP": 1, "FLAT": -1, "NATURAL": 0, "DOUBLE_SHARP": 2, "DOUBLE_FLAT": -2}.get(a.shape, 0)
        if alter is not None:
            altered[(staff, diatonic)] = alter
        else:
            alter = altered.get((staff, diatonic))
        if alter is None:
            fifths = keys.get(staff, 0)
            name = "CDEFGAB"[step]
            alter = (1 if name in SHARPS[:fifths] else 0) if fifths >= 0 else (-1 if name in SHARPS[::-1][:-fifths] else 0)
        shift = sum(o for s, x0, x1, o in shifts if s == staff and x0 - 5 <= h.box.x <= x1 + 5)
        h.diatonic, h.alter = diatonic + 7 * shift, alter
        h.midi = 12 * (octave + 1 + shift) + NATURAL_PCS[step] + alter
    for staff in list(clefs):
        clefs[staff] = [(0.0, clefs[staff][-1][1])]


def _alone(c: Chord, m_el: ET.Element, sheet: Sheet, staff_index: dict[int, int]) -> bool:
    """A whole rest with nothing else on its staff in the measure is a measure rest."""
    for cid in _ints(m_el.findtext("head-chords")) + _ints(m_el.findtext("rest-chords")):
        inter = sheet.inters.get(cid)
        if cid != c.id and inter is not None and staff_index.get(inter.staff, 1) == c.staff:
            return False
    return True
