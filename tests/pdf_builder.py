"""Builders for small synthetic vector PDFs of piano music, for the PDF check tests (test_pdf_check.py).

They draw what Finale and Sibelius exports hold: staff lines, barlines and stems as lines, beams as filled
parallelograms, ties as filled curves, and the symbols as text in a font laid out like Sonata (œ a black head, ˙ a
half head, & a treble clef...). No music font ships with the pipeline, so the symbols are drawn with Liberation Serif,
which has those characters, and the tests register it as a Sonata-layout font (FONT_NAME).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf

import pianopipe

FONT = Path(pianopipe.__file__).parent / "fonts" / "LiberationSerif-Regular.ttf"
FONT_NAME = "LiberationSerif"
SP = 5.0  # staff space
SIZE = 9.0  # font size: a black head about 1.3 spaces wide
LEFT = 40.0
HEADER = 45.0  # clef, key and time signature at a system's start
BAR = 180.0  # width of a bar's notes
QUARTER = 36.0  # space per quarter note
STEPS = "CDEFGAB"
TOP_LINE = {"treble": 38, "bass": 26}  # diatonic step (C4 = 28) of the top line
_FONT = pymupdf.Font(fontfile=str(FONT))
_W = _FONT.text_length("œ", SIZE)  # a black head's width


def diatonic(name: str) -> int:
    return 7 * int(name[-1]) + STEPS.index(name[0])


@dataclass
class Head:
    x: float
    y: float
    staff: int


class Sheet:
    """One page with systems of a grand staff (and optionally a vocal staff above each). `systems` gives each
    system's bar count; `clefs` the staves' clefs at the start."""

    def __init__(self, systems: tuple[int, ...] = (2,), key: int = 0, time: tuple[int, int] | None = (4, 4),
                 vocal: bool = False, clefs: tuple[str, str] = ("treble", "bass")):
        self.doc = pymupdf.open()
        self.page = self.doc.new_page(width=612, height=792)
        self.shape = self.page.new_shape()
        self.tops: list[dict[int, float]] = []  # per system: staff -> top line y
        self.lefts: list[list[float]] = []  # per system: each bar's left barline x
        self.clef = {1: clefs[0], 2: clefs[1]}
        y = 80.0
        for k, bars in enumerate(systems):
            if vocal:
                self._staff(y, bars)
                self._barlines([y], [y + 4 * SP], bars)
                self.glyph("&", LEFT + 4, y + 3 * SP)
                y += 12 * SP
            tops = {1: y, 2: y + 12 * SP}
            self.tops.append(tops)
            for top in tops.values():
                self._staff(top, bars)
            self.lefts.append(self._barlines([tops[1]], [tops[2] + 4 * SP], bars))
            for staff, top in tops.items():
                treble = self.clef[staff] == "treble"
                self.glyph("&" if treble else "?", LEFT + 4, top + (3 if treble else 1) * SP)
                order = [0, 3, -1, 2, 5, 1, 4] if key > 0 else [4, 1, 5, 2, 6, 3, 7]
                for i in range(abs(key)):
                    self.glyph("#" if key > 0 else "b", LEFT + 18 + 5 * i, top + (order[i] + 2 * (not treble)) * SP / 2)
                if time and k == 0:
                    self.glyph(str(time[0]), LEFT + 36, top + SP)
                    self.glyph(str(time[1]), LEFT + 36, top + 3 * SP)
            y += 30 * SP

    def _staff(self, top: float, bars: int) -> None:
        right = LEFT + HEADER + bars * BAR
        for i in range(5):
            self.line(LEFT, top + i * SP, right, top + i * SP)

    def _barlines(self, tops: list[float], bottoms: list[float], bars: int) -> list[float]:
        xs = [LEFT] + [LEFT + HEADER + i * BAR for i in range(1, bars + 1)]
        for x in xs:
            self.line(x, tops[0], x, bottoms[-1])
        return [LEFT] + xs[1:-1]

    # ------------------------------------------------------------ drawing

    def line(self, x0: float, y0: float, x1: float, y1: float, width: float = 0.5) -> None:
        self.shape.draw_line((x0, y0), (x1, y1))
        self.shape.finish(color=(0, 0, 0), width=width)

    def glyph(self, ch: str, x: float, y: float, size: float = SIZE) -> None:
        self.page.insert_text((x, y), ch, fontsize=size, fontname="music", fontfile=str(FONT))

    def text(self, s: str, x: float, y: float, italic: bool = True) -> None:
        self.page.insert_text((x, y), s, fontsize=8, fontname="tiit" if italic else "tiro")

    def x(self, bar: int, beat: float, system: int = 0) -> float:
        """Where a note on `beat` (1-based) of `bar` (1-based within the system) goes."""
        start = self.lefts[system][bar - 1] + (HEADER if bar == 1 else 0)
        return start + 10 + (beat - 1) * QUARTER

    def y(self, name: str, staff: int, system: int = 0, clef: str | None = None) -> float:
        steps = TOP_LINE[clef or self.clef[staff]] - diatonic(name)
        return self.tops[system][staff] + steps * SP / 2

    def chord(self, bar: int, beat: float, staff: int, names: list[str], value: str = "q", up: bool = True,
              dots: int = 0, system: int = 0, clef: str | None = None, accidentals: dict[str, str] | None = None,
              small: bool = False, beamed: bool = False) -> list[Head]:
        """Heads (with a stem unless whole), a flag for an unbeamed "8" or "16", dots and accidentals. Values: w h q
        8 16 (beamed notes get their beams from beam())."""
        x = self.x(bar, beat, system)
        size = SIZE * (0.65 if small else 1)
        char = {"w": "w", "h": "˙"}.get(value, "œ")
        w = _FONT.text_length(char, size)
        heads = []
        for n in names:
            y = self.y(n, staff, system, clef)
            self.glyph(char, x, y, size)
            heads.append(Head(x, y, staff))
            if accidentals and n in accidentals:
                self.glyph(accidentals[n], x - 1.4 * SP, y, size)
            for d in range(dots):
                on_line = round((y - self.tops[system][staff]) / (SP / 2)) % 2 == 0
                self.glyph(".", x + w + 0.6 * SP + d * SP, y - (SP / 2 if on_line else 0), size)
        if value == "w":
            return heads
        ys = [h.y for h in heads]
        stem_x = x + w if up else x
        tip = min(ys) - 3.5 * SP if up else max(ys) + 3.5 * SP
        self.line(stem_x, max(ys) if up else min(ys), stem_x, tip)
        if value in ("8", "16") and not beamed:
            self.glyph({("8", True): "J", ("8", False): "j", ("16", True): "R", ("16", False): "r"}[(value, up)],
                       stem_x, tip, size)
        for h in heads:
            h.stem = (stem_x, tip)  # type: ignore[attr-defined]
        return heads

    def beam(self, groups: list[list[Head]], count: int = 1) -> None:
        """Beams joining the stems of these chords (each a chord's heads), at their stem tips."""
        stems = [g[0].stem for g in groups]  # type: ignore[attr-defined]
        (x0, y0), (x1, y1) = stems[0], stems[-1]
        up = y0 < min(h.y for h in groups[0])
        for k in range(count):
            dy = k * 0.75 * SP * (1 if up else -1)
            pts = [(x0, y0 + dy), (x1, y1 + dy), (x1, y1 + dy + (0.5 * SP if up else -0.5 * SP)),
                   (x0, y0 + dy + (0.5 * SP if up else -0.5 * SP)), (x0, y0 + dy)]
            self.shape.draw_polyline(pts)
            self.shape.finish(fill=(0, 0, 0), color=None, closePath=True)

    def rest(self, bar: int, beat: float, staff: int, value: str, system: int = 0, dots: int = 0) -> None:
        x = self.x(bar, beat, system)
        y = self.tops[system][staff] + 2 * SP
        self.glyph({"w": "∑", "h": "Ó", "q": "Œ", "8": "‰", "16": "≈"}[value], x, y)
        for d in range(dots):
            self.glyph(".", x + _FONT.text_length("Œ", SIZE) + 0.6 * SP + d * SP, y - SP / 2)

    def tie(self, a: Head, b: Head) -> None:
        x0, x1, y = a.x + _W + 1, b.x - 1, a.y + 0.8 * SP
        self.shape.draw_bezier((x0, y), (x0 + 4, y + 3), (x1 - 4, y + 3), (x1, y))
        self.shape.finish(fill=(0, 0, 0), color=(0, 0, 0), width=0.3)

    def ledger(self, x: float, y: float) -> None:
        self.line(x - 2, y, x + _W + 2, y, 0.8)

    def clef_change(self, bar: int, beat: float, staff: int, clef: str, system: int = 0) -> None:
        self.glyph("&" if clef == "treble" else "?", self.x(bar, beat, system) - 9,
                   self.tops[system][staff] + (3 if clef == "treble" else 1) * SP, SIZE * 0.7)
        self.clef[staff] = clef

    def octave_line(self, bar: int, beat0: float, beat1: float, staff: int = 1, system: int = 0,
                    down: bool = False, glyphs: bool = False) -> None:
        """An 8va line over the staff (8vb under it), its sign as text or, as Sibelius's Opus Special draws it, as
        an "8" glyph (“) and a "va" (”) or "vb" (‘) glyph."""
        x0, x1 = self.x(bar, beat0, system), self.x(bar, beat1, system) + _W
        y = self.tops[system][staff] + (9 * SP if down else -5 * SP)
        if glyphs:
            self.glyph("“", x0, y + 2)
            self.glyph("‘" if down else "”", x0 + 5, y)
        else:
            self.text("8vb" if down else "8va", x0, y + 2)
        x = x0 + 14
        while x < x1:
            self.line(x, y, min(x + 3, x1), y)
            x += 5

    def save(self, path: Path) -> Path:
        self.shape.commit()
        self.doc.save(str(path))
        return path
