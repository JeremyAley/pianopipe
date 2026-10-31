"""Repairing the piano part of an Audiveris MusicXML export with the geometry in its .omr project.

Audiveris reads pitches well but often gets rhythm and voices wrong on piano scores: a missing beam turns sixteenths
into quarters, notes land in the wrong voice, chords are left out of every voice (and so out of the export), and a
measure whose voices don't add up shifts everything after it. This module re-times such measures from what is on
the page:

- chords in the same column (across both staves) sound together;
- each chord's duration comes from its own symbols: head, beams on its stem (and beams found in the page image
  that Audiveris didn't keep), flags, dots, tuplets;
- a sweep along the columns starts each one when the earliest note still sounding ends;
- when that doesn't fill the bar, a few readings are tried (a missed beam, dot or triplet, stray rests left out,
  timing from both ends around a gap), and the one whose horizontal spacing fits best wins.

Every bar is reported: "ok" (left as Audiveris read it), "repaired", "partial" (re-timed, but notes are probably
missing: for review) or "review" (left as read). It also undoes structural mistakes: a system Audiveris split in
two (the vocal staff in one, the piano in the next, which adds empty piano measures), a missed time signature, and
pitches its export got wrong from a misread clef. Exports that don't line up with the project are put right first:
movements Audiveris split off (an indented coda system becomes <song>.mvt2.mxl) are joined, and a measure its
exporter dropped (a gap in the measure numbers) gets an empty one, rebuilt from the page and flagged for review. The
design and the results on a real songbook are in docs/catalog-expansion.md §3.2.

Standard library only.
"""

from __future__ import annotations

import copy
import glob
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from . import omr
from .musicxml import STEPS, SUFFIXES, read_root

COLUMN_TOLERANCE = 0.6  # chords whose anchors are closer than this many staff spaces share a column
COLLISION_TOLERANCE = 1.4  # ... and so do two voices' chords this close, printed side by side to avoid a collision
SECOND_TOLERANCE = 3.0  # ... and two voices a unison or second apart on one staff, moved a whole head aside
SPACING_EXPONENT = 0.6  # space grows about 1.5 times for each doubling of duration (2 ** 0.6 = 1.52)
SPACING_OUTLIER = math.log(3)  # a pair of columns 3 times too far apart (or close) for its time: something's missing
CHANGE_COST = 0.1  # charge per assumption (a rest left out, a beam added), in units of mean log spacing error
GAP_COST = 0.3  # charge for timing a bar from both ends
CUT_COST = 0.08  # charge per note cut short where the next one starts
EARLY_END_COST = 0.3  # charge per staff whose notes start after the barline or stop before the next one
MATCH_TOLERANCE = 0.5  # staff spaces between a MusicXML note's default-x and its chord in the .omr
MAX_SKIPPED = 4  # a gap in the measure numbers wider than this isn't measures the exporter dropped
FILLED = "pianopipe-filled"  # (while repairing) a measure put in for one the export skipped: its number there
TYPES = [(Fraction(4), "whole"), (Fraction(2), "half"), (Fraction(1), "quarter"), (Fraction(1, 2), "eighth"),
         (Fraction(1, 4), "16th"), (Fraction(1, 8), "32nd"), (Fraction(1, 16), "64th")]


@dataclass
class BarResult:
    number: int  # measure number after the structural fix (1-based)
    status: str  # "ok" (left as Audiveris read it), "repaired", "partial" (re-timed around missing notes) or "review"
    reasons: list[str] = field(default_factory=list)


@dataclass
class Report:
    bars: list[BarResult] = field(default_factory=list)
    removed: list[int] = field(default_factory=list)  # measure numbers (before the fix) dropped as split-system copies
    warnings: list[str] = field(default_factory=list)
    movements: int = 1  # export files joined

    @property
    def needs_review(self) -> list[int]:
        """Bars a person (or a vision check) should look at: left as read, or re-timed around missing notes."""
        return [b.number for b in self.bars if b.status in ("review", "partial")]

    @property
    def repaired(self) -> list[int]:
        return [b.number for b in self.bars if b.status == "repaired"]

    def to_json(self) -> dict:
        """For a library's report.json: the song is needs_review when any bar is."""
        return {
            "status": "needs_review" if self.needs_review or self.warnings else "ok",
            "needs_review": self.needs_review,
            "repaired": self.repaired,
            "removed": self.removed,
            "movements": self.movements,
            "bars": {str(b.number): {"status": b.status, "reasons": b.reasons} for b in self.bars if b.status != "ok"},
            "warnings": self.warnings,
        }

    def summary(self) -> str:
        n = Counter(b.status for b in self.bars)
        lines = [f"{len(self.bars)} bars: {n['ok']} fine as read, {n['repaired']} repaired, {n['partial']} re-timed "
                 f"around missing notes, {n['review']} left as read"]
        if self.movements > 1:
            lines.append(f"joined {self.movements} movement exports")
        if self.removed:
            lines.append(f"removed {len(self.removed)} empty piano bars from a split system: {self.removed}")
        for b in self.bars:
            if b.status != "ok":
                lines.append(f"  bar {b.number} {b.status}: {'; '.join(b.reasons)}")
        lines.extend(self.warnings)
        return "\n".join(lines)


@dataclass
class Event:
    """A chord or rest with its place in time, ready to write."""
    staff: int
    onset: Fraction
    duration: Fraction
    pitches: list[tuple[str, int, int]]  # (step, alter, octave); empty for a rest
    ties: list[set[str]]  # per pitch: {"start", "stop"} copied from the export
    stem_up: bool | None = None


# ---------------------------------------------------------------- MusicXML side

def _piano_part(root: ET.Element) -> ET.Element | None:
    best = None
    for part in root.findall("part"):
        staves = max((int(s.text) for s in part.iter("staves") if s.text and s.text.strip().isdigit()), default=1)
        if staves >= 2:
            best = part
    return best


def _xml_notes(m_el: ET.Element, divisions: int) -> tuple[dict[str, Fraction], list[ET.Element]]:
    """Each voice's total duration (counting <forward>), and the measure's notes."""
    per_voice: dict[str, Fraction] = {}
    notes = []
    for el in m_el:
        if el.tag == "forward":
            v = el.findtext("voice", "1")
            per_voice[v] = per_voice.get(v, Fraction(0)) + Fraction(int(el.findtext("duration", "0")), divisions)
        elif el.tag == "note":
            notes.append(el)
            if el.find("chord") is not None or el.find("grace") is not None:
                continue
            v = el.findtext("voice", "1")
            per_voice[v] = per_voice.get(v, Fraction(0)) + Fraction(int(el.findtext("duration", "0")), divisions)
    return per_voice, notes


def _midi(note: ET.Element) -> int | None:
    p = note.find("pitch")
    if p is None:
        return None
    return 12 * (int(p.findtext("octave", "4")) + 1) + STEPS.get(p.findtext("step", "C"), 0) + round(float(p.findtext("alter", "0")))


# ---------------------------------------------------------------- the solver

@dataclass
class Solution:
    """Where each chord starts and how long it lasts, and how sure we are."""
    placed: list[tuple[omr.Chord, Fraction, Fraction]]  # (chord, onset, duration)
    complete: bool  # every column chained from the start to the end; False: timed from both ends around a gap
    changes: list[str] = field(default_factory=list)  # what had to be assumed, for the report


def _layer(c: omr.Chord) -> tuple[int, bool | None]:
    """A chord's layer on its staff: a voice that shares a staff with another goes the other way with its stems."""
    return c.staff, c.stem_up if c.heads else None


def columns(chords: list[omr.Chord], interline: float) -> list[list[omr.Chord]]:
    """Chords grouped by x. A chord joins the previous column when its anchor is within COLUMN_TOLERANCE staff
    spaces of the column's first chord, or within COLLISION_TOLERANCE when its layer isn't in the column yet (two
    voices on one note are printed side by side)."""
    cols: list[list[omr.Chord]] = []
    for c in sorted(chords, key=lambda c: c.anchor):
        if cols:
            gap = c.anchor - cols[-1][0].anchor
            if gap <= COLUMN_TOLERANCE * interline or (
                    gap <= COLLISION_TOLERANCE * interline and _layer(c) not in {_layer(o) for o in cols[-1]}) or (
                    gap <= SECOND_TOLERANCE * interline and any(_second(o, c) for o in cols[-1])):
                cols[-1].append(c)
                continue
        cols.append([c])
    return cols


def _second(a: omr.Chord, b: omr.Chord) -> bool:
    """Two chords of one staff with stems opposite ways and a unison or second between them: the engraver moves
    one aside by a head's width, though they sound together."""
    if a.staff != b.staff or not a.heads or not b.heads or a.stem_up is None or a.stem_up == b.stem_up:
        return False
    return any(abs(h.step - k.step) <= 1 for h in a.heads for k in b.heads)


def forward(cols: list[list[omr.Chord]], value: dict[int, Fraction], length: Fraction) -> tuple[list[Fraction], str]:
    """Onsets from the start: each column starts when the earliest chord still sounding ends.

    Returns the onsets of the columns it could place (all of them on success) and why it stopped ("" if it didn't).
    """
    onsets: list[Fraction] = []
    open_ends: list[Fraction] = []
    for k, col in enumerate(cols):
        if k == 0:
            t = Fraction(0)
        else:
            later = [e for e in open_ends if e > onsets[-1]]
            if not later:
                return onsets, f"nothing sounds into column {k + 1}"
            t = min(later)
        if t >= length:
            return onsets, f"column {k + 1} would start at the end of the bar"
        onsets.append(t)
        open_ends = [e for e in open_ends if e > t] + [t + value[c.id] for c in col]
    if max(open_ends, default=length) < length:
        return onsets, "the notes end before the bar does"
    return onsets, ""


def backward(cols: list[list[omr.Chord]], value: dict[int, Fraction], length: Fraction) -> list[Fraction] | None:
    """Onsets from the end, for columns after a gap: each column starts as late as it can while one of its chords
    still ends where a later column starts (or at the end of the bar)."""
    onsets: list[Fraction] = []
    starts = [length]
    for col in reversed(cols):
        limit = onsets[0] if onsets else length + 1
        cands = [s - value[c.id] for c in col for s in starts if Fraction(0) <= s - value[c.id] < limit]
        if not cands:
            return None
        t = max(cands)
        onsets.insert(0, t)
        starts.append(t)
    return onsets


def _durations(cols: list[list[omr.Chord]], onsets: list[Fraction], value: dict[int, Fraction],
               length: Fraction) -> tuple[list[tuple[omr.Chord, Fraction, Fraction]], int]:
    """Each chord's duration, cut where the next chord of its layer starts or where the bar ends (a duration read
    too long: a missed beam or flag). Returns the placed chords and how many were cut."""
    placed = [(c, t) for col, t in zip(cols, onsets) for c in col]
    out, cut = [], 0
    for c, t in placed:
        end = min([t + value[c.id], length] + [u for o, u in placed if u > t and _layer(o) == _layer(c)])
        if end < t + value[c.id]:
            cut += 1
        out.append((c, t, end - t))
    return out, cut


Mods = dict[int, tuple[int, int, Fraction | None]]  # chord id -> (extra beams, extra dots, tuplet)


def _hypotheses(chords: list[omr.Chord]) -> list[tuple[list[str], frozenset[int], Mods]]:
    """Readings to try: (what's assumed, chords left out, symbols added). Each adds at most one thing Audiveris
    missed (a beam on a beam group, a triplet on three equal beamed notes, a dot), with or without leaving out
    rests it had put in no voice."""
    stray = frozenset(c.id for c in chords if c.rest and c.voice is None)
    groups: dict[tuple[int, ...], list[omr.Chord]] = {}
    for c in sorted(chords, key=lambda c: c.anchor):
        if c.beam_ids:
            groups.setdefault(c.beam_ids, []).append(c)
    single: list[tuple[str, Mods]] = []
    for g in groups.values():
        single.append((f"added a beam Audiveris missed to {len(g)} beamed notes", {c.id: (1, 0, None) for c in g}))
    for g in groups.values():
        for size, name in ((3, "a triplet"), (6, "a sextuplet")):
            for k in range(len(g) - size + 1):
                run = g[k:k + size]
                if not any(c.tuplet for c in run) and len({c.value for c in run}) == 1:
                    single.append((f"read {size} beamed notes as {name}", {c.id: (0, 0, Fraction(2, 3)) for c in run}))
    for c in chords:
        if not c.dots and not c.measure_rest:
            single.append(("added a dot Audiveris missed", {c.id: (0, 1, None)}))
    out: list[tuple[list[str], frozenset[int], Mods]] = [([], frozenset(), {})]
    if stray:
        out.append(([f"left out {len(stray)} rests that Audiveris had put in no voice"], stray, {}))
    for label, drop, _ in list(out):
        for what, mods in single:
            out.append((label + [what], drop, mods))
    return out


def _unfilled(placed: list[tuple[omr.Chord, Fraction, Fraction]], length: Fraction) -> list[int]:
    """Staves whose notes start after the barline or stop before the next one: something is missing."""
    return sorted(s for s in {c.staff for c, _, _ in placed}
                  if max(t + d for c, t, d in placed if c.staff == s) < length
                  or min(t for c, t, _ in placed if c.staff == s) > 0)


def spacing(cols: list[list[omr.Chord]], times: list[Fraction], right: float, length: Fraction) -> tuple[float, float, int]:
    """How well the columns' x positions fit their times: (mean, worst, index of the worst pair).

    Engravers give a note that lasts twice as long about 1.5 times the space, so dx / dt**SPACING_EXPONENT is about
    constant along a bar; each pair of neighbouring columns (and the last one with the barline) is compared with the
    bar's median in log terms.
    """
    xs = [sum(c.anchor for c in col) / len(col) for col in cols] + [right]
    ts = list(times) + [length]
    logs = []
    for k in range(len(xs) - 1):
        dx, dt = xs[k + 1] - xs[k], ts[k + 1] - ts[k]
        if dx <= 0 or dt <= 0:
            return math.inf, math.inf, k
        logs.append(math.log(dx) - SPACING_EXPONENT * math.log(dt))
    if len(logs) < 2:
        return 0.0, 0.0, 0
    mid = sorted(logs)[len(logs) // 2]
    devs = [abs(v - mid) for v in logs]
    worst = max(range(len(devs)), key=devs.__getitem__)
    return sum(devs) / len(devs), devs[worst], worst


def solve(m: omr.Measure, length: Fraction) -> tuple[Solution | None, str]:
    """Time every chord of the measure, or say why it can't be done.

    Every reading in _hypotheses is tried; a reading that chains from the start to the end of the bar is complete,
    one that only works timed from both ends around a gap is partial. The one whose spacing fits best wins, with
    a small charge for each assumption and a larger one for a gap.
    """
    chords = [c for c in m.chords if not c.measure_rest]
    if not chords:
        return Solution([], True), ""
    first_why = ""
    best: tuple[float, Solution] | None = None
    seen = sum(1 for c in chords if c.seen_beams)
    found = [f"found beams on {seen} notes in the page image that Audiveris missed"] if seen else []
    for changes, drop, mods in _hypotheses(chords):
        kept = [c for c in chords if c.id not in drop]
        value = {c.id: c.value_with(*mods.get(c.id, (0, 0, None))) for c in kept}
        cols = columns(kept, m.interline)
        onsets, why = forward(cols, value, length)
        candidates = []
        if not why:
            candidates.append((onsets, None))
            placed, _ = _durations(cols, onsets, value, length)
            if not _unfilled(placed, length):
                onsets = []  # it chains and fills both staves: no need to try it from the end
        else:
            first_why = first_why or why
        if onsets or why:
            # time the columns after the failure (or a gap) from the end of the bar, meeting the forward times at any
            # column
            for split in range(min(len(onsets), len(cols) - 1), -1, -1):
                back = backward(cols[split:], value, length)
                if back and (split == 0 or back[0] > onsets[split - 1]):
                    candidates.append((onsets[:split] + back, split))
        for times, split in candidates:
            mean, worst, at = spacing(cols, times, m.right, length)
            if math.isinf(mean):
                continue
            placed, cut = _durations(cols, times, value, length)
            early = _unfilled(placed, length)
            score = (mean + CHANGE_COST * len(changes) + CUT_COST * cut + EARLY_END_COST * len(early)
                     + (GAP_COST if split is not None else 0))
            if best is not None and score >= best[0]:
                continue
            notes = found + changes
            if cut:
                notes.append(f"shortened {cut} notes that ran into the next one")
            complete = split is None and not early
            if split is not None:
                start = times[split - 1] if split else Fraction(0)
                notes.append(f"timed from both ends: a note or rest Audiveris missed is likely between beats "
                             f"{_beat(start)} and {_beat(times[split])}")
            elif early:
                notes.append(f"the {' and '.join('right hand' if s == 1 else 'left hand' for s in sorted(early))} "
                             f"doesn't fill the bar")
            elif worst > SPACING_OUTLIER and at + 1 < len(times):
                complete = False
                notes.append(f"the spacing doesn't fit the rhythm between beats {_beat(times[at])} and {_beat(times[at + 1])}")
            best = (score, Solution(placed, complete, notes))
    if best is None:
        return None, first_why or "no reading fits the bar"
    return best[1], ""


def _beat(t: Fraction) -> str:
    b = t + 1
    return str(int(b)) if b.denominator == 1 else f"{float(b):g}"


# ---------------------------------------------------------------- writing

def _type(duration: Fraction) -> tuple[str | None, int]:
    for dots in range(3):
        for value, name in TYPES:
            if value * (2 - Fraction(1, 2 ** dots)) == duration:
                return name, dots
    return None, 0


def _events(placed: list[tuple[omr.Chord, Fraction, Fraction]], xml_notes: list[ET.Element], m: omr.Measure) -> list[Event]:
    """Events with pitches from the .omr, and ties from the .omr and the export's matching notes."""
    tol = MATCH_TOLERANCE * m.interline
    xml_x = []
    for n in xml_notes:
        if n.get("default-x") is not None and n.find("pitch") is not None:
            xml_x.append((m.left + float(n.get("default-x")) * m.interline / 10, int(n.findtext("staff", "1")), _midi(n), n))
    out = []
    for c, t, d in placed:
        pitches, ties = [], []
        for h in sorted(c.heads, key=lambda h: h.midi or 0):
            if h.diatonic is None:
                continue
            octave, step = divmod(h.diatonic, 7)
            pitches.append(("CDEFGAB"[step], h.alter, octave))
            match = next((n for x, s, p, n in xml_x if s == c.staff and p == h.midi and abs(x - c.x) <= tol + h.box.w), None)
            ties.append(set(h.ties) | ({tie.get("type") for tie in match.findall("tie")} if match is not None else set()))
        if c.heads and not pitches:
            continue
        out.append(Event(c.staff, t, d, pitches, ties, c.stem_up))
    return out


def _voices(events: list[Event]) -> list[list[Event]]:
    """Pack one staff's events into voices: each voice a sequence that never overlaps itself."""
    voices: list[list[Event]] = []
    for e in sorted(events, key=lambda e: (e.onset, not e.stem_up, -max((p[2] * 7 for p in e.pitches), default=0))):
        same = next((v for v in voices if v[-1].onset == e.onset and v[-1].duration == e.duration and v[-1].pitches and e.pitches
                     and v[-1].stem_up == e.stem_up), None)
        if same is not None:  # one chord: same start and length, same stem direction
            last = same[-1]
            last.pitches = last.pitches + e.pitches
            last.ties = last.ties + e.ties
            continue
        free = next((v for v in voices if v[-1].onset + v[-1].duration <= e.onset), None)
        if free is None:
            voices.append([e])
        else:
            free.append(e)
    return voices


def _note_el(e: Event, k: int, voice: int, divisions: int) -> ET.Element:
    n = ET.Element("note")
    if k > 0:
        ET.SubElement(n, "chord")
    if e.pitches:
        step, alter, octave = e.pitches[k]
        p = ET.SubElement(n, "pitch")
        ET.SubElement(p, "step").text = step
        if alter:
            ET.SubElement(p, "alter").text = str(alter)
        ET.SubElement(p, "octave").text = str(octave)
    else:
        ET.SubElement(n, "rest")
    ET.SubElement(n, "duration").text = str(int(e.duration * divisions))
    for kind in sorted(e.ties[k]) if e.pitches else []:
        ET.SubElement(n, "tie", type=kind)
    ET.SubElement(n, "voice").text = str(voice)
    name, dots = _type(e.duration)
    if name:
        ET.SubElement(n, "type").text = name
        for _ in range(dots):
            ET.SubElement(n, "dot")
    if e.stem_up is not None and e.pitches:
        ET.SubElement(n, "stem").text = "up" if e.stem_up else "down"
    ET.SubElement(n, "staff").text = str(e.staff)
    if e.pitches and any(e.ties[k]):
        notations = ET.SubElement(n, "notations")
        for kind in sorted(e.ties[k]):
            ET.SubElement(notations, "tied", type=kind)
    return n


def _rewrite(m_el: ET.Element, events: list[Event], length: Fraction, divisions: int) -> None:
    """Replace the measure's notes, backups and forwards with the events, one voice after another."""
    keep = [el for el in m_el if el.tag not in ("note", "backup", "forward")]
    for el in list(m_el):
        m_el.remove(el)
    for el in keep:
        if el.tag != "barline" or el.get("location") == "left":
            m_el.append(el)
    first = True
    for staff in (1, 2):
        for k, voice in enumerate(_voices([e for e in events if e.staff == staff])):
            number = (0 if staff == 1 else 4) + min(k, 3) + 1
            pos = Fraction(0)
            for e in voice:
                if e.onset > pos:
                    fw = ET.SubElement(m_el, "forward")
                    ET.SubElement(fw, "duration").text = str(int((e.onset - pos) * divisions))
                    ET.SubElement(fw, "voice").text = str(number)
                    ET.SubElement(fw, "staff").text = str(staff)
                for i in range(max(1, len(e.pitches))):
                    m_el.append(_note_el(e, i, number, divisions))
                pos = e.onset + e.duration
            if pos < length:  # every voice fills the bar, as notation programs expect
                fw = ET.SubElement(m_el, "forward")
                ET.SubElement(fw, "duration").text = str(int((length - pos) * divisions))
                ET.SubElement(fw, "voice").text = str(number)
                ET.SubElement(fw, "staff").text = str(staff)
                pos = length
            first = False
            bk = ET.SubElement(m_el, "backup")
            ET.SubElement(bk, "duration").text = str(int(pos * divisions))
    if first:  # no notes at all: keep the measure's length
        fw = ET.SubElement(m_el, "forward")
        ET.SubElement(fw, "duration").text = str(int(length * divisions))
    for el in keep:
        if el.tag == "barline" and el.get("location") != "left":
            m_el.append(el)


# ---------------------------------------------------------------- structure

def _number(m_el: ET.Element) -> int | None:
    match = re.match(r"\d+", m_el.get("number", ""))
    return int(match.group()) if match else None


def fill_gaps(root: ET.Element) -> list[int]:
    """Put an empty measure wherever a part's measure numbers skip some (51, 52, 53, 55): Audiveris's exporter drops a
    measure it fails on ("Error visiting Measure{#9} in {Page#4.1}" in its log), which puts every later measure off
    the project's. The new measure is marked FILLED for repair_tree. Returns the skipped numbers."""
    skipped: set[int] = set()
    for part in root.findall("part"):
        prev = None
        for m_el in part.findall("measure"):
            num = _number(m_el)
            if num is None:
                continue
            if prev is not None and 1 < num - prev <= MAX_SKIPPED + 1:
                at = list(part).index(m_el)
                for k, missing in enumerate(range(prev + 1, num)):
                    part.insert(at + k, ET.Element("measure", {"number": str(missing), FILLED: str(missing)}))
                    skipped.add(missing)
            prev = num
    return sorted(skipped)


def _part_name(root: ET.Element, part: ET.Element) -> str:
    sp = next((s for s in root.iter("score-part") if s.get("id") == part.get("id")), None)
    return (sp.findtext("part-name") or "").strip() if sp is not None else ""


def join(roots: list[ET.Element], warnings: list[str]) -> ET.Element:
    """One score from Audiveris's movement exports (<song>.mvt1.mxl, .mvt2.mxl, ...: it makes an indented system,
    such as a coda, a movement of its own). Each part's measures follow on, matched by part name (else by place); the
    first measure of each later movement starts a new system, where the importer looks for a coda the export has no
    sign for (musicxml._guess_coda). A part missing from a movement gets empty measures there."""
    root = roots[0]
    parts = root.findall("part")
    for k, other in enumerate(roots[1:], 2):
        others = other.findall("part")
        names = [_part_name(other, p) for p in others]
        length = max((len(p.findall("measure")) for p in others), default=0)
        used: set[int] = set()
        for j, part in enumerate(parts):
            name = _part_name(root, part)
            match = next((i for i, n in enumerate(names) if n == name and i not in used), None)
            if match is None and j < len(others) and j not in used and len(others) == len(parts):
                match = j
            if match is None:
                measures = [ET.Element("measure") for _ in range(length)]
                warnings.append(f"movement {k} has no part {name or part.get('id')!r}: empty measures put in")
            else:
                used.add(match)
                measures = others[match].findall("measure")
            for i, m_el in enumerate(measures):
                if i == 0:
                    pr = m_el.find("print")
                    if pr is None:
                        pr = ET.Element("print")
                        m_el.insert(0, pr)
                    if pr.get("new-page") != "yes":
                        pr.set("new-system", "yes")
                part.append(m_el)
        for i, p in enumerate(others):
            if i not in used:
                warnings.append(f"movement {k}'s part {names[i] or p.get('id')!r} isn't in the first movement: left out")
    return root


def export_paths(specs: list[str], omr_path: str | Path) -> list[Path]:
    """The export files a command line names: files, globs, or a folder (the exports in it named after the project,
    <song>.mxl or its movements <song>.mvt1.mxl, <song>.mvt2.mxl), movements in order."""
    def by_movement(paths) -> list[Path]:
        def key(p: Path):
            m = re.search(r"\.mvt(\d+)$", p.stem)
            return (int(m.group(1)) if m else 0, p.name)
        return sorted(paths, key=key)

    out: list[Path] = []
    stem = Path(omr_path).stem
    for spec in specs:
        p = Path(spec)
        if p.is_dir():
            found = [f for f in p.iterdir() if f.suffix.lower() in SUFFIXES
                     and (f.stem == stem or re.fullmatch(rf"{re.escape(stem)}\.mvt\d+", f.stem))]
            found = [f for f in found if f.suffix.lower() == ".mxl"] or found  # not a repaired copy written beside them
            movements = [f for f in found if f.stem != stem]
            out.extend(by_movement(movements or found))
        elif any(c in spec for c in "*?["):
            out.extend(by_movement(Path(f) for f in glob.glob(spec)))
        else:
            out.append(p)
    if not out:
        raise FileNotFoundError(f"no exports found in {', '.join(specs)}")
    return out


def _split_systems(score: omr.Score) -> list[tuple[int, int]]:
    """(first empty measure index, length) for each run of measures where the piano is absent and the next run of
    the same length has the piano alone: Audiveris split one system into a vocal one and a piano one."""
    runs = []
    ms = score.measures
    i = 0
    while i < len(ms):
        if ms[i].absent:
            j = i
            while j < len(ms) and ms[j].absent:
                j += 1
            k = j - i
            if j + k <= len(ms) and all(m.alone and not m.absent for m in ms[j:j + k]):
                runs.append((i, k))
            i = j
        else:
            i += 1
    return runs


def _rest_only(m_el: ET.Element) -> bool:
    return all(n.find("rest") is not None for n in m_el.findall("note"))


def _merge_split(root: ET.Element, piano: ET.Element, score: omr.Score, report: Report) -> None:
    parts = root.findall("part")
    for start, k in reversed(_split_systems(score)):
        others_ok = all(_rest_only(p.findall("measure")[start + k + j]) for p in parts if p is not piano for j in range(k))
        piano_empty = all(_rest_only(piano.findall("measure")[start + j]) for j in range(k))
        if not (others_ok and piano_empty):
            report.warnings.append(f"measures {start + 1}-{start + k} look like a split system but weren't merged")
            continue
        pm = piano.findall("measure")
        for j in range(k):
            empty, full = pm[start + j], pm[start + k + j]
            if full.get(FILLED):
                empty.set(FILLED, full.get(FILLED))
            for el in list(empty):
                if el.tag in ("note", "backup", "forward"):
                    empty.remove(el)
            for el in full:
                if el.tag in ("note", "backup", "forward", "direction", "harmony"):
                    empty.append(copy.deepcopy(el))
        for p in parts:
            pms = p.findall("measure")
            for j in range(k):
                p.remove(pms[start + k + j])
        report.removed.extend(range(start + k + 1, start + 2 * k + 1))
        score.measures[start:start + k] = score.measures[start + k:start + 2 * k]
        del score.measures[start + k:start + 2 * k]
    for p in parts:
        for n, m_el in enumerate(p.findall("measure"), 1):
            m_el.set("number", str(n))
    for n, m in enumerate(score.measures, 1):
        m.number = n


# ---------------------------------------------------------------- entry point

def repair_tree(root: ET.Element, score: omr.Score) -> Report:
    """Repair the piano part of a parsed MusicXML export in place, using the project's geometry. Measures fill_gaps
    put in are rebuilt from the page and flagged for review."""
    report = Report()
    try:
        _repair(root, score, report)
    finally:
        for m_el in root.iter("measure"):
            m_el.attrib.pop(FILLED, None)
    return report


def _repair(root: ET.Element, score: omr.Score, report: Report) -> None:
    piano = _piano_part(root)
    if piano is None:
        report.warnings.append("no two-staff part to repair")
        return
    if len(piano.findall("measure")) != len(score.measures):
        report.warnings.append(f"the export has {len(piano.findall('measure'))} measures and the project "
                               f"{len(score.measures)}; nothing repaired (if Audiveris wrote movements, "
                               f"<song>.mvt1.mxl, .mvt2.mxl, pass them all)")
        report.bars = [BarResult(n + 1, "review", ["measures don't line up with the project"])
                       for n in range(len(piano.findall("measure")))]
        return
    _merge_split(root, piano, score, report)
    divisions = 1
    length = Fraction(4)
    meters = {Fraction(4 * int(t.findtext("beats")), int(t.findtext("beat-type", "4")))
              for t in piano.iter("time") if (t.findtext("beats") or "").isdigit()}
    for m_el, m in zip(piano.findall("measure"), score.measures):
        attrs = m_el.find("attributes")
        if attrs is not None:
            if attrs.findtext("divisions"):
                divisions = int(attrs.findtext("divisions"))
            t = attrs.find("time")
            if t is not None and t.findtext("beats"):
                length = Fraction(4 * int(t.findtext("beats")), int(t.findtext("beat-type", "4")))
        per_voice, notes = _xml_notes(m_el, divisions)
        dropped = [c for c in m.chords if c.voice is None and not c.measure_rest]
        adds_up = all(v == length for v in per_voice.values())
        exported = Counter((int(n.findtext("staff", "1")), _midi(n)) for n in notes if n.find("pitch") is not None)
        read = Counter((c.staff, h.midi) for c in m.chords if c.voice is not None for h in c.heads)
        reasons = []
        filled = m_el.get(FILLED)
        if filled:
            reasons.append(f"Audiveris's export dropped this measure (it had no measure {filled})")
        if not adds_up:
            reasons.append("voices didn't add up")
        if dropped:
            reasons.append(f"{len(dropped)} chords were left out of every voice")
        if m.abnormal:
            reasons.append("Audiveris marked it abnormal")
        if exported != read:
            reasons.append("the export's pitches differ from the page's clefs")
        if m.unread:
            reasons.append(f"{len(m.unread)} marks on the staves look like note heads Audiveris didn't read")
        if not reasons:
            report.bars.append(BarResult(m.number, "ok"))
            continue
        solution, why = solve(m, length)
        if solution is None or not solution.complete:
            # Audiveris misses time signatures: a bar that only fits another meter of the piece is in that meter
            for meter in sorted(meters - {length}):
                other, _ = solve(m, meter)
                if other is not None and other.complete:
                    beats, beat_type = _meter(meter)
                    _set_time(m_el, beats, beat_type)
                    other.changes.insert(0, f"it's in {beats}/{beat_type}: Audiveris missed the time signature")
                    length, solution = meter, other
                    break
        if solution is None:
            report.bars.append(BarResult(m.number, "review", reasons + [f"{'a bar of rests put in' if filled else 'left as read'}: {why}"]))
            if filled:  # so later bars keep their place
                _rewrite(m_el, [], length, divisions)
            continue
        status = "repaired" if solution.complete and not m.unread and not filled else "partial"
        report.bars.append(BarResult(m.number, status, reasons + solution.changes))
        placed = solution.placed
        events = _events(placed, notes, m)
        grid = math.lcm(divisions, *(e.onset.denominator for e in events), *(e.duration.denominator for e in events),
                        length.denominator)
        if grid != divisions:
            _set_divisions(piano, m_el, grid, divisions)
            divisions_here = grid
        else:
            divisions_here = divisions
        _rewrite(m_el, events, length, divisions_here)
    return report


def _meter(length: Fraction) -> tuple[int, int]:
    """(beats, beat type) for a bar length in quarter notes: 2 -> 2/4, 3/2 -> 3/8."""
    beat_type = 4
    while (length * beat_type / 4).denominator != 1:
        beat_type *= 2
    return int(length * beat_type / 4), beat_type


def _set_time(m_el: ET.Element, beats: int, beat_type: int) -> None:
    attrs = m_el.find("attributes")
    if attrs is None:
        attrs = ET.Element("attributes")
        m_el.insert(next((k for k, el in enumerate(m_el) if el.tag != "print"), len(m_el)), attrs)
    for old in attrs.findall("time"):
        attrs.remove(old)
    time = ET.Element("time")
    ET.SubElement(time, "beats").text = str(beats)
    ET.SubElement(time, "beat-type").text = str(beat_type)
    # schema order: divisions, key, time, staves, ...
    at = sum(1 for el in attrs if el.tag in ("footnote", "level", "divisions", "key"))
    attrs.insert(at, time)


def _set_divisions(part: ET.Element, m_el: ET.Element, new: int, old: int) -> None:
    """Give one measure finer divisions, and put the old ones back in the next measure."""
    attrs = m_el.find("attributes")
    if attrs is None:
        attrs = ET.Element("attributes")
        m_el.insert(0, attrs)
    d = attrs.find("divisions")
    if d is None:
        d = ET.Element("divisions")
        attrs.insert(0, d)
    d.text = str(new)
    ms = part.findall("measure")
    i = ms.index(m_el)
    if i + 1 < len(ms):
        nxt = ms[i + 1]
        na = nxt.find("attributes")
        if na is None:
            na = ET.Element("attributes")
            nxt.insert(next((k for k, el in enumerate(nxt) if el.tag not in ("print",)), 0), na)
        if na.find("divisions") is None:
            nd = ET.Element("divisions")
            nd.text = str(old)
            na.insert(0, nd)


def read_exports(exports: str | Path | list[str | Path], warnings: list[str]) -> ET.Element:
    """One export, or several movement exports joined (in the order given), with dropped measures put back."""
    paths = [exports] if isinstance(exports, (str, Path)) else list(exports)
    roots = [read_root(p) for p in paths]
    for path, root in zip(paths, roots):
        skipped = fill_gaps(root)
        if skipped:
            warnings.append(f"{Path(path).name} has no measure {', '.join(map(str, skipped))} (Audiveris's exporter "
                            f"dropped {'it' if len(skipped) == 1 else 'them'}): an empty measure put in, rebuilt from the page where it can be")
    return join(roots, warnings)


def repair(exports: str | Path | list[str | Path], omr_path: str | Path, out: str | Path) -> Report:
    """Write a repaired copy of an Audiveris export (.mxl or .xml), or of its movements joined, as uncompressed
    MusicXML."""
    warnings: list[str] = []
    root = read_exports(exports, warnings)
    report = repair_tree(root, omr.read(omr_path))
    report.warnings[:0] = warnings
    report.movements = 1 if isinstance(exports, (str, Path)) else len(exports)
    tree = ET.ElementTree(root)
    ET.indent(tree)
    Path(out).write_bytes(b'<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="utf-8"))
    return report
