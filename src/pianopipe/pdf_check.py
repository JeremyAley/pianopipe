"""Checking an OMR export against the publisher's vector PDF, bar by bar, and rewriting the bars that differ.

omr_repair fixes timing from Audiveris's own geometry, so it can't see what Audiveris got wrong in a way that still
adds up: bass notes filed on the treble staff and read in the treble clef (E2+B2 became C#4+G4), a half note it never
read, a leading rest it dropped from an inner voice. A publisher's PDF that holds real vector notation (Sheet
Music Direct's Finale, Sibelius and MuseScore exports) has every head at an exact place, so pdf_score reads each
bar exactly, and this module

- audits: compares each bar's notes per staff with the export: pitches, then onsets and durations when the PDF
  bar's timing reads cleanly;
- fills: rewrites a bar that differs from the PDF when its timing reads without any assumption: every column
  chains from the barline to the next, both staves are covered, no note runs past the barline, and the spacing
  fits. A bar the export lost is put back the same way. Everything else stays flagged. A bar omr-repair left for
  review that matches the PDF is cleared.

Bars are matched to the PDF in order when the counts agree (the export's system breaks are cross-checked), and by
aligning the two on their pitches when they don't. Every bar is reported:
"ok" (matches the PDF), "pitches-ok" (pitches match; the PDF's timing didn't read cleanly, so timing is unchecked),
"filled" (rewritten from the PDF), "differs" (left as it was: for review), "unmatched" (no PDF bar for it) or
"inserted" (a bar the export lost, put back from the PDF). Bar numbers are the output's.
The design and the results are in docs/catalog-expansion.md §3.2. Needs pymupdf (the [omr] extra).
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from . import omr, omr_repair, pdf_score
from .musicxml import STEPS, read_root

AMBIGUOUS_FIT = 0.08  # two timings whose spacing fits closer than this (mean log error) are a coin toss
NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


@dataclass
class Note:
    staff: int
    onset: Fraction
    duration: Fraction
    midi: int
    grace: bool = False


@dataclass
class BarCheck:
    number: int  # the export's measure, 1-based
    status: str  # ok, pitches-ok, filled, differs, unmatched, inserted
    reasons: list[str] = field(default_factory=list)
    pdf_bar: int | None = None  # the PDF's bar (1-based, print order)
    page: int | None = None
    flagged: bool = False  # omr-repair left it for review


@dataclass
class Report:
    bars: list[BarCheck] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing: list[int] = field(default_factory=list)  # PDF bars the file lacks and that couldn't be put back

    @property
    def needs_review(self) -> list[int]:
        """Bars still wrong or unchecked: differing, unmatched, or flagged by omr-repair with timing unchecked."""
        return [b.number for b in self.bars if b.status in ("differs", "unmatched")
                or (b.status == "pitches-ok" and b.flagged)]

    def count(self, status: str) -> int:
        return sum(1 for b in self.bars if b.status == status)

    @property
    def found(self) -> list[int]:
        """Bars the audit found wrong (before any fill), and bars the export lost."""
        return [b.number for b in self.bars if b.status in ("filled", "differs", "inserted")]

    def to_json(self) -> dict:
        return {
            "status": "needs_review" if self.needs_review or self.missing else "ok",
            "needs_review": self.needs_review,
            "missing_pdf_bars": self.missing,
            "found": self.found,
            "filled": [b.number for b in self.bars if b.status == "filled"],
            "inserted": [b.number for b in self.bars if b.status == "inserted"],
            "bars": {str(b.number): {"status": b.status, "pdf_bar": b.pdf_bar, "page": b.page, "reasons": b.reasons}
                     for b in self.bars if b.status != "ok"},
            "warnings": self.warnings,
        }

    def summary(self) -> str:
        lines = [f"{len(self.bars)} bars: {self.count('ok')} match the PDF, {self.count('pitches-ok')} match in "
                 f"pitch (timing unchecked), {len(self.found)} wrong: {self.count('filled')} rewritten from the PDF, "
                 f"{self.count('inserted')} lost bars put back, {self.count('differs')} left for review; "
                 f"{self.count('unmatched')} not matched to the PDF"]
        for b in self.bars:
            if b.status in ("filled", "differs", "unmatched", "inserted") or b.number in self.needs_review:
                lines.append(f"  bar {b.number} {b.status}"
                             + (f" (PDF bar {b.pdf_bar}, page {b.page})" if b.pdf_bar else "") + f": {'; '.join(b.reasons)}")
        lines.extend(self.warnings)
        return "\n".join(lines)


def name(midi: int) -> str:
    return NAMES[midi % 12] + str(midi // 12 - 1)


# ---------------------------------------------------------------- the two readings of a bar

def export_notes(m_el: ET.Element, divisions: int) -> list[Note]:
    """The pitched notes of a measure with their onsets, following MusicXML's cursor (backup, forward, chord)."""
    out, pos, last = [], Fraction(0), Fraction(0)
    for el in m_el:
        if el.tag == "backup":
            pos -= Fraction(int(el.findtext("duration", "0")), divisions)
        elif el.tag == "forward":
            pos += Fraction(int(el.findtext("duration", "0")), divisions)
        elif el.tag == "note":
            grace = el.find("grace") is not None
            dur = Fraction(0) if grace else Fraction(int(el.findtext("duration", "0")), divisions)
            if el.find("chord") is not None:
                onset = last
            else:
                onset = pos
                last = pos
                pos += dur
            p = el.find("pitch")
            if p is not None:
                midi = (12 * (int(p.findtext("octave", "4")) + 1) + STEPS.get(p.findtext("step", "C"), 0)
                        + round(float(p.findtext("alter", "0") or 0)))
                out.append(Note(int(el.findtext("staff", "1")), onset, dur, midi, grace))
    return out


@dataclass
class Reading:
    placed: list[tuple[omr.Chord, Fraction, Fraction]] | None  # None: the timing didn't read cleanly
    why: str  # why not
    notes: list[Note]  # pitched notes (onsets and durations only when placed)


def columns(chords: list[pdf_score.Chord], interline: float) -> list[list[pdf_score.Chord]]:
    """Chords grouped by x, as omr_repair.columns but from exact positions. A chord joins the column when its
    anchor is within 0.6 staff spaces of a chord's in it, or, on the same staff, when its
    heads (or rest) overlap a chord's in it, or touch them from another layer: two voices a second apart, one pushed
    a head's width aside. Notes one after the other never touch."""
    cols: list[list[pdf_score.Chord]] = []
    for c in sorted(chords, key=lambda c: c.extent[0]):
        if cols and _joins(c, cols[-1], interline):
            cols[-1].append(c)
        else:
            cols.append([c])
    return cols


def _joins(c: pdf_score.Chord, col: list[pdf_score.Chord], il: float) -> bool:
    if any(abs(p - q) <= 0.6 * il for o in col for p in _points(c) for q in _points(o)):
        return True
    a, b = c.extent
    for o in col:
        if o.staff != c.staff:
            continue
        overlap = min(b, o.extent[1]) - max(a, o.extent[0])
        if overlap > 0.3 * il or (overlap > -0.1 * il and c.heads and o.heads
                                  and omr_repair._layer(o) != omr_repair._layer(c)):
            return True
    return False


def _points(c: pdf_score.Chord) -> list[float]:
    """Where a chord stands: its anchor, or for a chord without a stem (whole notes, a second side by side) each
    head's centre."""
    if c.heads and c.stem_x is None:
        return [h.box.cx for h in c.heads]
    return [c.anchor]


def read_bar(bar: pdf_score.Bar, length: Fraction) -> Reading:
    """The PDF bar timed from its symbols alone: no beam, dot or triplet assumed, every column chained from the
    barline, every staff filling the bar, no note running past it, and spacing that fits. Notes keep their printed
    lengths: two voices on one side of their stems can overlap (a held note under a moving one)."""
    m = bar.measure
    chords = [c for c in m.chords if not c.measure_rest]
    graces = [Note(g.staff, Fraction(0), Fraction(0), midi, True) for g in bar.graces for midi in g.midis]
    heads = {id(h.glyph): (c.staff, h.midi) for c in chords for h in c.heads}  # a head two stems share counts once
    untimed = [Note(s, Fraction(-1 - k), Fraction(0), midi) for k, (s, midi) in enumerate(heads.values())] + graces
    if bar.problems:
        return Reading(None, "; ".join(bar.problems), untimed)
    if not chords:
        return Reading([], "", graces)
    value = {c.id: c.value for c in chords}
    base = columns(chords, m.interline)
    # two voices in unison or a second apart are printed side by side, as far apart as two beats can be: where the
    # geometry allows either, try both, and keep the bar only if exactly one timing reads
    pairs = [k for k in range(len(base) - 1) if _collision(base[k], base[k + 1], m.interline)]
    variants = [base] + [_merged(base, {k}) for k in pairs] + ([_merged(base, set(pairs))] if len(pairs) > 1 else [])
    found: dict[tuple, tuple] = {}  # timing -> (spacing fit, placed)
    first_why = ""
    for cols in variants:
        placed, why = _time(cols, value, bar, length)
        if placed is None:
            first_why = first_why or why
            continue
        onsets = sorted({t for _, t, _ in placed})
        fit = omr_repair.spacing([[c for c, t, _ in placed if t == u] for u in onsets], onsets, m.right, length)[0]
        found.setdefault(tuple(sorted((c.id, t) for c, t, _ in placed)), (fit, placed))
    if not found:
        return Reading(None, f"the PDF's timing doesn't read: {first_why}", untimed)
    ranked = sorted(found.values(), key=lambda f: f[0])
    if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < AMBIGUOUS_FIT:
        return Reading(None, "the PDF's timing reads two ways (voices printed side by side)", untimed)
    placed = ranked[0][1]
    notes = [Note(c.staff, t, d, h.midi) for c, t, d in placed for h in c.heads] + graces
    return Reading(placed, "", notes)


def _time(cols, value, bar: pdf_score.Bar, length: Fraction):
    """(placed chords, "") when the columns chain from barline to barline, no note runs past it, both staves are
    covered and the spacing fits; else (None, why)."""
    onsets, why = omr_repair.forward(cols, value, length)
    if why:
        return None, why
    placed = [(c, t, value[c.id]) for col, t in zip(cols, onsets) for c in col]
    if any(t + d > length for _, t, d in placed):
        return None, "a note runs past the barline"
    gap = _gap(placed, bar, length)
    if gap:
        return None, gap
    mean, worst, at = omr_repair.spacing(cols, onsets, bar.measure.right, length)
    if worst > omr_repair.SPACING_OUTLIER and at + 1 < len(onsets):
        return None, "the spacing doesn't fit the rhythm"
    return placed, ""


def _collision(a: list[pdf_score.Chord], b: list[pdf_score.Chord], il: float) -> bool:
    """Neighbouring columns that may be one: on one staff, two layers, a unison or second between them, close."""
    for c in a:
        for o in b:
            if (c.staff == o.staff and c.heads and o.heads and omr_repair._layer(c) != omr_repair._layer(o)
                    and o.extent[0] - c.extent[1] <= 1.0 * il
                    and any(abs(h.diatonic - k.diatonic) <= 1 for h in c.heads for k in o.heads)):
                return True
    return False


def _merged(cols: list[list], ks: set[int]) -> list[list]:
    out: list[list] = []
    for k, col in enumerate(cols):
        if k - 1 in ks:
            out[-1] = out[-1] + col
        else:
            out.append(list(col))
    return out


def _gap(placed, bar: pdf_score.Bar, length: Fraction) -> str:
    """Where a staff has nothing printed: neither its own notes and rests nor the other staff's notes under a beam
    they share (a cross-staff run). "" when both staves are covered from barline to barline."""
    for staff in (1, 2):
        if any(c.measure_rest and c.staff == staff for c in bar.measure.chords):
            continue
        beams = {b for c, _, _ in placed if c.staff == staff for b in c.beam_ids}
        spans = sorted((t, t + d) for c, t, d in placed if c.staff == staff or set(c.beam_ids) & beams)
        if not spans and not any(c.staff == staff for c in bar.measure.chords):
            continue  # nothing at all on this staff: an empty staff, as some arrangements print
        end = Fraction(0)
        for a, b in spans + [(length, length)]:
            if a > end:
                hand = "right hand" if staff == 1 else "left hand"
                return f"the {hand} has nothing printed between beats {_beat(end)} and {_beat(a)}"
            end = max(end, b)
    return ""


def _sounding(notes: list[Note], staff: int) -> dict[tuple[Fraction, int], Fraction]:
    """(onset, pitch) -> duration of a staff's notes; two voices on one key at once are one note, the longer."""
    out: dict[tuple[Fraction, int], Fraction] = {}
    for n in notes:
        if n.staff == staff and not n.grace:
            out[(n.onset, n.midi)] = max(out.get((n.onset, n.midi), Fraction(0)), n.duration)
    return out


def compare(pdf: Reading, export: list[Note]) -> list[str]:
    """How the export differs from the PDF bar, per staff: pitches first; onsets and durations when timed."""
    out = []
    for staff in (1, 2):
        hand = "right hand" if staff == 1 else "left hand"
        ps, xs = _sounding(pdf.notes, staff), _sounding(export, staff)
        p, x = Counter(m for _, m in ps), Counter(m for _, m in xs)
        if p != x:
            out.append(f"{hand}: the PDF has {_names(p - x) or 'nothing more'}, the file {_names(x - p) or 'nothing more'}")
            continue
        if pdf.placed is not None:
            if set(ps) != set(xs):
                moved = sorted(set(xs) - set(ps))
                out.append(f"{hand}: notes start at other times ({', '.join(f'{name(m)} at beat {_beat(t)}' for t, m in moved[:4])} "
                           f"in the file; the PDF has {', '.join(f'beat {_beat(t)}' for t, _ in sorted(set(ps) - set(xs))[:4])})")
                continue
            if ps != xs:
                diff = sorted(k for k in ps if ps[k] != xs[k])
                out.append(f"{hand}: other lengths ({', '.join(f'{name(m)} at beat {_beat(t)} lasts {_beats(ps[(t, m)])} in the PDF, {_beats(xs[(t, m)])} in the file' for t, m in diff[:3])})")
                continue
        pg = Counter(n.midi for n in pdf.notes if n.staff == staff and n.grace)
        xg = Counter(n.midi for n in export if n.staff == staff and n.grace)
        if pg != xg:
            out.append(f"{hand}: grace notes differ (the PDF has {_names(pg - xg) or 'none more'}, "
                       f"the file {_names(xg - pg) or 'none more'})")
    return out


def _beats(d: Fraction) -> str:
    return f"{d} beat" + ("" if d == 1 else "s")


def _names(c: Counter) -> str:
    return " ".join(name(m) for m in sorted(c.elements()))


def _beat(t: Fraction) -> str:
    b = t + 1
    return str(int(b)) if b.denominator == 1 else f"{float(b):g}"


# ---------------------------------------------------------------- matching bars

def _systems(part: ET.Element) -> list[int]:
    """Index of each system's first measure in the export, from its <print new-system/new-page>."""
    starts = [0]
    for i, m in enumerate(part.findall("measure")):
        pr = m.find("print")
        if i and pr is not None and (pr.get("new-system") == "yes" or pr.get("new-page") == "yes"):
            starts.append(i)
    return starts


def match(part: ET.Element, score: pdf_score.PdfScore, report: Report) -> list[int | None]:
    """The PDF bar (index) for each of the export's measures. In order when the counts agree; otherwise by
    aligning the two sequences on their pitches, so a bar the export lost or added costs only that bar."""
    measures = part.findall("measure")
    n, k = len(measures), len(score.bars)
    xs, ps = _systems(part), score.systems
    if n == k:
        if xs != ps:
            report.warnings.append(f"the export's system breaks (bars {[i + 1 for i in xs]}) differ from the PDF's "
                                   f"({[i + 1 for i in ps]}); bars matched in order")
        return list(range(n))
    report.warnings.append(f"the export has {n} bars and the PDF {k}: bars matched on their pitches")
    divisions, mine = 1, []
    for m_el in measures:
        d = m_el.find("attributes/divisions")
        if d is not None and (d.text or "").isdigit():
            divisions = int(d.text)
        mine.append(Counter(n.midi for n in export_notes(m_el, divisions) if not n.grace))
    theirs = [Counter(h.midi for h in {id(h.glyph): h for c in b.measure.chords for h in c.heads}.values())
              for b in score.bars]
    out = _align(mine, theirs)
    lost = [j for j in range(k) if j not in set(out)]
    for j in lost:
        b = score.bars[j]
        report.warnings.append(f"PDF bar {j + 1} (page {b.page}) has no bar in the file")
    return out


def _similar(a: Counter, b: Counter) -> float:
    """How alike two bars' pitches are, 0 to 1: shared notes, or failing that shared pitch classes (a bar read in
    the wrong clef still has its rhythm and most of its letters)."""
    if not a and not b:
        return 0.6
    if not a or not b:
        return 0.0
    exact = 2 * sum((a & b).values()) / (sum(a.values()) + sum(b.values()))
    pa, pb = Counter(m % 12 for m in a.elements()), Counter(m % 12 for m in b.elements())
    return max(exact, 0.5 * 2 * sum((pa & pb).values()) / (sum(pa.values()) + sum(pb.values())))


def _align(mine: list[Counter], theirs: list[Counter], gap: float = 0.4) -> list[int | None]:
    """Needleman-Wunsch over the two bar sequences: each pairing scores its similarity, each bar left out costs
    gap. The PDF bar for each of mine, or None."""
    n, k = len(mine), len(theirs)
    score = [[0.0] * (k + 1) for _ in range(n + 1)]
    step = [[""] * (k + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0], step[i][0] = -gap * i, "up"
    for j in range(1, k + 1):
        score[0][j], step[0][j] = -gap * j, "left"
    for i in range(1, n + 1):
        for j in range(1, k + 1):
            score[i][j], step[i][j] = max((score[i - 1][j - 1] + _similar(mine[i - 1], theirs[j - 1]), "pair"),
                                          (score[i - 1][j] - gap, "up"), (score[i][j - 1] - gap, "left"))
    out: list[int | None] = [None] * n
    i, j = n, k
    while i or j:
        move = step[i][j]
        if move == "pair":
            out[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif move == "up":
            i -= 1
        else:
            j -= 1
    return out


# ---------------------------------------------------------------- writing

def _events(placed: list[tuple[omr.Chord, Fraction, Fraction]], bar: pdf_score.Bar, length: Fraction) -> list[omr_repair.Event]:
    events = []
    for c, t, d in placed:
        heads = sorted(c.heads, key=lambda h: h.midi)
        events.append(omr_repair.Event(c.staff, t, d, [pdf_score._name(h) for h in heads],
                                       [set(h.ties) for h in heads], c.stem_up))
    for c in bar.measure.chords:
        if c.measure_rest:
            events.append(omr_repair.Event(c.staff, Fraction(0), length, [], []))
    return events


def _graces(m_el: ET.Element, bar: pdf_score.Bar, placed, divisions: int) -> int:
    """Put the PDF's grace notes before the note they lead to. Returns how many couldn't be placed."""
    missed = 0
    for g in sorted(bar.graces, key=lambda g: g.x):  # each goes right before its note: groups keep their order
        after = [(c.anchor, t) for c, t, _ in placed if c.staff == g.staff and c.heads and c.anchor > g.x]
        if not after:
            missed += 1
            continue
        target = min(after)[1]
        pos, k = Fraction(0), None
        for i, el in enumerate(list(m_el)):
            if el.tag == "backup":
                pos -= Fraction(int(el.findtext("duration")), divisions)
            elif el.tag == "forward":
                pos += Fraction(int(el.findtext("duration")), divisions)
            elif el.tag == "note" and el.find("chord") is None and el.find("grace") is None:
                if el.findtext("staff") == str(g.staff) and el.find("pitch") is not None and pos == target:
                    k = i
                    break
                pos += Fraction(int(el.findtext("duration")), divisions)
        if k is None:
            missed += 1
            continue
        voice = m_el[k].findtext("voice")
        for j, (step, alter, octave) in enumerate(g.pitches):
            n = ET.Element("note")
            ET.SubElement(n, "grace", slash="yes")
            if j:
                ET.SubElement(n, "chord")
            p = ET.SubElement(n, "pitch")
            ET.SubElement(p, "step").text = step
            if alter:
                ET.SubElement(p, "alter").text = str(alter)
            ET.SubElement(p, "octave").text = str(octave)
            ET.SubElement(n, "voice").text = voice
            ET.SubElement(n, "type").text = "eighth"
            ET.SubElement(n, "staff").text = str(g.staff)
            m_el.insert(k + j, n)
    return missed


def fill(part: ET.Element, m_el: ET.Element, bar: pdf_score.Bar, reading: Reading, length: Fraction,
         divisions: int) -> tuple[int, int]:
    """Rewrite the measure from the PDF bar. Returns the divisions the measure now uses and how many grace notes
    couldn't be placed."""
    events = _events(reading.placed, bar, length)
    grid = math.lcm(divisions, *(e.onset.denominator for e in events), *(e.duration.denominator for e in events),
                    length.denominator)
    if grid != divisions:
        omr_repair._set_divisions(part, m_el, grid, divisions)
    omr_repair._rewrite(m_el, events, length, grid)
    return grid, _graces(m_el, bar, reading.placed, grid)


# ---------------------------------------------------------------- entry point

def check_tree(root: ET.Element, score: pdf_score.PdfScore, flagged: list[int] | None = None,
               write: bool = True) -> Report:
    """Audit (and with write, fill) the piano part of a parsed MusicXML file in place."""
    report = Report(warnings=list(score.warnings))
    flagged = set(flagged or [])
    part = omr_repair._piano_part(root)
    if part is None:
        report.warnings.append("no two-staff part to check")
        return report
    matched = match(part, score, report)
    measures = part.findall("measure")
    divisions, meter = 1, (4, 4)
    printed = score.bars[0].time if score.bars else None  # the PDF's meter in force
    for i, m_el in enumerate(measures):
        attrs = m_el.find("attributes")
        if attrs is not None:
            if attrs.findtext("divisions"):
                divisions = int(attrs.findtext("divisions"))
            meter = _meter_of(attrs) or meter
        length = _quarters(meter)
        number = i + 1
        k = matched[i]
        if k is None:
            report.bars.append(BarCheck(number, "unmatched", ["no PDF bar matched to it"], flagged=number in flagged))
            continue
        bar = score.bars[k]
        printed = bar.time or printed
        here = _quarters(printed) if printed else length
        exported = export_notes(m_el, divisions)
        pickup = m_el.get("implicit") == "yes"
        if pickup:  # as long as its notes
            per_voice, _ = omr_repair._xml_notes(m_el, divisions)
            here = max(per_voice.values(), default=length)
        reading = read_bar(bar, here)
        diffs = compare(reading, exported)
        retime = printed is not None and printed != meter and not pickup
        if retime:
            diffs.insert(0, f"time signature: {printed[0]}/{printed[1]} in the PDF, {meter[0]}/{meter[1]} in the file")
        check = BarCheck(number, "ok", [], k + 1, bar.page, number in flagged)
        if not diffs:
            if reading.placed is None:
                check.status = "pitches-ok"
                check.reasons.append(reading.why)
            report.bars.append(check)
            continue
        check.reasons = diffs
        if reading.placed is None or not write:
            check.status = "differs"
            if reading.placed is None:
                check.reasons.append(reading.why)
            report.bars.append(check)
            continue
        nxt = measures[i + 1] if i + 1 < len(measures) else None
        saved = [(el, ET.fromstring(ET.tostring(el))) for el in (m_el, nxt) if el is not None]
        if retime:
            omr_repair._set_time(m_el, *printed)
            meter = printed
        new_div, missed = fill(part, m_el, bar, reading, here, divisions)
        after = compare(reading, export_notes(m_el, new_div))
        if missed and not after:
            check.reasons.append(f"{missed} grace notes couldn't be placed")
        if after:  # the rewrite didn't come out as read: put the bar back
            for el, copy in saved:
                el.clear()
                el.attrib.update(copy.attrib)
                el.extend(list(copy))
            check.status = "differs"
            check.reasons.append("rewriting it from the PDF didn't verify: " + "; ".join(after))
        else:
            check.status = "filled"
        report.bars.append(check)
    if write:
        _put_back(root, part, score, matched, report)
    else:
        report.missing = [j + 1 for j in range(len(score.bars)) if j not in set(matched)]
    return report


def _put_back(root: ET.Element, part: ET.Element, score: pdf_score.PdfScore, matched: list[int | None],
              report: Report) -> None:
    """Insert the PDF bars the export lost, where their timing reads cleanly (other parts get a bar's rest), and
    number the measures afresh."""
    have = {k for k in matched if k is not None}
    lost = [j for j in range(len(score.bars)) if j not in have]
    if not lost:
        return
    measures = part.findall("measure")
    others = [p for p in root.findall("part") if p is not part]
    inserted: list[tuple[int, int]] = []  # (index in the output, PDF bar)
    for j in reversed(lost):
        after = max((i for i, k in enumerate(matched) if k is not None and k < j), default=-1)
        divisions, meter = _in_force(measures, after)
        printed = next((b.time for b in reversed(score.bars[:j + 1]) if b.time), None)
        here = _quarters(printed or meter)
        reading = read_bar(score.bars[j], here)
        if reading.placed is None:
            report.warnings.append(f"PDF bar {j + 1} is missing from the file and couldn't be put back: {reading.why}")
            report.missing.append(j + 1)
            continue
        m_el = ET.Element("measure")
        if score.bars[j].number - 1 in score.systems:
            m_el.append(ET.Element("print", {"new-system": "yes"}))
        if printed and printed != meter:
            omr_repair._set_time(m_el, *printed)
            nxt = measures[after + 1] if after + 1 < len(measures) else None
            if nxt is not None and nxt.find("attributes/time") is None:  # the bars after keep the file's meter
                omr_repair._set_time(nxt, *meter)
        _insert(part, after, m_el)
        _, missed = fill(part, m_el, score.bars[j], reading, here, divisions)
        if missed:
            report.warnings.append(f"PDF bar {j + 1}: {missed} grace notes couldn't be placed")
        for p in others:
            pm = p.findall("measure")
            pdiv, _ = _in_force(pm, after)
            rest = ET.Element("measure")
            n = ET.SubElement(rest, "note")
            ET.SubElement(n, "rest", measure="yes")
            ET.SubElement(n, "duration").text = str(int(here * pdiv))
            ET.SubElement(n, "voice").text = "1"
            _insert(p, after, rest)
        inserted.append((after + 1, j))
    for p in root.findall("part"):
        for n, m_el in enumerate(p.findall("measure"), 1):
            m_el.set("number", str(n))
    for pos, j in sorted(inserted, reverse=True):  # the last first, so earlier insertions push it along
        for b in report.bars:
            if b.number > pos:
                b.number += 1
        report.bars.append(BarCheck(pos + 1, "inserted", ["the file had lost this bar; put back from the PDF"],
                                    j + 1, score.bars[j].page))
    report.bars.sort(key=lambda b: b.number)


def _in_force(measures: list[ET.Element], upto: int) -> tuple[int, tuple[int, int]]:
    """Divisions and meter in force at the end of measure upto (index; -1: before the first)."""
    divisions, meter = 1, (4, 4)
    for m_el in measures[:upto + 1] if upto >= 0 else measures[:1]:
        a = m_el.find("attributes")
        if a is None:
            continue
        if (a.findtext("divisions") or "").isdigit():
            divisions = int(a.findtext("divisions"))
        meter = _meter_of(a) or meter
    return divisions, meter


def _meter_of(attrs: ET.Element) -> tuple[int, int] | None:
    t = attrs.find("time")
    if t is not None and (t.findtext("beats") or "").isdigit():
        return int(t.findtext("beats")), int(t.findtext("beat-type", "4"))
    return None


def _quarters(meter: tuple[int, int]) -> Fraction:
    return Fraction(4 * meter[0], meter[1])


def _insert(part: ET.Element, after: int, m_el: ET.Element) -> None:
    ms = part.findall("measure")
    at = list(part).index(ms[after]) + 1 if after >= 0 else list(part).index(ms[0])
    part.insert(at, m_el)


def check(pdf: str | Path, musicxml: str | Path, out: str | Path | None = None,
          repair_report: str | Path | None = None) -> Report:
    """Audit a MusicXML file against the PDF it was read from; with out, write a copy with the differing bars
    rewritten from the PDF where that's unambiguous."""
    root = read_root(musicxml)
    flagged = []
    if repair_report:
        flagged = json.loads(Path(repair_report).read_text(encoding="utf-8")).get("needs_review", [])
    report = check_tree(root, pdf_score.read(pdf), flagged, write=out is not None)
    if out is not None:
        tree = ET.ElementTree(root)
        ET.indent(tree)
        Path(out).write_bytes(b'<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="utf-8"))
    return report
