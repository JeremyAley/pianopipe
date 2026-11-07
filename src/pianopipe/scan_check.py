"""Two OMR readings of a scanned song compared bar by bar, merged, and corrected by hand where they can't be trusted.

A scan has no vector notation for pdf_check to read, and neither reader is right often enough alone: on the
Contemporary Disney scan (2026-11-04) copista split a bar in two at a stray mark, read pencilled note names as notes
and dropped key-signature flats; Audiveris read the same bars right and got others wrong that copista read right.
Where both readers agree on a bar's notes (staff, onset, length, pitch) the bar is very likely right; where they
differ, one of them usually is. So:

- **align** the two piano parts on their pitches (pdf_check's Needleman-Wunsch), and catch a bar one reader split
  in two (two bars of one reading that together match one of the other's);
- **compare** each pair per staff;
- **merge**: copista's reading, with Audiveris's bar wherever copista split a bar, lost one, or read one that
  can't be right (a staff that doesn't fill its bar, a staff with no notes where the other reader has some);
- **flag** every bar where the two disagree (`needs_review`), with the reasons, for the vision or hand check;
- **fix**: songs.json's per-song `fixes` replace bars by hand, in the text form `bar_text` writes (so a correction
  survives a re-run of the readers as long as the bar numbers do).

The bar text: per staff (`RH`, `LH`), chords as `onset:note,note`, each note `<step><alter><octave>/<length>` in
quarter notes (fractions allowed), `~` for a tie into the next note, a `g` before a grace note (no length);
staves split with `|`: `RH 0:A3/1,C4/1/2 1/2:F4/1/2~ | LH 0:F2/1,F3/1`. `time=2/4` at the start sets the bar's
time signature. Rests are what's left over. The design and results: docs/catalog-expansion.md §3.2 "Scans".
"""

from __future__ import annotations

import copy
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction

from . import musicxml, omr_repair, pdf_check
from .musicxml import STEPS

FINE = 1680  # divisions a quarter note is made a multiple of before bars are rewritten (fine): 48 x 5 x 7, for
# sixteenths, 32nds, triplets, sextuplets, quintuplets and septuplets (a 7-note run in Just Around the Riverbend)
SPLIT_GAIN = 0.05  # joining two bars must match the other reading this much better than the bar alone


@dataclass
class Bar:
    notes: list[pdf_check.Note]
    length: Fraction  # the time signature in force, in quarter notes
    element: ET.Element
    divisions: int


@dataclass
class BarResult:
    number: int  # the merged score's bar, 1-based
    status: str  # same, differs, filled (Audiveris's notes in copista's rests), split (one reader's two bars
    # joined), only-copista, only-audiveris, fixed (checked by hand)
    source: str  # copista, audiveris, hand
    copista: list[int] = field(default_factory=list)  # 1-based bars of each reading behind it
    audiveris: list[int] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


@dataclass
class Report:
    bars: list[BarResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # what the merge took from Audiveris beyond notes

    @property
    def needs_review(self) -> list[int]:
        return [b.number for b in self.bars if b.status not in ("same", "fixed", "filled")]

    def to_json(self) -> dict:
        return {"needs_review": self.needs_review,
                "bars": {str(b.number): {"status": b.status, "source": b.source, "copista": b.copista,
                                         "audiveris": b.audiveris, "reasons": b.reasons} for b in self.bars},
                "notes": self.notes, "warnings": self.warnings}


# ---------------------------------------------------------------- reading the parts

def piano_part(root: ET.Element) -> ET.Element | None:
    return omr_repair._piano_part(root)


def read_bars(part: ET.Element) -> list[Bar]:
    out, divisions, length = [], 1, Fraction(4)
    for m_el in part.findall("measure"):
        d = m_el.find("attributes/divisions")
        if d is not None and (d.text or "").strip().isdigit():
            divisions = int(d.text)
        t = m_el.find("attributes/time")
        if t is not None and (t.findtext("beats") or "").isdigit():
            length = Fraction(4 * int(t.findtext("beats")), int(t.findtext("beat-type")))
        out.append(Bar(pdf_check.export_notes(m_el, divisions), length, m_el, divisions))
    return out


def _pitches(bar: Bar) -> Counter:
    return Counter(n.midi for n in bar.notes if not n.grace)


def _shifted(notes: list[pdf_check.Note], by: Fraction) -> list[pdf_check.Note]:
    return [pdf_check.Note(n.staff, n.onset + by, n.duration, n.midi, n.grace) for n in notes]


# ---------------------------------------------------------------- comparing

def differences(a: list[pdf_check.Note], b: list[pdf_check.Note], names: tuple[str, str] = ("copista", "Audiveris")) -> list[str]:
    """How two readings of a bar differ, per staff: pitches, then onsets, then lengths, then grace notes."""
    out = []
    for staff in (1, 2):
        hand = "right hand" if staff == 1 else "left hand"
        sa, sb = pdf_check._sounding(a, staff), pdf_check._sounding(b, staff)
        pa, pb = Counter(m for _, m in sa), Counter(m for _, m in sb)
        if pa != pb:
            out.append(f"{hand}: {names[0]} has {pdf_check._names(pa - pb) or 'nothing more'}, "
                       f"{names[1]} {pdf_check._names(pb - pa) or 'nothing more'}")
        elif set(sa) != set(sb):
            moved = sorted(set(sa) - set(sb))
            out.append(f"{hand}: notes start at other times ({', '.join(f'{pdf_check.name(m)} at beat {pdf_check._beat(t)}' for t, m in moved[:4])} "
                       f"for {names[0]}; beats {', '.join(pdf_check._beat(t) for t, _ in sorted(set(sb) - set(sa))[:4])} for {names[1]})")
        elif sa != sb:
            diff = sorted(k for k in sa if sa[k] != sb[k])
            out.append(f"{hand}: other lengths (" + ", ".join(
                f"{pdf_check.name(m)} at beat {pdf_check._beat(t)}: {sa[(t, m)]} or {sb[(t, m)]}" for t, m in diff[:3]) + ")")
        ga = Counter(n.midi for n in a if n.staff == staff and n.grace)
        gb = Counter(n.midi for n in b if n.staff == staff and n.grace)
        if ga != gb:
            out.append(f"{hand}: grace notes differ ({names[0]} {pdf_check._names(ga - gb) or 'none more'}, "
                       f"{names[1]} {pdf_check._names(gb - ga) or 'none more'})")
    return out


def implausible(bar: Bar) -> list[str]:
    """What can't be right about a reading of a bar on its own: a staff whose notes run past the bar's end."""
    out = []
    for staff in (1, 2):
        ends = [n.onset + n.duration for n in bar.notes if n.staff == staff and not n.grace]
        if ends and max(ends) > bar.length:
            out.append(f"{'right' if staff == 1 else 'left'} hand runs past the bar's end")
    return out


# ---------------------------------------------------------------- aligning

def align(a: list[Bar], b: list[Bar]) -> list[tuple[list[int], list[int]]]:
    """Groups of bar indices (a's, b's) that are one bar of music: one to one, two to one where a reader split a
    bar in two, one to none where a reader lost or added one. In order."""
    pairs = pdf_check._align([_pitches(x) for x in a], [_pitches(x) for x in b])
    of_b: dict[int, int] = {j: i for i, j in enumerate(pairs) if j is not None}
    groups: list[tuple[list[int], list[int]]] = []
    i = j = 0
    while i < len(a) or j < len(b):
        if i < len(a) and pairs[i] is not None and pairs[i] == j:
            groups.append(([i], [j]))
            i, j = i + 1, j + 1
        elif i < len(a) and (pairs[i] is None or pairs[i] > j) and (j >= len(b) or j in of_b):
            groups.append(([i], []))
            i += 1
        else:
            groups.append(([], [j]))
            j += 1
    return _joins(groups, a, b)


def _joins(groups: list[tuple[list[int], list[int]]], a: list[Bar], b: list[Bar]) -> list[tuple[list[int], list[int]]]:
    """Fold a lone bar into its neighbour's group when the two together, one after the other, match the other
    reading's bar better than the neighbour alone (a bar split in two). Notes are matched with their onsets: the
    pieces of a split bar line up with the whole bar only when joined."""
    def similar(seq: list[Bar], idx: list[int], target: Counter) -> float:
        return max(_similar_timed(_timed(joined_notes(seq, idx, overlay)[0]), target) for overlay in (False, True))

    out = list(groups)
    changed = True
    while changed:
        changed = False
        for k, (ia, ib) in enumerate(out):
            if bool(ia) == bool(ib):
                continue
            lone_a = bool(ia)
            for n in (k - 1, k + 1):
                if not 0 <= n < len(out) or not out[n][0] or not out[n][1]:
                    continue
                na, nb = out[n]
                mine = (a, na, ia) if lone_a else (b, nb, ib)
                other = (b, nb) if lone_a else (a, na)
                seq, have, extra = mine
                target = _timed(joined_notes(other[0], other[1])[0])
                both = sorted(have + extra)
                if similar(seq, both, target) >= similar(seq, have, target) + SPLIT_GAIN:
                    out[n] = (both, nb) if lone_a else (na, both)
                    del out[k]
                    changed = True
                    break
            if changed:
                break
    return out


def _similar_timed(a: Counter, b: Counter) -> float:
    """Shared (onset, pitch) notes of two bars, 0 to 1."""
    if not a or not b:
        return 0.0
    return 2 * sum((a & b).values()) / (sum(a.values()) + sum(b.values()))


def _timed(notes: list[pdf_check.Note]) -> Counter:
    return Counter((n.onset, n.midi) for n in notes if not n.grace)


def joined_notes(bars: list[Bar], idx: list[int], overlay: bool = False) -> tuple[list[pdf_check.Note], Fraction]:
    """The notes of consecutive bars as one bar, and its length. A bar split in two comes in two ways: each piece
    timed from its own start (the second follows the first's last note), or the second piece already timed from
    the bar's start, after a leading gap (`overlay`: the pieces laid over each other). Each piece's time
    signature can't be trusted, so a joined bar is as long as its notes."""
    notes, at = [], Fraction(0)
    for x in idx:
        notes += _shifted(bars[x].notes, Fraction(0) if overlay else at)
        ends = [n.onset + n.duration for n in bars[x].notes if not n.grace]
        at = max(at, max(ends, default=0)) if overlay else at + (max(ends, default=bars[x].length) if len(idx) > 1 else bars[x].length)
    if len(idx) == 1:
        at = bars[idx[0]].length
    return notes, at


def best_join(bars: list[Bar], idx: list[int], target: list[pdf_check.Note]) -> tuple[list[pdf_check.Note], Fraction]:
    """joined_notes in whichever way matches the other reading's bar better."""
    return max((joined_notes(bars, idx, overlay) for overlay in (False, True)),
               key=lambda j: _similar_timed(_timed(j[0]), _timed(target)))


# ---------------------------------------------------------------- the bar text

def note_name(step: str, alter: int, octave: int) -> str:
    return step + {-2: "bb", -1: "b", 0: "", 1: "#", 2: "##"}[alter] + str(octave)


def _frac(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def bar_text(m_el: ET.Element, divisions: int) -> str:
    """A measure's notes in the bar text (see the module's docstring), spelled as the file spells them; grace notes
    in the order they are played."""
    pos, last = Fraction(0), Fraction(0)
    chords: dict[int, dict[Fraction, list[tuple[bool, int, str]]]] = {1: {}, 2: {}}
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
                onset, last = pos, pos
                pos += dur
            p = el.find("pitch")
            if p is None:
                continue
            name = note_name(p.findtext("step", "C"), round(float(p.findtext("alter", "0") or 0)), int(p.findtext("octave", "4")))
            tie = "~" if any(t.get("type") == "start" for t in el.findall("tie")) else ""
            midi = (12 * (int(p.findtext("octave", "4")) + 1) + STEPS.get(p.findtext("step", "C"), 0)
                    + round(float(p.findtext("alter", "0") or 0)))
            chords[int(el.findtext("staff", "1"))].setdefault(onset, []).append(
                (not grace, 0 if grace else midi, ("g" + name if grace else f"{name}/{_frac(dur)}") + tie))
    t = m_el.find("attributes/time")
    head = f"time={t.findtext('beats')}/{t.findtext('beat-type')} " if t is not None else ""
    staves = [hand + " " + " ".join(f"{_frac(on)}:{','.join(n for *_, n in sorted(ns, key=lambda x: x[:2]))}" for on, ns in sorted(chords[st].items()))
              for st, hand in ((1, "RH"), (2, "LH"))]
    return head + " | ".join(s.strip() for s in staves)


NOTE = re.compile(r"^(g?)([A-G])(bb|b|##|#|n?)(-?\d)(?:/([\d/]+))?(~?)$")


def parse_bar(text: str) -> tuple[tuple[int, int] | None, list[tuple[int, Fraction, Fraction, tuple[str, int, int], bool, bool]]]:
    """The bar text read: (time signature or None, [(staff, onset, length, (step, alter, octave), tie, grace)])."""
    time = None
    m = re.match(r"\s*time=(\d+)/(\d+)\s*", text)
    if m:
        time = (int(m.group(1)), int(m.group(2)))
        text = text[m.end():]
    notes = []
    for part in text.split("|"):
        part = part.strip()
        if not part:
            continue
        hand, _, rest = part.partition(" ")
        if hand not in ("RH", "LH"):
            raise ValueError(f"each staff starts RH or LH: {part!r}")
        staff = 1 if hand == "RH" else 2
        for tok in rest.split():
            onset, _, ns = tok.partition(":")
            if not ns:
                raise ValueError(f"a chord is onset:notes: {tok!r}")
            for n in ns.split(","):
                nm = NOTE.match(n)
                if not nm:
                    raise ValueError(f"not a note: {n!r} in {tok!r}")
                grace, step, acc, octave, length, tie = nm.groups()
                if not grace and not length:
                    raise ValueError(f"a note needs a length: {n!r}")
                alter = {"bb": -2, "b": -1, "": 0, "n": 0, "#": 1, "##": 2}[acc]
                notes.append((staff, Fraction(onset), Fraction(length) if length else Fraction(0), (step, alter, int(octave)),
                              bool(tie), bool(grace)))
    return time, notes


def set_bar(m_el: ET.Element, text: str, divisions: int, length: Fraction, clip: bool = False) -> Fraction:
    """Rewrite a measure's notes from the bar text (its time signature too, if the text sets one). Returns the bar's
    length. A note running past the bar's end is an error, or with `clip` (a reader's bar written into the merge) cut
    at the end; a grace note with no note to lead to is then dropped."""
    time, notes = parse_bar(text)
    if time:
        omr_repair._set_time(m_el, *time)
        length = Fraction(4 * time[0], time[1])
    lcm = divisions
    for _, on, ln, *_ in notes:
        for x in (on, ln):
            lcm = lcm * x.denominator // _gcd(lcm, x.denominator)
    if lcm != divisions:
        raise ValueError(f"a length or onset finer than the part's divisions ({divisions} a quarter)")
    events: dict[tuple[int, Fraction, Fraction], omr_repair.Event] = {}
    graces = []
    for staff, on, ln, pitch, tie, grace in notes:
        if grace:
            graces.append((staff, on, pitch))
            continue
        if on + ln > length:
            if not clip:
                raise ValueError(f"{note_name(*pitch)} at {on} runs past the bar's end ({length})")
            if on >= length:
                continue
            ln = length - on
        e = events.setdefault((staff, on, ln), omr_repair.Event(staff, on, ln, [], []))
        e.pitches.append(pitch)
        e.ties.append({"start"} if tie else set())
    omr_repair._rewrite(m_el, list(events.values()), length, divisions)
    _put_graces(m_el, graces, divisions, clip)
    return length


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def retie(part: ET.Element) -> None:
    """Make every tie stop match a tie start: a note tied from the note before it on its staff (same pitch, ending
    where it starts) gets its stop, and a stop with no start goes. Bars rewritten from either reader or by hand
    carry their own starts; the stops are put right here, across bar lines."""
    divisions, at = 1, Fraction(0)
    notes = []  # (staff, onset, end, midi, element), the whole part in time
    for m_el in part.findall("measure"):
        d = m_el.find("attributes/divisions")
        if d is not None and (d.text or "").strip().isdigit():
            divisions = int(d.text)
        pos, last, end = Fraction(0), Fraction(0), Fraction(0)
        for el in m_el:
            if el.tag == "backup":
                pos -= Fraction(int(el.findtext("duration", "0")), divisions)
            elif el.tag == "forward":
                pos += Fraction(int(el.findtext("duration", "0")), divisions)
            elif el.tag == "note":
                if el.find("grace") is not None:
                    continue
                dur = Fraction(int(el.findtext("duration", "0")), divisions)
                onset = last if el.find("chord") is not None else pos
                if el.find("chord") is None:
                    last = pos
                    pos += dur
                end = max(end, pos)
                p = el.find("pitch")
                if p is not None:
                    midi = (12 * (int(p.findtext("octave", "4")) + 1) + STEPS.get(p.findtext("step", "C"), 0)
                            + round(float(p.findtext("alter", "0") or 0)))
                    notes.append((int(el.findtext("staff", "1")), at + onset, at + onset + dur, midi, el))
        t = m_el.find("attributes/time")
        at += max(end, pos)
    starts = {(s, e, m) for s, _, e, m, el in notes if any(t.get("type") == "start" for t in el.findall("tie"))}
    for s, on, _, m, el in notes:
        want = (s, on, m) in starts
        for tie in el.findall("tie"):
            if tie.get("type") == "stop":
                el.remove(tie)
        nots = el.find("notations")
        if nots is not None:
            for t in nots.findall("tied"):
                if t.get("type") == "stop":
                    nots.remove(t)
        if want:
            before = next((k for k, c in enumerate(el) if c.tag in ("voice", "type", "stem", "staff", "notations")), len(el))
            el.insert(before, ET.Element("tie", type="stop"))
            if nots is None:
                nots = ET.SubElement(el, "notations")
            nots.insert(0, ET.Element("tied", type="stop"))


# ---------------------------------------------------------------- merging

def merge(cop_root: ET.Element, aud_root: ET.Element, choose: dict[int, str] | None = None) -> tuple[ET.Element, Report]:
    """copista's score with the piano part checked against Audiveris's, bar by bar (see the module's docstring).
    The other parts (the voices) follow the piano's bars: a bar dropped or added in the piano is dropped or added
    in them too. `choose` overrides the choice for merged bars checked by hand: {bar: "copista" or "audiveris"}."""
    choose = choose or {}
    report = Report()
    out = copy.deepcopy(cop_root)
    cp, ap = piano_part(out), piano_part(aud_root)
    if cp is None or ap is None:
        report.warnings.append("no piano part in " + ("copista's" if cp is None else "Audiveris's")
                               + " reading: left as copista read it")
        return out, report
    fine(cp)
    a, b = read_bars(cp), read_bars(ap)
    others = [p for p in out.findall("part") if p is not cp]
    other_bars = [p.findall("measure") for p in others]
    new_piano: list[ET.Element] = []
    new_others: list[list[ET.Element]] = [[] for _ in others]
    groups = align(a, b)
    for ia, ib in groups:
        res = BarResult(len(new_piano) + 1, "same", "copista", [i + 1 for i in ia], [j + 1 for j in ib])
        use_b, nb = False, []
        if ia and ib:
            na, _ = joined_notes(a, ia)
            nb, _ = joined_notes(b, ib)
            if len(ia) > 1:
                na, _ = best_join(a, ia, nb)
            elif len(ib) > 1:
                nb, _ = best_join(b, ib, na)
            res.reasons = differences(na, nb)
            bad_a = implausible(a[ia[0]]) if len(ia) == 1 else []
            bad_b = implausible(b[ib[0]]) if len(ib) == 1 else []
            if len(ia) > 1 or len(ib) > 1:
                res.status = "split"
                res.reasons.insert(0, ("copista" if len(ia) > 1 else "Audiveris") + f" read it as {max(len(ia), len(ib))} bars")
            elif res.reasons:
                res.status = "differs"
            use_b = (len(ia) > 1 and not bad_b) or bool(bad_a and not bad_b) or bool(res.reasons and _empty_staff(na, nb))
            if (res.status == "differs" and not use_b and not bad_b and a[ia[0]].length == b[ib[0]].length
                    and _fills_gaps(na, nb) and _same_ties(a[ia[0]], b[ib[0]])):
                use_b, res.status = True, "filled"
                res.reasons.append("Audiveris has notes copista missed, in copista's rests: taken")
            res.reasons += [f"copista: {x}" for x in bad_a] + [f"Audiveris: {x}" for x in bad_b]
            if (bad_a or bad_b) and res.status == "same":
                res.status = "differs"
        elif ia:
            res.status, res.reasons = "only-copista", ["Audiveris has no bar here"]
        else:
            res.status, res.reasons, use_b = "only-audiveris", ["copista has no bar here"], True
        pick = choose.get(res.number)
        if pick in ("copista", "audiveris"):
            if (pick == "audiveris" and len(ib) != 1) or (pick == "copista" and not ia):
                raise ValueError(f"bar {res.number}: {pick} has no bar of its own here")
            use_b = pick == "audiveris"
            res.status = "fixed"
        if use_b and len(ib) == 1:
            m_el = _adopt(b[ib[0]], a[ia[0]] if ia else (a[-1] if a else None))
            res.source = "audiveris"
        elif len(ia) > 1:  # Audiveris read it as two bars too: copista's pieces joined
            res.reasons.append("copista's pieces joined")
            m_el = _joined_element(a, ia, nb)
        else:
            m_el = a[ia[0]].element
        new_piano.append(m_el)
        for k, ms in enumerate(other_bars):
            src = [ms[i] for i in ia if i < len(ms)]
            new_others[k].append(src[0] if src else ET.Element("measure"))
        report.bars.append(res)
    aud_parts = [p.findall("measure") for p in aud_root.findall("part")]
    for k, ((_, ib), m_el) in enumerate(zip(groups, new_piano)):
        carried = _carry_jumps(m_el, aud_parts, ib, k == 0)
        if carried:
            report.notes.append(f"bar {k + 1}: marks from Audiveris ({', '.join(carried)})")
    _replace_measures(cp, new_piano)
    for p, ms in zip(others, new_others):
        _replace_measures(p, ms)
    retie(cp)
    return out, report


def _jump_kinds(el: ET.Element) -> set[str]:
    """The jumps and signs a <direction> marks: <segno>, <coda>, <sound> attributes, or words (D.S., To Coda...)."""
    kinds = {k for k in ("segno", "coda") if el.find(f".//{k}") is not None}
    snd = el.find("sound")
    if snd is not None:
        kinds |= {k for k in musicxml.JUMP_FLAGS if snd.get(k) is not None}
    text = re.sub(r"^[\W_]+", "", " ".join(w.text or "" for w in el.iter("words")).strip())
    kinds |= {flag for flag, rx in musicxml._JUMP_WORDS if rx.match(text)}
    if text and el.get("placement") != "below" and (musicxml._tempo_word(text) or musicxml._TEMPO_PRIMO.search(text)):
        kinds.add("tempo")
    return kinds


def _carry_jumps(m_el: ET.Element, aud_parts: list[list[ET.Element]], ib: list[int], first: bool = False) -> list[str]:
    """Audiveris's jump marks and tempo words in its bars (any part: it often files "D.S. al Coda" with the vocal
    line) copied to the merged bar when copista has none of their kind there. copista reads no words, so without
    these a scan's D.S. al Coda is lost and the song stops early, and a change of tempo ("Moderate Calypso" after a
    free opening) never comes. Not the first bar's tempo: the opening tempo is set from the OCR's words apart."""
    have = set().union(*(_jump_kinds(d) for d in m_el.findall("direction"))) | ({"tempo"} if first else set())
    carried = []
    for ms in aud_parts:
        for j in ib:
            if j >= len(ms):
                continue
            for d in ms[j].findall("direction"):
                kinds = _jump_kinds(d) - have
                if kinds:
                    new = copy.deepcopy(d)
                    for tag in ("offset", "staff", "voice"):
                        for x in new.findall(tag):
                            new.remove(x)
                    m_el.insert(next((k for k, c in enumerate(m_el) if c.tag == "note"), len(m_el)), new)
                    have |= kinds
                    carried += sorted(kinds)
    return carried


def _fills_gaps(a: list[pdf_check.Note], b: list[pdf_check.Note]) -> bool:
    """b is a with notes added, each where a's staff is silent (copista drops a note, often one of a triplet, and
    leaves a rest; Part of Your World lost six that way)."""
    for staff in (1, 2):
        sa, sb = pdf_check._sounding(a, staff), pdf_check._sounding(b, staff)
        if any(sb.get(k) != d for k, d in sa.items()):
            return False
        for (t, m), d in sb.items():
            if (t, m) not in sa and any(on < t + d and t < on + dur for (on, _), dur in sa.items()):
                return False
    ga = Counter(n.midi for n in a if n.grace)
    gb = Counter(n.midi for n in b if n.grace)
    return ga == gb and _timed(b) != _timed(a)


def _tie_starts(bar: Bar) -> set[tuple[int, Fraction, int]]:
    """(staff, onset, pitch) of the bar's notes tied into the next."""
    out = set()
    for line in bar_text(bar.element, bar.divisions).split("|"):
        line = line.split(" ", 1)[-1] if line.startswith("time=") else line
        staff = 1 if line.strip().startswith("RH") else 2
        for tok in line.split()[1:]:
            on, _, ns = tok.partition(":")
            for n in ns.split(","):
                if n.endswith("~"):
                    m = NOTE.match(n)
                    step, acc, octave = m.group(2), m.group(3), int(m.group(4))
                    alter = {"bb": -2, "b": -1, "": 0, "n": 0, "#": 1, "##": 2}[acc]
                    out.add((staff, Fraction(on), 12 * (octave + 1) + STEPS[step] + alter))
    return out


def _same_ties(a: Bar, b: Bar) -> bool:
    """Both readings tie the notes they share the same way (taking b's bar mustn't lose or add a tie of a's)."""
    shared = {(n.staff, n.onset, n.midi) for n in a.notes if not n.grace}
    return {t for t in _tie_starts(a) if t in shared} == {t for t in _tie_starts(b) if t in shared}


def _empty_staff(a: list[pdf_check.Note], b: list[pdf_check.Note]) -> bool:
    """copista has a staff empty that Audiveris has notes on (copista dropped a staff's notes)."""
    return any(not any(n.staff == s for n in a) and any(n.staff == s for n in b) for s in (1, 2))


def _adopt(src: Bar, like: Bar | None) -> ET.Element:
    """Audiveris's bar as a measure of copista's part: its notes in copista's divisions (made finer if they must),
    copista's attributes (clefs, key) kept; Audiveris's time signature where the two differ."""
    m_el = copy.deepcopy(like.element) if like is not None else ET.Element("measure")
    divisions = like.divisions if like is not None else src.divisions
    length = like.length if like is not None else src.length
    text = bar_text(src.element, src.divisions).split(" ", 1)[-1] if bar_text(src.element, src.divisions).startswith("time=") \
        else bar_text(src.element, src.divisions)
    if src.length != length:
        text = f"time={_meter(src.length)} " + text
    set_bar(m_el, text, divisions, length, clip=True)
    return m_el


def _needed_divisions(text: str, divisions: int) -> int:
    lcm = divisions
    for _, on, ln, *_ in parse_bar(text)[1]:
        for x in (on, ln):
            lcm = lcm * x.denominator // _gcd(lcm, x.denominator)
    return lcm


def _meter(length: Fraction) -> str:
    beats, beat_type = omr_repair._meter(length)
    return f"{beats}/{beat_type}"


def fine(part: ET.Element) -> None:
    """Every measure of the part in divisions that are a multiple of FINE a quarter, so a bar written from the
    other reader or by hand (triplets, sixteenths, 32nds) fits."""
    divisions = 1
    for m_el in part.findall("measure"):
        d = m_el.find("attributes/divisions")
        if d is not None and (d.text or "").strip().isdigit():
            divisions = int(d.text)
            d.text = str(divisions * FINE // _gcd(divisions, FINE))
        factor = FINE // _gcd(divisions, FINE)
        if factor > 1:
            for el in m_el.iter("duration"):
                if el.text and el.text.strip().isdigit():
                    el.text = str(int(el.text) * factor)


def _joined_element(bars: list[Bar], idx: list[int], other: list[pdf_check.Note]) -> ET.Element:
    """Consecutive bars of one reading as one measure (a bar it split in two), its time signature from the notes."""
    m_el = copy.deepcopy(bars[idx[0]].element)
    notes, length = best_join(bars, idx, other)
    text = f"time={_meter(length)} " + _notes_text(notes)
    set_bar(m_el, text, bars[idx[0]].divisions, length, clip=True)
    return m_el


def _notes_text(notes: list[pdf_check.Note]) -> str:
    """Bar text from pitch-only notes, spelled with sharps (a bar rebuilt from pieces)."""
    staves = []
    for st, hand in ((1, "RH"), (2, "LH")):
        chords: dict[Fraction, list[str]] = {}
        for n in sorted(notes, key=lambda n: (n.onset, -1 if n.grace else n.midi)):  # grace notes keep their order
            if n.staff == st:
                name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][n.midi % 12] + str(n.midi // 12 - 1)
                chords.setdefault(n.onset, []).append(("g" + name) if n.grace else f"{name}/{_frac(n.duration)}")
        staves.append(hand + " " + " ".join(f"{_frac(on)}:{','.join(ns)}" for on, ns in sorted(chords.items())))
    return " | ".join(s.strip() for s in staves)


def _replace_measures(part: ET.Element, measures: list[ET.Element]) -> None:
    for m in part.findall("measure"):
        part.remove(m)
    for k, m in enumerate(measures, 1):
        m.set("number", str(k))
        part.append(m)


# ---------------------------------------------------------------- hand fixes

def apply_fixes(root: ET.Element, fixes: dict[str, str | list[str]], report: Report | None = None) -> list[str]:
    """songs.json's `fixes` on the piano part: {"<bar>": "<bar text>"} replaces a bar's notes; "ok" says the merged
    bar was checked and is right; "delete" drops the bar from every part; "<bar>+" adds a bar after it (empty in the
    other parts), or several, given as a list of bar texts (a passage the print repeats a varying number of times,
    "Repeat as needed", written out). ("copista" and "audiveris", which reading to take, are merge's.) Bar numbers are the merged
    reading's, before any fix. Returns what was done."""
    part = piano_part(root)
    if part is None or not fixes:
        return []
    others = [p for p in root.findall("part") if p is not part]
    fine(part)
    measures = part.findall("measure")
    other_ms = [p.findall("measure") for p in others]
    bars = read_bars(part)
    rows = list(report.bars) if report is not None else []
    done, deletes, inserts = [], set(), {}
    for key, text in fixes.items():
        n = int(key.rstrip("+"))
        if not 1 <= n <= len(measures):
            raise ValueError(f"fix for bar {key}: the song has {len(measures)} bars")
        if key.endswith("+"):
            inserts[n] = text
        elif text.strip() in ("ok", "copista", "audiveris"):
            if report is not None and n <= len(report.bars):
                report.bars[n - 1].status = "fixed"
        elif text.strip() == "delete":
            deletes.add(n)
        else:
            bar = bars[n - 1]
            set_bar(measures[n - 1], text, bar.divisions, bar.length)
            done.append(f"bar {n}")
            if report is not None and n <= len(report.bars):
                report.bars[n - 1].status, report.bars[n - 1].source = "fixed", "hand"
    for n, texts in sorted(inserts.items(), reverse=True):
        bar = bars[n - 1]
        texts = [texts] if isinstance(texts, str) else texts
        for text in reversed(texts):  # each goes in right after bar n: the last first
            new = ET.Element("measure")
            set_bar(new, text, bar.divisions, bar.length)
            part.insert(list(part).index(measures[n - 1]) + 1, new)
            for p, ms in zip(others, other_ms):
                if n - 1 < len(ms):
                    p.insert(list(p).index(ms[n - 1]) + 1, ET.Element("measure"))
            if report is not None and n <= len(rows):
                report.bars.insert(report.bars.index(rows[n - 1]) + 1, BarResult(0, "fixed", "hand"))
        done.append(f"a bar after {n}" if len(texts) == 1 else f"{len(texts)} bars after {n}")
    for n in sorted(deletes, reverse=True):
        live = part.findall("measure")
        _pass_on(live, live.index(measures[n - 1]))
        part.remove(measures[n - 1])
        for p, ms in zip(others, other_ms):
            if n - 1 < len(ms):
                live = p.findall("measure")
                _pass_on(live, live.index(ms[n - 1]))
                p.remove(ms[n - 1])
        if report is not None and n <= len(rows):
            report.bars.remove(rows[n - 1])
        done.append(f"bar {n} deleted")
    for p in root.findall("part"):
        for k, m in enumerate(p.findall("measure"), 1):
            m.set("number", str(k))
    if report is not None:  # the report's bars stay the song's, numbered as it now stands
        for k, b in enumerate(report.bars, 1):
            b.number = k
    retie(part)
    return done


def _put_graces(m_el: ET.Element, graces: list, divisions: int, clip: bool = False) -> None:
    """Grace notes go just before the first note of their staff at their onset."""
    if not graces:
        return
    pos, last = Fraction(0), Fraction(0)
    targets = []
    for el in list(m_el):
        if el.tag == "backup":
            pos -= Fraction(int(el.findtext("duration", "0")), divisions)
        elif el.tag == "forward":
            pos += Fraction(int(el.findtext("duration", "0")), divisions)
        elif el.tag == "note":
            if el.find("chord") is None:
                targets.append((int(el.findtext("staff", "1")), pos, el))
                pos += Fraction(int(el.findtext("duration", "0")), divisions)
    for staff, on, (step, alter, octave) in graces:
        t = next((el for s, p, el in targets if s == staff and p == on), None)
        if t is None:
            if clip:
                continue
            raise ValueError(f"no note at {on} on staff {staff} for the grace note {note_name(step, alter, octave)}")
        g = ET.Element("note")
        ET.SubElement(g, "grace", slash="yes")
        p = ET.SubElement(g, "pitch")
        ET.SubElement(p, "step").text = step
        if alter:
            ET.SubElement(p, "alter").text = str(alter)
        ET.SubElement(p, "octave").text = str(octave)
        ET.SubElement(g, "voice").text = t.findtext("voice", "1")
        ET.SubElement(g, "type").text = "eighth"
        ET.SubElement(g, "staff").text = str(staff)
        m_el.insert(list(m_el).index(t), g)


def _pass_on(measures: list[ET.Element], i: int) -> None:
    """The attributes of measure i (divisions, key, time, staves, clefs) copied into the next measure where it has
    none of its own, before measure i is deleted: a phantom bar copista made at a change of key carries the change,
    and a song's first bar carries its divisions."""
    if i + 1 >= len(measures):
        return
    gone = measures[i].find("attributes")
    if gone is None:
        return
    nxt = measures[i + 1]
    attrs = nxt.find("attributes")
    if attrs is None:
        attrs = ET.Element("attributes")
        nxt.insert(next((k for k, c in enumerate(nxt) if c.tag not in ("print",)), len(nxt)), attrs)
    order = ("divisions", "key", "time", "staves", "clef")
    for el in gone:
        if el.tag in order and not any(
                c.tag == el.tag and c.get("number") == el.get("number") for c in attrs):
            at = next((k for k, c in enumerate(attrs) if c.tag in order and order.index(c.tag) > order.index(el.tag)), len(attrs))
            attrs.insert(at, copy.deepcopy(el))


# ---------------------------------------------------------------- repeats and endings

def set_repeats(root: ET.Element, spec: dict[str, str]) -> list[str]:
    """songs.json's `repeats`: the song's repeat signs and voltas as printed, replacing every part's own. Keys are a
    bar or a range of bars (the song's as it now stands, after the fixes), values tokens: `|:` a forward repeat at
    the range's start, `:|` a backward repeat at its end, `1.` (or `1,2.`) a volta over the range, closed at its end
    (open, "discontinue", unless a `:|` closes it), `x3` with a `:|` to play the passage three times. E.g.
    {"5": "|:", "9-12": "1. :|", "13": "2."}. Returns what was set."""
    parts = root.findall("part")
    piano = piano_part(root) or (parts[-1] if parts else None)
    if piano is None:
        return []
    for p in parts:
        for m in p.findall("measure"):
            for b in m.findall("barline"):
                marks = b.findall("repeat") + b.findall("ending")
                if marks:  # a repeat's barline goes with it; a song's final barline stays
                    m.remove(b)
    measures = piano.findall("measure")
    left: dict[int, dict] = {}  # bar -> {"forward": bool, "volta": str}
    right: dict[int, dict] = {}  # bar -> {"backward": bool, "volta": str}
    done = []
    for key, value in sorted(spec.items(), key=lambda kv: int(kv[0].split("-")[0])):
        lo, _, hi = key.partition("-")
        a, b = int(lo), int(hi or lo)
        if not 1 <= a <= b <= len(measures):
            raise ValueError(f"repeats {key}: the song has {len(measures)} bars")
        tokens = value.split()
        volta = next((t.rstrip(".") for t in tokens if re.fullmatch(r"\d+(,\d+)*\.?", t)), None)
        if "|:" in tokens:
            left.setdefault(a, {})["forward"] = True
        if volta:
            left.setdefault(a, {})["volta"] = volta
            right.setdefault(b, {})["volta"] = volta
        if ":|" in tokens:
            right.setdefault(b, {})["backward"] = True
        times = next((int(t.strip("x")) for t in tokens if re.fullmatch(r"x\d+|\d+x", t)), None)
        if times:
            right.setdefault(b, {})["times"] = times
        done.append(f"{key} {value}")
    for n, marks in left.items():
        bl = _barline(measures[n - 1], "left")
        if marks.get("forward"):
            ET.SubElement(bl, "bar-style").text = "heavy-light"
            ET.SubElement(bl, "repeat", direction="forward")
        if marks.get("volta"):
            ET.SubElement(bl, "ending", number=marks["volta"], type="start")
    for n, marks in right.items():
        br = _barline(measures[n - 1], "right")
        # a volta numbered 1,2 is played twice before the repeat is left: three passes
        times = (marks.get("times") or (max(int(x) for x in marks["volta"].split(",")) + 1 if marks.get("volta") else 2)
                 if marks.get("backward") else 0)
        if times:
            ET.SubElement(br, "bar-style").text = "light-heavy"
        if marks.get("volta"):
            ET.SubElement(br, "ending", number=marks["volta"], type="stop" if times else "discontinue")
        if times:
            ET.SubElement(br, "repeat", {"direction": "backward", **({"times": str(times)} if times > 2 else {})})
    return done


def _barline(m_el: ET.Element, location: str) -> ET.Element:
    for b in m_el.findall("barline"):
        if b.get("location", "right") == location:
            for x in list(b):
                b.remove(x)
            return b
    b = ET.Element("barline", location=location)
    if location == "left":
        m_el.insert(next((k for k, c in enumerate(m_el) if c.tag not in ("print", "attributes")), len(m_el)), b)
    else:
        m_el.append(b)
    return b


# ---------------------------------------------------------------- the vocal line

def check_voices(root: ET.Element, aud_root: ET.Element, report: Report) -> list[str]:
    """Each vocal bar copista read kept only where it can be trusted: Audiveris read the bar the same, or the
    piano's right hand (checked by now) plays its every note (same beat, same note name). Otherwise Audiveris's
    reading of the bar if the piano doubles that; otherwise the bar's notes become rests, so levels 1 and 2 take
    that bar's tune from the piano's top line (the importer does where a voice has nothing). copista reads the
    guitar chord boxes over a songbook's vocal staff as notes. The importer's own test, a bar that adds up, passes every copista bar: copista pads them.
    `report` is the merge's, its bars numbered as the song now stands (apply_fixes keeps them so). Returns what was
    left out, per part."""
    piano = piano_part(root)
    if piano is None:
        return []
    aud_piano = piano_part(aud_root)
    aud_voices = [read_bars(p) for p in aud_root.findall("part") if p is not aud_piano]
    rh = [{(n.onset, n.midi % 12) for n in b.notes if n.staff == 1 and not n.grace} for b in read_bars(piano)]
    names = {sp.get("id"): sp.findtext("part-name", "") for sp in root.iter("score-part")}
    out = []
    for vp in [p for p in root.findall("part") if p is not piano]:
        dropped, taken = [], []
        fine(vp)
        for k, vb in enumerate(read_bars(vp)):
            notes = [n for n in vb.notes if not n.grace]
            if not notes:
                continue
            ib = report.bars[k].audiveris if k < len(report.bars) else []
            same = len(ib) == 1 and any(ib[0] - 1 < len(av) and _lengths(av[ib[0] - 1].notes) == _lengths(notes)
                                        for av in aud_voices)
            def doubled(ns: list[pdf_check.Note]) -> bool:
                ns = [n for n in ns if not n.grace]
                return bool(ns) and k < len(rh) and all((n.onset, n.midi % 12) in rh[k] for n in ns)

            if same or doubled(notes):
                continue
            other = next((av[ib[0] - 1] for av in aud_voices if len(ib) == 1 and ib[0] - 1 < len(av)
                          and av[ib[0] - 1].length == vb.length and doubled(av[ib[0] - 1].notes)), None)
            if other is not None:
                text = bar_text(other.element, other.divisions)
                set_bar(vb.element, text.split(" ", 1)[1] if text.startswith("time=") else text, vb.divisions, vb.length,
                        clip=True)
                taken.append(k + 1)
            else:
                _silence(vb.element)
                dropped.append(k + 1)
        retie(vp)
        name = names.get(vp.get("id")) or vp.get("id")
        if taken:
            out.append(f"{name}: {len(taken)} bars as Audiveris read them (the piano doubles them): {_bar_list(taken)}")
        if dropped:
            out.append(f"{name}: {len(dropped)} bars left to the piano's top line "
                       f"(the readers differ and the piano doesn't double them): {_bar_list(dropped)}")
    return out


def _lengths(notes: list[pdf_check.Note]) -> Counter:
    return Counter((n.onset, n.midi, n.duration) for n in notes if not n.grace)


def _silence(m_el: ET.Element) -> None:
    """A measure's notes made rests of the same lengths (a chord's other notes dropped)."""
    for n in m_el.findall("note"):
        if n.find("chord") is not None or n.find("grace") is not None:
            m_el.remove(n)
            continue
        p = n.find("pitch")
        if p is None:
            continue
        at = list(n).index(p)
        n.remove(p)
        n.insert(at, ET.Element("rest"))
        for tag in ("tie", "notations", "lyric", "accidental", "stem", "beam"):
            for x in n.findall(tag):
                n.remove(x)


def _bar_list(bars: list[int]) -> str:
    runs, start = [], None
    for k, b in enumerate(bars):
        if start is None:
            start = b
        if k + 1 == len(bars) or bars[k + 1] != b + 1:
            runs.append(str(start) if start == b else f"{start}-{b}")
            start = None
    return ", ".join(runs)
