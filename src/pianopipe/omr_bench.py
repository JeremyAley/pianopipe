"""The OMR benchmark's scoring: a reading of a piano score against its correct answer, bar by bar (docs/omr-engine.md §5).

A bar counts as right only when every note of the piano part matches: its hand (the staff it is written on), its onset
in the bar, its written length and its pitch; a grace note on hand, onset (its main note's) and pitch. The piano part is the one the
importer takes its hands from (musicxml.load), so a reading whose piano bars went into other parts, or that has no
two-staff piano part at all, loses those bars, as it would in the app.

The two scores' bars are aligned on their pitches (Needleman-Wunsch), so a lost or added bar costs only that bar.
Each wrong bar gets a cause, the largest of: missing notes, extra notes, pitches an octave off (a clef or 8va),
other pitches, hands, onsets and lengths; a bar with no counterpart is "lost", and one read with no notes "empty".

Also here: the XML steps that make a benchmark page's source from a public-domain score (an excerpt, a sung line over
the piano with lyrics, chord names). Standard library only; engraving and damage are in omr_pages.py.
"""
from __future__ import annotations

import copy
import csv
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction
from pathlib import Path

from . import musicxml

CAUSES = ("lost", "empty", "missing", "extra", "octave", "pitch", "hand", "onset", "duration")
GAP = -0.2  # Needleman-Wunsch: a bar left unmatched (two of these lose to matching any pair in order)


@dataclass(frozen=True, order=True)
class BarNote:
    hand: str  # "R" for the upper staff, "L" for the lower
    onset: Fraction  # quarter notes from the start of the bar
    length: Fraction  # written length in quarter notes; 0 for a grace note
    pitch: int
    grace: bool = False


@dataclass
class Reading:
    """A score's piano bars as written, and each bar's measure number (to match omr-repair's bar numbers)."""
    bars: list[list[BarNote]]
    numbers: list[int]
    problem: str = ""  # why there are no bars (no two-staff piano part, a file that doesn't load)
    pickup: bool = False  # the first bar is shorter than its time signature


@dataclass
class BarResult:
    bar: int  # the correct answer's bar, 1-based
    read: int | None  # the reading's bar matched to it, 1-based (None: lost)
    notes: int  # notes in the correct bar
    read_notes: int
    right: int  # notes matched exactly
    duration: int = 0
    hand: int = 0
    onset: int = 0
    octave: int = 0
    pitch: int = 0
    missing: int = 0
    extra: int = 0
    cause: str = ""  # "" for a right bar, else one of CAUSES
    flagged: bool | None = None  # the chain flagged the read bar for review (None: no flags known)

    @property
    def exact(self) -> bool:
        return not self.cause


@dataclass
class Comparison:
    bars: list[BarResult] = field(default_factory=list)
    extra_bars: list[int] = field(default_factory=list)  # read bars matched to no correct bar, 1-based
    problem: str = ""

    @property
    def right(self) -> int:
        return sum(b.exact for b in self.bars)

    @property
    def score(self) -> float:
        """Bars right over the correct bars plus the bars the reading added."""
        total = len(self.bars) + len(self.extra_bars)
        return self.right / total if total else 1.0

    def causes(self) -> Counter:
        c = Counter(b.cause for b in self.bars if b.cause)
        if self.extra_bars:
            c["added"] = len(self.extra_bars)
        return c


# ---------------------------------------------------------------- reading a score's piano bars

def piano_bars(path: str | Path, answer: bool = False) -> Reading:
    """The piano part's bars as written: the two-staff keyboard part the importer would take its hands from. An
    `answer` may instead be two one-staff parts, the upper the right hand (kern converted by music21 is written so);
    a reading may not: the importer drops those, and so does the app."""
    try:
        score = musicxml.load(path)
    except (musicxml.MusicXMLError, ET.ParseError, OSError, KeyError, ValueError, ZeroDivisionError) as e:
        return Reading([], [], f"doesn't load: {type(e).__name__}: {e}")
    root = musicxml.read_root(path)
    info = musicxml._part_info(root)
    els = root.findall("part")
    if not score.keyboard:
        parts = [musicxml._parse_part(e, info.get(e.get("id", ""), {})) for e in els]
        if not (answer and len(parts) == 2 and all(p.staves == 1 for p in parts)):
            return Reading([], [], f"no two-staff piano part (parts: {', '.join(score.parts)}; hands from {score.hand_source})")
        for m_lo, m_hi in zip(parts[1].measures, parts[0].measures):  # the lower part on the lower staff
            m_hi.notes += [replace(n, staff=2) for n in m_lo.notes]
        el, part = els[0], parts[0]
    else:
        el = next(p for p in els if p.get("id", "") == score.keyboard)
        part = musicxml._parse_part(el, info.get(score.keyboard, {}))
    bars, numbers = [], []
    first = part.measures[0] if part.measures else None
    pickup = bool(first and first.time and first.length < Fraction(4 * first.time[0], first.time[1]))
    for k, (m, m_el) in enumerate(zip(part.measures, el.findall("measure"))):
        bars.append(sorted(BarNote("R" if n.staff == 1 else "L", n.pos, Fraction(0) if n.grace else n.dur, n.pitch,
                                   bool(n.grace)) for n in m.notes))
        num = re.match(r"\d+", m_el.get("number", ""))
        numbers.append(int(num.group()) if num else k + 1)
    return Reading(bars, numbers, pickup=pickup)


# ---------------------------------------------------------------- comparing

def _similarity(a: list[BarNote], b: list[BarNote]) -> float:
    if not a and not b:
        return 1.0
    pa, pb = Counter(n.pitch for n in a), Counter(n.pitch for n in b)
    same = sum((Counter(a) & Counter(b)).values())  # breaks ties between bars of the same pitches
    return (2 * sum((pa & pb).values()) + 0.01 * 2 * same) / (len(a) + len(b))


def align(truth: list[list[BarNote]], read: list[list[BarNote]]) -> list[tuple[int | None, int | None]]:
    """Pairs (truth index, read index) in order; None on one side for a bar the other lacks."""
    n, m = len(truth), len(read)
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    step = [[""] * (m + 1) for _ in range(n + 1)]  # how each cell was reached: "d"iagonal, "u"p (truth bar alone), "l"eft
    for i in range(1, n + 1):
        score[i][0], step[i][0] = i * GAP, "u"
    for j in range(1, m + 1):
        score[0][j], step[0][j] = j * GAP, "l"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score[i][j], step[i][j] = max((score[i - 1][j - 1] + _similarity(truth[i - 1], read[j - 1]), "d"),
                                          (score[i - 1][j] + GAP, "u"), (score[i][j - 1] + GAP, "l"),
                                          key=lambda t: t[0])  # the first of equals: the diagonal
    pairs = []
    i, j = n, m
    while i or j:
        if step[i][j] == "d":
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif step[i][j] == "u":
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    return pairs[::-1]


def _take(left: list[BarNote], right: list[BarNote], key) -> int:
    """Removes pairs that agree on key() from both lists; returns how many."""
    pool: dict = {}
    for k, n in enumerate(right):
        pool.setdefault(key(n), []).append(k)
    used_l, used_r = [], set()
    for k, n in enumerate(left):
        free = pool.get(key(n))
        if free:
            used_l.append(k)
            used_r.add(free.pop(0))
    for k in reversed(used_l):
        del left[k]
    right[:] = [n for k, n in enumerate(right) if k not in used_r]
    return len(used_l)


def compare_bar(truth: list[BarNote], read: list[BarNote]) -> BarResult:
    """Note by note: exact matches first, then the closest kind of mistake for what's left."""
    t, r = list(truth), list(read)
    res = BarResult(0, None, len(truth), len(read), 0)
    res.right = _take(t, r, lambda n: (n.hand, n.onset, n.length, n.pitch, n.grace))
    res.duration = _take(t, r, lambda n: (n.hand, n.onset, n.pitch, n.grace))
    res.hand = _take(t, r, lambda n: (n.onset, n.pitch, n.grace))
    res.octave = _take(t, r, lambda n: (n.hand, n.onset, n.pitch % 12, n.grace))
    res.onset = _take(t, r, lambda n: (n.hand, n.pitch, n.grace))
    res.pitch = _take(t, r, lambda n: (n.hand, n.onset, n.grace))
    res.missing, res.extra = len(t), len(r)
    if not read and truth:
        res.cause = "empty"
    else:
        counts = {c: getattr(res, c) for c in ("missing", "extra", "octave", "pitch", "hand", "onset", "duration")}
        worst = max(counts.values())
        res.cause = next(c for c, v in counts.items() if v == worst) if worst else ""
    return res


def _pickup(truth: list[BarNote], read: list[BarNote]) -> BarResult | None:
    """A pickup bar read as a full bar with rests before its notes: right if that's the only difference."""
    if not truth or not read:
        return None
    shift = min(n.onset for n in read) - min(n.onset for n in truth)
    if shift <= 0:
        return None
    res = compare_bar(truth, [BarNote(n.hand, n.onset - shift, n.length, n.pitch, n.grace) for n in read])
    return None if res.cause else res


def compare(truth: Reading, read: Reading, only: set[int] | None = None, flagged: set[int] | None = None) -> Comparison:
    """Every correct bar (or those numbered in `only`, 1-based) against the reading. `flagged`: the read bars' measure
    numbers the chain flagged for review."""
    out = Comparison(problem=read.problem)
    pairs = align(truth.bars, read.bars) if read.bars else [(i, None) for i in range(len(truth.bars))]
    for ti, ri in pairs:
        if ti is None:
            if only is None:
                out.extra_bars.append(ri + 1)
            continue
        if only is not None and ti + 1 not in only:
            continue
        if ri is None:
            res = BarResult(ti + 1, None, len(truth.bars[ti]), 0, 0, missing=len(truth.bars[ti]), cause="lost",
                            flagged=None if flagged is None else False)
        else:
            res = compare_bar(truth.bars[ti], read.bars[ri])
            if ti == ri == 0 and res.cause and truth.pickup:
                res = _pickup(truth.bars[0], read.bars[0]) or res
            res.bar, res.read = ti + 1, ri + 1
            if flagged is not None:
                res.flagged = read.numbers[ri] in flagged
        out.bars.append(res)
    return out


# ---------------------------------------------------------------- results

ROW = ["case", "source", "engraver", "layout", "damage", "stage", "bar", "read", "exact", "cause", "flagged", "notes",
       "read_notes", "right", "duration", "hand", "onset", "octave", "pitch", "missing", "extra"]


def rows(case: dict, stage: str, cmp: Comparison) -> list[dict]:
    head = {k: case.get(k, "") for k in ("case", "source", "engraver", "layout", "damage")}
    out = []
    for b in cmp.bars:
        d = asdict(b)
        out.append({**head, "stage": stage, **{k: d[k] for k in ROW if k in d}, "exact": int(b.exact),
                    "flagged": "" if b.flagged is None else int(b.flagged)})
    for k in cmp.extra_bars:
        out.append({**head, "stage": stage, "bar": "", "read": k, "exact": 0, "cause": "added"})
    return out


def write_csv(path: str | Path, all_rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ROW, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)


def summarise(all_rows: list[dict], by: str, stage: str) -> list[dict]:
    """Per value of `by` (damage, engraver...): bars, right, added, the score, and the bars each cause cost."""
    groups: dict[str, list[dict]] = {}
    for r in all_rows:
        if r["stage"] == stage:
            groups.setdefault(str(r[by]), []).append(r)
    out = []
    for key, rs in groups.items():
        bars = [r for r in rs if r["cause"] != "added"]
        added = len(rs) - len(bars)
        right = sum(int(r["exact"]) for r in bars)
        causes = Counter(r["cause"] for r in rs if r["cause"])
        wrong = [r for r in bars if not int(r["exact"])]
        flagged = [r for r in bars if r.get("flagged") not in ("", None)]
        out.append({
            by: key, "cases": len({r["case"] for r in rs}), "bars": len(bars), "right": right, "added": added,
            "score": right / (len(bars) + added) if bars or added else 1.0, "causes": dict(causes.most_common()),
            # the review flags: wrong bars nobody would look at, and right bars flagged anyway
            "wrong_unflagged": sum(1 for r in wrong if r.get("flagged") == 0),
            "right_flagged": sum(1 for r in flagged if int(r["exact"]) and r["flagged"] == 1),
            "flagged": sum(1 for r in flagged if r["flagged"] == 1),
        })
    return sorted(out, key=lambda d: d["score"])


def lost_to(all_rows: list[dict], stage: str, base: str = "clean") -> list[dict]:
    """Per damage: bars right on the clean page but wrong on the damaged one, over the same pieces and engravings, so
    the damages can be ranked by what they cost. `added`: bars added beyond those the clean reading added."""
    key = lambda r: (r["source"], r["engraver"], r["layout"])  # noqa: E731
    clean = {key(r) + (r["bar"],): int(r["exact"])
             for r in all_rows if r["stage"] == stage and r["damage"] == base and r["cause"] != "added"}
    clean_added = Counter(key(r) for r in all_rows if r["stage"] == stage and r["damage"] == base and r["cause"] == "added")
    out: dict[str, dict] = {}
    added: dict[str, Counter] = {}
    seen: dict[str, set] = {}  # each damage's pieces and engravings
    for r in all_rows:
        if r["stage"] != stage or r["damage"] == base:
            continue
        d = out.setdefault(r["damage"], {"damage": r["damage"], "paired": 0, "lost": 0, "gained": 0, "added": 0})
        seen.setdefault(r["damage"], set()).add(key(r))
        if r["cause"] == "added":
            added.setdefault(r["damage"], Counter())[key(r)] += 1
            continue
        was = clean.get(key(r) + (r["bar"],))
        if was is None:
            continue
        d["paired"] += 1
        d["lost"] += was and not int(r["exact"])
        d["gained"] += (not was) and int(r["exact"])
    for dmg, d in out.items():
        d["added"] = sum(added.get(dmg, Counter())[k] - clean_added[k] for k in seen[dmg])
        d["cost"] = (d["lost"] - d["gained"] + d["added"]) / d["paired"] if d["paired"] else 0.0
    return sorted(out.values(), key=lambda d: -d["cost"])


# ---------------------------------------------------------------- making a benchmark page's source

def excerpt(root: ET.Element, bars: int) -> ET.Element:
    """The first `bars` measures of every part, without the source's page and system breaks (the engraver lays it
    out) and ending with a final barline."""
    out = copy.deepcopy(root)
    for part in out.findall("part"):
        for m in part.findall("measure")[bars:]:
            part.remove(m)
        for m in part.findall("measure"):
            for p in m.findall("print"):
                m.remove(p)
        last = part.findall("measure")[-1]
        for b in last.findall("barline"):
            if b.get("location", "right") == "right":
                last.remove(b)
        bl = ET.SubElement(last, "barline", {"location": "right"})
        ET.SubElement(bl, "bar-style").text = "light-heavy"
    return out


LYRICS = "Sing we now of glad - ness, joy and peace to all the world, while the bells ring out a - bove us".split()
_TYPES = [(Fraction(4), "whole", 0), (Fraction(3), "half", 1), (Fraction(2), "half", 0), (Fraction(3, 2), "quarter", 1),
          (Fraction(1), "quarter", 0), (Fraction(3, 4), "eighth", 1), (Fraction(1, 2), "eighth", 0),
          (Fraction(1, 4), "16th", 0), (Fraction(1, 8), "32nd", 0)]


def _rest(length: Fraction, divisions: int) -> list[ET.Element]:
    """Rests filling `length` quarter notes, longest first."""
    out = []
    while length > 0:
        q, kind, dots = next(((q, k, d) for q, k, d in _TYPES if q <= length), (length, None, 0))
        n = ET.Element("note")
        ET.SubElement(n, "rest")
        ET.SubElement(n, "duration").text = str(int(q * divisions))
        ET.SubElement(n, "voice").text = "1"
        if kind:
            ET.SubElement(n, "type").text = kind
            for _ in range(dots):
                ET.SubElement(n, "dot")
        out.append(n)
        length -= q
    return out


def _measure_length(attrs: dict) -> Fraction:
    return Fraction(4 * attrs.get("beats", 4), attrs.get("beat_type", 4))


def add_voice(root: ET.Element, chords: bool = True) -> ET.Element:
    """A songbook page from a piano score: a sung line over the piano (its top line, with words under it) and chord
    names over the piano, as in a piano/vocal/guitar book. The piano part is left as it was: it stays the answer."""
    out = copy.deepcopy(root)
    piano = next((p for p in out.findall("part") if p.find("measure/attributes/staves") is not None), None)
    if piano is None:
        return out
    pl = out.find("part-list")
    sp = ET.Element("score-part", {"id": "Pvoc"})
    ET.SubElement(sp, "part-name").text = "Voice"
    pl.insert(0, sp)
    voc = ET.Element("part", {"id": "Pvoc"})
    out.insert(list(out).index(piano), voc)
    attrs: dict = {"divisions": 1}
    word = 0
    for m_el in piano.findall("measure"):
        vm = ET.SubElement(voc, "measure", {k: v for k, v in m_el.attrib.items() if k in ("number", "implicit")})
        pos, last, first_voice, sung = Fraction(0), Fraction(0), None, []
        bass: tuple[Fraction, ET.Element] | None = None
        for el in m_el:
            if el.tag == "attributes":
                d = el.findtext("divisions")
                if d:
                    attrs["divisions"] = int(d)
                t = el.find("time")
                if t is not None and t.findtext("beats", "").isdigit():
                    attrs["beats"], attrs["beat_type"] = int(t.findtext("beats")), int(t.findtext("beat-type", "4"))
                va = ET.SubElement(vm, "attributes")
                for tag in ("divisions", "key", "time"):
                    if el.find(tag) is not None:
                        va.append(copy.deepcopy(el.find(tag)))
                if el.find("clef") is not None:
                    clef = ET.SubElement(va, "clef")
                    ET.SubElement(clef, "sign").text = "G"
                    ET.SubElement(clef, "line").text = "2"
                if not len(va):
                    vm.remove(va)
            elif el.tag in ("backup", "forward"):
                d = Fraction(int(el.findtext("duration", "0")), attrs["divisions"])
                pos = max(Fraction(0), pos - d) if el.tag == "backup" else pos + d
            elif el.tag == "note":
                chord, grace = el.find("chord") is not None, el.find("grace") is not None
                d = Fraction(int(el.findtext("duration", "0")), attrs["divisions"])
                start = last if chord else pos
                if not chord and not grace:
                    last, pos = pos, pos + d
                staff = el.findtext("staff", "1").strip()
                voice = el.findtext("voice", "1").strip()
                if staff == "2" and el.find("pitch") is not None and not grace and (bass is None or start < bass[0]):
                    bass = (start, el.find("pitch"))
                if staff != "1" or grace or chord or el.find("cue") is not None:
                    continue
                first_voice = first_voice or voice
                if voice == first_voice:
                    sung.append((start, d, el))
        length = _measure_length(attrs)
        at = Fraction(0)
        for start, d, el in sung:
            if start < at:
                continue
            for r in _rest(start - at, attrs["divisions"]):
                vm.append(r)
            n = ET.SubElement(vm, "note")
            for tag in ("pitch", "rest", "duration"):
                if el.find(tag) is not None:
                    n.append(copy.deepcopy(el.find(tag)))
            for t in el.findall("tie"):
                n.append(copy.deepcopy(t))
            ET.SubElement(n, "voice").text = "1"
            for tag in ("type", "dot", "accidental", "time-modification"):
                for x in el.findall(tag):
                    n.append(copy.deepcopy(x))
            ties = {t.get("type") for t in el.findall("tie")}
            if el.find("pitch") is not None and "stop" not in ties:
                ly = ET.SubElement(n, "lyric", {"number": "1"})
                ET.SubElement(ly, "syllabic").text = "single"
                ET.SubElement(ly, "text").text = LYRICS[word % len(LYRICS)]
                word += 1
            at = start + d
        for r in _rest(length - at, attrs["divisions"]) if at < length else []:
            vm.append(r)
        if chords and bass is not None:
            h = ET.Element("harmony")
            rt = ET.SubElement(h, "root")
            ET.SubElement(rt, "root-step").text = bass[1].findtext("step", "C")
            if bass[1].findtext("alter") not in (None, "0"):
                ET.SubElement(rt, "root-alter").text = bass[1].findtext("alter")
            ET.SubElement(h, "kind").text = "major"
            first_note = next((k for k, x in enumerate(m_el) if x.tag == "note"), len(m_el))
            m_el.insert(first_note, h)
    return out
