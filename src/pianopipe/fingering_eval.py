"""Hand-written fingering test cases and an agreement score for fingering.py.

The PIG fingering dataset allows only nonprofit academic use (docs/decisions.md #12), so the model is checked
against standard textbook fingerings written out here: scales, five-finger tunes, arpeggios, chords, jumps and
the opening of the Minuet in G (BWV Anh. 114). Where editions differ, a note lists every accepted finger.
`pianopipe eval-fingering` prints the report. At the time of writing, 90.5% of notes agree, and these are the
known differences (all playable, none the textbook choice):
- Two-octave left-hand C major puts the 4-over-thumb crossing in the first octave instead of the second (the two
  cost the same).
- Chromatic scales use groups of 1-2-3(-4) with thumbs on white keys (close to the "modern" chromatic fingering)
  instead of the classic 1-3 alternation, which the span rules price higher.
- A harmonic minor right hand ends 1-2-3-4 on top (the augmented second fits 2-3) instead of 3-4-5.
- An octave bass line takes the fifth below with 4 rather than 5; the root chord after V in the left hand takes
  4-2-1.
- The minuet's bar 7 stays in position (F#-G-A-B-G as 2-3-4-5-3) instead of crossing.
The model plans one hand at a time and looks only one onset back for its costs, so it doesn't see phrase ends,
and it judges legato from the MIDI note lengths, which score-generated files make fully legato.

Notation: notes are `C4` (middle C) / `F#5` / `Bb3`, chords are `C4+E4+G4`, `r` is a rest, and
`:0.5` after a note sets its length in beats (default 1). Fingers follow the same order, `1+3+5` for a chord in
the order its notes are written, `3|4` when either is fine.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from .difficulty import make_levels
from .fingering import FINGERS, MIRROR, REPEAT_WINDOW, SPANS, _black, _onsets, assign_fingering, margins
from .model import Note
from .process import load_notes

TPB = 480
MARGIN_BUCKETS = (0, 1, 2, 4, 8)
_NAME = re.compile(r"^([A-G])([#b]?)(-?\d)$")
_STEP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def pitch(name: str) -> int:
    m = _NAME.match(name)
    if not m:
        raise ValueError(f"bad note name {name!r}")
    letter, accidental, octave = m.groups()
    return 12 * (int(octave) + 1) + _STEP[letter] + {"#": 1, "b": -1, "": 0}[accidental]


@dataclass
class Case:
    name: str
    hand: str
    notes: str
    fingers: str
    group: str = ""

    def build(self) -> tuple[list[Note], list[set[int]]]:
        """The notes (each event `length` beats long, legato) and the accepted fingers for each note, in order."""
        notes, accepted, tick = [], [], 0
        tokens, answers = self.notes.split(), self.fingers.split()
        if len([t for t in tokens if not t.startswith("r")]) != len(answers):
            raise ValueError(f"{self.name}: {len(tokens)} events but {len(answers)} fingerings")
        answers_iter = iter(answers)
        for token in tokens:
            name, _, length = token.partition(":")
            ticks = round(float(length or 1) * TPB)
            if name != "r":
                options = next(answers_iter).split("+")
                for note_name, option in zip(name.split("+"), options, strict=True):
                    notes.append(Note(tick, tick + ticks, pitch(note_name), hand=self.hand))
                    accepted.append({int(f) for f in option.split("|")})
            tick += ticks
        return notes, accepted


CASES = [
    # Scales (ABRSM / Hanon fingerings).
    Case("C major RH up, 2 oct", "R", "C4 D4 E4 F4 G4 A4 B4 C5 D5 E5 F5 G5 A5 B5 C6", "1 2 3 1 2 3 4 1 2 3 1 2 3 4 5", "scale"),
    Case("C major RH down, 2 oct", "R", "C6 B5 A5 G5 F5 E5 D5 C5 B4 A4 G4 F4 E4 D4 C4", "5 4 3 2 1 3 2 1 4 3 2 1 3 2 1", "scale"),
    Case("C major LH up, 2 oct", "L", "C3 D3 E3 F3 G3 A3 B3 C4 D4 E4 F4 G4 A4 B4 C5", "5 4 3 2 1 3 2 1 4 3 2 1 3 2 1", "scale"),
    Case("C major LH down, 2 oct", "L", "C5 B4 A4 G4 F4 E4 D4 C4 B3 A3 G3 F3 E3 D3 C3", "1 2 3 1 2 3 4 1 2 3 1 2 3 4 5", "scale"),
    Case("C major RH up and down", "R", "C4 D4 E4 F4 G4 A4 B4 C5 B4 A4 G4 F4 E4 D4 C4", "1 2 3 1 2 3 4 5 4 3 2 1 3 2 1", "scale"),
    Case("C major LH up and down", "L", "C3 D3 E3 F3 G3 A3 B3 C4 B3 A3 G3 F3 E3 D3 C3", "5 4 3 2 1 3 2 1 2 3 1 2 3 4 5", "scale"),
    Case("G major RH up", "R", "G4 A4 B4 C5 D5 E5 F#5 G5", "1 2 3 1 2 3 4 5", "scale"),
    Case("G major LH up", "L", "G2 A2 B2 C3 D3 E3 F#3 G3", "5 4 3 2 1 3 2 1", "scale"),
    Case("D major RH up", "R", "D4 E4 F#4 G4 A4 B4 C#5 D5", "1 2 3 1 2 3 4 5", "scale"),
    Case("D major LH up", "L", "D3 E3 F#3 G3 A3 B3 C#4 D4", "5 4 3 2 1 3 2 1", "scale"),
    Case("F major RH up", "R", "F4 G4 A4 Bb4 C5 D5 E5 F5", "1 2 3 4 1 2 3 4", "scale"),
    Case("F major LH up", "L", "F2 G2 A2 Bb2 C3 D3 E3 F3", "5 4 3 2 1 3 2 1", "scale"),
    Case("A harmonic minor RH up", "R", "A4 B4 C5 D5 E5 F5 G#5 A5", "1 2 3 1 2 3 4 5", "scale"),
    Case("A harmonic minor LH up", "L", "A2 B2 C3 D3 E3 F3 G#3 A3", "5 4 3 2 1 3 2 1", "scale"),
    Case("Chromatic RH up", "R", "C4:0.25 C#4:0.25 D4:0.25 D#4:0.25 E4:0.25 F4:0.25 F#4:0.25 G4:0.25 G#4:0.25 A4:0.25 A#4:0.25 B4:0.25 C5", "1 3 1 3 1 2 3 1 3 1 3 1 2", "scale"),
    Case("Chromatic RH down", "R", "C5:0.25 B4:0.25 A#4:0.25 A4:0.25 G#4:0.25 G4:0.25 F#4:0.25 F4:0.25 E4:0.25 D#4:0.25 D4:0.25 C#4:0.25 C4", "1|2 1 3 1 3 1 3 2 1 3 1 3 1", "scale"),
    Case("Chromatic LH up", "L", "C3:0.25 C#3:0.25 D3:0.25 D#3:0.25 E3:0.25 F3:0.25 F#3:0.25 G3:0.25 G#3:0.25 A3:0.25 A#3:0.25 B3:0.25 C4", "1 3 1 3 2 1 3 1 3 1 3 2 1", "scale"),
    # Five-finger patterns and tunes.
    Case("C position RH", "R", "C4 D4 E4 F4 G4 F4 E4 D4 C4", "1 2 3 4 5 4 3 2 1", "five-finger"),
    Case("C position LH", "L", "C3 D3 E3 F3 G3 F3 E3 D3 C3", "5 4 3 2 1 2 3 4 5", "five-finger"),
    Case("G position RH skips", "R", "G4 B4 A4 C5 B4 D5 C5 A4 G4", "1 3 2 4 3 5 4 2 1", "five-finger"),
    Case("Ode to Joy RH", "R", "E4 E4 F4 G4 G4 F4 E4 D4 C4 C4 D4 E4 E4:1.5 D4:0.5 D4:2", "3 3 4 5 5 4 3 2 1 1 2 3 3 2 2", "five-finger"),
    Case("Mary had a little lamb RH", "R", "E4 D4 C4 D4 E4 E4 E4:2 D4 D4 D4:2 E4 G4 G4:2", "3 2 1 2 3 3 3 2 2 2 3 5 5", "five-finger"),
    # Arpeggios and broken chords.
    Case("C major arpeggio RH up", "R", "C4:0.5 E4:0.5 G4:0.5 C5:0.5 E5:0.5 G5:0.5 C6", "1 2 3 1 2 3 5", "arpeggio"),
    Case("C major arpeggio RH down", "R", "C6:0.5 G5:0.5 E5:0.5 C5:0.5 G4:0.5 E4:0.5 C4", "5 3 2 1 3 2 1", "arpeggio"),
    Case("C major arpeggio LH up", "L", "C3:0.5 E3:0.5 G3:0.5 C4:0.5 E4:0.5 G4:0.5 C5", "5 4|3 2 1 4|3 2 1", "arpeggio"),
    Case("A minor arpeggio RH up", "R", "A3:0.5 C4:0.5 E4:0.5 A4:0.5 C5:0.5 E5:0.5 A5", "1 2 3 1 2 3 5", "arpeggio"),
    Case("C broken chord RH", "R", "C4 E4 G4 E4 C4", "1 3 5 3 1", "arpeggio"),
    Case("Alberti bass C LH", "L", "C3:0.25 G3:0.25 E3:0.25 G3:0.25 C3:0.25 G3:0.25 E3:0.25 G3:0.25", "5 1 3 1 5 1 3 1", "arpeggio"),
    Case("Alberti bass F/C LH", "L", "C3:0.25 A3:0.25 F3:0.25 A3:0.25 C3:0.25 A3:0.25 F3:0.25 A3:0.25", "5 1 2|3 1 5 1 2|3 1", "arpeggio"),
    # Chords.
    Case("C triad inversions RH", "R", "C4+E4+G4 r E4+G4+C5 r G4+C5+E5", "1+3+5 1+2+5 1+3+5", "chord"),
    Case("C triad inversions LH", "L", "C3+E3+G3 r E3+G3+C4 r G3+C4+E4", "5+3+1 5+3|2+1 5+2+1", "chord"),
    Case("I IV V I RH", "R", "C4+E4+G4 C4+F4+A4 B3+D4+G4 C4+E4+G4", "1+3+5 1+4|3+5 1+2+5 1+3+5", "chord"),
    Case("I IV V I LH", "L", "C3+E3+G3 C3+F3+A3 B2+D3+G3 C3+E3+G3", "5+3+1 5+2+1 5+3|4+1 5+3+1", "chord"),
    Case("Octaves RH", "R", "C4+C5 D4+D5 E4+E5 G4+G5", "1+5 1+5 1+5 1+5", "chord"),
    # Jumps.
    Case("Jump up then down RH", "R", "C4 D4 E4 F4 G4 C6 B5 A5 G5 F5", "1 2 3 4 5 5 4 3 2 1", "jump"),
    Case("Jump down then up RH", "R", "G5 F5 E5 D5 C5 C4 D4 E4 F4 G4", "5 4 3 2 1 1 2 3 4 5", "jump"),
    Case("Octave bass LH", "L", "C2 C3 G2 G3 C2 C3", "5 1 5 1 5 1", "jump"),
    # Minuet in G (Petzold, BWV Anh. 114), bars 1-8; common teaching-edition fingering.
    Case(
        "Minuet in G RH bars 1-8", "R",
        "D5 G4:0.5 A4:0.5 B4:0.5 C5:0.5 D5 G4 G4 E5 C5:0.5 D5:0.5 E5:0.5 F#5:0.5 G5 G4 G4 "
        "C5 D5:0.5 C5:0.5 B4:0.5 A4:0.5 B4 C5:0.5 B4:0.5 A4:0.5 G4:0.5 F#4 G4:0.5 A4:0.5 B4:0.5 G4:0.5 A4:3",
        "5 1 2 3 4 5 1 1 3 1 2 3 4 5 1 1 4 5 4 3 2 3 4 3 2 1 2 1 2 3 1 2",
        "minuet",
    ),
    Case(
        "Minuet in G LH bars 1-8", "L",
        "G3+B3+D4:2 A3 B3:3 C4:3 B3:3 A3:3 G3:3 D4 B3 G3 D4:2 D3",
        "5+3+1 4 3 2 3 4 5 1 3|2 5|4 1 5",
        "minuet",
    ),
]


@dataclass
class Result:
    case: Case
    predicted: list[int]
    accepted: list[set[int]]

    @property
    def agree(self) -> int:
        return sum(p in a for p, a in zip(self.predicted, self.accepted))

    @property
    def total(self) -> int:
        return len(self.accepted)


def run_case(case: Case) -> Result:
    notes, accepted = case.build()
    fingered = assign_fingering(notes, TPB)
    return Result(case, [n.finger for n in fingered], accepted)


def evaluate(cases: list[Case] | None = None) -> list[Result]:
    return [run_case(c) for c in (cases or CASES)]


def printed_pieces(raw: Path, sources: Path, fingering_dir: Path | None = None, only_printed: bool = True) -> list[Piece]:
    """The catalog's pieces as the build reads them (hands, printed fingers on `Note.finger`): every manifest entry
    whose file is under `raw`, with the Mutopia fingers from `fingering_dir` (data/catalog/fingering). With
    `only_printed`, just the pieces that have printed fingers."""
    out = []
    for e in json.loads(Path(sources).read_text(encoding="utf-8")):
        src = Path(raw) / e["file"]
        if not src.exists():
            continue
        song, notes, _, _ = load_notes(src, satb=bool(e.get("satb")))
        f = Path(fingering_dir) / f"{e['id']}.json" if fingering_dir else None
        if f is not None and f.exists():
            given = {(t, p): fg for t, p, fg in json.loads(f.read_text(encoding="utf-8"))["fingers"]}
            notes = [replace(n, finger=given.get((n.start, n.pitch), n.finger)) for n in notes]
        if not only_printed or any(n.finger in FINGERS for n in notes):
            out.append(Piece(e["id"], song.ticks_per_beat, notes, song.meta))
    return out


@dataclass
class Piece:
    id: str
    ticks_per_beat: int
    notes: list[Note]
    meta: list | tuple = ()  # the song's tempo, time and key signatures, which make_levels reads


@dataclass
class Tally:
    n: int = 0
    exact: int = 0
    near: int = 0  # within one finger

    def add(self, printed: int, got: int) -> None:
        self.n += 1
        self.exact += printed == got
        self.near += abs(printed - got) <= 1


def evaluate_printed(pieces: list[Piece]) -> dict[str, Tally]:
    """Generated against printed fingers: each piece fingered as if it had none (assign_fingering, as the catalog
    fingers a piece without them), scored on the notes the edition fingers, by kind of spot. `margin a-b` groups the
    notes by search margin (fingering.margins), the model's own confidence."""
    tallies: dict[str, Tally] = {}
    for piece in pieces:
        got = assign_fingering([replace(n, finger=None) for n in piece.notes], piece.ticks_per_beat)
        by_note = {id(n): g.finger for n, g in zip(piece.notes, got)}
        for hand in ("R", "L"):
            hn = sorted((n for n in piece.notes if n.hand == hand), key=lambda n: (n.start, n.pitch))
            _, margin = margins([replace(n, finger=None) for n in hn], hand, piece.ticks_per_beat)
            onsets = _onsets(hn, piece.ticks_per_beat)
            for k, chord in enumerate(onsets):
                for i in chord:
                    n = hn[i]
                    if n.finger not in FINGERS:
                        continue
                    kinds = ["all", f"hand {hand}", "chord" if len(chord) > 1 else "single", "black key" if _black(n.pitch) else "white key"]
                    if k and len(chord) == 1 and len(onsets[k - 1]) == 1:
                        prev = hn[onsets[k - 1][0]]
                        d = (n.pitch - prev.pitch) * (1 if hand == "R" else -1)
                        kinds.append("repeated note" if d == 0 else "step" if abs(d) <= 2 else "skip" if abs(d) <= 7 else "leap")
                        if prev.finger in FINGERS and d * (n.finger - prev.finger) < 0:
                            kinds.append("printed crossing")
                    if n.finger == 1 and _black(n.pitch):
                        kinds.append("printed thumb on black")
                    lo = max((b for b in MARGIN_BUCKETS if b <= margin[i]), default=0)
                    kinds.append(f"margin {lo}+")
                    for kind in kinds:
                        tallies.setdefault(kind, Tally()).add(n.finger, by_note[id(n)])
    return tallies


def wrong_hand_chords(pieces: list[Piece]) -> tuple[int, int]:
    """(onsets with two or more printed fingers in one hand, those whose printed fingers don't rise with the pitch in
    that hand): an edition's chord that the hand can't play as split is a note from the other hand."""
    total = wrong = 0
    for piece in pieces:
        for hand, sign in (("R", 1), ("L", -1)):
            hn = sorted((n for n in piece.notes if n.hand == hand), key=lambda n: (n.start, n.pitch))
            for chord in _onsets(hn, piece.ticks_per_beat):
                fingers = [hn[i].finger for i in sorted(chord, key=lambda i: sign * hn[i].pitch) if hn[i].finger in FINGERS]
                if len(fingers) > 1:
                    total += 1
                    wrong += any(a >= b for a, b in zip(fingers, fingers[1:]))
    return total, wrong


def lint(pieces: list[Piece], levels: bool = False) -> Counter:
    """Generated fingering checked against standard practice, on every piece (and with `levels`, every level as the
    catalog makes it): counts of spots a teacher would question. Only generated fingers count (printed ones are the
    edition's); legato pairs are single notes with less than a quarter beat between them, within a fifth."""
    c: Counter = Counter()
    for piece in pieces:
        tpb = piece.ticks_per_beat
        for notes in (make_levels(piece.notes, tpb, piece.meta).values() if levels else [piece.notes]):
            fingered = assign_fingering(notes, tpb)
            for hand in ("R", "L"):
                hn = sorted((n for n in fingered if n.hand == hand), key=lambda n: (n.start, n.pitch))
                if hn:
                    _lint_hand(hn, hand, tpb, c)
    return c


def _lint_hand(hn: list[Note], hand: str, tpb: int, c: Counter) -> None:
    rp = (lambda p: p) if hand == "R" else (lambda p: MIRROR - p)
    onsets = [[hn[i] for i in chord] for chord in _onsets(hn, tpb)]
    c["generated fingers"] += sum(not n.printed for n in hn)
    for k, chord in enumerate(onsets):
        if len(chord) > 1:
            if all(n.printed for n in chord):
                continue  # the edition's chord
            low, high = min(chord, key=lambda n: rp(n.pitch)), max(chord, key=lambda n: rp(n.pitch))
            span = rp(high.pitch) - rp(low.pitch)
            if span > 16:
                c["chord over a tenth in one hand (hand split)"] += 1
            elif low.finger < high.finger and span > SPANS[(low.finger, high.finger)][5]:
                c["chord beyond the practical span"] += 1
            continue
        b = chord[0]
        if b.finger == 1 and _black(b.pitch) and not b.printed:
            near = [o[0] for o in (onsets[k - 1] if k else [], onsets[k + 1] if k + 1 < len(onsets) else [])
                    if len(o) == 1 and abs(o[0].pitch - b.pitch) <= 4 and not _black(o[0].pitch)]
            if near:
                c["thumb on a black key in a line"] += 1
        if not k or len(onsets[k - 1]) != 1:
            continue
        a = onsets[k - 1][0]
        d = rp(b.pitch) - rp(a.pitch)
        if (a.printed and b.printed) or b.start - a.end >= max(1, tpb // 4) or abs(d) > 7:
            continue
        c["legato pairs"] += 1
        if d and d * (b.finger - a.finger) < 0:
            lo, hi = sorted((a.finger, b.finger))
            c["thumb crossing" if lo == 1 and hi < 5 else "5 crossing the thumb" if lo == 1 else "finger over finger"] += 1
        if d and a.finger == b.finger:
            c["same finger on two keys legato"] += 1
        if d and a.finger != b.finger:
            lo, hi = sorted((a.finger, b.finger))
            span = abs(d) if (b.finger > a.finger) == (d > 0) else -abs(d)
            if not SPANS[(lo, hi)][0] <= span <= SPANS[(lo, hi)][5]:
                c["step beyond the practical span"] += 1
    # The same REPEAT_WINDOW onsets (pitches and rhythm) coming back with other fingers than the first time.
    seen: dict[tuple, tuple] = {}
    for k in range(len(onsets) - REPEAT_WINDOW + 1):
        run = onsets[k:k + REPEAT_WINDOW]
        t0 = run[0][0].start
        sig = tuple((n.start - t0, n.pitch) for o in run for n in o)
        fingers = tuple(n.finger for o in run for n in o)
        if sig in seen:
            c["repeated passages"] += 1
            c["repeat fingered differently"] += fingers != seen[sig]
        else:
            seen[sig] = fingers


def printed_report(tallies: dict[str, Tally], wrong_hand: tuple[int, int]) -> str:
    lines = [f"{'spot':26} {'printed':>7} {'same':>6} {'within one':>10}"]
    order = ["all", "hand R", "hand L", "single", "chord", "white key", "black key", "step", "skip", "leap",
             "repeated note", "printed crossing", "printed thumb on black"] + [f"margin {b}+" for b in MARGIN_BUCKETS]
    for kind in (k for k in order if k in tallies):
        t = tallies[kind]
        lines.append(f"{kind:26} {t.n:7d} {t.exact / t.n:6.1%} {t.near / t.n:10.1%}")
    total, wrong = wrong_hand
    lines.append(f"printed chords against the hand split: {wrong} of {total}")
    return "\n".join(lines)


def lint_report(c: Counter) -> str:
    n = c["generated fingers"]
    lines = [f"{n} generated fingers, {c['legato pairs']} legato pairs"]
    for kind, v in sorted(c.items()):
        if kind not in ("generated fingers", "legato pairs", "repeated passages"):
            lines.append(f"{kind:44} {v:6d}  {1000 * v / max(1, n):6.2f} per 1000")
    lines.append(f"repeats fingered differently: {c['repeat fingered differently'] / max(1, c['repeated passages']):.1%}"
                 f" of {c['repeated passages']} repeated {REPEAT_WINDOW}-onset passages")
    return "\n".join(lines)


def report(results: list[Result]) -> str:
    lines = []
    groups: dict[str, list[Result]] = {}
    for r in results:
        groups.setdefault(r.case.group, []).append(r)
        mark = "ok " if r.agree == r.total else "   "
        lines.append(f"{mark}{r.case.name:32} {r.agree:3d}/{r.total:<3d}  got {' '.join(map(str, r.predicted))}")
        if r.agree != r.total:
            want = " ".join("|".join(map(str, sorted(a))) for a in r.accepted)
            lines.append(f"   {'':32}          want {want}")
    lines.append("")
    for group, rs in groups.items():
        agree, total = sum(r.agree for r in rs), sum(r.total for r in rs)
        exact = sum(r.agree == r.total for r in rs)
        lines.append(f"{group:12} {agree / total:6.1%} of notes, {exact}/{len(rs)} cases exact")
    agree, total = sum(r.agree for r in results), sum(r.total for r in results)
    lines.append(f"{'all':12} {agree / total:6.1%} of notes ({agree}/{total}), "
                 f"{sum(r.agree == r.total for r in results)}/{len(results)} cases exact")
    return "\n".join(lines)
