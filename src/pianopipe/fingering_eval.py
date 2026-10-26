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

import re
from dataclasses import dataclass

from .fingering import assign_fingering
from .model import Note

TPB = 480
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
