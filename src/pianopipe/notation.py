"""Notation for the sheet music player: an app-ready level (docs/formats.md) as a MusicXML score (docs/sheet-music.md).

The score is written from the level file itself, the notes the app plays, so its bars are the app's bars: the bar grid
follows the file's time signatures from tick 0 exactly as the app's MidiFile.Build does (a meter change in the middle of
a bar ends that bar there), through the last note-off. Steps:

- Quantise: each hand's note starts and ends snap to a 32nd grid, or, in a quarter where that fits clearly worse, to a
  16th-triplet grid (triplets). Compound meters (6/8, 9/8, 12/8, 3/8) use the 32nd grid only.
- Lines: in levels 1-2 (one line per hand) a note ends where the hand's next note starts. In the other levels a note
  that overlaps the next start by a 32nd or less ends there (legato in a performance), and each hand has at most two
  voices: at a start whose notes end at different times, the top notes go to voice 1 (stems up) and the rest to voice 2
  (stems down); a start with no free voice cuts the voice that ends soonest.
- Rhythm: notes cross bar lines as ties; in a bar, a note or rest that starts off the beat ends at the next beat, and
  one that starts on the beat takes the longest written value that fits (whole, half, quarter, eighth, 16th, 32nd,
  dotted), so beats stay visible. Triplet quarters are written as 3:2 tuplets. Eighths and shorter are beamed per beat.
- Spelling: the key signature (or, without one, the best-fitting key) picks each pitch's name: the spelling closest to
  the key's centre on the line of fifths, sharps on a tie (the raised leading tone in minor keys). Accidentals follow
  the usual rule: they last to the end of the bar, per staff and octave.
- Staves: right hand on the upper staff, left hand on the lower; a level with one hand gets one staff. Each staff's
  clef is treble or bass from its notes' median pitch, changing where the hand moves well into the other clef for two
  beats or more, at a beat inside a bar if that's where it moves.
- Grace notes: one to four notes shorter than a 32nd, just before a longer note of the same hand, are written as grace
  notes (slashed) before it rather than as 64ths squeezed into the rhythm.
- 8va and 8vb: a stretch of the treble staff whose top line stays above the staff for a quarter or more and reaches
  four ledger lines (A6) is written an octave lower under an 8va line; the bass staff likewise under 8vb (reaching F1).
- A first bar shorter than the second whose meter differs is a pickup: it shows the second bar's time signature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction as Q
from statistics import median
from xml.sax.saxutils import escape

from . import features
from .model import Note, Song

DIVISIONS = 24  # per quarter: 32nds (3) and 16th triplets (4)
DUPLE, TRIPLET = Q(1, 8), Q(1, 6)  # grid units in quarters
LEGATO_TIDY = Q(1, 8)  # an overlap this short (quarters) is legato, not a second voice
GRACE_MAX = Q(3, 32)  # notes this short (quarters, under a 32nd) just before a longer one are grace notes
GRACE_GROUP = 4  # at most this many in a row (more is a written-out run or trill)
GRACE_MAIN = Q(1, 4)  # the note they lead to lasts at least a 16th
CLEF_RUN = 2  # beats a hand must stay in the other clef for the clef to change
# 8va: the staff's top line stays above the staff (G5 and up), no note goes below G4 (an octave lower, three ledger
# lines under the staff) and one reaches A6 (four ledger lines). 8vb: the bottom line stays under the staff (F2 and
# down), no note above E3, one at F1 or below. A stretch lasts a quarter or more, unless it reaches C7 or D1 (five
# ledger lines, too many to read even for a moment).
OTTAVA_LINE, OTTAVA_FLOOR, OTTAVA_PEAK, OTTAVA_ALONE = 79, 67, 93, 96
OTTAVA_BASSA_LINE, OTTAVA_BASSA_CEILING, OTTAVA_BASSA_PEAK, OTTAVA_BASSA_ALONE = 41, 52, 29, 26
OTTAVA_MIN = Q(1)

# Major key names (as mido writes them) to the number of sharps (+) or flats (-).
FIFTHS = {"Cb": -7, "Gb": -6, "Db": -5, "Ab": -4, "Eb": -3, "Bb": -2, "F": -1, "C": 0, "G": 1, "D": 2, "A": 3, "E": 4,
          "B": 5, "F#": 6, "C#": 7}
STEPS = "FCGDAEB"  # the line of fifths: F = -1, C = 0, ... B = 5
STEP_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# Written values in quarters: (duration, MusicXML type, dots).
VALUES = [
    (Q(6), "whole", 1), (Q(4), "whole", 0), (Q(3), "half", 1), (Q(2), "half", 0), (Q(3, 2), "quarter", 1),
    (Q(1), "quarter", 0), (Q(3, 4), "eighth", 1), (Q(1, 2), "eighth", 0), (Q(3, 8), "16th", 1), (Q(1, 4), "16th", 0),
    (Q(1, 8), "32nd", 0),
]
# In a triplet quarter (3:2), written value by length in sixths of the quarter.
TRIPLET_VALUES = {1: ("16th", 0), 2: ("eighth", 0), 3: ("eighth", 1), 4: ("quarter", 0)}
BEAMED = {"eighth", "16th", "32nd"}


def key_fifths(name: str) -> int:
    """Sharps (+) or flats (-) of a key named as mido names them ("G", "Em", "Bb")."""
    major = features.RELATIVE_MAJOR.get(name, name)
    return FIFTHS.get(major, 0)


def spell(pitch: int, fifths: int, minor: bool = False, step_to: int = 0) -> tuple[str, int, int]:
    """(step, alter, octave) of a MIDI pitch in a key. A note outside the key that steps a semitone to the next note
    (step_to +1 or -1) is raised or lowered towards it (E D# E, A Ab G); otherwise the spelling nearest the key's centre
    on the line of fifths wins (sharps on a tie). A minor key's centre sits further sharp, so its raised sixth and
    seventh (F# G# in A minor) and the D# of E major chords are sharps, while Bb stays flat."""
    pc = pitch % 12
    centre = fifths + (3.5 if minor else 2)
    cands = []
    for p in range(-15, 20):
        if (p * 7) % 12 != pc:
            continue
        alter = (p + 1) // 7
        if abs(alter) <= 1:
            cands.append(p)
    inside = [p for p in cands if fifths - 1 <= p <= fifths + 5]
    if inside:
        p = inside[0]
    elif step_to in (1, -1) and len(cands) > 1:
        p = max(cands) if step_to > 0 else min(cands)
    else:
        p = min(cands, key=lambda c: (abs(c - centre), -c))
    step = STEPS[(p + 1) % 7]
    alter = (p + 1) // 7
    natural = pitch - alter
    return step, alter, natural // 12 - 1


def key_alters(fifths: int) -> dict[str, int]:
    """Each step's alteration under a key signature."""
    out = {s: 0 for s in STEPS}
    if fifths > 0:
        for s in "FCGDAEB"[:fifths]:
            out[s] = 1
    elif fifths < 0:
        for s in "BEADGCF"[:-fifths]:
            out[s] = -1
    return out


@dataclass
class Bar:
    number: int  # 1-based, the app's bar number
    start: int  # ticks
    end: int
    num: int
    den: int
    fifths: int = 0
    minor: bool = False


def bar_grid(meta, tpb: int, last_tick: int) -> list[Bar]:
    """The app's bar grid (MidiFile.Build): 4/4 until the first time signature, a change mid-bar ends that bar, bars
    through the last note event."""
    meters = sorted(((t, m.numerator, m.denominator) for t, m in meta if m.type == "time_signature"), key=lambda e: e[0])
    starts: list[tuple[int, int, int]] = []
    mi, num, den, bar = 0, 4, 4, 0
    while True:
        while mi < len(meters) and meters[mi][0] <= bar:
            _, num, den = meters[mi]
            mi += 1
        if bar >= last_tick and starts:
            break
        starts.append((bar, num, den))
        if len(starts) > 20000:
            break
        nxt = bar + max(1, tpb * 4 * num // den)
        if mi < len(meters) and meters[mi][0] < nxt:
            nxt = meters[mi][0]
        bar = nxt
    bars = []
    for i, (s, n, d) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else bar
        bars.append(Bar(i + 1, s, end, n, d))
    return bars


def compound(num: int, den: int) -> bool:
    return den >= 8 and num % 3 == 0


def beat_length(num: int, den: int) -> Q:
    unit = Q(4, den)
    return unit * 3 if compound(num, den) else unit


@dataclass
class Chord:
    on: Q  # quarters from the start of the piece
    off: Q
    pitches: list[int]
    fingers: list[int | None]
    staccato: bool = False
    graces: list[list[int]] = field(default_factory=list)  # grace notes before it: each one's pitches, high to low


@dataclass
class Piece:
    """One written note, chord or rest in a measure."""
    at: Q  # quarters from the bar's start
    dur: Q
    kind: str  # MusicXML type
    dots: int = 0
    pitches: list[int] = field(default_factory=list)  # empty: a rest
    fingers: list[int | None] = field(default_factory=list)
    tie_start: bool = False
    tie_stop: bool = False
    tuplet: str = ""  # "", "start", "stop", "mid"
    staccato: bool = False
    beam: str = ""  # "", "begin", "continue", "end"
    whole_bar: bool = False
    graces: list[list[int]] = field(default_factory=list)


@dataclass
class Staff:
    hand: str
    clefs: list[str]  # per bar, at its start: "G" or "F"
    voices: list[list[Chord]]
    triplets: set[int]  # quarter indexes written as triplets
    clef_changes: dict[int, list[tuple[Q, str]]] = field(default_factory=dict)  # bar index: (quarters in, clef) inside it
    ottavas: list[tuple[Q, Q, int]] = field(default_factory=list)  # (start, end) in quarters, +1 8va or -1 8vb

    def clef_at_end(self, bi: int) -> str:
        changes = self.clef_changes.get(bi)
        return changes[-1][1] if changes else self.clefs[bi]


def _snap(x: Q, unit: Q) -> Q:
    return Q(round(x / unit)) * unit


def split_graces(notes: list[Note], tpb: int) -> tuple[list[Note], dict[int, list[list[int]]]]:
    """One hand's notes without its grace notes, and the grace notes by the tick of the note they lead to (each grace's
    pitches, high to low). A grace note is shorter than a 32nd (GRACE_MAX) and comes in a row of one to four such
    starts, each within GRACE_MAX of the next, just before a start whose notes last a 16th or longer: an ornament played
    as 64ths, which written in the rhythm would squeeze the beat. A longer row of short notes is a written-out run."""
    at: dict[int, list[Note]] = {}
    for n in notes:
        at.setdefault(n.start, []).append(n)
    times = sorted(at)
    short = [all(Q(n.end - n.start, tpb) <= GRACE_MAX for n in at[t]) for t in times]
    graces: dict[int, list[list[int]]] = {}
    i = 0
    while i < len(times):
        if not short[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(times) and short[j + 1] and Q(times[j + 1] - times[j], tpb) <= GRACE_MAX:
            j += 1
        main = times[j + 1] if j + 1 < len(times) else None
        if (main is not None and j - i < GRACE_GROUP and not short[j + 1] and Q(main - times[j], tpb) <= GRACE_MAX
                and Q(max(n.end - n.start for n in at[main]), tpb) >= GRACE_MAIN):
            graces[main] = [sorted({n.pitch for n in at[t]}, reverse=True) for t in times[i:j + 1]]
        i = j + 1
    dropped = set()
    for main, gs in graces.items():
        k = times.index(main)
        dropped.update(times[k - len(gs):k])
    return [n for n in notes if n.start not in dropped], graces


def _attach_graces(vs: list[list[Chord]], graces: dict[int, list[list[int]]], tpb: int) -> None:
    """Puts each row of grace notes on the chord it leads to: the chord nearest its tick (within a 16th), preferring
    voice 1."""
    chords = sorted((c for v in vs for c in v), key=lambda c: c.on)
    for tick, gs in sorted(graces.items()):
        at = Q(tick, tpb)
        near = [c for c in chords if abs(c.on - at) <= GRACE_MAIN]
        if not near:
            continue
        best = min(near, key=lambda c: (abs(c.on - at), 0 if any(c is x for x in vs[0]) else 1))
        best.graces = gs + best.graces


def quantise(notes: list[Note], tpb: int, bars: list[Bar]) -> tuple[list[tuple[Q, Q, int, int | None]], set[int]]:
    """Notes as (on, off, pitch, finger) in quarters on the grid, and the quarters that use the triplet grid."""
    pos = [(Q(n.start, tpb), Q(n.end, tpb), n.pitch, n.finger) for n in notes]
    by_quarter: dict[int, list[Q]] = {}
    for on, off, _, _ in pos:
        by_quarter.setdefault(int(on), []).append(on)
        by_quarter.setdefault(int(off), []).append(off)

    def meter_at(q: int) -> tuple[int, int]:
        tick = q * tpb
        for b in bars:
            if b.start <= tick < b.end:
                return b.num, b.den
        return bars[-1].num, bars[-1].den

    triplets = set()
    for q, xs in by_quarter.items():
        if compound(*meter_at(q)):
            continue
        err_d = sum(abs(x - _snap(x, DUPLE)) for x in xs)
        err_t = sum(abs(x - _snap(x, TRIPLET)) for x in xs)
        if err_t < err_d and any(_snap(x, TRIPLET) % DUPLE for x in xs):
            triplets.add(q)

    def snap(x: Q) -> Q:
        return _snap(x, TRIPLET if int(x) in triplets else DUPLE)

    out = []
    for on, off, pitch, finger in pos:
        a = snap(on)
        b = snap(off)
        if b <= a:
            b = a + (TRIPLET if int(a) in triplets else DUPLE)
        out.append((a, b, pitch, finger))
    out.sort(key=lambda e: (e[0], -e[2]))
    return out, triplets


def voices(qnotes: list[tuple[Q, Q, int, int | None]], single_line: bool) -> list[list[Chord]]:
    """One hand's quantised notes as at most two voices of chords (see the module docstring)."""
    groups: list[list[tuple[Q, Q, int, int | None]]] = []
    for n in qnotes:
        if groups and groups[-1][0][0] == n[0]:
            if n[2] not in [g[2] for g in groups[-1]]:
                groups[-1].append(n)
        else:
            groups.append([n])
    # Tidy ends: one line per hand, or a short legato overlap.
    for i, g in enumerate(groups):
        nxt = groups[i + 1][0][0] if i + 1 < len(groups) else None
        if nxt is None:
            continue
        for j, (on, off, p, f) in enumerate(g):
            if off > nxt and (single_line or off - nxt <= LEGATO_TIDY):
                g[j] = (on, nxt, p, f)
    vs: list[list[Chord]] = [[], []]
    ends = [Q(0), Q(0)]

    def put(v: int, on: Q, notes: list) -> None:
        if vs[v] and vs[v][-1].off > on:
            vs[v][-1].off = on  # cut the voice that was still sounding
            if vs[v][-1].off <= vs[v][-1].on:
                vs[v].pop()
        off = min(n[1] for n in notes)
        notes = sorted(notes, key=lambda n: -n[2])
        vs[v].append(Chord(on, off, [n[2] for n in notes], [n[3] for n in notes]))
        ends[v] = off

    for g in groups:
        on = g[0][0]
        offs = sorted({n[1] for n in g})
        if single_line or len(offs) == 1:
            parts = [g]
        else:
            # The notes ending with the top note are voice 1; the rest are voice 2 (cut to their shortest end).
            top = max(g, key=lambda n: n[2])
            parts = [[n for n in g if n[1] == top[1]], [n for n in g if n[1] != top[1]]]
        free = [v for v in (0, 1) if ends[v] <= on]
        if single_line:
            put(0, on, g)
        elif len(parts) == 2 and len(free) == 2:
            put(0, on, parts[0])
            put(1, on, parts[1])
        elif free:
            put(free[0], on, g)
        else:
            put(0 if ends[0] <= ends[1] else 1, on, g)
    for v in vs:
        _staccato(v)
    return [v for v in vs if v] or [[]]


PLAIN = {Q(1, 8), Q(1, 4), Q(1, 2), Q(1), Q(2), Q(4)}


def _staccato(chords: list[Chord]) -> None:
    """A short note followed by a gap, both within a quarter, is written to the next note with a staccato dot (a
    played-short quarter is a staccato quarter, not a dotted 16th and two rests), unless its own length is a plain value
    at least half of that span (an eighth then an eighth rest stays)."""
    for i, c in enumerate(chords[:-1]):
        nxt = chords[i + 1].on
        span, dur = nxt - c.on, c.off - c.on
        if c.off < nxt and span <= 1 and (dur not in PLAIN or dur * 2 < span):
            c.off = nxt
            c.staccato = True


def _split_duple(at: Q, dur: Q, beat: Q, rest: bool = False) -> list[tuple[Q, Q, str, int]]:
    """Written values for a note or rest at `at` (quarters in the bar) lasting dur, in a stretch without triplets. A rest
    takes only values that start on a multiple of their own length (a quarter rest and a half rest, not a dotted half
    rest from beat 2), and dotted ones only in compound meters."""
    out = []
    while dur > 0:
        pos = at % beat
        if rest:
            ok = [v for v in VALUES if v[0] <= dur and at % v[0] == 0 and (not v[2] or beat % 3 == 0 or beat.denominator == 2)]
            ok = ok or [v for v in VALUES if v[0] <= dur and (not v[2])]
        elif pos == 0:
            ok = [v for v in VALUES if v[0] <= dur and (v[0] < beat or v[0] % beat == 0 or (v[2] and (v[0] * 2 / 3) % beat == 0))]
        else:
            room = beat - pos
            ok = [v for v in VALUES if v[0] <= min(dur, room)]
        if not ok:  # off the grid (shouldn't happen after quantising): take the rest as a 32nd
            v = (dur, "32nd", 0)
        else:
            v = ok[0]
        out.append((at, v[0], v[1], v[2]))
        at += v[0]
        dur -= v[0]
    return out


def _split_triplet(at: Q, dur: Q) -> list[tuple[Q, Q, str, int]]:
    """Written values (3:2) inside one triplet quarter."""
    out = []
    six = int(dur * 6)
    while six > 0:
        n = 4 if six >= 4 else six
        kind, dots = TRIPLET_VALUES[n]
        out.append((at, Q(n, 6), kind, dots))
        at += Q(n, 6)
        six -= n
    return out


def measure_pieces(chords: list[Chord], bar: Bar, tpb: int, triplets: set[int], bar_q0: Q,
                   cut: tuple[Q, ...] = ()) -> list[Piece]:
    """One voice's written pieces in one bar: notes (tied across the bar lines and beats) and rests between them,
    also split (tied) at `cut`, quarters into the bar where something is written between them (a clef change)."""
    length = Q(bar.end - bar.start, tpb)
    beat = beat_length(bar.num, bar.den)
    events: list[tuple[Q, Q, Chord | None, bool, bool]] = []  # (at, end) in the bar, chord, tied from before, tied on
    for c in chords:
        a, b = c.on - bar_q0, c.off - bar_q0
        if b <= 0 or a >= length:
            continue
        events.append((max(a, Q(0)), min(b, length), c, a < 0, b > length))
    events.sort(key=lambda e: e[0])
    filled: list[tuple[Q, Q, Chord | None, bool, bool]] = []
    t = Q(0)
    for e in events:
        if e[0] > t:
            filled.append((t, e[0], None, False, False))
        filled.append(e)
        t = e[1]
    if t < length:
        filled.append((t, length, None, False, False))
    if all(e[2] is None for e in filled) and not cut:
        return [Piece(Q(0), length, "whole", whole_bar=True)]

    pieces: list[Piece] = []
    for a, b, chord, tied_in, tied_out in filled:
        # Cut at triplet quarters' edges (and at `cut`), then write each stretch.
        near = [q for q in range(int(bar_q0 + a) - 1, int(bar_q0 + b) + 1) if q in triplets]
        cuts = sorted({a, b} | {Q(e) - bar_q0 for q in near for e in (q, q + 1) if a < Q(e) - bar_q0 < b}
                      | {c for c in cut if a < c < b})
        parts = []
        for x, y in zip(cuts, cuts[1:]):
            q = int(bar_q0 + x)
            if q in triplets and int(bar_q0 + y - Q(1, 1000)) == q and not (x == Q(q) - bar_q0 and y - x == 1):
                parts += [(p, True) for p in _split_triplet(x, y - x)]
            else:
                parts += [(p, False) for p in _split_duple(x, y - x, beat, chord is None)]
        for i, ((at, dur, kind, dots), trip) in enumerate(parts):
            pc = Piece(at, dur, kind, dots)
            pc.tuplet = "mid" if trip else ""
            if chord is not None:
                pc.pitches = list(chord.pitches)
                pc.tie_stop = tied_in or i > 0
                pc.tie_start = tied_out or i < len(parts) - 1
                pc.fingers = list(chord.fingers) if not pc.tie_stop else [None] * len(chord.pitches)
                pc.staccato = chord.staccato and i == len(parts) - 1
                if i == 0 and not tied_in:
                    pc.graces = chord.graces
            pieces.append(pc)
    _mark_tuplets(pieces, bar_q0)
    _beam(pieces, beat, bar_q0, triplets)
    return pieces


def _mark_tuplets(pieces: list[Piece], bar_q0: Q) -> None:
    by_q: dict[int, list[Piece]] = {}
    for p in pieces:
        if p.tuplet:
            by_q.setdefault(int(bar_q0 + p.at), []).append(p)
    for group in by_q.values():
        group[0].tuplet = "start"
        group[-1].tuplet = "stop" if len(group) > 1 else "single"
        for p in group[1:-1]:
            p.tuplet = "mid"


def _beam(pieces: list[Piece], beat: Q, bar_q0: Q, triplets: set[int]) -> None:
    """Beams eighths and shorter within a beat (a triplet quarter is its own group); rests break them."""
    def group_of(p: Piece):
        q = int(bar_q0 + p.at)
        return ("t", q) if q in triplets and p.tuplet else ("b", int(p.at // beat))

    run: list[Piece] = []

    def close():
        if len(run) >= 2:
            run[0].beam = "begin"
            for p in run[1:-1]:
                p.beam = "continue"
            run[-1].beam = "end"
        run.clear()

    for p in pieces:
        if not p.pitches or p.kind not in BEAMED:
            close()
            continue
        if run and group_of(run[-1]) != group_of(p):
            close()
        run.append(p)
    close()


def _vote(pitches: list[int], clef: str) -> str:
    """The clef notes ask for while `clef` is shown: the other one only if they sit well inside it."""
    if pitches:
        m = median(pitches)
        if clef == "F" and min(pitches) >= 60 and m >= 67:
            return "G"
        if clef == "G" and max(pitches) <= 60 and m <= 53:
            return "F"
    return clef


def _clefs(notes: list[Note], hand: str, bars: list[Bar], tpb: int) -> tuple[list[str], dict[int, list[tuple[Q, str]]]]:
    """The clef at each bar's start, and changes inside bars (bar index: [(quarters into the bar, clef)]). The hand's
    usual clef comes from its median pitch; it changes for a bar whose notes sit well inside the other clef (a left
    hand crossing over to play high), and only back once they leave it. The change moves to a beat inside the bar where
    the bar's first part is still well inside the old clef, and a bar that stays in its clef changes at a beat after
    which its last CLEF_RUN beats or more sit well inside the other; either only where no note of the hand is held
    across the change (a bass note held under the crossing keeps the bar in its clef)."""
    m = median(n.pitch for n in notes)
    clef = ("F" if m < 55 else "G") if hand == "R" else ("G" if m >= 64 else "F")
    starts: list[str] = []
    changes: dict[int, list[tuple[Q, str]]] = {}
    i = 0
    sounding: list[Note] = []
    notes = sorted(notes, key=lambda n: n.start)
    for bi, b in enumerate(bars):
        here: list[tuple[Q, int]] = []
        while i < len(notes) and notes[i].start < b.end:
            if notes[i].start >= b.start:
                here.append((Q(notes[i].start - b.start, tpb), notes[i].pitch))
            sounding.append(notes[i])
            i += 1
        sounding = [n for n in sounding if n.end > b.start]
        # Notes sounding in the bar, (start, end) in quarters into it: a clef can't change inside one of them.
        held = [(Q(n.start - b.start, tpb), Q(n.end - b.start, tpb)) for n in sounding]
        pitches = [p for _, p in here]
        beat, length = beat_length(b.num, b.den), Q(b.end - b.start, tpb)
        cuts = [k * beat for k in range(1, int(length / beat) + 1) if k * beat < length]
        other = "F" if clef == "G" else "G"

        def split(s: Q) -> tuple[list[int], list[int], Q]:
            """The pitches before and after a beat, and the first start after it (where the clef would change)."""
            post = [(q, p) for q, p in here if q >= s]
            return [p for q, p in here if q < s], [p for _, p in post], min((q for q, _ in post), default=length)

        def free(at: Q) -> bool:
            """No note is held across `at` (a legato overlap of up to LEGATO_TIDY aside): it would be drawn in the
            other clef from there."""
            return not any(a < at < e - LEGATO_TIDY for a, e in held)
        start = clef
        if _vote(pitches, clef) != clef:
            start = other
            for s in reversed(cuts):
                pre, post, at = split(s)
                if pre and post and free(at) and _vote(pre, other) == clef and _vote(post, clef) == other:
                    start, changes[bi] = clef, [(at, other)]
                    break
            clef = other
        elif pitches:
            for s in cuts:
                pre, post, at = split(s)
                if length - at >= CLEF_RUN * beat - LEGATO_TIDY and pre and post and free(at) and _vote(post, clef) == other:
                    changes[bi] = [(at, other)]
                    clef = other
                    break
        starts.append(start)
    return starts, changes


def _ottavas(staff: Staff, bars: list[Bar], tpb: int) -> list[tuple[Q, Q, int]]:
    """Stretches of a staff written an octave from where they sound (see the module docstring): runs of consecutive
    starts in the treble clef where the highest note sounding is at OTTAVA_LINE or above and every note struck at
    OTTAVA_FLOOR or above (8va), or likewise below in the bass clef (8vb); reaching OTTAVA_PEAK (OTTAVA_BASSA_PEAK) and
    lasting OTTAVA_MIN (or reaching the _ALONE pitches), with no other note of the staff sounding across their ends."""
    chords = sorted((c for v in staff.voices for c in v), key=lambda c: c.on)
    if not chords:
        return []
    bar_q = [Q(b.start, tpb) for b in bars]

    def clef_at(q: Q) -> str:
        bi = max(0, sum(1 for s in bar_q if s <= q) - 1)
        clef = staff.clefs[bi]
        for at, c in staff.clef_changes.get(bi, []):
            if bar_q[bi] + at <= q:
                clef = c
        return clef
    def struck(c: Chord) -> list[int]:
        return c.pitches + [g for gs in c.graces for g in gs]

    # The highest and lowest note sounding when each chord starts.
    span: dict[int, tuple[int, int]] = {}
    active: list[Chord] = []
    for c in chords:
        active = [o for o in active if o.off > c.on] + [c]
        ps = [p for o in active for p in o.pitches]
        span[id(c)] = (min(ps), max(ps))

    def sounding(c: Chord) -> tuple[int, int]:
        return span[id(c)]
    out: list[tuple[Q, Q, int]] = []
    for shift, clef, line, inside, peak, alone in (
            (1, "G", lambda c: sounding(c)[1] >= OTTAVA_LINE, lambda p: p >= OTTAVA_FLOOR,
             lambda p: p >= OTTAVA_PEAK, lambda p: p >= OTTAVA_ALONE),
            (-1, "F", lambda c: sounding(c)[0] <= OTTAVA_BASSA_LINE, lambda p: p <= OTTAVA_BASSA_CEILING,
             lambda p: p <= OTTAVA_BASSA_PEAK, lambda p: p <= OTTAVA_BASSA_ALONE)):
        run: list[Chord] = []

        def close():
            ps = [p for c in run for p in struck(c)]
            a, b = (run[0].on, max(c.off for c in run)) if run else (0, 0)
            if any(peak(p) for p in ps) and (b - a >= OTTAVA_MIN or any(alone(p) for p in ps)):
                mine = {id(c) for c in run}
                # No note outside the run starts in it or sounds across its ends (it would be drawn an octave off).
                if not any(id(c) not in mine and (c.on < a < c.off or c.on < b < c.off or a < c.on < b) for c in chords):
                    out.append((a, b, shift))
            run.clear()
        for c in chords:
            if all(inside(p) for p in struck(c)) and line(c) and clef_at(c.on) == clef:
                run.append(c)
            else:
                close()
        close()
    return sorted(out)


@dataclass
class Score:
    bars: list[Bar]
    staves: list[Staff]
    tpb: int
    pickup: bool
    title: str = ""


def build(song: Song, level: int = 4, title: str = "") -> Score:
    """A Score from an app-ready level (tracks named RH and LH; otherwise all notes are the right hand)."""
    tpb = song.ticks_per_beat
    hands = {"R": [], "L": []}
    for t in song.tracks:
        hand = "L" if t.name == "LH" else "R"
        hands[hand].extend(n for n in t.notes)
    all_notes = hands["R"] + hands["L"]
    last = max((n.end for n in all_notes), default=tpb * 4)
    bars = bar_grid(song.meta, tpb, last)
    sigs = sorted(((t, m.key) for t, m in song.meta if m.type == "key_signature"), key=lambda e: e[0])
    if not sigs:
        guess = features.keys(all_notes, song.meta)
        sigs = [(0, guess[0] if guess else "C")]
    # Major or minor per signature, from the notes it covers (files often write A minor as C).
    modes = {}
    for i, (t, k) in enumerate(sigs):
        stop = sigs[i + 1][0] if i + 1 < len(sigs) else None
        part = [n for n in all_notes if n.start >= t and (stop is None or n.start < stop)] or all_notes
        major = features.RELATIVE_MAJOR.get(k, k)
        minor = features.RELATIVE_MINOR.get(major)
        modes[t] = bool(part) and minor is not None and             features._mode_for(features.pitch_class_weights(part), major, minor) == minor
    for b in bars:
        t0, name = sigs[0]
        for t, k in sigs:
            if t <= b.start:
                t0, name = t, k
        b.fifths, b.minor = key_fifths(name), modes[t0]
    single = level in (1, 2)
    staves = []
    for hand in ("R", "L"):
        notes = sorted(hands[hand], key=lambda n: (n.start, -n.pitch))
        if not notes:
            continue
        rest, graces = split_graces(notes, tpb)
        q, trip = quantise(rest, tpb, bars)
        vs = voices(q, single)
        _attach_graces(vs, graces, tpb)
        clefs, changes = _clefs(rest, hand, bars, tpb)
        # On the notes' quantised grid (a change at a note played a tick late would split a piece to nothing).
        for bi, cs in list(changes.items()):
            q0, length = Q(bars[bi].start, tpb), Q(bars[bi].end - bars[bi].start, tpb)
            cs = [(_snap(q0 + at, TRIPLET if int(q0 + at) in trip else DUPLE) - q0, c) for at, c in cs]
            changes[bi] = [(at, c) for at, c in cs if 0 < at < length]
            if not changes[bi]:  # snapped onto a bar line: the bar starts in the new clef, or the next one does
                if cs[-1][0] <= 0:
                    clefs[bi] = cs[-1][1]
                del changes[bi]
        st = Staff(hand, clefs, vs, trip, changes)
        st.ottavas = _ottavas(st, bars, tpb)
        staves.append(st)
    pickup = len(bars) > 1 and (bars[0].num, bars[0].den) != (bars[1].num, bars[1].den) and \
        Q(bars[0].end - bars[0].start) < Q(bars[1].end - bars[1].start)
    return Score(bars, staves, tpb, pickup, title)


_ACCIDENTALS = {-2: "flat-flat", -1: "flat", 0: "natural", 1: "sharp", 2: "double-sharp"}


def _dur(q: Q) -> int:
    return int(q * DIVISIONS)


def to_musicxml(score: Score, fingering: bool = True) -> str:
    """The score as MusicXML 4.0 (partwise, one piano part)."""
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<score-partwise version="4.0">',
           f"<work><work-title>{escape(score.title)}</work-title></work>",
           '<part-list><score-part id="P1"><part-name print-object="no">Piano</part-name></score-part></part-list>',
           '<part id="P1">']
    nstaves = len(score.staves)
    prev_key = prev_meter = None
    prev_names: dict[int, dict[int, tuple[str, int, int]]] = {}  # per staff: how the last notes of the bar were spelled
    for bi, bar in enumerate(score.bars):
        bar_q0 = Q(bar.start, score.tpb)
        attrs = []
        implicit = ' implicit="yes"' if bi == 0 and score.pickup else ""
        out.append(f'<measure number="{bar.number}"{implicit}>')
        if bi == 0:
            attrs.append(f"<divisions>{DIVISIONS}</divisions>")
        if bar.fifths != prev_key:
            attrs.append(f"<key><fifths>{bar.fifths}</fifths></key>")
            prev_key = bar.fifths
        meter = (score.bars[1].num, score.bars[1].den) if bi == 0 and score.pickup else (bar.num, bar.den)
        if bi == 1 and score.pickup:
            meter = prev_meter
        if meter != prev_meter:
            attrs.append(f"<time><beats>{meter[0]}</beats><beat-type>{meter[1]}</beat-type></time>")
            prev_meter = meter
        if bi == 0 and nstaves > 1:
            attrs.append(f"<staves>{nstaves}</staves>")
        for si, st in enumerate(score.staves):
            if bi == 0 or st.clefs[bi] != st.clef_at_end(bi - 1):
                attrs.append(_clef_xml(si, st.clefs[bi]))
        if attrs:
            out.append("<attributes>" + "".join(attrs) + "</attributes>")
        length = _dur(Q(bar.end - bar.start, score.tpb))
        first = True
        for si, st in enumerate(score.staves):
            alters = key_alters(bar.fifths)
            state: dict[tuple[str, int], int] = {}
            per_voice = []
            for vi, chords in enumerate(st.voices):
                cut = tuple(at for at, _ in st.clef_changes.get(bi, [])) if vi == 0 else ()
                pieces = measure_pieces(chords, bar, score.tpb, st.triplets, bar_q0, cut)
                if vi > 0 and all(not p.pitches for p in pieces):
                    continue  # an empty second voice is left out
                per_voice.append((vi, pieces))
            two = len(per_voice) > 1
            names = _spellings(per_voice, bar, prev_names.get(si, {}))
            prev_names[si] = {pitch: names[(vi, len(ps) - 1, pitch)] for vi, ps in per_voice if ps for pitch in ps[-1].pitches}
            # Accidentals in time order across the staff's voices (grace notes before the note they lead to).
            shown: dict[tuple, str] = {}
            timeline = sorted(((p.at, vi, k, p) for vi, ps in per_voice for k, p in enumerate(ps)), key=lambda e: (e[0], e[1]))
            for _, vi, k, p in timeline:
                heads = [((vi, (k, gi), pitch), False) for gi, g in enumerate(p.graces) for pitch in g]
                for key, tied in heads + [((vi, k, pitch), p.tie_stop) for pitch in p.pitches]:
                    step, alter, octave = names[key]
                    cur = state.get((step, octave), alters[step])
                    if tied:
                        continue
                    if alter != cur:
                        shown[key] = _ACCIDENTALS[alter]
                    state[(step, octave)] = alter
            # Clef changes and 8va lines inside the bar go in the first voice, before the first note at or after them
            # (its notes are split, tied, where a clef changes: measure_pieces' `cut`).
            marks = _staff_marks(st, si, nstaves, bar, bi, bar_q0, score.tpb)
            for vi, pieces in per_voice:
                if not first:
                    out.append(f"<backup><duration>{length}</duration></backup>")
                first = False
                voice = si * 4 + vi + 1
                todo = marks if vi == per_voice[0][0] else []
                for k, p in enumerate(pieces):
                    while todo and todo[0][0] <= p.at:
                        out.append(todo.pop(0)[2])
                    out.append(_note_xml(p, voice, si + 1 if nstaves > 1 else 0, st.hand, two and vi, two,
                                         names, shown, (vi, k), fingering, length))
                out += [m[2] for m in todo]
        out.append("</measure>")
    out.append("</part></score-partwise>")
    return "\n".join(out)


def _clef_xml(si: int, clef: str) -> str:
    return f'<clef number="{si + 1}"><sign>{clef}</sign><line>{2 if clef == "G" else 4}</line></clef>'


def _staff_marks(st: Staff, si: int, nstaves: int, bar: Bar, bi: int, bar_q0: Q, tpb: int) -> list[tuple[Q, int, str]]:
    """A staff's clef changes and 8va/8vb starts and stops inside a bar: (quarters into the bar, order, MusicXML),
    sorted; a stop goes before a clef change and a start at the same place."""
    staff_xml = f"<staff>{si + 1}</staff>" if nstaves > 1 else ""
    length = Q(bar.end - bar.start, tpb)
    out = [(at, 1, f"<attributes>{_clef_xml(si, clef)}</attributes>") for at, clef in st.clef_changes.get(bi, [])]
    for a, b, shift in st.ottavas:
        if 0 <= a - bar_q0 < length:
            kind, place = ("down", "above") if shift > 0 else ("up", "below")
            out.append((a - bar_q0, 2, f'<direction placement="{place}"><direction-type><octave-shift type="{kind}" '
                                       f'size="8" number="{si + 1}"/></direction-type>{staff_xml}</direction>'))
        if 0 < b - bar_q0 <= length:
            out.append((b - bar_q0, 0, f'<direction><direction-type><octave-shift type="stop" size="8" number="{si + 1}"/>'
                                       f"</direction-type>{staff_xml}</direction>"))
    return sorted(out, key=lambda m: (m[0], m[1]))


def _spellings(per_voice, bar: Bar, carried: dict) -> dict[tuple, tuple[str, int, int]]:
    """Every note's spelling in a bar, keyed (voice, piece, pitch), or (voice, (piece, grace), pitch) for a grace note:
    a note looks at its voice's next note to see whether it steps a semitone; a tied note keeps the spelling it was tied
    from (also from the last bar, `carried`)."""
    names = {}
    for vi, pieces in per_voice:
        seen: dict[int, tuple[str, int, int]] = {}  # a pitch played again in the bar keeps its name
        for k, p in enumerate(pieces):
            nxt = next((q for q in pieces[k + 1:] if q.pitches and not q.tie_stop), None)
            for gi, grace in enumerate(p.graces):
                then = p.graces[gi + 1] if gi + 1 < len(p.graces) else p.pitches
                for pitch in grace:
                    if pitch not in seen:
                        seen[pitch] = spell(pitch, bar.fifths, bar.minor, min((q - pitch for q in then), key=abs))
                    names[(vi, (k, gi), pitch)] = seen[pitch]
            for pitch in p.pitches:
                if p.tie_stop:
                    prev = names.get((vi, k - 1, pitch)) if k > 0 else carried.get(pitch)
                    if prev:
                        names[(vi, k, pitch)] = seen[pitch] = prev
                        continue
                if pitch in seen:
                    names[(vi, k, pitch)] = seen[pitch]
                    continue
                step_to = min((q - pitch for q in nxt.pitches), key=abs) if nxt else 0
                names[(vi, k, pitch)] = seen[pitch] = spell(pitch, bar.fifths, bar.minor, step_to)
    return names


def _note_xml(p: Piece, voice: int, staff: int, hand: str, lower: bool, two: bool, names, shown, key, fingering: bool,
              bar_len: int) -> str:
    staff_xml = f"<staff>{staff}</staff>" if staff else ""
    if not p.pitches:
        if p.whole_bar:
            return f'<note><rest measure="yes"/><duration>{bar_len}</duration><voice>{voice}</voice>{staff_xml}</note>'
        return (f"<note><rest/><duration>{_dur(p.dur)}</duration><voice>{voice}</voice><type>{p.kind}</type>"
                + "<dot/>" * p.dots + _tuplet_xml(p) + staff_xml + _tuplet_notations(p) + "</note>")
    parts = []
    # Grace notes: slashed (Verovio then plays them just before the beat, so the main note's start stays on it); a
    # single one as an eighth, a group as beamed 16ths.
    kind = "eighth" if len(p.graces) == 1 else "16th"
    for gi, grace in enumerate(p.graces):
        for i, pitch in enumerate(grace):
            step, alter, octave = names[(key[0], (key[1], gi), pitch)]
            x = ['<note><grace slash="yes"/>', "<chord/>" if i else "",
                 f"<pitch><step>{step}</step>" + (f"<alter>{alter}</alter>" if alter else "") + f"<octave>{octave}</octave></pitch>",
                 f"<voice>{voice}</voice><type>{kind}</type>"]
            acc = shown.get((key[0], (key[1], gi), pitch))
            if acc:
                x.append(f"<accidental>{acc}</accidental>")
            if two:
                x.append(f"<stem>{'down' if lower else 'up'}</stem>")
            x.append(staff_xml)
            if i == 0 and len(p.graces) > 1:
                x.append(f'<beam number="1">{"begin" if gi == 0 else "end" if gi == len(p.graces) - 1 else "continue"}</beam>')
            x.append("</note>")
            parts.append("".join(x))
    for i, pitch in enumerate(p.pitches):
        step, alter, octave = names[(key[0], key[1], pitch)]
        x = ["<note>"]
        if i > 0:
            x.append("<chord/>")
        x.append(f"<pitch><step>{step}</step>" + (f"<alter>{alter}</alter>" if alter else "") + f"<octave>{octave}</octave></pitch>")
        x.append(f"<duration>{_dur(p.dur)}</duration>")
        if p.tie_stop:
            x.append('<tie type="stop"/>')
        if p.tie_start:
            x.append('<tie type="start"/>')
        x.append(f"<voice>{voice}</voice><type>{p.kind}</type>" + "<dot/>" * p.dots)
        acc = shown.get((key[0], key[1], pitch))
        if acc:
            x.append(f"<accidental>{acc}</accidental>")
        x.append(_tuplet_xml(p))
        if two:
            x.append(f"<stem>{'down' if lower else 'up'}</stem>")
        x.append(staff_xml)
        if i == 0 and p.beam:
            x.append(f'<beam number="1">{p.beam}</beam>')
        notations = []
        if p.tie_stop:
            notations.append('<tied type="stop"/>')
        if p.tie_start:
            notations.append('<tied type="start"/>')
        if i == 0:
            notations.append(_tuplet_notation(p))
        if i == 0 and p.staccato:
            notations.append("<articulations><staccato/></articulations>")
        f = p.fingers[i] if i < len(p.fingers) else None
        if fingering and f:
            place = "above" if hand == "R" else "below"
            notations.append(f'<technical><fingering placement="{place}">{f}</fingering></technical>')
        notations = [n for n in notations if n]
        if notations:
            x.append("<notations>" + "".join(notations) + "</notations>")
        x.append("</note>")
        parts.append("".join(x))
    return "".join(parts)


def _tuplet_xml(p: Piece) -> str:
    if not p.tuplet:
        return ""
    return "<time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification>"


def _tuplet_notation(p: Piece) -> str:
    if p.tuplet in ("start", "single"):
        return '<tuplet type="start" bracket="no"/>' + ('<tuplet type="stop"/>' if p.tuplet == "single" else "")
    if p.tuplet == "stop":
        return '<tuplet type="stop"/>'
    return ""


def _tuplet_notations(p: Piece) -> str:
    t = _tuplet_notation(p)
    return f"<notations>{t}</notations>" if t else ""
