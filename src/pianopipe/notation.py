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
  clef is treble or bass from its notes' median pitch.
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


@dataclass
class Staff:
    hand: str
    clefs: list[str]  # per bar: "G" or "F"
    voices: list[list[Chord]]
    triplets: set[int]  # quarter indexes written as triplets


def _snap(x: Q, unit: Q) -> Q:
    return Q(round(x / unit)) * unit


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


def measure_pieces(chords: list[Chord], bar: Bar, tpb: int, triplets: set[int], bar_q0: Q) -> list[Piece]:
    """One voice's written pieces in one bar: notes (tied across the bar lines and beats) and rests between them."""
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
    if all(e[2] is None for e in filled):
        return [Piece(Q(0), length, "whole", whole_bar=True)]

    pieces: list[Piece] = []
    for a, b, chord, tied_in, tied_out in filled:
        # Cut at triplet quarters' edges, then write each stretch.
        near = [q for q in range(int(bar_q0 + a) - 1, int(bar_q0 + b) + 1) if q in triplets]
        cuts = sorted({a, b} | {Q(e) - bar_q0 for q in near for e in (q, q + 1) if a < Q(e) - bar_q0 < b})
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


def _clefs(notes: list[Note], hand: str, bars: list[Bar]) -> list[str]:
    """A clef per bar: the hand's usual clef from its median pitch, changing for a bar whose notes sit well inside the
    other clef (a left hand crossing over to play high), and only back once they leave it."""
    m = median(n.pitch for n in notes)
    clef = ("F" if m < 55 else "G") if hand == "R" else ("G" if m >= 64 else "F")
    out = []
    i = 0
    notes = sorted(notes, key=lambda n: n.start)
    for b in bars:
        pitches = []
        while i < len(notes) and notes[i].start < b.end:
            if notes[i].start >= b.start:
                pitches.append(notes[i].pitch)
            i += 1
        if pitches:
            bm = median(pitches)
            if clef == "F" and min(pitches) >= 60 and bm >= 67:
                clef = "G"
            elif clef == "G" and max(pitches) <= 60 and bm <= 53:
                clef = "F"
        out.append(clef)
    return out


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
        q, trip = quantise(notes, tpb, bars)
        staves.append(Staff(hand, _clefs(notes, hand, bars), voices(q, single), trip))
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
            if bi == 0 or st.clefs[bi] != st.clefs[bi - 1]:
                clef = st.clefs[bi]
                attrs.append(f'<clef number="{si + 1}"><sign>{clef}</sign><line>{2 if clef == "G" else 4}</line></clef>')
        if attrs:
            out.append("<attributes>" + "".join(attrs) + "</attributes>")
        length = _dur(Q(bar.end - bar.start, score.tpb))
        first = True
        for si, st in enumerate(score.staves):
            alters = key_alters(bar.fifths)
            state: dict[tuple[str, int], int] = {}
            per_voice = []
            for vi, chords in enumerate(st.voices):
                pieces = measure_pieces(chords, bar, score.tpb, st.triplets, bar_q0)
                if vi > 0 and all(not p.pitches for p in pieces):
                    continue  # an empty second voice is left out
                per_voice.append((vi, pieces))
            two = len(per_voice) > 1
            names = _spellings(per_voice, bar, prev_names.get(si, {}))
            prev_names[si] = {pitch: names[(vi, len(ps) - 1, pitch)] for vi, ps in per_voice if ps for pitch in ps[-1].pitches}
            # Accidentals in time order across the staff's voices.
            shown: dict[tuple[int, int, int], str] = {}
            timeline = sorted(((p.at, vi, k, p) for vi, ps in per_voice for k, p in enumerate(ps)), key=lambda e: (e[0], e[1]))
            for _, vi, k, p in timeline:
                for pitch in p.pitches:
                    step, alter, octave = names[(vi, k, pitch)]
                    cur = state.get((step, octave), alters[step])
                    if p.tie_stop:
                        continue
                    if alter != cur:
                        shown[(vi, k, pitch)] = _ACCIDENTALS[alter]
                    state[(step, octave)] = alter
            for vi, pieces in per_voice:
                if not first:
                    out.append(f"<backup><duration>{length}</duration></backup>")
                first = False
                voice = si * 4 + vi + 1
                for k, p in enumerate(pieces):
                    out.append(_note_xml(p, voice, si + 1 if nstaves > 1 else 0, st.hand, two and vi, two,
                                         names, shown, (vi, k), fingering, length))
        out.append("</measure>")
    out.append("</part></score-partwise>")
    return "\n".join(out)


def _spellings(per_voice, bar: Bar, carried: dict) -> dict[tuple[int, int, int], tuple[str, int, int]]:
    """Every note's spelling in a bar, keyed (voice, piece, pitch): a note looks at its voice's next note to see whether it
    steps a semitone; a tied note keeps the spelling it was tied from (also from the last bar, `carried`)."""
    names = {}
    for vi, pieces in per_voice:
        seen: dict[int, tuple[str, int, int]] = {}  # a pitch played again in the bar keeps its name
        for k, p in enumerate(pieces):
            nxt = next((q for q in pieces[k + 1:] if q.pitches and not q.tie_stop), None)
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
