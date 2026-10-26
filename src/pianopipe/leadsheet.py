"""House arrangements: a lead sheet (melody + chord symbols) becomes a two-hand piano part.

Lead sheets are written in a small subset of ABC notation (data/house/*.abc in the repo):

- Header fields: T: title, C: composer, M: meter, L: unit note length, Q: tempo (1/4=120), K: key (G, F, Bb, Am, Em...).
- Notes C D E F G A B (the octave from middle C), c d e f g a b (the octave above), ' and , move an octave;
  ^ sharp, _ flat, = natural (accidentals last to the end of the bar, as in printed music); z is a rest.
- Lengths: a number multiplies the unit length (A2, A3/2), a slash divides it (A/, A/2, A3/4).
- Ties A2-A2, triplets (3ABC, bars |, repeats |: :| with first and second endings [1 [2.
- Chord symbols in double quotes before a note ("G", "Am", "D7", "Bdim", "C/E", "Fmaj7", "Gsus4"); a chord lasts
  until the next one.

The melody is the right hand. The left hand is generated from the chords, a pattern per meter: 3/4 plays the root
then the chord twice (oom-pah-pah), 6/8 a broken chord in eighths, 2/4 and 4/4 the root on strong beats and the
chord on the others. Chords are voiced in close position below the melody, moving to the nearest inversion, with
the root an octave or so below. The pipeline's levels then reduce this full arrangement like any other piece.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import mido

from .model import Note, Song, TrackInfo

TPB = 480
SUFFIXES = (".abc",)
NOTE_PCS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
SHARPS_ORDER = "FCGDAEB"
KEY_FIFTHS = {"Cb": -7, "Gb": -6, "Db": -5, "Ab": -4, "Eb": -3, "Bb": -2, "F": -1, "C": 0, "G": 1, "D": 2, "A": 3,
              "E": 4, "B": 5, "F#": 6, "C#": 7}
CHORD_TYPES = {
    "": (0, 4, 7), "m": (0, 3, 7), "7": (0, 4, 7, 10), "m7": (0, 3, 7, 10), "maj7": (0, 4, 7, 11), "dim": (0, 3, 6),
    "dim7": (0, 3, 6, 9), "aug": (0, 4, 8), "sus4": (0, 5, 7), "sus2": (0, 2, 7), "6": (0, 4, 7, 9), "m6": (0, 3, 7, 9),
}
_CHORD = re.compile(r"^([A-G])([#b]?)(maj7|m7|m6|dim7|dim|aug|sus4|sus2|m|7|6)?(?:/([A-G])([#b]?))?$")
_TOKEN = re.compile(r'"[^"]*"|\[\d|:\||\|:|\|\]|\|\||\||\(\d|[\^_=]*[A-Ga-gz][,\']*\d*/*\d*|-|\s+|.')


class LeadSheetError(ValueError):
    pass


@dataclass
class Chord:
    start: Fraction  # quarter notes
    root: int  # pitch class
    tones: tuple[int, ...]  # pitch classes, root first
    bass: int | None = None  # pitch class of a slash bass


@dataclass
class LeadSheet:
    title: str = ""
    composer: str = ""
    meter: tuple[int, int] = (4, 4)
    tempo: float = 100.0
    key_fifths: int = 0
    key_minor: bool = False
    melody: list[tuple[Fraction, Fraction, int]] = field(default_factory=list)  # (start, length, pitch), quarter notes
    chords: list[Chord] = field(default_factory=list)
    bars: list[tuple[Fraction, Fraction]] = field(default_factory=list)  # (start, end) of each played bar
    length: Fraction = Fraction(0)


def parse_chord(text: str, at: Fraction) -> Chord:
    m = _CHORD.match(text.strip())
    if not m:
        raise LeadSheetError(f"unknown chord symbol {text!r}")
    root = (NOTE_PCS[m.group(1)] + {"#": 1, "b": -1, "": 0}[m.group(2)]) % 12
    tones = tuple((root + i) % 12 for i in CHORD_TYPES[m.group(3) or ""])
    bass = (NOTE_PCS[m.group(4)] + {"#": 1, "b": -1, "": 0}[m.group(5) or ""]) % 12 if m.group(4) else None
    return Chord(at, root, tones, bass)


def _key(text: str) -> tuple[int, bool]:
    text = text.strip().split()[0] if text.strip() else "C"
    minor = text.endswith("m") and not text.endswith("maj")
    name = text[:-1] if minor else text
    if minor:  # relative major: a minor third up
        pc = (NOTE_PCS[name[0]] + {"#": 1, "b": -1}.get(name[1:2], 0) + 3) % 12
        name = next(k for k, f in KEY_FIFTHS.items() if (NOTE_PCS[k[0]] + {"#": 1, "b": -1}.get(k[1:2], 0)) % 12 == pc and abs(f) <= 6)
    if name not in KEY_FIFTHS:
        raise LeadSheetError(f"unknown key {text!r}")
    return KEY_FIFTHS[name], minor


def _key_accidentals(fifths: int) -> dict[str, int]:
    if fifths >= 0:
        return {SHARPS_ORDER[i]: 1 for i in range(fifths)}
    return {SHARPS_ORDER[6 - i]: -1 for i in range(-fifths)}


def _length(spec: str) -> Fraction:
    """'' -> 1, '2' -> 2, '/' -> 1/2, '/2' -> 1/2, '3/2' -> 3/2, '//' -> 1/4."""
    if not spec:
        return Fraction(1)
    m = re.fullmatch(r"(\d*)(/*)(\d*)", spec)
    num = int(m.group(1)) if m.group(1) else 1
    slashes = len(m.group(2))
    if slashes == 0:
        return Fraction(num)
    den = int(m.group(3)) if m.group(3) else 2 ** slashes
    return Fraction(num, den)


def parse(text: str) -> LeadSheet:
    sheet = LeadSheet()
    unit = Fraction(1, 8)
    body = []
    for line in text.splitlines():
        line = line.split("%", 1)[0].rstrip()
        hm = re.match(r"^([A-Za-z]):\s*(.*)$", line)
        if hm and hm.group(1) in "XTCMLQKZN":
            field_, value = hm.group(1), hm.group(2).strip()
            if field_ == "T" and not sheet.title:
                sheet.title = value
            elif field_ == "C":
                sheet.composer = value
            elif field_ == "M":
                n, d = (value.replace("C", "4/4") if value in ("C", "C|") else value).split("/")
                sheet.meter = (int(n), int(d))
            elif field_ == "L":
                n, d = value.split("/")
                unit = Fraction(int(n), int(d))
            elif field_ == "Q":
                qm = re.match(r"(?:(\d+)/(\d+)=)?(\d+)", value)
                beat = Fraction(int(qm.group(1)), int(qm.group(2))) if qm.group(1) else Fraction(1, 4)
                sheet.tempo = float(int(qm.group(3)) * beat * 4)
            elif field_ == "K":
                sheet.key_fifths, sheet.key_minor = _key(value)
            continue
        if line.strip():
            body.append(line)
    tokens = [t for t in _TOKEN.findall(" ".join(body)) if not t.isspace()]
    _play(sheet, tokens, unit)
    return sheet


def _play(sheet: LeadSheet, tokens: list[str], unit: Fraction) -> None:
    """Walk the tokens, unrolling repeats and endings, into melody notes and chords (quarter-note times)."""
    # First split into bars with their repeat/ending marks, then unroll.
    bars: list[dict] = []
    cur = {"items": [], "start_repeat": False, "end_repeat": False, "ending": None}
    for t in tokens:
        if t in ("|", "||", "|]", ":|", "|:"):
            if t == ":|":
                cur["end_repeat"] = True
            if cur["items"] or t != "|:":
                bars.append(cur)
                cur = {"items": [], "start_repeat": False, "end_repeat": False, "ending": None}
            if t == "|:":
                cur["start_repeat"] = True
        elif re.fullmatch(r"\[\d", t):
            cur["ending"] = int(t[1])
        else:
            cur["items"].append(t)
    if cur["items"]:
        bars.append(cur)
    order: list[int] = []
    i, section, pass_no, taken = 0, 0, 1, set()
    while i < len(bars):
        b = bars[i]
        if b["start_repeat"]:
            section = i if i not in taken else section
        if b["ending"] is not None and b["ending"] != pass_no:
            i += 1
            continue
        order.append(i)
        if b["end_repeat"] and i not in taken:
            taken.add(i)
            pass_no = 2
            i = section
            continue
        if b["end_repeat"]:
            pass_no = 1
        i += 1
    key_acc = _key_accidentals(sheet.key_fifths)
    t = Fraction(0)
    tied: dict[int, int] = {}  # pitch -> index of the melody note a tie carries on
    for bi in order:
        bar_start = t
        acc = dict(key_acc)
        bar_acc: dict[tuple[str, int], int] = {}
        triplet = 0
        items = bars[bi]["items"]
        for ti, tok in enumerate(items):
            if tok.startswith('"'):
                sheet.chords.append(parse_chord(tok.strip('"'), t))
            elif re.fullmatch(r"\(\d", tok):
                triplet = int(tok[1])
            elif tok == "-":
                continue
            else:
                nm = re.fullmatch(r"([\^_=]*)([A-Ga-gz])([,']*)(\d*/*\d*)", tok)
                if not nm:
                    continue
                accs, letter, octs, lens = nm.groups()
                dur = _length(lens) * unit * 4
                if triplet:
                    dur = dur * Fraction(2, 3)
                    triplet -= 1
                if letter != "z":
                    step = letter.upper()
                    octave = 4 + (1 if letter.islower() else 0) + octs.count("'") - octs.count(",")
                    if accs:
                        alter = accs.count("^") - accs.count("_")
                        bar_acc[(step, octave)] = alter
                    alter = bar_acc.get((step, octave), acc.get(step, 0))
                    pitch = 12 * (octave + 1) + NOTE_PCS[step] + alter
                    if pitch in tied:
                        idx = tied.pop(pitch)
                        s, d, p = sheet.melody[idx]
                        sheet.melody[idx] = (s, d + dur, p)
                    else:
                        sheet.melody.append((t, dur, pitch))
                        idx = len(sheet.melody) - 1
                    tied.clear()
                    if ti + 1 < len(items) and items[ti + 1] == "-":
                        tied[pitch] = idx
                t += dur
        if t > bar_start:
            sheet.bars.append((bar_start, t))
    sheet.length = t


def _voice(tones: tuple[int, ...], below: int, previous: list[int] | None) -> list[int]:
    """A close-position voicing of the chord's tones with its top note under `below`, nearest to `previous`."""
    options = []
    triad = list(tones[:4])
    for inv in range(len(triad)):
        pcs = triad[inv:] + triad[:inv]
        for base_oct in range(3, 6):
            v, last = [], None
            for pc in pcs:
                p = 12 * base_oct + pc
                while last is not None and p <= last:
                    p += 12
                v.append(p)
                last = p
            if 50 <= v[0] and v[-1] < below and v[-1] - v[0] <= 12:
                options.append(v)
    if not options:
        v = sorted(12 * 4 + pc for pc in triad)
        while v[-1] >= below and v[0] > 40:
            v = [p - 12 for p in v]
        return v
    if previous:
        return min(options, key=lambda v: (sum(abs(a - b) for a, b in zip(v, previous)), -v[-1]))
    return max(options, key=lambda v: v[-1])


def arrange(sheet: LeadSheet) -> Song:
    """The two-hand piano part: melody in the right hand, the generated accompaniment in the left."""
    q = TPB
    num, den = sheet.meter
    beat = Fraction(4, den)  # a beat in quarter notes
    bar_len = beat * num
    rh = [Note(round(s * q), round((s + d) * q), p, 84, "R", 0, 0) for s, d, p in sheet.melody]
    lh: list[Note] = []
    if not sheet.chords:
        raise LeadSheetError("no chord symbols: the left hand needs them")

    def chord_at(t: Fraction) -> Chord | None:
        current = None
        for c in sheet.chords:
            if c.start <= t:
                current = c
            else:
                break
        return current

    def melody_low(t0: Fraction, t1: Fraction) -> int:
        notes = [p for s, d, p in sheet.melody if s < t1 and s + d > t0]
        return min(notes) if notes else 72

    previous: list[int] | None = None
    first = sheet.chords[0].start
    six_eight = num % 3 == 0 and den == 8
    for bar_no, (t, bar_end) in enumerate(sheet.bars):
        if bar_no == 0 and bar_end - t < bar_len:
            continue  # a pickup: the melody alone
        if six_eight:
            steps = [(t + Fraction(3, 2) * k, Fraction(3, 2)) for k in range(num // 3)]
        else:
            steps = [(t + beat * k, beat) for k in range(num)]
        for k, (s, d) in enumerate(steps):
            if s >= bar_end:
                break
            c = chord_at(s)
            if c is None or s < first:
                continue
            below = min(melody_low(s, s + d) - 1, 66)
            voicing = _voice(c.tones, below, previous)
            previous = voicing
            root_pc = c.bass if c.bass is not None else c.root
            root = voicing[0] - ((voicing[0] - root_pc) % 12 or 12)  # the chord's root just below the voicing
            while root > 50:
                root -= 12
            if six_eight:
                # Root, fifth, third... broken in eighths across the dotted beat.
                third = voicing[1] if len(voicing) > 1 else voicing[0]
                fifth = next((p for p in voicing if (p - root) % 12 == 7), voicing[-1])
                seq = [root, fifth, third]
                for j, p in enumerate(seq):
                    st = s + Fraction(1, 2) * j
                    lh.append(Note(round(st * q), round((st + Fraction(1, 2)) * q), p, 70, "L", 1, 1))
                continue
            strong = k == 0 or (num == 4 and k == 2)
            end = s + d
            if strong:
                lh.append(Note(round(s * q), round(end * q), root, 74, "L", 1, 1))
            else:
                for p in voicing:
                    lh.append(Note(round(s * q), round((s + d * Fraction(3, 4)) * q), p, 62, "L", 1, 1))
    song = Song(ticks_per_beat=q)
    song.meta = [
        (0, mido.MetaMessage("set_tempo", tempo=int(round(mido.bpm2tempo(sheet.tempo))))),
        (0, mido.MetaMessage("time_signature", numerator=num, denominator=den)),
        (0, mido.MetaMessage("key_signature", key=_key_name(sheet.key_fifths, sheet.key_minor))),
    ]
    song.tracks = [TrackInfo(0, "RH", rh), TrackInfo(1, "LH", sorted(lh, key=lambda n: (n.start, n.pitch)))]
    return song


def _key_name(fifths: int, minor: bool) -> str:
    major = ["Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#"]
    minors = ["Abm", "Ebm", "Bbm", "Fm", "Cm", "Gm", "Dm", "Am", "Em", "Bm", "F#m", "C#m", "G#m", "D#m", "A#m"]
    return (minors if minor else major)[fifths + 7]


def load(path: str | Path) -> tuple[Song, LeadSheet]:
    sheet = parse(Path(path).read_text(encoding="utf-8"))
    return arrange(sheet), sheet
