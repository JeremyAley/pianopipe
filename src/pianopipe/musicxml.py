"""Reading MusicXML scores (.musicxml, .xml, .mxl) into a Song with hands from the staves.

Standard library only (zipfile + xml.etree), like the rest of the pipeline. The design is in
docs/catalog-expansion.md §2; in short:

- Hands: a keyboard part's staff 1 is the right hand and staff 2 the left. A note belongs to the home staff of its
  voice (where most of the voice's notes are; MuseScore numbers staff 1's voices 1-4 and staff 2's 5-8), so
  cross-staff notes stay with the hand that plays them. Two one-staff keyboard parts are right and left by pitch. Choir scores (soprano, alto, tenor, bass)
  use the SATB rule: soprano and alto to the right hand, tenor and bass to the left (hand_source "satb"; the tenor
  is moved up where the left hand would span more than an octave by `satb.reach`, which process.load_notes runs).
- Repeats, voltas, D.C., D.S., To Coda and Fine are unrolled (no repeats after a jump), so the song is one timeline.
- Ties merge on <tie> (playback), unisons within a hand become one note, grace notes take time from the previous
  note (acciaccatura) or the next one (appoggiatura).
- Tempo from <sound tempo>, else <metronome>, else tempo words; time and key signatures become meta events.
- Printed fingering (<fingering>) is kept in Note.finger.
"""

from __future__ import annotations

import io
import math
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import mido

from .model import Note, Song, TrackInfo

SUFFIXES = (".musicxml", ".xml", ".mxl")
TPB = 480
MIN_BPM, MAX_BPM = 20, 300
GRACE_LENGTH = Fraction(1, 8)  # an acciaccatura lasts a 32nd note (in quarter notes)
TIE_GAP = Fraction(4)  # the longest gap (quarter notes) a tie may bridge
DEFAULT_VELOCITY = 80

STEPS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
MAJOR_KEYS = ["Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#"]
MINOR_KEYS = ["Abm", "Ebm", "Bbm", "Fm", "Cm", "Gm", "Dm", "Am", "Em", "Bm", "F#m", "C#m", "G#m", "D#m", "A#m"]
# Tempo words used when a marking has no <sound tempo> or <metronome> (quarter notes per minute).
TEMPO_WORDS = [
    ("prestissimo", 200), ("presto", 180), ("vivace", 160), ("allegro", 132), ("allegretto", 112),
    ("moderato", 108), ("andantino", 92), ("andante", 76), ("adagio", 60), ("larghetto", 60), ("lento", 52),
    ("largo", 46), ("grave", 40),
]
_KEYBOARD = re.compile(r"piano|klavier|pianoforte|фортепиано|keyboard|keys|harpsichord|clavecin|cembalo|organ|orgel|"
                       r"celesta|clavichord|pno", re.I)
_CHOIR = {
    "S": re.compile(r"^\s*(s\.?|sop\.?|soprano|sopran|soprani|treble|cantus|superius)\s*\d*\s*$", re.I),
    "A": re.compile(r"^\s*(a\.?|alt\.?|alto|alt|contralto|altus)\s*\d*\s*$", re.I),
    "T": re.compile(r"^\s*(t\.?|ten\.?|tenor|tenore|tenori)\s*\d*\s*$", re.I),
    "B": re.compile(r"^\s*(b\.?|bass|basso|bassi|basse|bassus|baritone)\s*\d*\s*$", re.I),
}
_CHOIR_PART = re.compile(r"choir|chorus|satb|chor\b|voices", re.I)
_SETTING_MARKERS = re.compile(r"\b(r\.?h\.?|m\.?d\.?|l\.?h\.?|m\.?s\.?)\b", re.I)


class MusicXMLError(ValueError):
    """The file isn't a score we can turn into two hands (timewise, empty, a piano duet...)."""


@dataclass
class _Note:
    pos: Fraction  # quarter notes from the start of the measure
    dur: Fraction
    pitch: int
    staff: int
    voice: str
    velocity: int
    tie_start: bool = False
    tie_stop: bool = False
    grace: str = ""  # "", "acciaccatura" or "appoggiatura"
    finger: int | None = None
    chord: bool = False


@dataclass
class _Measure:
    length: Fraction = Fraction(0)
    notes: list[_Note] = field(default_factory=list)
    tempos: list[tuple[Fraction, float, str]] = field(default_factory=list)  # (pos, bpm, kind)
    time: tuple[int, int] | None = None
    key: str | None = None
    forward: bool = False
    backward: int = 0  # times to play, 0 = no backward repeat
    ending_start: set[int] | None = None
    ending_stop: bool = False
    segno: bool = False
    coda: bool = False
    dacapo: bool = False
    dalsegno: bool = False
    tocoda: bool = False
    fine: bool = False


@dataclass
class _Part:
    id: str
    name: str
    program: int | None  # GM program, 1-based as in MusicXML
    percussion: bool
    staves: int = 1
    measures: list[_Measure] = field(default_factory=list)

    @property
    def keyboard(self) -> bool:
        return bool(_KEYBOARD.search(self.name)) or self.staves >= 2 or (self.program is not None and 1 <= self.program <= 8)

    @property
    def choir_voice(self) -> str | None:
        return next((v for v, rx in _CHOIR.items() if rx.match(self.name)), None)

    def pitched_notes(self) -> int:
        return sum(len(m.notes) for m in self.measures)


@dataclass
class Score:
    song: Song
    hand_source: str  # "staves", "satb" or "none" (hands left to the automatic splitter)
    title: str = ""
    composer: str = ""
    arranger: str = ""
    parts: list[str] = field(default_factory=list)  # part names, for reports
    warnings: list[str] = field(default_factory=list)
    bars: list[int] = field(default_factory=list)  # tick where each played (unrolled) measure starts, then the end

    @property
    def notes(self) -> list[Note]:
        return self.song.notes


def read_root(path: str | Path) -> ET.Element:
    """The <score-partwise> element of a .musicxml/.xml file or a compressed .mxl."""
    data = Path(path).read_bytes()
    if data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = z.namelist()
            name = None
            if "META-INF/container.xml" in names:
                rootfile = ET.fromstring(z.read("META-INF/container.xml")).find(".//{*}rootfile")
                if rootfile is not None:
                    name = rootfile.get("full-path")
            if not name:
                name = next((n for n in names if n.lower().endswith((".xml", ".musicxml")) and not n.startswith("META-INF")), None)
            if not name:
                raise MusicXMLError("no score inside the .mxl file")
            data = z.read(name)
    root = ET.fromstring(data)
    _strip_namespaces(root)
    if root.tag == "score-timewise":
        raise MusicXMLError("timewise MusicXML isn't supported (export it as partwise)")
    if root.tag != "score-partwise":
        raise MusicXMLError(f"not a MusicXML score (<{root.tag}>)")
    return root


def _strip_namespaces(root: ET.Element) -> None:
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]


def _num(text: str | None, default: float = 0.0) -> float:
    try:
        return float(text) if text is not None else default
    except ValueError:
        return default


def _key_name(fifths: int, mode: str) -> str:
    fifths = max(-7, min(7, fifths))
    return (MINOR_KEYS if mode == "minor" else MAJOR_KEYS)[fifths + 7]


def _tempo_word(text: str) -> float | None:
    low = text.lower()
    return next((float(bpm) for word, bpm in TEMPO_WORDS if re.search(rf"\b{word}\b", low)), None)


def _parse_part(part_el: ET.Element, info: dict) -> _Part:
    part = _Part(part_el.get("id", ""), info.get("name", ""), info.get("program"), info.get("percussion", False))
    divisions = 1
    transpose = 0
    velocity = DEFAULT_VELOCITY
    for m_el in part_el.findall("measure"):
        m = _Measure()
        pos = Fraction(0)
        last_start = Fraction(0)
        furthest = Fraction(0)
        for el in m_el:
            tag = el.tag
            if tag == "attributes":
                d = el.findtext("divisions")
                if d:
                    divisions = max(1, int(_num(d, 1)))
                s = el.findtext("staves")
                if s:
                    part.staves = max(part.staves, int(_num(s, 1)))
                t = el.find("time")
                if t is not None and t.findtext("beats"):
                    beats = sum(int(_num(b, 0)) for b in re.split(r"[+]", t.findtext("beats")) if b.strip())
                    beat_type = int(_num(t.findtext("beat-type"), 4))
                    if beats > 0 and beat_type > 0 and beat_type & (beat_type - 1) == 0:
                        m.time = (beats, beat_type)
                k = el.find("key")
                if k is not None and k.findtext("fifths") is not None:
                    m.key = _key_name(int(_num(k.findtext("fifths"))), (k.findtext("mode") or "major").strip().lower())
                tr = el.find("transpose")
                if tr is not None:
                    transpose = int(_num(tr.findtext("chromatic"))) + 12 * int(_num(tr.findtext("octave-change")))
            elif tag in ("backup", "forward"):
                d = Fraction(int(_num(el.findtext("duration"))), divisions)
                pos = pos - d if tag == "backup" else pos + d
                pos = max(pos, Fraction(0))
                furthest = max(furthest, pos)
            elif tag == "direction":
                at = pos + Fraction(int(_num(el.findtext("offset"))), divisions)
                velocity = _direction(el, m, at, velocity)
            elif tag == "sound":
                velocity = _sound(el, m, pos, velocity)
            elif tag == "barline":
                _barline(el, m)
            elif tag == "note":
                is_chord = el.find("chord") is not None
                grace_el = el.find("grace")
                dur = Fraction(int(_num(el.findtext("duration"))), divisions)
                start = last_start if is_chord else pos
                if grace_el is None and not is_chord:
                    last_start = pos
                    pos += dur
                    furthest = max(furthest, pos)
                elif is_chord:
                    furthest = max(furthest, start + dur)
                pitch_el = el.find("pitch")
                if pitch_el is None or el.find("cue") is not None or el.find("rest") is not None:
                    continue
                pitch = (12 * (int(_num(pitch_el.findtext("octave"), 4)) + 1) + STEPS.get(pitch_el.findtext("step", "C"), 0)
                         + round(_num(pitch_el.findtext("alter"))) + transpose)
                if not 0 <= pitch <= 127:
                    continue
                ties = {t.get("type") for t in el.findall("tie")}
                grace = ""
                if grace_el is not None:
                    grace = "acciaccatura" if grace_el.get("slash") == "yes" or grace_el.get("steal-time-previous") else "appoggiatura"
                vel = round(_num(el.get("dynamics")) * 0.9) if el.get("dynamics") else velocity
                finger = None
                for f in el.iter("fingering"):
                    digits = re.findall(r"[1-5]", f.text or "")
                    if digits and f.get("substitution") != "yes":
                        finger = int(digits[0])
                        break
                m.notes.append(_Note(
                    start, dur, pitch, int(_num(el.findtext("staff"), 1)), (el.findtext("voice") or "1").strip(),
                    max(1, min(127, vel)), "start" in ties, "stop" in ties, grace, finger, is_chord,
                ))
        m.length = max(furthest, pos)
        part.measures.append(m)
    return part


def _sound(el: ET.Element, m: _Measure, at: Fraction, velocity: int) -> int:
    if el.get("tempo"):
        m.tempos.append((at, _num(el.get("tempo")), "sound"))
    if el.get("dynamics"):
        velocity = max(1, min(127, round(_num(el.get("dynamics")) * 0.9)))
    m.segno |= el.get("segno") is not None
    m.coda |= el.get("coda") is not None
    m.dacapo |= el.get("dacapo") == "yes"
    m.dalsegno |= el.get("dalsegno") is not None
    m.tocoda |= el.get("tocoda") is not None
    m.fine |= el.get("fine") is not None
    return velocity


def _direction(el: ET.Element, m: _Measure, at: Fraction, velocity: int) -> int:
    snd = el.find("sound")
    if snd is not None:
        velocity = _sound(snd, m, at, velocity)
    if snd is None or not snd.get("tempo"):
        met = el.find(".//metronome")
        if met is not None and met.findtext("per-minute"):
            unit = {"whole": 4, "half": 2, "quarter": 1, "eighth": Fraction(1, 2), "16th": Fraction(1, 4)}.get(
                met.findtext("beat-unit", "quarter"), 1)
            if met.find("beat-unit-dot") is not None:
                unit = unit * Fraction(3, 2)
            per_minute = re.findall(r"[\d.]+", met.findtext("per-minute"))
            if per_minute:
                m.tempos.append((at, float(per_minute[0]) * float(unit), "metronome"))
        else:
            words = " ".join(w.text or "" for w in el.iter("words"))
            bpm = _tempo_word(words) if words else None
            if bpm:
                m.tempos.append((at, bpm, "words"))
    m.segno |= el.find(".//segno") is not None
    m.coda |= el.find(".//coda") is not None
    return velocity


def _barline(el: ET.Element, m: _Measure) -> None:
    r = el.find("repeat")
    if r is not None:
        if r.get("direction") == "forward":
            m.forward = True
        elif r.get("direction") == "backward":
            m.backward = max(2, int(_num(r.get("times"), 2)))
    e = el.find("ending")
    if e is not None:
        kind = e.get("type")
        if kind == "start":
            nums = {int(x) for x in re.findall(r"\d+", e.get("number", "1"))}
            m.ending_start = nums or {1}
        elif kind in ("stop", "discontinue"):
            m.ending_stop = True
    snd = el.find("sound")
    if snd is not None:
        _sound(snd, m, Fraction(0), DEFAULT_VELOCITY)


def play_order(measures: list[_Measure]) -> list[int]:
    """Measure indices in playing order: repeats and voltas unrolled, then D.C./D.S. with To Coda and Fine.

    Repeats aren't taken again after a D.C. or D.S. (the usual convention); in a volta group the last ending is
    played instead."""
    n = len(measures)
    # Which ending numbers each measure belongs to, and the last number of its volta group.
    ending_of: list[set[int] | None] = [None] * n
    group_last: list[int] = [0] * n
    current: set[int] | None = None
    group: list[int] = []
    group_max = 0
    for i, m in enumerate(measures):
        if m.ending_start is not None:
            current = m.ending_start
            group_max = max(group_max, max(current))
        elif current is None and group:
            for j in group:
                group_last[j] = group_max
            group, group_max = [], 0
        if current is not None:
            ending_of[i] = current
            group.append(i)
        if m.ending_stop:
            current = None
    for j in group:
        group_last[j] = group_max

    segno = next((i for i, m in enumerate(measures) if m.segno), 0)
    # The coda sign after the To Coda measure (some files also put a coda glyph on the To Coda direction).
    to_coda = next((i for i, m in enumerate(measures) if m.tocoda), -1)
    coda = next((i for i, m in enumerate(measures) if m.coda and i > to_coda and not m.tocoda), None)
    order: list[int] = []
    i, section_start, passes, jumped = 0, 0, 1, False
    taken: dict[int, int] = {}
    limit = 20 * n + 20
    while i < n and len(order) < limit:
        m = measures[i]
        if m.forward and i != section_start:
            section_start, passes = i, 1
        endings = ending_of[i]
        if endings is not None and ((jumped and group_last[i] not in endings) or (not jumped and passes not in endings)):
            i += 1
            continue
        order.append(i)
        if jumped and m.fine:
            break
        if jumped and m.tocoda and coda is not None and coda > i:
            i = coda
            continue
        if m.backward and not jumped:
            done = taken.get(i, 1)
            if done < m.backward:
                taken[i] = done + 1
                passes += 1
                i = section_start
                continue
            taken[i] = 1
            section_start, passes = i + 1, 1
        if (m.dacapo or m.dalsegno) and not jumped:
            jumped = True
            i = segno if m.dalsegno else 0
            section_start, passes = i, 1
            continue
        i += 1
    return order


def _part_info(root: ET.Element) -> dict[str, dict]:
    info = {}
    for sp in root.iter("score-part"):
        name = (sp.findtext("part-name") or "").strip()
        if not name or name.lower() in ("mvmt", "staff", "music"):
            name = (sp.findtext(".//instrument-name") or name).strip()
        program = sp.findtext(".//midi-program")
        channel = sp.findtext(".//midi-channel")
        percussion = channel == "10" or "drum" in (sp.findtext(".//instrument-sound") or "")
        info[sp.get("id", "")] = {"name": name, "program": int(_num(program)) if program else None, "percussion": percussion}
    return info


def _credit(root: ET.Element, kind: str) -> str:
    names = [c.text.strip() for c in root.iter("creator") if c.get("type") == kind and c.text and c.text.strip()]
    return "; ".join(names)


def _unroll(parts: list[_Part], order: list[int]) -> tuple[list[Fraction], Fraction]:
    """Start of each played measure (quarter notes) on the unrolled timeline, and the total length."""
    starts, t = [], Fraction(0)
    for mi in order:
        starts.append(t)
        t += max((p.measures[mi].length for p in parts if mi < len(p.measures)), default=Fraction(0))
    return starts, t


@dataclass
class _Event:
    start: Fraction
    end: Fraction
    pitch: int
    velocity: int
    part: int
    staff: int
    voice: str
    finger: int | None


def _events(parts: list[_Part], order: list[int], starts: list[Fraction]) -> list[_Event]:
    """Every sounding note on the unrolled timeline, with ties merged and grace notes placed."""
    out: list[_Event] = []
    for pi, part in enumerate(parts):
        open_ties: dict[tuple, _Event] = {}
        for mi, t0 in zip(order, starts):
            if mi >= len(part.measures):
                continue
            pending: dict[str, list[_Note]] = {}  # grace notes waiting for their main note, per voice
            for n in part.measures[mi].notes:
                if n.grace:
                    pending.setdefault(n.voice, []).append(n)
                    continue
                start, end = t0 + n.pos, t0 + n.pos + n.dur
                graces = pending.pop(n.voice, [])
                if graces:
                    start = _place_graces(graces, n, start, end, pi, out)
                key = (n.voice, n.pitch)
                prev = None
                if n.tie_stop:
                    prev = open_ties.pop(key, None)
                    if prev is None:  # tied from another voice
                        other = next((k for k in open_ties if k[1] == n.pitch), None)
                        prev = open_ties.pop(other) if other else None
                # A tie may skip over other notes of its voice ("hold the D while the A sounds"); MuseScore plays
                # that as one held note, so any gap up to a bar is bridged.
                if prev is not None and start - prev.end <= TIE_GAP:
                    prev.end = max(prev.end, end)
                    ev = prev
                else:
                    ev = _Event(start, end, n.pitch, n.velocity, pi, n.staff, n.voice, n.finger)
                    out.append(ev)
                if n.tie_start:
                    open_ties[key] = ev
    return out


def _place_graces(graces: list[_Note], main: _Note, start: Fraction, end: Fraction, part: int, out: list[_Event]) -> Fraction:
    """Add grace notes before `main`; returns the main note's (possibly delayed) start."""
    onsets: list[list[_Note]] = []
    for g in graces:
        if g.chord and onsets:
            onsets[-1].append(g)
        else:
            onsets.append([g])
    if graces[0].grace == "acciaccatura":
        step = GRACE_LENGTH
        t = max(Fraction(0), start - step * len(onsets))
        for group in onsets:
            for g in group:
                out.append(_Event(t, t + step, g.pitch, g.velocity, part, g.staff, g.voice, g.finger))
            t += step
        return start
    # Appoggiatura: the graces share the first half of the main note.
    share = (end - start) / 2 / len(onsets)
    t = start
    for group in onsets:
        for g in group:
            out.append(_Event(t, t + share, g.pitch, g.velocity, part, g.staff, g.voice, g.finger))
        t += share
    return t


def _voice_homes(part: _Part) -> dict[str, int]:
    """Each voice's home staff: the staff holding at least two thirds of its notes. MuseScore numbers staff 1's voices
    1-4 and staff 2's 5-8, so a voice's cross-staff notes belong to the hand of its home staff. Voices spread over
    both staves (other programs reuse numbers per staff) get no home, and their notes follow their own staff."""
    counts: dict[str, dict[int, int]] = {}
    for m in part.measures:
        for n in m.notes:
            counts.setdefault(n.voice, {}).setdefault(n.staff, 0)
            counts[n.voice][n.staff] += 1
    homes = {}
    for voice, by_staff in counts.items():
        staff, top = max(by_staff.items(), key=lambda kv: kv[1])
        if top * 3 >= 2 * sum(by_staff.values()):
            homes[voice] = staff
    return homes


def load(path: str | Path) -> Score:
    root = read_root(path)
    info = _part_info(root)
    parts = [_parse_part(p, info.get(p.get("id", ""), {})) for p in root.findall("part")]
    parts = [p for p in parts if not p.percussion and p.pitched_notes()]
    if not parts:
        raise MusicXMLError("no pitched notes")
    warnings: list[str] = []
    title = (root.findtext("work/work-title") or root.findtext("movement-title") or "").strip()

    # Structure (repeats, jumps, tempo) comes from every part's barlines and directions, merged per measure.
    n = max(len(p.measures) for p in parts)
    merged = []
    for i in range(n):
        mm = _Measure()
        for p in parts:
            if i < len(p.measures):
                pm = p.measures[i]
                mm.forward |= pm.forward
                mm.backward = max(mm.backward, pm.backward)
                if pm.ending_start is not None and mm.ending_start is None:
                    mm.ending_start = pm.ending_start
                mm.ending_stop |= pm.ending_stop
                for flag in ("segno", "coda", "dacapo", "dalsegno", "tocoda", "fine"):
                    setattr(mm, flag, getattr(mm, flag) or getattr(pm, flag))
        merged.append(mm)
    order = play_order(merged)
    starts, total = _unroll(parts, order)

    keyboard = [p for p in parts if p.keyboard and not p.choir_voice]
    choir = [p for p in parts if p.choir_voice]
    events = _events(parts, order, starts)
    hand_of: dict[int, object] = {}
    satb = False
    if len(choir) >= 3 and {"S", "B"} <= {p.choir_voice for p in choir}:
        satb = True
        use = {parts.index(p): ("R" if p.choir_voice in ("S", "A") else "L") for p in choir}
        hand = lambda e: use.get(e.part)  # noqa: E731
    elif len([p for p in keyboard if p.staves >= 2]) >= 2:
        raise MusicXMLError("piano duet (two keyboard parts with two staves each)")
    elif any(p.staves >= 2 for p in keyboard):
        kp = next(p for p in keyboard if p.staves >= 2)
        pi = parts.index(kp)
        homes = _voice_homes(kp)
        satb = bool(_CHOIR_PART.search(kp.name))

        def hand(e: _Event, pi=pi, homes=homes):
            if e.part != pi:
                return None
            return "R" if homes.get(e.voice, e.staff) == 1 else "L"
        others = [p.name or p.id for p in parts if p is not kp]
        if others:
            warnings.append(f"used the keyboard part {kp.name or kp.id!r}; left out {', '.join(others)}")
    elif len(keyboard) == 2:
        means = {parts.index(p): sum(nn.pitch for m in p.measures for nn in m.notes) / p.pitched_notes() for p in keyboard}
        hi = max(means, key=means.get)
        hand = lambda e, hi=hi: None if e.part not in means else ("R" if e.part == hi else "L")  # noqa: E731
    elif len(parts) == 1 and parts[0].staves == 1:
        hand = lambda e: "R"  # noqa: E731 - a melody alone; the gate decides whether that's enough
        warnings.append("single staff: right hand only")
    else:
        hand = lambda e: "?"  # noqa: E731 - no structure to read hands from
    tpb = TPB
    notes: list[Note] = []
    for e in events:
        h = hand(e)
        if h is None:
            continue
        notes.append(Note(
            round(e.start * tpb), round(e.end * tpb), e.pitch, e.velocity, None if h == "?" else h,
            0 if h != "L" else 1, 0 if h != "L" else 1, 0, e.finger,
        ))
    notes = [nt for nt in notes if nt.end > nt.start]
    hand_source = "none" if any(nt.hand is None for nt in notes) else ("satb" if satb else "staves")
    notes = merge_unisons(notes)
    song = Song(ticks_per_beat=tpb, meta=_meta(parts, merged, order, starts, tpb, warnings, total))
    song.tracks = [
        TrackInfo(0, "RH", sorted((x for x in notes if x.hand == "R"), key=lambda x: (x.start, x.pitch))),
        TrackInfo(1, "LH", sorted((x for x in notes if x.hand == "L"), key=lambda x: (x.start, x.pitch))),
    ]
    if hand_source == "none":
        song.tracks = [TrackInfo(0, "", sorted(notes, key=lambda x: (x.start, x.pitch)))]
    return Score(song, hand_source, title, _credit(root, "composer"), _credit(root, "arranger"),
                 [p.name or p.id for p in parts], warnings, [round(t * tpb) for t in starts] + [round(total * tpb)])


def merge_unisons(notes: list[Note]) -> list[Note]:
    """One key can't sound twice: same-pitch notes that start together become the longer one (in the left hand below
    middle C, else the right, when both staves have it), and a note still sounding when its key is struck again by
    the same hand is cut there."""
    starts: dict[tuple, set] = {}
    for n in notes:
        starts.setdefault((n.start, n.pitch), set()).add(n.hand)
    keep_hand = {k: ("L" if k[1] < 60 else "R") for k, hands in starts.items() if len(hands) > 1}
    moved = []
    for n in notes:
        k = (n.start, n.pitch)
        if k in keep_hand and n.hand != keep_hand[k]:
            n = Note(n.start, n.end, n.pitch, n.velocity, keep_hand[k], 1 if keep_hand[k] == "L" else 0,
                     1 if keep_hand[k] == "L" else 0, n.program, n.finger)
        moved.append(n)
    out: list[Note] = []
    by_key: dict[tuple, list[Note]] = {}
    for n in moved:
        by_key.setdefault((n.hand, n.pitch), []).append(n)
    for group in by_key.values():
        group.sort(key=lambda n: (n.start, -n.end))
        kept: list[Note] = []
        for n in group:
            if kept and n.start == kept[-1].start:
                if n.end > kept[-1].end:
                    kept[-1] = Note(kept[-1].start, n.end, n.pitch, max(n.velocity, kept[-1].velocity), n.hand,
                                    kept[-1].track, kept[-1].channel, kept[-1].program, kept[-1].finger or n.finger)
                continue
            if kept and n.start < kept[-1].end:
                p = kept[-1]
                kept[-1] = Note(p.start, n.start, p.pitch, p.velocity, p.hand, p.track, p.channel, p.program, p.finger)
            kept.append(n)
        out.extend(kept)
    return sorted(out, key=lambda n: (n.start, n.pitch))


def partial_meter(length: Fraction, den: int) -> tuple[int, int]:
    """The time signature of a bar `length` quarters long (a pickup), in the meter's own unit where it fits."""
    while den < 64 and (length * den / 4).denominator != 1:
        den *= 2
    return max(1, round(length * den / 4)), den


def _meta(parts: list[_Part], merged: list[_Measure], order: list[int], starts: list[Fraction], tpb: int,
          warnings: list[str], total: Fraction | None = None) -> list:
    """Tempo, time and key signature events on the unrolled timeline. A played measure shorter than its time signature
    (a pickup, the bar that completes it, a bar split at a repeat sign) gets a time signature of its own length, and the
    real one again at the next measure: MIDI has no partial bars, and the app counts bars by time signature
    (MidiFile.Build), so without it every later bar line would be off (docs/sheet-music.md)."""
    tempos: dict[int, float] = {}
    times: dict[int, tuple[int, int]] = {}
    keys: dict[int, str] = {}
    cur_time, cur_key = None, None
    for mi, t0 in zip(order, starts):
        tick0 = round(t0 * tpb)
        for p in parts:
            if mi >= len(p.measures):
                continue
            pm = p.measures[mi]
            if pm.time and pm.time != cur_time:
                times[tick0] = pm.time
            if pm.key and pm.key != cur_key:
                keys[tick0] = pm.key
            for pos, bpm, kind in sorted(pm.tempos, key=lambda x: (x[0], x[2] != "words")):
                if MIN_BPM <= bpm <= MAX_BPM:
                    tempos[round((t0 + pos) * tpb)] = bpm
                else:
                    warnings.append(f"ignored tempo {bpm:g} in measure {mi + 1}")
        cur_time = times.get(tick0, cur_time)
        cur_key = keys.get(tick0, cur_key)
    # Partial bars.
    ends = list(starts[1:]) + ([total] if total is not None else [])
    meter = None
    for k, (t0, t1) in enumerate(zip(starts, ends)):
        tick0 = round(t0 * tpb)
        meter = times.get(tick0, meter)
        if meter is None:
            continue
        nominal = Fraction(4 * meter[0], meter[1])
        if 0 < t1 - t0 < nominal:
            times[tick0] = partial_meter(t1 - t0, meter[1])
            nxt = round(t1 * tpb)
            if k + 1 < len(starts) and nxt not in times:
                times[nxt] = meter
            # Keep the real meter for what follows (times[tick0] is now the partial one).
    meta = []
    for tick, bpm in sorted(tempos.items()):
        meta.append((tick, mido.MetaMessage("set_tempo", tempo=int(round(mido.bpm2tempo(bpm))))))
    if not tempos:
        warnings.append("no tempo marking: 100 per minute")
        meta.append((0, mido.MetaMessage("set_tempo", tempo=int(mido.bpm2tempo(100)))))
    for tick, (num, den) in sorted(times.items()):
        meta.append((tick, mido.MetaMessage("time_signature", numerator=num, denominator=den)))
    for tick, key in sorted(keys.items()):
        meta.append((tick, mido.MetaMessage("key_signature", key=key)))
    meta.sort(key=lambda e: e[0])
    return meta
