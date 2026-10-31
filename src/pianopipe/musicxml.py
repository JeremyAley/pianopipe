"""Reading MusicXML scores (.musicxml, .xml, .mxl) into a Song with hands from the staves.

Standard library only (zipfile + xml.etree), like the rest of the pipeline. The design is in
docs/catalog-expansion.md §2; in short:

- Hands: a keyboard part's staff 1 is the right hand and staff 2 the left. A note belongs to the home staff of its
  voice (where most of the voice's notes are; MuseScore numbers staff 1's voices 1-4 and staff 2's 5-8), so
  cross-staff notes stay with the hand that plays them, unless that hand would stretch past a tenth or leap away to
  reach one (hands.repair_hands, which also splits chords one hand can't play). Two one-staff keyboard parts are right and left by pitch. Choir scores (soprano, alto, tenor, bass)
  use the SATB rule: soprano and alto to the right hand, tenor and bass to the left (hand_source "satb"; the tenor
  is moved up where the left hand would span more than an octave by `satb.reach`; so is a two-staff hymn's with
  `load(satb=True)`).
- Repeats, voltas, D.C., D.S., To Coda and Fine are unrolled (no repeats after a jump), so the song is one timeline.
  Jumps written only as words (Audiveris's OCR) count too; a missing segno or coda sign is guessed, with a warning.
- Ties merge on <tie> (playback), unisons within a hand become one note, grace notes take time from the previous
  note (acciaccatura) or the next one (appoggiatura).
- Tempo from <sound tempo>, else <metronome>, else tempo words (Italian, or English above the staff: "Moderately
  fast", "Bright Rock"; "Tempo I" goes back to the first); time and key signatures become meta events.
- Printed fingering (<fingering>) is kept in Note.finger.
- With `load(melody=...)`, a song's voice or melody part is merged into the right hand as its top voice (melody.py),
  in the bars whose note values add up; the others keep the piano's own top line. Merged or not, a voice or
  melody-instrument part's line is Score.tune, which levels 1 and 2 take as their tune (difficulty.make_levels).
"""

from __future__ import annotations

import io
import math
from bisect import bisect_right
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path

import mido

from .hands import repair_hands
from .melody import merge as merge_melody, top_line
from .model import Note, Song, TrackInfo
from .satb import reach

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
# English markings on US publisher scores (pop arrangements in 4/4 mostly). Matched only above the staff: Audiveris
# exports lyrics below it as words too.
ENGLISH_TEMPO_WORDS = [
    ("very slowly", 50), ("very slow", 50), ("slowly", 66), ("slow", 66), ("broadly", 60), ("freely", 72),
    ("tenderly", 72), ("deliberately", 72), ("majestically", 72), ("expressively", 76), ("with expression", 76),
    ("gently", 76), ("ballad", 72), ("slow ballad", 63), ("rock ballad", 76), ("pop ballad", 76), ("slow rock", 76),
    ("moderately slow", 80), ("moderately slowly", 80), ("not too slow", 80), ("leisurely", 84), ("relaxed", 88),
    ("smoothly", 88), ("flowing", 92), ("gospel", 80), ("moderately", 100), ("moderate", 100), ("steadily", 100),
    ("medium", 108), ("easy swing", 108), ("moderate rock", 112), ("shuffle", 112), ("march", 116),
    ("not too fast", 116), ("not fast", 116), ("swing", 126), ("medium swing", 120), ("rock", 120),
    ("moderately fast", 126), ("moderately bright", 132), ("spirited", 132), ("briskly", 132), ("lively", 144),
    ("waltz", 144), ("slow waltz", 108), ("bright", 152), ("brightly", 152), ("bright rock", 152),
    ("bright swing", 160), ("fast", 160), ("quickly", 160), ("fast swing", 176), ("very fast", 184),
]
_TEMPO_PRIMO = re.compile(r"\btempo\s+(i|1|primo)\b", re.I)
# Jumps written as words, for exports that leave out <sound dalsegno/tocoda/...> (Audiveris's OCR): the words must
# start the direction, so lyrics exported as words don't count.
_JUMP_WORDS = [
    ("dalsegno", re.compile(r"^(d\.?\s?[s%]\.?|dal\s+segno)(\W|$)", re.I)),
    ("dacapo", re.compile(r"^(d\.?\s?c\.?|da\s+capo)(\W|$)", re.I)),
    ("tocoda", re.compile(r"^to\s+coda\b", re.I)),
    ("coda", re.compile(r"^coda\W*$", re.I)),
    ("segno", re.compile(r"^segno\W*$", re.I)),
    ("fine_word", re.compile(r"^fine\W*$", re.I)),
]
JUMP_FLAGS = ("segno", "coda", "dacapo", "dalsegno", "tocoda", "fine")
INDENT = 10  # tenths: a system whose left margin is this much wider than the score's usual one is indented
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
    fill: dict[str, Fraction] = field(default_factory=dict)  # each voice's notes, rests and forwards, summed
    rested: set[str] = field(default_factory=set)  # voices with a whole-measure rest
    fine_word: bool = False  # "Fine" as words only: honoured where a jump goes to a Fine (lyrics may say "fine")
    al_fine: bool = False  # a D.C. or D.S. read from words that says "al Fine"
    al_coda: bool = False  # ... or "al Coda"
    jump_sound: bool = False  # a D.C. or D.S. from a <sound> attribute
    system: bool = False  # starts a system (new system or page)
    margin: float | None = None  # its system's left margin (tenths)


@dataclass
class _Part:
    id: str
    name: str
    program: int | None  # GM program, 1-based as in MusicXML
    percussion: bool
    staves: int = 1
    measures: list[_Measure] = field(default_factory=list)
    clef_octave: int = 0  # an octave clef's change (-1 for a tenor's treble clef with an 8 below)

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
    order: list[int] = field(default_factory=list)  # each played measure's written measure (0-based), in playing order
    keyboard: str = ""  # id of the two-staff keyboard part the hands came from, if they came from one
    repaired: list[int] = field(default_factory=list)  # ticks of chords split between the hands to be playable
    melody: dict = field(default_factory=dict)  # load(melody=...): the parts merged, their bars merged and skipped
    tune: list[Note] = field(default_factory=list)  # a voice part's tune for levels 1 and 2 (§2.6), in tune_bars
    tune_bars: list[tuple[int, int]] = field(default_factory=list)  # (start, end) ticks of the bars the tune covers

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


def _tempo_word(text: str, english: bool = True) -> float | None:
    """The tempo of a marking's words: its earliest tempo word, the longest of those starting there ("Moderately fast"
    before "Moderately"). `english`: English words count too (not below the staff, where lyrics are), also as OCR
    misreads them with I for l ("SIOWIY")."""
    low = text.lower()
    tries = [(low, TEMPO_WORDS + (ENGLISH_TEMPO_WORDS if english else []))]
    if english:
        tries.append((re.sub(r"[i1|]", "l", low), ENGLISH_TEMPO_WORDS))
    for words, table in tries:
        best: tuple[tuple[int, int], int] | None = None
        for word, bpm in table:
            m = re.search(rf"\b{word.replace(' ', r'[\s,-]+')}\b", words)
            if m and (best is None or (m.start(), -len(word)) < best[0]):
                best = ((m.start(), -len(word)), bpm)
        if best:
            return float(best[1])
    return None


def _jump_words(text: str, m: _Measure) -> None:
    """Jumps and signs written only as words (no <sound> attribute): D.S., D.C., To Coda, Coda, Segno, Fine."""
    text = re.sub(r"^[\W_]+", "", text.strip())
    for flag, rx in _JUMP_WORDS:
        if rx.match(text):
            setattr(m, flag, True)
            if flag in ("dalsegno", "dacapo"):
                m.al_fine |= bool(re.search(r"\bfine\b", text, re.I))
                m.al_coda |= bool(re.search(r"\bcoda\b", text, re.I))


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
                for c in el.findall("clef"):
                    part.clef_octave = int(_num(c.findtext("clef-octave-change")))
            elif tag in ("backup", "forward"):
                d = Fraction(int(_num(el.findtext("duration"))), divisions)
                if tag == "forward" and el.findtext("voice"):
                    voice = el.findtext("voice").strip()
                    m.fill[voice] = m.fill.get(voice, Fraction(0)) + d
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
            elif tag == "print":
                m.system |= "yes" in (el.get("new-system"), el.get("new-page")) or el.find("system-layout") is not None
                margin = el.findtext("system-layout/system-margins/left-margin")
                if margin is not None:
                    m.margin = _num(margin)
            elif tag == "note":
                is_chord = el.find("chord") is not None
                grace_el = el.find("grace")
                dur = Fraction(int(_num(el.findtext("duration"))), divisions)
                start = last_start if is_chord else pos
                if grace_el is None and not is_chord:
                    last_start = pos
                    pos += dur
                    furthest = max(furthest, pos)
                    voice = (el.findtext("voice") or "1").strip()
                    m.fill[voice] = m.fill.get(voice, Fraction(0)) + dur
                    rest = el.find("rest")
                    if rest is not None and rest.get("measure") == "yes":
                        m.rested.add(voice)
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
    m.jump_sound |= el.get("dacapo") == "yes" or el.get("dalsegno") is not None
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
            bpm = _tempo_word(words, english=el.get("placement") != "below") if words else None
            if bpm:
                m.tempos.append((at, bpm, "words"))
            elif _TEMPO_PRIMO.search(words):
                m.tempos.append((at, 0.0, "primo"))  # the first tempo again, found in _meta
    text = " ".join(w.text or "" for w in el.iter("words"))
    if snd is None or not any(snd.get(flag) is not None for flag in JUMP_FLAGS):
        _jump_words(text, m)
    elif snd.get("dacapo") == "yes" or snd.get("dalsegno") is not None:
        m.al_coda |= bool(re.search(r"\bcoda\b", text, re.I))
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


def play_order(measures: list[_Measure], warnings: list[str] | None = None) -> list[int]:
    """Measure indices in playing order: repeats and voltas unrolled, then D.C./D.S. with To Coda and Fine.

    Repeats aren't taken again after a D.C. or D.S. (the usual convention); in a volta group the last ending is
    played instead. A D.S. with no segno, or a To Coda with no coda sign (OCR'd exports have the words but not the
    signs), is guessed and said in `warnings` (_guess_segno, _guess_coda)."""
    warnings = [] if warnings is None else warnings
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
            if group and current <= {k for j in group for k in ending_of[j]}:
                # an ending numbered like one before it in its group (OCR reads a "2." as "1."): the next number
                current = {group_max + 1}
                warnings.append(f"the ending in measure {i + 1} has the number of an earlier one: read as "
                                f"ending {group_max + 1}")
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

    jump = next((i for i, m in enumerate(measures) if m.dacapo or m.dalsegno), None)
    segno = next((i for i, m in enumerate(measures) if m.segno), None)
    if segno is None:
        segno = 0
        if jump is not None and measures[jump].dalsegno:
            segno = _guess_segno(measures, jump, warnings)
    # The coda sign after the To Coda measure (some files also put a coda glyph on the To Coda direction).
    to_coda = next((i for i, m in enumerate(measures) if m.tocoda), -1)
    coda = next((i for i, m in enumerate(measures) if m.coda and i > to_coda and not m.tocoda), None)
    if to_coda >= 0 and coda is None:
        if jump is None or jump < to_coda:
            warnings.append(f"To Coda in measure {to_coda + 1} but no D.C. or D.S. after it: the To Coda is ignored")
        else:
            coda = _guess_coda(measures, to_coda, jump, warnings)
    if jump is not None and measures[jump].al_coda and to_coda < 0:
        warnings.append(f"al Coda in measure {jump + 1} but no To Coda: played to the end (check the score)")
    if jump is not None and measures[jump].al_fine and not any(m.fine for m in measures):
        warnings.append(f"al Fine in measure {jump + 1} but no Fine: played to the end (check the score)")
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


def _guess_segno(measures: list[_Measure], jump: int, warnings: list[str]) -> int:
    """Where a D.S. with no segno goes back to: the first forward repeat before it (a D.S. isn't a D.C., so the sign is
    after the start, and songs put it where a section starts), else the start."""
    segno = next((i for i, m in enumerate(measures[:jump]) if m.forward and i > 0), None)
    if segno is None:
        warnings.append(f"D.S. in measure {jump + 1} but no segno: played from the start (check the score)")
        return 0
    warnings.append(f"D.S. in measure {jump + 1} but no segno: guessed measure {segno + 1}, the first forward repeat "
                    f"(check the score)")
    return segno


def _guess_coda(measures: list[_Measure], to_coda: int, jump: int, warnings: list[str]) -> int | None:
    """Where a To Coda with no coda sign goes: the first indented system after the D.C. or D.S. (a coda printed as a
    section of its own; Audiveris makes it a movement of its own), else the measure after the jump if it starts a
    system. None (and the To Coda ignored) when neither fits."""
    margins = Counter(m.margin for m in measures if m.system and m.margin is not None)
    usual = margins.most_common(1)[0][0] if margins else None
    coda = next((i for i in range(jump + 1, len(measures)) if measures[i].system and usual is not None
                 and measures[i].margin is not None and measures[i].margin > usual + INDENT), None)
    how = "the first indented system after the jump"
    if coda is None and jump + 1 < len(measures) and measures[jump + 1].system:
        coda, how = jump + 1, "the system after the jump"
    if coda is None:
        warnings.append(f"To Coda in measure {to_coda + 1} but no coda sign: the To Coda is ignored (check the score)")
    else:
        warnings.append(f"To Coda in measure {to_coda + 1} but no coda sign: guessed measure {coda + 1}, {how} "
                        f"(check the score)")
    return coda


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


MEASURE_SLACK = Fraction(1, 16)  # a measure this close to its time signature's length is that long (tuplet rounding)


def _unroll(parts: list[_Part], order: list[int], timing: list[_Part] | None = None) -> tuple[list[Fraction], Fraction]:
    """Start of each played measure (quarter notes) on the unrolled timeline, and the total length. A measure within
    MEASURE_SLACK of its time signature's length counts as exactly that long: durations in divisions are rounded, so
    a measure of odd tuplets can come out a hair over (Liszt's 961/480 quarters in 2/4), which would put every later bar
    line off the score's. A measure is as long as its longest part, or with `timing` (the keyboard part the hands come
    from), as that part's measure where it has notes or rests: a scan's voice part whose bar Audiveris read too long
    would otherwise stretch the piano's bar."""
    if timing:
        timed = lambda mi: [p for p in timing if mi < len(p.measures) and p.measures[mi].length] or parts  # noqa: E731
    else:
        timed = lambda mi: parts  # noqa: E731
    meters: list[Fraction | None] = []
    cur = None
    for i in range(max((len(p.measures) for p in parts), default=0)):
        cur = next((p.measures[i].time for p in parts if i < len(p.measures) and p.measures[i].time), cur)
        meters.append(Fraction(4 * cur[0], cur[1]) if cur else None)
    starts, t = [], Fraction(0)
    for mi in order:
        starts.append(t)
        length = max((p.measures[mi].length for p in timed(mi) if mi < len(p.measures)), default=Fraction(0))
        if meters[mi] is not None and 0 < abs(length - meters[mi]) <= MEASURE_SLACK:
            length = meters[mi]
        t += length
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


def load(path: str | Path, satb: bool = False, melody=False) -> Score:
    """`satb`: the score is a four-part hymn or chorale on two staves (hand_source "satb", the tenor reach).
    `melody`: merge a voice or melody instrument part into the keyboard part's right hand as its top voice (§2.6):
    True for the score's voice and melody parts (melody_parts), or a part's id or name, or a list of them."""
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
                for flag in JUMP_FLAGS + ("fine_word", "al_fine", "al_coda", "jump_sound", "system"):
                    setattr(mm, flag, getattr(mm, flag) or getattr(pm, flag))
                mm.margin = pm.margin if mm.margin is None else mm.margin
        merged.append(mm)
    if any(m.al_fine or (m.jump_sound and not m.al_coda) for m in merged):
        for m in merged:
            m.fine |= m.fine_word
    order = play_order(merged, warnings)
    keyboard = [p for p in parts if p.keyboard and not p.choir_voice]
    choir = [p for p in parts if p.choir_voice]
    two_staff = [p for p in keyboard if p.staves >= 2]
    choir_rule = len(choir) >= 3 and {"S", "B"} <= {p.choir_voice for p in choir}
    starts, total = _unroll(parts, order, two_staff if len(two_staff) == 1 and not choir_rule else None)
    events = _events(parts, order, starts)
    for e in events:
        if 0 < e.end - total <= MEASURE_SLACK:  # the last measure snapped to its time signature: so does its last note
            e.end = total
    kp_id = ""
    staff_hand = lambda e: None  # noqa: E731 - only a two-staff keyboard part has cross-staff notes
    tpb = TPB
    line: list[Note] = []  # the melody to merge into the right hand
    melody_report: dict = {}
    tune: list[Note] = []  # levels 1 and 2's tune, from a voice part, in tune_bars
    tune_bars: list[tuple[int, int]] = []
    if choir_rule:
        satb = True
        use = {parts.index(p): ("R" if p.choir_voice in ("S", "A") else "L") for p in choir}
        hand = lambda e: use.get(e.part)  # noqa: E731
    elif len([p for p in keyboard if p.staves >= 2]) >= 2:
        raise MusicXMLError("piano duet (two keyboard parts with two staves each)")
    elif any(p.staves >= 2 for p in keyboard):
        kp = next(p for p in keyboard if p.staves >= 2)
        pi = parts.index(kp)
        homes = _voice_homes(kp)
        satb = satb or bool(_CHOIR_PART.search(kp.name))
        kp_id = kp.id

        def hand(e: _Event, pi=pi, homes=homes):
            if e.part != pi:
                return None
            return "R" if homes.get(e.voice, e.staff) == 1 else "L"

        def staff_hand(e: _Event, pi=pi):
            return None if e.part != pi else ("R" if e.staff == 1 else "L")
        chosen: list[_Part] = []
        if melody:
            chosen = melody_parts(parts, kp, melody)
            line, melody_report = _melody_line(chosen, kp, order, starts, tpb)
            warnings += _fold_warnings(melody_report)
            if not chosen:
                warnings.append("melody: no voice or melody part to merge")
        # Levels 1 and 2 take their tune from a part named as a voice or melody instrument, merged or not (§2.6).
        sung = chosen or melody_parts(parts, kp, True, named=True)
        tune_line, tune_report = (line, melody_report) if chosen else _melody_line(sung, kp, order, starts, tpb)
        if tune_line:
            played = [k for k, mi in enumerate(order) if mi in tune_report["ok"]]
            tune_bars = _spans(played, starts, total, tpb)
            tune = _clip(top_line(tune_line, tpb), tune_bars)
            warnings.append(f"melody: levels 1 and 2 take the tune from {', '.join(tune_report['parts'])} in "
                            f"{len(played)} of {len(order)} bars, the piano's top line in the rest")
        others = [p.name or p.id for p in parts if p is not kp and p not in chosen]
        if others or chosen:
            with_melody = f" with the melody of {', '.join(p.name or p.id for p in chosen)}" if chosen else ""
            warnings.append(f"used the keyboard part {kp.name or kp.id!r}{with_melody}"
                            + (f"; left out {', '.join(others)}" if others else ""))
    elif len(keyboard) == 2:
        means = {parts.index(p): sum(nn.pitch for m in p.measures for nn in m.notes) / p.pitched_notes() for p in keyboard}
        hi = max(means, key=means.get)
        hand = lambda e, hi=hi: None if e.part not in means else ("R" if e.part == hi else "L")  # noqa: E731
    elif len(parts) == 1 and parts[0].staves == 1:
        hand = lambda e: "R"  # noqa: E731 - a melody alone; the gate decides whether that's enough
        warnings.append("single staff: right hand only")
    else:
        hand = lambda e: "?"  # noqa: E731 - no structure to read hands from
    if melody and not kp_id:
        warnings.append("melody: no two-staff keyboard part to merge it into")
    notes: list[Note] = []
    cross_staff: dict[tuple[int, int], str] = {}  # (start, pitch) -> the hand of the staff a cross-staff note is printed on
    for e in events:
        h = hand(e)
        if h is None:
            continue
        notes.append(Note(
            round(e.start * tpb), round(e.end * tpb), e.pitch, e.velocity, None if h == "?" else h,
            0 if h != "L" else 1, 0 if h != "L" else 1, 0, e.finger,
        ))
        if h in ("R", "L") and staff_hand(e) not in (None, h):
            cross_staff[notes[-1].start, e.pitch] = staff_hand(e)
    notes = [nt for nt in notes if nt.end > nt.start]
    hand_source = "none" if any(nt.hand is None for nt in notes) else ("satb" if satb else "staves")
    notes = merge_unisons(notes)
    repaired: list[int] = []
    if hand_source == "satb":
        notes = reach(notes, tpb)  # before the repair, which would otherwise split what the SATB rule moves
    if hand_source in ("staves", "satb"):
        # The voice rule is a convention: a cross-staff note goes to its staff's hand where the voice's hand would
        # stretch past a tenth or leap away to reach it, and a chord one hand can't play is split (§2.2).
        notes = repair_hands(notes, tpb, [cross_staff.get((nt.start, nt.pitch), nt.hand) != nt.hand for nt in notes],
                             repaired)
    if line:
        # Into the hands as repaired, so cross-staff notes are already with the hand that plays them.
        notes, stats = merge_melody(notes, top_line(line, tpb), tpb)
        notes = merge_unisons(notes)  # a note moved to the left hand may run into a later one on its key
        melody_report.update(dropped=stats.dropped, moved=stats.moved, cut=stats.cut)
        warnings.append(f"melody: left out {stats.dropped} accompaniment notes above it or doubling it, moved "
                        f"{stats.moved} more than a tenth under it to the left hand, shortened {stats.cut}")
    song = Song(ticks_per_beat=tpb, meta=_meta(parts, merged, order, starts, tpb, warnings, total))
    song.tracks = [
        TrackInfo(0, "RH", sorted((x for x in notes if x.hand == "R"), key=lambda x: (x.start, x.pitch))),
        TrackInfo(1, "LH", sorted((x for x in notes if x.hand == "L"), key=lambda x: (x.start, x.pitch))),
    ]
    if hand_source == "none":
        song.tracks = [TrackInfo(0, "", sorted(notes, key=lambda x: (x.start, x.pitch)))]
    return Score(song, hand_source, title, _credit(root, "composer"), _credit(root, "arranger"),
                 [p.name or p.id for p in parts], warnings, [round(t * tpb) for t in starts] + [round(total * tpb)],
                 order, kp_id if hand_source == "staves" else "", repaired, melody_report, tune, tune_bars)


# Parts that carry a song's tune (§2.6): voices and melody instruments, not a guitar's chords, a bass or drums.
_MELODY = re.compile(r"voice|vocal|vox|\bvoc\b|sing|soprano|mezzo|alto|tenor|baritone|melody|lead|solo|gesang|stimme|"
                     r"canto|chant|flute|violin|oboe|clarinet|trumpet|sax|recorder", re.I)
_NOT_MELODY = re.compile(r"guitar|gtr|bass|drum|perc|chord|harp", re.I)


def melody_parts(parts: list[_Part], kp: _Part, melody, named: bool = False) -> list[_Part]:
    """The parts whose tune `load(melody=...)` merges into `kp`'s right hand: for True (or "auto"), the score's voice
    and melody-instrument parts by name, else (unless `named`) every other one-staff part that isn't a guitar, bass or
    drums; for a part's id or name, or a list of them, those parts."""
    others = [p for p in parts if p is not kp and not p.keyboard]
    if melody is True or (isinstance(melody, str) and melody.strip().lower() in ("true", "auto", "yes", "part")):
        found = [p for p in others if _MELODY.search(p.name) and not _NOT_MELODY.search(p.name)]
        return found or ([] if named else [p for p in others if p.staves == 1 and not _NOT_MELODY.search(p.name)])
    wanted = {melody.strip().lower()} if isinstance(melody, str) else {str(w).strip().lower() for w in melody}
    return [p for p in others if p.id.lower() in wanted or p.name.lower() in wanted]


def _bar_ok(part: _Part, kp: _Part, mi: int) -> bool | None:
    """Whether a melody part's written measure can be merged: every voice with notes in it fills the keyboard part's
    measure exactly (notes, rests and forwards; a whole-measure rest counts as filling it), and no note runs past it.
    OMR reads a voice's rhythm worse than the piano's (lyrics spread its notes, tuplets and dots go missing), and a bar
    that doesn't add up has notes in the wrong place. None for a bar with no notes."""
    m = part.measures[mi] if mi < len(part.measures) else None
    if m is None or not m.notes:
        return None
    expected = kp.measures[mi].length if mi < len(kp.measures) and kp.measures[mi].length else m.length
    if any(n.pos + n.dur > expected + MEASURE_SLACK for n in m.notes):
        return False
    return all(v in m.rested or abs(m.fill.get(v, Fraction(0)) - expected) <= MEASURE_SLACK for v in {n.voice for n in m.notes})


RH_LOW = 60  # a tune whose middle note sits below middle C is raised by octaves to sit there or above (§2.6)


def _melody_line(chosen: list[_Part], kp: _Part, order: list[int], starts: list[Fraction], tpb: int
                 ) -> tuple[list[Note], dict]:
    """The melody parts' notes in the bars that add up (_bar_ok), as right-hand notes, and the report: per part, the
    written bars used and skipped and the octaves it was raised; `ok`, the written measures (0-based) any part was
    used in. A part in a treble clef with an 8 below (a tenor's) is raised an octave to where it's printed, and a part
    whose middle note is still below middle C (a baritone in the bass clef) by octaves until it isn't: a piano plays a
    song's tune in the right hand, where it's printed for a voice in the treble clef."""
    report: dict = {"parts": [], "bars": {}, "raised": {}, "ok": set()}
    names = [p.name or p.id for p in chosen]
    clean: list[_Part] = []
    shift: list[int] = []
    for p in chosen:
        label = f"{p.name} ({p.id})" if not p.name or names.count(p.name) > 1 else p.name
        status = [_bar_ok(p, kp, mi) for mi in range(len(p.measures))]
        # A skipped bar's notes are left out, and ties into or out of it are broken, so a note tied across it
        # doesn't hold through the bar the piano plays.
        measures = []
        for mi, (m, s) in enumerate(zip(p.measures, status)):
            before = mi > 0 and not status[mi - 1]
            after = mi + 1 < len(status) and not status[mi + 1]
            measures.append(replace(m, notes=[replace(n, tie_stop=n.tie_stop and not before,
                                                      tie_start=n.tie_start and not after) for n in m.notes] if s else []))
        clean.append(replace(p, measures=measures))
        up = -12 * p.clef_octave if p.clef_octave < 0 else 0
        pitches = sorted(n.pitch + up for m in measures for n in m.notes)
        if pitches and pitches[len(pitches) // 2] < RH_LOW:
            up += 12 * math.ceil((RH_LOW - pitches[len(pitches) // 2]) / 12)
        shift.append(up)
        report["parts"].append(label)
        report["bars"][label] = {"merged": [mi + 1 for mi, s in enumerate(status) if s],
                                 "skipped": [mi + 1 for mi, s in enumerate(status) if s is False]}
        report["raised"][label] = up // 12
        report["ok"] |= {mi for mi, s in enumerate(status) if s}
    line = [Note(round(e.start * tpb), round(e.end * tpb), e.pitch + shift[e.part], e.velocity, "R")
            for e in _events(clean, order, starts) if e.end > e.start and 0 <= e.pitch + shift[e.part] <= 127]
    return line, report


def _fold_warnings(report: dict) -> list[str]:
    """A warning per part merged into the right hand (load(melody=...))."""
    out = []
    for label in report["parts"]:
        merged, skipped = report["bars"][label]["merged"], report["bars"][label]["skipped"]
        text = f"melody: merged {label} into the right hand in {len(merged)} of the {len(merged) + len(skipped)} bars it has notes in"
        if skipped:
            text += f"; skipped bars {_bar_list(skipped)}, whose note values don't fill the bar (the piano's own top line plays there)"
        if report["raised"][label]:
            text += f"; raised {report['raised'][label]} octave(s)"
        out.append(text)
    return out


def _spans(played: list[int], starts: list[Fraction], total: Fraction, tpb: int) -> list[tuple[int, int]]:
    """The played bars `played` (indices into `starts`) as tick spans, neighbours joined."""
    ends = list(starts[1:]) + [total]
    out: list[tuple[int, int]] = []
    for k in played:
        a, b = round(starts[k] * tpb), round(ends[k] * tpb)
        if out and out[-1][1] == a:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def _clip(notes: list[Note], spans: list[tuple[int, int]]) -> list[Note]:
    """The notes starting inside `spans`, ending by the end of theirs."""
    firsts = [a for a, _ in spans]
    out = []
    for n in notes:
        i = bisect_right(firsts, n.start) - 1
        if i >= 0 and n.start < spans[i][1]:
            out.append(replace(n, end=min(n.end, spans[i][1])))
    return out


def _bar_list(bars: list[int]) -> str:
    """[2, 3, 4, 9] -> "2-4, 9"."""
    runs: list[list[int]] = []
    for b in bars:
        if runs and b == runs[-1][-1] + 1:
            runs[-1].append(b)
        else:
            runs.append([b])
    return ", ".join(f"{r[0]}-{r[-1]}" if len(r) > 1 else str(r[0]) for r in runs)


# Elements dropped from a score engraved from its own notation: page layout and credits (the app's pages are laid out
# afresh and show no header), and the other parts' entries.
_LAYOUT = ("defaults", "credit", "identification", "work", "movement-number", "movement-title")


def keyboard_notation(path: str | Path, part_id: str) -> str:
    """The score's keyboard part alone, as MusicXML text for engraving its full level from the source's own notation
    (docs/sheet-music.md §1.1): spelling, voices, slurs, dynamics, grace notes, repeats as written. Other parts, credits,
    page and system breaks and layout, cue notes and metronome marks are left out; the part's name is hidden."""
    root = read_root(path)
    for tag in _LAYOUT:
        for el in root.findall(tag):
            root.remove(el)
    part_list = root.find("part-list")
    if part_list is not None:
        for el in list(part_list):
            if el.tag != "score-part" or el.get("id") != part_id:
                part_list.remove(el)
        sp = part_list.find("score-part")
        if sp is not None:
            for el in list(sp):
                if el.tag not in ("part-name",):
                    sp.remove(el)
            name = sp.find("part-name")
            if name is not None:
                name.set("print-object", "no")
    for part in root.findall("part"):
        if part.get("id") != part_id:
            root.remove(part)
            continue
        if any(_num(t.findtext("chromatic")) or _num(t.findtext("octave-change")) for t in part.iter("transpose")):
            raise MusicXMLError("a transposing keyboard part")
        for m in part.findall("measure"):
            for el in m.findall("print"):
                m.remove(el)
            # Metronome marks: Verovio writes their note in the music font as text, which resvg draws as empty boxes.
            for d in m.findall("direction"):
                for dt in d.findall("direction-type"):
                    if dt.find("metronome") is not None:
                        d.remove(dt)
                if d.find("direction-type") is None:
                    m.remove(d)
            # Cue notes aren't played (the level has none): their time stays as a <forward>.
            for k, el in reversed(list(enumerate(m))):
                if el.tag == "note" and el.find("cue") is not None:
                    m.remove(el)
                    if el.find("chord") is None and el.find("grace") is None:
                        fwd = ET.Element("forward")
                        ET.SubElement(fwd, "duration").text = el.findtext("duration") or "0"
                        for tag in ("voice", "staff"):
                            if el.find(tag) is not None:
                                ET.SubElement(fwd, tag).text = el.findtext(tag)
                        m.insert(k, fwd)
    return ET.tostring(root, encoding="unicode")


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
    (a pickup, the bar that completes it, a bar split at a repeat sign) or longer (a cadenza, an unmarked 5/4 bar in
    4/4) gets a time signature of its own length, and the real one again at the next measure: MIDI has no partial bars,
    and the app counts bars by time signature (MidiFile.Build), so without it every later bar line would be off
    (docs/sheet-music.md)."""
    tempos: dict[int, float] = {}
    times: dict[int, tuple[int, int]] = {}
    keys: dict[int, str] = {}
    cur_time, cur_key = None, None
    first: float | None = None  # the first tempo, for "Tempo I"
    for mi, t0 in zip(order, starts):
        tick0 = round(t0 * tpb)
        marks = [t for p in parts if mi < len(p.measures) for t in p.measures[mi].tempos]
        # A metronome mark or <sound tempo> in the measure wins over tempo words ("Moderately" over "♩ = 92").
        marked = any(kind not in ("words", "primo") for _, _, kind in marks)
        for p in parts:
            if mi >= len(p.measures):
                continue
            pm = p.measures[mi]
            if pm.time and pm.time != cur_time:
                times[tick0] = pm.time
            if pm.key and pm.key != cur_key:
                keys[tick0] = pm.key
            for pos, bpm, kind in sorted(pm.tempos, key=lambda x: (x[0], x[2] != "words")):
                if kind in ("words", "primo") and marked:
                    continue
                if kind == "primo":
                    if first is None:
                        continue
                    bpm = first
                if MIN_BPM <= bpm <= MAX_BPM:
                    tempos[round((t0 + pos) * tpb)] = bpm
                    first = bpm if first is None else first
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
        if 0 < t1 - t0 != nominal:
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
