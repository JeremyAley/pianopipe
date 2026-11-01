"""Reading cards and read-ahead passages (docs/learning-mode.md §7 "Reading flash cards", §14 L8).

A card is a short notated fragment answered on the keys: single landmark notes first (middle C, treble G, bass F, ...),
then steps and skips from a landmark, then two-note intervals and triads, then one-bar patterns in five-finger
positions. Each card is a one-bar level (app-ready MIDI) and a one-page sheet on a card-shaped page (CARD: wider than
tall, the notation large and centred, one staff in its own clef, no time signature but on the patterns). The
read-ahead passages are the first pieces of the reading collection at Initial to Grade 2 (sightread.build_piece,
the same pieces as `reading/` in the official catalog).

`build(out)` writes the bank in the app's layout (`pianopipe cards --out unity/.../Resources/Reading`):
cards.json and cards/<id>.mid.bytes, <id>.sheet.json, <id>-p1.png.bytes; passages.json and passages/<id>/... the same
way. Binary files end in `.bytes` so Unity keeps them as they are (tools/bundle_songs.py does the same for songs).
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from fractions import Fraction as Q
from pathlib import Path

import mido

from . import engrave, midi_io, notation, sightread
from .model import Note

TPB = 480
VERSION = 1
# A card's page: about 37 x 24 cm at the book's Medium size; a staff space of 18 * 2.2 = 40 px (2.8 times a book page's).
CARD = engrave.Layout(width=1000, height=640, scale=220, margin=30, margin_top=30)
STAGES = ("landmarks", "steps", "intervals", "patterns")
STAGE_TITLES = {"landmarks": "Landmarks", "steps": "Steps and skips", "intervals": "Intervals and triads",
                "patterns": "One-bar patterns"}
# Seconds a right answer may take to count as read (fluent): more notes, more time.
SECONDS = {"landmarks": 4.0, "steps": 5.0, "intervals": 5.0, "triads": 6.0, "patterns": 8.0}

_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
_WHITE = (0, 2, 4, 5, 7, 9, 11)


def note_name(pitch: int) -> str:
    return f"{_NAMES[pitch % 12]}{pitch // 12 - 1}"


def _white_step(pitch: int, steps: int) -> int:
    """The white key `steps` letter names above (below, negative) a white key."""
    octave, i = divmod(pitch, 12)
    k = _WHITE.index(i) + steps
    return (octave + k // 7) * 12 + _WHITE[k % 7]


@dataclass(frozen=True)
class Landmark:
    key: str
    title: str
    pitch: int
    hand: str  # "R": treble clef, "L": bass clef


# In the order they are introduced: the three the brief names first, then the outer ones.
LANDMARKS = (
    Landmark("c4-treble", "middle C (treble)", 60, "R"),
    Landmark("c4-bass", "middle C (bass)", 60, "L"),
    Landmark("g4", "treble G", 67, "R"),
    Landmark("f3", "bass F", 53, "L"),
    Landmark("c5", "treble C", 72, "R"),
    Landmark("c3", "bass C", 48, "L"),
    Landmark("g5", "high G", 79, "R"),
    Landmark("f2", "low F", 41, "L"),
)


@dataclass
class Card:
    id: str
    stage: str
    title: str  # what it shows, for logs and the session's summary (never shown while the card is up)
    hand: str
    chords: list[tuple[Q, Q, tuple[int, ...]]]  # (start, length in quarters, pitches struck together)
    seconds: float
    meter: bool = False  # print the time signature (the patterns: a bar of four beats)
    order: int = 0  # position in the bank: new cards are introduced in this order
    landmark: str = ""  # the landmark a card reads from (steps and skips), for the summary

    @property
    def pitches(self) -> list[int]:
        return [p for _, _, ps in self.chords for p in ps]

    def notes(self) -> list[Note]:
        return [Note(int(at * TPB), int((at + d) * TPB), p, velocity=72, hand=self.hand)
                for at, d, ps in self.chords for p in ps]

    def manifest(self) -> dict:
        return {"id": self.id, "stage": STAGES.index(self.stage), "order": self.order, "title": self.title,
                "hand": self.hand, "seconds": self.seconds, "landmark": self.landmark,
                "answer": " ".join(note_name(p) for p in self.pitches)}


def _interval_name(steps: int) -> str:
    return {1: "2nd", 2: "3rd", 3: "4th", 4: "5th", 5: "6th", 6: "7th", 7: "octave"}[steps]


def bank() -> list[Card]:
    """The card bank, in the order cards are introduced."""
    cards: list[Card] = []
    whole = Q(4)
    for lm in LANDMARKS:
        cards.append(Card(f"lm-{lm.key}", "landmarks", lm.title[0].upper() + lm.title[1:], lm.hand, [(Q(0), whole, (lm.pitch,))],
                          SECONDS["landmarks"]))
    # Steps, then skips, from the first six landmarks (low F and high G are far enough out already): the landmark as a
    # half note, then the note a step or skip away, so the reading goes from a note known to the next.
    for steps, word in ((1, "Step"), (2, "Skip")):
        for lm in LANDMARKS[:6]:
            for d in (1, -1):
                target = _white_step(lm.pitch, d * steps)
                cards.append(Card(f"st-{lm.key}-{'up' if d > 0 else 'down'}{steps}", "steps",
                                  f"{word} {'up' if d > 0 else 'down'} from {lm.title}: {note_name(target)}", lm.hand,
                                  [(Q(0), Q(2), (lm.pitch,)), (Q(2), Q(2), (target,))], SECONDS["steps"], landmark=lm.title))
    # Two-note intervals struck together, from the landmarks: 3rds and 5ths first, then 2nds, 4ths, 6ths and octaves.
    bases = (("c4", "middle C", 60, "R"), ("c3", "bass C", 48, "L"), ("g4", "treble G", 67, "R"), ("f3", "bass F", 53, "L"))
    for steps in (2, 4, 1, 3, 5, 7):
        for key, name, pitch, hand in bases:
            if key in ("g4", "f3") and steps not in (2, 4):
                continue
            top = _white_step(pitch, steps)
            cards.append(Card(f"iv-{key}-{_interval_name(steps)}", "intervals", f"A {_interval_name(steps)} from {name}",
                              hand, [(Q(0), whole, (pitch, top))], SECONDS["intervals"], landmark=name))
    # Root-position triads on the white keys (C, F and G major first), treble and bass.
    for root in ("C", "F", "G", "D", "E", "A"):
        for hand, octave in (("R", 4), ("L", 3)):
            pitch = (octave + 1) * 12 + _NAMES.index(root)
            if hand == "R" and root == "A":
                pitch -= 12  # A3-C4-E4: in the treble, not above it
            if hand == "L" and root == "A":
                pitch -= 12  # A2-C3-E3
            chord = (pitch, _white_step(pitch, 2), _white_step(pitch, 4))
            quality = "minor" if root in ("D", "E", "A") else "major"
            clef = "treble" if hand == "R" else "bass"
            cards.append(Card(f"tr-{root.lower()}-{clef}", "intervals", f"{root} {quality} triad ({clef})", hand,
                              [(Q(0), whole, chord)], SECONDS["triads"]))
    # One-bar patterns: four quarters in a five-finger position (scale degrees 0-4 of it), right hand in the treble,
    # left in the bass.
    positions = (("c-rh", "C position, right hand", 60, "R"), ("c-lh", "C position, left hand", 48, "L"),
                 ("g-rh", "G position, right hand", 67, "R"), ("g-lh", "G position, left hand", 43, "L"))
    shapes = ((0, 1, 2, 3), (4, 3, 2, 1), (0, 2, 4, 2), (0, 1, 0, 2), (2, 3, 4, 2), (4, 2, 3, 1))
    for i, shape in enumerate(shapes):
        for key, name, base, hand in positions:
            ps = [_white_step(base, s) for s in shape]
            cards.append(Card(f"pt-{key}-{i + 1}", "patterns", f"{name}: {' '.join(note_name(p) for p in ps)}", hand,
                              [(Q(k), Q(1), (p,)) for k, p in enumerate(ps)], SECONDS["patterns"], meter=True))
    for i, c in enumerate(cards):
        c.order = i
    return cards


def problems(cards: list[Card]) -> list[str]:
    """What's wrong with a bank: repeated ids or answers, notes off the white keys or out of the clef's reach."""
    out, ids, answers = [], set(), set()
    for c in cards:
        if c.id in ids:
            out.append(f"{c.id}: repeated id")
        ids.add(c.id)
        key = (c.hand, tuple((at, ps) for at, _, ps in c.chords))
        if key in answers:
            out.append(f"{c.id}: the same notes as an earlier card")
        answers.add(key)
        for p in c.pitches:
            if p % 12 not in _WHITE:
                out.append(f"{c.id}: {note_name(p)} is not a white key")
            lo, hi = (55, 84) if c.hand == "R" else (36, 64)
            if not lo <= p <= hi:
                out.append(f"{c.id}: {note_name(p)} is outside the {'treble' if c.hand == 'R' else 'bass'} clef's ledger lines")
        if max(at + d for at, d, _ in c.chords) > 4:
            out.append(f"{c.id}: longer than a bar")
    return out


# --- building ----------------------------------------------------------------------------------------------------


# A card is answered as fast as it's read: at this tempo its notes are at most a quarter of a second apart, so the app's
# wait mode (a press up to 0.4 s before a note is due counts) takes them as quickly as they're played.
CARD_BPM = 480


def _meta() -> list:
    return [(0, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(CARD_BPM))),
            (0, mido.MetaMessage("key_signature", key="C")),
            (0, mido.MetaMessage("time_signature", numerator=4, denominator=4))]


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def engrave_card(card: Card, out_dir: str | Path) -> dict:
    """Writes out_dir/<id>.mid, <id>.sheet.json and <id>-p1.png (the notation centred on the card) and returns the
    sheet data."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    midi_io.write_app_midi(out_dir / f"{card.id}.mid", TPB, _meta(), card.notes())
    song = midi_io.load(out_dir / f"{card.id}.mid")
    score = notation.build(song, 1, card.title)
    xml = notation.to_musicxml(score, fingering=False)
    if not card.meter:
        xml = xml.replace("<time>", '<time print-object="no">')
    starts = [Q(b.start, score.tpb) for b in score.bars] + [Q(score.bars[-1].end, score.tpb)]
    sheet = engrave.engrave_notation(xml, out_dir, card.id, 1, starts, None, layout=CARD)
    if len(sheet.data["pages"]) != 1 or len(sheet.data["measures"]) != 1:
        raise ValueError(f"{card.id}: engraved {len(sheet.data['pages'])} pages, {len(sheet.data['measures'])} bars")
    return _centre(out_dir, card.id, sheet.data)


def _centre(out_dir: Path, stem: str, data: dict) -> dict:
    """Moves the notation to the middle of its page (Verovio sets a lone bar at the left, at the top) and the bar
    map with it; rewrites the page and sheet.json."""
    from PIL import Image, ImageOps

    page = out_dir / data["pages"][0]["file"]
    img = Image.open(page).convert("L")
    box = ImageOps.invert(img).getbbox()
    if box is None:
        raise ValueError(f"{stem}: an empty card")
    x0, y0, x1, y1 = box
    dx = (img.width - (x1 - x0)) // 2 - x0
    dy = (img.height - (y1 - y0)) // 2 - y0
    out = Image.new("L", img.size, engrave.PAPER)
    out.paste(img.crop(box), (x0 + dx, y0 + dy))
    # The four levels the page was drawn in (the 2-bit PNG reads back as 0, 85, 170, 255).
    levels = (engrave.INK, engrave.STAFF, engrave.INK_EDGE, engrave.PAPER)
    out = out.point(lambda v: min(levels, key=lambda l: abs(l - v)))
    data["pages"][0]["sha256"] = _write(page, engrave._grey2(out))
    fx, fy = dx / img.width, dy / img.height
    for s in data["systems"]:
        s["top"], s["bottom"] = round(s["top"] + fy, 4), round(s["bottom"] + fy, 4)
    for m in data["measures"]:
        m["left"], m["right"] = round(m["left"] + fx, 4), round(m["right"] + fx, 4)
        m["onsets"] = [round(v + fx, 4) if i % 2 else v for i, v in enumerate(m["onsets"])]
        m["notes"] = [round(v + (fx if i % 4 == 2 else fy if i % 4 == 3 else 0), 4) if i % 4 >= 2 else v
                      for i, v in enumerate(m["notes"])]
    (out_dir / f"{stem}.sheet.json").write_bytes((json.dumps(data, separators=(",", ":")) + "\n").encode("utf-8"))
    return data


PASSAGE_GRADES = (0, 1, 2)
PASSAGES_PER_GRADE = 6


def build_passages(out_dir: str | Path, grades=PASSAGE_GRADES, per_grade: int = PASSAGES_PER_GRADE) -> list[dict]:
    """The read-ahead passages: the reading collection's first pieces at each grade, under out_dir/songs/<id>/."""
    out = []
    for grade in grades:
        for number, seed in enumerate(sightread.bank_seeds(grade, per_grade), 1):
            entry = sightread.build_piece(grade, seed, number, out_dir)
            lv = entry["levels"][0]
            out.append({"id": entry["id"], "grade": grade, "title": entry["title"], "bpm": lv["tempo_bpm"],
                        "notes": lv["notes"]})
    return out


def build(out: str | Path, passages: bool = True, log=print) -> dict:
    """The bank in the app's layout under `out` (Resources/Reading): cards.json, cards/, passages.json, passages/."""
    out = Path(out)
    cards = bank()
    bad = problems(cards)
    if bad:
        raise ValueError("card bank: " + "; ".join(bad))
    tmp = out / "_build"
    if tmp.exists():
        shutil.rmtree(tmp)
    written: set[Path] = set()
    for c in cards:
        engrave_card(c, tmp / "cards")
        written |= _bundle(tmp / "cards", c.id, out / "cards", c.id)
    manifest = {"version": VERSION, "stages": [STAGE_TITLES[s] for s in STAGES], "cards": [c.manifest() for c in cards]}
    (out / "cards.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"cards: {len(cards)} ({', '.join(f'{sum(c.stage == s for c in cards)} {s}' for s in STAGES)})")
    if passages:
        found = build_passages(tmp / "passages")
        for p in found:
            written |= _bundle(tmp / "passages" / "songs" / p["id"], "level1", out / "passages", p["id"])
        (out / "passages.json").write_text(json.dumps({"version": VERSION, "passages": found}, indent=1, ensure_ascii=False) + "\n",
                                           encoding="utf-8")
        log(f"passages: {len(found)} (grades {', '.join(str(g) for g in PASSAGE_GRADES)})")
    shutil.rmtree(tmp)
    # Files from an earlier bank that this one doesn't have, and their Unity .meta files; the others' .meta files stay,
    # so a rebuild keeps their GUIDs.
    for d in ("cards", "passages") if passages else ("cards",):
        for f in list((out / d).iterdir()):
            if f.suffix != ".meta" and f not in written:
                f.unlink()
                Path(f"{f}.meta").unlink(missing_ok=True)
    return manifest


def _bundle(src: Path, stem: str, dst: Path, name: str) -> set[Path]:
    """A level and its sheet from src/<stem>.* to dst/<name>.mid.bytes, <name>.sheet.json and <name>-pN.png.bytes (the
    sheet.json's page names follow). Returns the files written."""
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / f"{stem}.mid", dst / f"{name}.mid.bytes")
    data = json.loads((src / f"{stem}.sheet.json").read_text(encoding="utf-8"))
    for i, page in enumerate(data["pages"], 1):
        new = f"{name}-p{i}.png"
        shutil.copyfile(src / page["file"], dst / f"{new}.bytes")
        page["file"] = new
    (dst / f"{name}.sheet.json").write_bytes((json.dumps(data, separators=(",", ":")) + "\n").encode("utf-8"))
    return {dst / f"{name}.mid.bytes", dst / f"{name}.sheet.json"} | {dst / f"{p['file']}.bytes" for p in data["pages"]}


def summary(manifest: dict) -> str:
    counts = [sum(c["stage"] == i for c in manifest["cards"]) for i in range(len(STAGES))]
    return "; ".join(f"{t}: {n}" for t, n in zip(manifest["stages"], counts))


__all__ = ["Card", "CARD", "LANDMARKS", "STAGES", "bank", "build", "build_passages", "engrave_card", "problems",
           "note_name", "summary"]
