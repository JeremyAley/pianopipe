"""Choosing the piano part of a MIDI file and reading hands from its structure."""

from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import mean

from .model import Note, Song

DRUM_CHANNEL = 9  # channel 10 in 1-based numbering
PIANO_PROGRAMS = range(0, 8)  # GM programs 1-8 (0-based 0-7)

# LilyPond exports name one track per voice after its staff: "up", "upII", "upA:", "down", "downIII:".
_RIGHT = re.compile(r"\b(rh|right|r\.?h\.?|treble|upper|oben|droite)\b|right hand|\bmelody\b|^\s*up(I+|[A-D])?\s*:?\s*$", re.I)
_LEFT = re.compile(r"\b(lh|left|l\.?h\.?|bass|lower|unten|gauche)\b|left hand|^\s*down(I+|[A-D])?\s*:?\s*$", re.I)
_PIANO = re.compile(r"piano|klavier|keys|pno", re.I)
# Instruments a track name can give away when the program says nothing (GM program 1, the default, on every track).
# Whole words only, and never on a LilyPond staff track ("up:viola" and "down:VoiceI" name voices, not instruments).
_OTHER = re.compile(
    r"\b(guitars?|gtr|flutes?|piccolo|violins?|viola|cello|contrabass|strings|drums?|perc(ussion)?|kick|snare|"
    r"hi-?hat|cymbals?|sax(ophone)?|trumpets?|trombones?|horns?|tuba|clarinets?|oboes?|bassoons?|harp|organ|synth|"
    r"pad|vocals?|lead)\b", re.I)
_LILYPOND_STAFF = re.compile(r"^\s*(up|down)\w*\s*:", re.I)


_CHOIR = (
    ("R", re.compile(r"^\s*(s|sop|soprano|sopran|treble|a|alt|alto|contralto)\.?\s*\d*\s*$", re.I)),
    ("L", re.compile(r"^\s*(t|ten|tenor|tenore|b|bass|basso|bassi|baritone)\.?\s*\d*\s*$", re.I)),
)


@dataclass
class PianoPart:
    notes: list[Note]
    # How hands were decided: "track_names", "two_tracks", "two_channels", "satb" or "none".
    hand_source: str


def _choir_hand(name: str) -> str | None:
    return next((hand for hand, rx in _CHOIR if rx.match(name)), None)


def _hand_from_name(name: str) -> str | None:
    r, l = bool(_RIGHT.search(name)), bool(_LEFT.search(name))
    if r and not l:
        return "R"
    if l and not r:
        return "L"
    return None


def _piano_track(name: str, notes: list[Note]) -> bool:
    """Whether a track is a piano part: mostly GM piano programs (1-8) and not named after another instrument (a file
    without program changes leaves every track at program 1), or another program but named piano."""
    if sum(n.program in PIANO_PROGRAMS for n in notes) / len(notes) >= 0.5:
        return bool(_PIANO.search(name) or _LILYPOND_STAFF.match(name)) or not _OTHER.search(name)
    return bool(_PIANO.search(name))


def pick_piano(song: Song) -> PianoPart:
    """Return the piano notes, with hands set when the file's structure gives them.

    Only piano tracks count (_piano_track): a band file's flute "Melody" and bass guitar "Bass" aren't two hands. Hand
    names ("Right", "Melody", "Bass") are read only on piano tracks, and drums (GM channel 10) are always dropped."""
    voiced = [(t, _choir_hand(t.name)) for t in song.tracks if t.notes]
    if len(voiced) >= 3 and all(h for _, h in voiced) and {"R", "L"} == {h for _, h in voiced}:
        # A four-part choir setting, one track per voice: soprano and alto right, tenor and bass left.
        out = [n.with_hand(h) for t, h in voiced for n in t.notes if n.channel != DRUM_CHANNEL]
        return PianoPart(sorted(out, key=lambda n: (n.start, n.pitch)), "satb")
    tracks = []
    for t in song.tracks:
        notes = [n for n in t.notes if n.channel != DRUM_CHANNEL]
        if not notes:
            continue
        if _piano_track(t.name, notes):
            tracks.append((t, notes))
    if not tracks:
        # No piano track: every non-drum note, hands left to the automatic split (a band file needs a track picker).
        notes = [n for t in song.tracks for n in t.notes if n.channel != DRUM_CHANNEL]
        return PianoPart(sorted(notes, key=lambda n: (n.start, n.pitch)), "none")

    named = {id(t): _hand_from_name(t.name) for t, _ in tracks}
    if len(tracks) >= 2 and all(named[id(t)] for t, _ in tracks) and len(set(named.values())) == 2:
        out = [n.with_hand(named[id(t)]) for t, notes in tracks for n in notes]
        return PianoPart(sorted(out, key=lambda n: (n.start, n.pitch)), "track_names")

    if len(tracks) == 2:
        (ta, na), (tb, nb) = tracks
        hi, lo = (na, nb) if mean(n.pitch for n in na) >= mean(n.pitch for n in nb) else (nb, na)
        out = [n.with_hand("R") for n in hi] + [n.with_hand("L") for n in lo]
        return PianoPart(sorted(out, key=lambda n: (n.start, n.pitch)), "two_tracks")

    all_notes = sorted((n for _, notes in tracks for n in notes), key=lambda n: (n.start, n.pitch))
    channels = sorted({n.channel for n in all_notes})
    if len(tracks) == 1 and len(channels) == 2:
        by_ch = {c: [n for n in all_notes if n.channel == c] for c in channels}
        a, b = channels
        hi, lo = (a, b) if mean(n.pitch for n in by_ch[a]) >= mean(n.pitch for n in by_ch[b]) else (b, a)
        out = [n.with_hand("R" if n.channel == hi else "L") for n in all_notes]
        return PianoPart(out, "two_channels")

    return PianoPart(all_notes, "none")
