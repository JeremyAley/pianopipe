"""Choosing the piano part of a MIDI file and reading hands from its structure."""

from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import mean

from .model import Note, Song

DRUM_CHANNEL = 9  # channel 10 in 1-based numbering
PIANO_PROGRAMS = range(0, 8)  # GM programs 1-8 (0-based 0-7)

_RIGHT = re.compile(r"\b(rh|right|r\.?h\.?|treble|upper|oben|droite)\b|right hand|\bmelody\b", re.I)
_LEFT = re.compile(r"\b(lh|left|l\.?h\.?|bass|lower|unten|gauche)\b|left hand", re.I)
_PIANO = re.compile(r"piano|klavier|keys|pno", re.I)


@dataclass
class PianoPart:
    notes: list[Note]
    # How hands were decided: "track_names", "two_tracks", "two_channels" or "none".
    hand_source: str


def _hand_from_name(name: str) -> str | None:
    r, l = bool(_RIGHT.search(name)), bool(_LEFT.search(name))
    if r and not l:
        return "R"
    if l and not r:
        return "L"
    return None


def pick_piano(song: Song) -> PianoPart:
    """Return the piano notes, with hands set when the file's structure gives them."""
    tracks = []
    for t in song.tracks:
        notes = [n for n in t.notes if n.channel != DRUM_CHANNEL]
        if not notes:
            continue
        piano_share = sum(n.program in PIANO_PROGRAMS for n in notes) / len(notes)
        if piano_share >= 0.5 or _PIANO.search(t.name) or _hand_from_name(t.name):
            tracks.append((t, notes))
    if not tracks:
        # No piano-like track: fall back to every non-drum note (a band file needs the track picker later).
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
