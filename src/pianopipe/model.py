"""Core data types shared by the pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

Hand = Literal["R", "L"]


@dataclass(frozen=True)
class Note:
    start: int  # absolute ticks
    end: int  # absolute ticks
    pitch: int  # MIDI note number
    velocity: int = 80
    hand: Hand | None = None
    track: int = 0  # source track index
    channel: int = 0  # source channel
    program: int = 0  # GM program (0-based) active when the note started
    finger: int | None = None  # 1 (thumb) to 5 (little finger), see fingering.py

    def with_hand(self, hand: Hand | None) -> "Note":
        return replace(self, hand=hand)

    @property
    def duration(self) -> int:
        return self.end - self.start


@dataclass
class TrackInfo:
    index: int
    name: str
    notes: list[Note] = field(default_factory=list)


@dataclass
class Song:
    ticks_per_beat: int
    # Tempo, time and key signature events as (absolute tick, mido.MetaMessage).
    meta: list = field(default_factory=list)
    tracks: list[TrackInfo] = field(default_factory=list)

    @property
    def notes(self) -> list[Note]:
        out = [n for t in self.tracks for n in t.notes]
        out.sort(key=lambda n: (n.start, n.pitch))
        return out
