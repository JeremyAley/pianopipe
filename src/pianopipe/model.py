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
    printed: bool = False  # the finger is the score's own (printed in its source), not computed: docs/formats.md

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
    # The sustain pedal: (down tick, up tick) spans in order, from the source's CC 64 or a score's pedal marks; empty
    # when it has none (docs/formats.md). Spans may touch (a change: up and down again at one tick), never overlap.
    pedal: list[tuple[int, int]] = field(default_factory=list)

    @property
    def notes(self) -> list[Note]:
        out = [n for t in self.tracks for n in t.notes]
        out.sort(key=lambda n: (n.start, n.pitch))
        return out
