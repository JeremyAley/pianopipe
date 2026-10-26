"""Quality gate: decide whether a processed song is good enough for a catalog.

A song is skipped (with reasons) when its hand data looks wrong, it has too few notes or is too short, a level came
out (nearly) empty, or too many of its chords can't be played by one hand. Smaller problems become warnings that go
into the catalog entry. The official catalog also requires hands from the file's structure (track names, two tracks,
two channels, a score's staves or a choir's voices) rather than automatic separation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

from .hands import group_chords, playability_flags
from .model import Note

STRUCTURAL_HANDS = ("track_names", "two_tracks", "two_channels", "staves", "satb", "leadsheet")


@dataclass
class GatePolicy:
    min_notes: int = 100  # in the full arrangement
    min_duration_s: float = 20.0
    min_level_notes: int = 16  # every level, including the melody-only one
    min_hand_share: float = 0.05  # each hand plays at least this share of the notes
    max_crossed_share: float = 0.25  # chords where the left hand reaches above the right (some pieces overlap a lot)
    max_unplayable_share: float = 0.05  # chords one hand can't play
    allowed_hand_sources: tuple[str, ...] = STRUCTURAL_HANDS + ("auto",)


OFFICIAL = GatePolicy(allowed_hand_sources=STRUCTURAL_HANDS)


@dataclass
class GateResult:
    reasons: list[str] = field(default_factory=list)  # why the song is skipped; empty means it passes
    warnings: list[str] = field(default_factory=list)  # kept in the catalog entry

    @property
    def ok(self) -> bool:
        return not self.reasons


class QualityError(ValueError):
    def __init__(self, reasons: list[str]):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


def _clock(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def check(
    notes: list[Note],
    levels: dict[int, list[Note]],
    hand_source: str,
    duration_s: float,
    ticks_per_beat: int,
    seconds=lambda tick: 0.0,
    policy: GatePolicy | None = None,
) -> GateResult:
    """Run every check on the full arrangement (`notes`, hands set) and its levels."""
    policy = policy or GatePolicy()
    result = GateResult()
    if hand_source not in policy.allowed_hand_sources:
        result.reasons.append(f"hands not in the file's structure ({hand_source})")
    if len(notes) < policy.min_notes:
        result.reasons.append(f"too few notes ({len(notes)} < {policy.min_notes})")
    if duration_s < policy.min_duration_s:
        result.reasons.append(f"too short ({duration_s:.0f} s < {policy.min_duration_s:.0f} s)")
    if not notes:
        return result

    rh = [n for n in notes if n.hand == "R"]
    lh = [n for n in notes if n.hand == "L"]
    for name, hand in (("right", rh), ("left", lh)):
        share = len(hand) / len(notes)
        if share < policy.min_hand_share:
            result.reasons.append(f"{name} hand has only {share:.0%} of the notes")
    if rh and lh and median(n.pitch for n in rh) < median(n.pitch for n in lh):
        result.reasons.append("right hand plays lower than the left (hands swapped?)")

    chords = group_chords(notes, window=max(1, ticks_per_beat // 32))
    crossed = 0
    for chord in chords:
        r = [n.pitch for n in chord if n.hand == "R"]
        l = [n.pitch for n in chord if n.hand == "L"]
        if r and l and max(l) > min(r):
            crossed += 1
    if chords and crossed / len(chords) > policy.max_crossed_share:
        result.reasons.append(f"hands cross in {crossed / len(chords):.0%} of chords (bad hand data?)")
    elif crossed:
        result.warnings.append(f"hands cross in {crossed} chords")

    flags = playability_flags(notes, ticks_per_beat)
    if chords and len(flags) / len(chords) > policy.max_unplayable_share:
        result.reasons.append(f"{len(flags)} of {len(chords)} chords can't be played by one hand")
    result.warnings += [f"{_clock(seconds(tick))} {text}" for tick, text in flags]

    for level, level_notes in levels.items():
        if len(level_notes) < policy.min_level_notes:
            result.reasons.append(f"level {level} has only {len(level_notes)} notes")
    return result
