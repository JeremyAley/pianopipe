"""Quality gate: decide whether a processed song is good enough for a catalog.

A song is skipped (with reasons) when its hand data looks wrong, it has too few notes or is too short, a level came
out (nearly) empty, or too many of its chords can't be played by one hand. Smaller problems become warnings that go
into the catalog entry. The official catalog also requires hands from the file's structure (track names, two tracks,
two channels, a score's staves or a choir's voices) rather than automatic separation.

The reduced levels are checked against the full arrangement (level_faults): a note of level 1 or 2 held into its
hand's next note where the piece doesn't hold that key, a note moved off the piece's rhythm, and a level with fewer
notes than the one below it. Warnings, and reasons to skip the song in the official catalog.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from statistics import median

from .difficulty import LEGATO, level_grid, on_level_grid
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
    # The share of a level's notes that may fail a level check (level_faults) before the song is skipped; None: the
    # level checks only warn.
    max_level_faults: float | None = None


OFFICIAL = GatePolicy(allowed_hand_sources=STRUCTURAL_HANDS, max_level_faults=0.0)


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


def level_faults(notes: list[Note], levels: dict[int, list[Note]], ticks_per_beat: int,
                 tune: list[Note] = ()) -> list[tuple[int, int, str]]:
    """What the reduced levels (every level below the highest) get wrong against the full arrangement `notes`, as
    (level, notes affected, text). `tune`: a voice part's notes that levels 1 and 2 play (difficulty.make_level);
    they count as part of the piece here.
      - Held (levels 1 and 2, one line in each hand): a note runs more than a 32nd into its hand's next note, though
        the piece isn't holding that key there. The app scores how long a note is held.
      - Rhythm (every reduced level): a note starts where nothing in the piece does (a figure's top note moved to
        where the figure starts is fine), and either that is off the level's grid (difficulty.on_level_grid) or the
        note came from a place that was on the grid already: a triplet written as sixteenths. Not seen: a note moved
        onto a start of the other hand's (triplets against sixteenths).
      - Count: a level has fewer notes than the level below it. (Not level 3 against level 2 of a song with a
        `tune`: the voice's line can be busier than the piano part it is sung over.)"""
    tpb = ticks_per_beat
    w = tpb // LEGATO
    piece = sorted(list(notes) + list(tune), key=lambda n: n.start)
    starts = {n.start for n in piece}
    # Each pitch's starts in the piece, with the latest end among its notes up to each one.
    by_pitch: dict[int, tuple[list[int], list[int]]] = {}
    for n in piece:
        at, ends = by_pitch.setdefault(n.pitch, ([], []))
        at.append(n.start)
        ends.append(max(n.end, ends[-1]) if ends else n.end)

    def sounding_past(pitch: int, tick: int) -> bool:
        """The piece holds `pitch`, struck before `tick`, more than a 32nd past it."""
        at, ends = by_pitch.get(pitch, ([], []))
        i = bisect_left(at, tick)
        return i > 0 and ends[i - 1] - tick > w

    faults = []
    order = sorted(levels)
    for level in order[:-1]:
        held = moved = 0
        reach = level_grid(level, tpb) / 2  # as far as snapping moves a note
        for hand in {n.hand for n in levels[level]}:
            mine = [n for n in levels[level] if n.hand == hand]
            onsets = sorted({n.start for n in mine})
            for n in mine:
                nxt = bisect_right(onsets, n.start)
                if level <= 2 and nxt < len(onsets) and n.end - onsets[nxt] > w and not sounding_past(n.pitch, onsets[nxt]):
                    held += 1
                if n.start in starts:
                    continue
                at = by_pitch.get(n.pitch, ([], []))[0]
                near = at[bisect_left(at, n.start - reach):bisect_right(at, n.start + reach)]  # where it came from
                if not on_level_grid(n.start, level, tpb) or near and all(on_level_grid(s, level, tpb) for s in near):
                    moved += 1
        if held:
            faults.append((level, held, f"level {level}: {held} notes held into the next note"))
        if moved:
            faults.append((level, moved, f"level {level}: {moved} notes moved off the rhythm"))
    for lower, higher in zip(order, order[1:]):
        if tune and lower <= 2 < higher:
            continue
        if len(levels[higher]) < len(levels[lower]):
            faults.append((lower, len(levels[lower]) - len(levels[higher]),
                           f"level {higher} has fewer notes than level {lower} ({len(levels[higher])}, {len(levels[lower])})"))
    return faults


def check(
    notes: list[Note],
    levels: dict[int, list[Note]],
    hand_source: str,
    duration_s: float,
    ticks_per_beat: int,
    seconds=lambda tick: 0.0,
    policy: GatePolicy | None = None,
    repaired: list[int] = (),
    tune: list[Note] = (),
) -> GateResult:
    """Run every check on the full arrangement (`notes`, hands set) and its levels. `repaired` are the ticks of chords
    the file gave one hand unplayably, split between the hands (hands.repair_hands): they count towards
    max_unplayable_share as written, so a file whose hands are wrong throughout is still skipped, and become warnings.
    `tune`: the voice part's notes that levels 1 and 2 play, if the song has one (level_faults)."""
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
    if chords and (len(flags) + len(repaired)) / len(chords) > policy.max_unplayable_share:
        result.reasons.append(f"{len(flags) + len(repaired)} of {len(chords)} chords can't be played by one hand as written")
    result.warnings += [f"{_clock(seconds(tick))} {text}" for tick, text in flags]
    result.warnings += [f"{_clock(seconds(tick))} chord split between the hands to be playable" for tick in repaired]

    for level, level_notes in levels.items():
        if len(level_notes) < policy.min_level_notes:
            result.reasons.append(f"level {level} has only {len(level_notes)} notes")
    for level, count, text in level_faults(notes, levels, ticks_per_beat, tune):
        skip = policy.max_level_faults is not None and count > policy.max_level_faults * len(levels[level])
        (result.reasons if skip else result.warnings).append(text)
    return result
