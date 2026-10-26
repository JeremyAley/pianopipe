"""Rule-based difficulty levels (the first step of milestone M4).

Each level is a full song the app can switch to with the difficulty slider:
  1 Melody          right-hand top line only, rhythm snapped to eighths
  2 Melody + bass   adds the lowest left-hand note of each chord, thinned to at most one per beat
  3 Simplified      right hand keeps top two notes, left hand keeps root and top, snapped to sixteenths
  4 Full            the original arrangement
The Nakamura and Yoshii statistical reduction can replace levels 1 to 3 later.
difficulty_score() rates any level on a fixed 1-10 scale for the song browser.
"""

from __future__ import annotations

import math
from dataclasses import replace

from .hands import group_chords
from .model import Note

LEVELS = {1: "Melody", 2: "Melody + bass", 3: "Simplified", 4: "Full"}


def _quantize(notes: list[Note], grid: int) -> list[Note]:
    """Snap starts and ends to the grid, except where that would merge notes that start at different times in one
    hand: those keep their own timing. (Snapping Für Elise's sixteenth-note E-D#-E onto an eighth grid turned the
    melody into two-note chords, 2026-10-26.)"""
    snapped = {id(n): round(n.start / grid) * grid for n in notes}
    starts_at: dict[tuple, set] = {}
    for n in notes:
        starts_at.setdefault((n.hand, snapped[id(n)]), set()).add(n.start)
    out = []
    for n in notes:
        start = snapped[id(n)]
        if len(starts_at[(n.hand, start)]) > 1:
            out.append(n)  # would collide with a note that starts elsewhere: keep the original rhythm
            continue
        end = max(start + grid, round(n.end / grid) * grid)
        out.append(replace(n, start=start, end=end))
    return _dedupe(out)


def _dedupe(notes: list[Note]) -> list[Note]:
    """Drop duplicate (start, pitch, hand) notes and trim overlaps of the same pitch in one hand."""
    seen: dict[tuple, Note] = {}
    for n in sorted(notes, key=lambda n: (n.start, n.pitch)):
        seen.setdefault((n.start, n.pitch, n.hand), n)
    out = sorted(seen.values(), key=lambda n: (n.hand or "", n.pitch, n.start))
    trimmed = []
    for i, n in enumerate(out):
        nxt = out[i + 1] if i + 1 < len(out) else None
        if nxt and nxt.hand == n.hand and nxt.pitch == n.pitch and nxt.start < n.end:
            n = replace(n, end=nxt.start)
        if n.end > n.start:
            trimmed.append(n)
    return sorted(trimmed, key=lambda n: (n.start, n.pitch))


def _per_chord(notes: list[Note], hand: str, keep, tpb: int) -> list[Note]:
    chords = group_chords([n for n in notes if n.hand == hand], window=max(1, tpb // 32))
    return [n for chord in chords for n in keep(chord)]


def make_level(notes: list[Note], level: int, ticks_per_beat: int) -> list[Note]:
    if level not in LEVELS:
        raise ValueError(f"unknown level {level}")
    if level == 4:
        return list(notes)
    tpb = ticks_per_beat
    melody = _per_chord(notes, "R", lambda c: c[-1:], tpb)
    if level == 1:
        return _quantize(melody, tpb // 2)
    if level == 2:
        bass = _per_chord(notes, "L", lambda c: c[:1], tpb)
        thinned, last = [], None
        for n in bass:
            if last is None or n.start - last >= tpb:
                thinned.append(n)
                last = n.start
        return _quantize(melody + thinned, tpb // 2)
    rh = _per_chord(notes, "R", lambda c: c[-2:], tpb)
    lh = _per_chord(notes, "L", lambda c: c[:1] + (c[-1:] if len(c) > 1 else []), tpb)
    return _quantize(rh + lh, tpb // 4)


def make_levels(notes: list[Note], ticks_per_beat: int) -> dict[int, list[Note]]:
    return {lvl: make_level(notes, lvl, ticks_per_beat) for lvl in LEVELS}


# Difficulty score: a fixed scale (not normalised per catalog), so a song's number doesn't change when others are added.
WINDOW_S = 4.0  # notes per second are counted over windows this long
RAW_EASY, RAW_HARD = 3.0, 50.0  # raw scores that map to 1 and 10


def difficulty_score(notes: list[Note], seconds, ticks_per_beat: int) -> float:
    """Rate how hard a level is to play, 1 (easiest) to 10.

    `seconds(tick)` converts ticks to seconds with the song's tempo map. The raw score is the busy-passage speed
    (90th percentile of notes per second over 4 s windows), raised for chords (notes per onset in each hand) and for
    jumps (mean pitch move between one hand's onsets beyond a third), and lowered when only one hand plays; it maps
    to 1-10 on a log scale.
    """
    if not notes:
        return 1.0
    onsets = sorted(seconds(n.start) for n in notes)
    windows: dict[int, int] = {}
    for t in onsets:
        windows[int(t // WINDOW_S)] = windows.get(int(t // WINDOW_S), 0) + 1
    rates = sorted(c / WINDOW_S for c in windows.values())
    busy = rates[min(len(rates) - 1, int(0.9 * len(rates)))]

    chord_sizes, jumps = [], []
    window = max(1, ticks_per_beat // 32)
    for hand in ("R", "L"):
        chords = group_chords([n for n in notes if n.hand == hand], window)
        chord_sizes += [len(c) for c in chords]
        centers = [(c[0].pitch + c[-1].pitch) / 2 for c in chords]
        jumps += [abs(b - a) for a, b in zip(centers, centers[1:])]
    chord_mean = sum(chord_sizes) / len(chord_sizes)
    jump_mean = sum(jumps) / len(jumps) if jumps else 0.0

    one_hand = 0.7 if len({n.hand for n in notes}) == 1 else 1.0
    raw = one_hand * busy * (1 + 0.3 * (chord_mean - 1)) * (1 + 0.1 * max(0.0, jump_mean - 4))
    if raw <= RAW_EASY:
        return 1.0
    score = 1 + 9 * math.log(raw / RAW_EASY) / math.log(RAW_HARD / RAW_EASY)
    return round(min(10.0, score), 1)
