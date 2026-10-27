"""Rule-based difficulty levels (the first step of milestone M4).

Each level is a full song the app can switch to with the difficulty slider:
  1 Melody          the right hand's outer line: the top note of each onset, skipping notes under a melody note that
                    is still held, except that a note held past its first beat over moving notes gives way to them
                    after that beat (_fill_held); a trill becomes one held note and a grace note gives way to its main
                    note
  2 Melody + bass   adds the left hand's bottom line (the same rule, mirrored), thinned to the first onset in each
                    beat of the time signature (at least a quarter long, see bass_grid)
  3 Simplified      right hand keeps top two notes, left hand keeps root and top
  4 Full            the original arrangement
Levels 1 and 2 are snapped to the eighth grid and level 3 to the sixteenth grid, except notes that would land on the
same grid point as another note of their hand: those keep their own timing (see _quantize).
The Nakamura and Yoshii statistical reduction can replace levels 1 to 3 later.
difficulty_score() rates any level on a fixed 1-10 scale for the song browser.
"""

from __future__ import annotations

import math
from dataclasses import replace

from . import features
from .hands import group_chords
from .model import Note

LEVELS = {1: "Melody", 2: "Melody + bass", 3: "Simplified", 4: "Full"}
# Melody (and bass line) onsets closer than this many 32nds compete, and one is dropped (outer_notes). A sixteenth
# lost real notes: the short note of each dotted pair in Dvorak's Humoresque is a 32nd before the next one.
MELODY_MIN_GAP = 1
TRILL_NOTES = 4  # this many notes or more, alternating between two pitches a step apart a 32nd or less apart: a trill


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


def _grid_level(tick: int, tpb: int) -> int:
    """3 on a quarter, 2 on an eighth, 1 on a sixteenth, 0 off the sixteenth grid (within a 128th)."""
    w = max(1, tpb // 32)
    for level, step in ((3, tpb), (2, tpb // 2), (1, tpb // 4)):
        pos = tick % step
        if min(pos, step - pos) <= w:
            return level
    return 0


def _stronger(a: Note, b: Note, tpb: int) -> bool:
    """Whether b should replace a when the two are too close: b is twice as long, or neither is and b sits on a
    stronger grid position."""
    if b.duration >= 2 * a.duration or a.duration >= 2 * b.duration:
        return b.duration > a.duration
    return _grid_level(b.start, tpb) > _grid_level(a.start, tpb)


def _trills(line: list[Note], tpb: int) -> list[Note]:
    """Each trill in a one-note-at-a-time line becomes its first note, held to the end of the trill."""
    fast = tpb // 8 + max(1, tpb // 32)
    out, i = [], 0
    while i < len(line):
        j = i + 1
        if j < len(line) and 0 < abs(line[j].pitch - line[i].pitch) <= 2:
            pair = (line[i].pitch, line[j].pitch)
            while j < len(line) and line[j].start - line[j - 1].start <= fast and line[j].pitch == pair[(j - i) % 2]:
                j += 1
        if j - i >= TRILL_NOTES:
            out.append(replace(line[i], end=max(n.end for n in line[i:j])))
            i = j
        else:
            out.append(line[i])
            i += 1
    return out


def outer_notes(notes: list[Note], hand: str, tpb: int, min_gap: int) -> list[Note]:
    """One note per onset of the hand's outer line (features.outer_line): the top note (right hand) or bottom note
    (left hand) of each chord not under (over) a note still held. Trills become one held note (_trills), and an
    onset less than `min_gap` after the last kept one competes with it (_stronger), so a grace note gives way to its
    main note."""
    w = max(1, tpb // 32)
    chords = group_chords([n for n in notes if n.hand == hand], window=w)
    line = [c[-1] if hand == "R" else c[0] for c in features.outer_line(chords, hand, w)]
    if hand == "R":
        line = _fill_held(line, chords, tpb)  # before trills merge into one long note, which isn't a real hold
    line = _trills(line, tpb)
    out: list[Note] = []
    for n in line:
        if out and n.start - out[-1].start < min_gap:
            if _stronger(out[-1], n, tpb):
                out[-1] = n
            continue
        out.append(n)
    return out


def _fill_held(line: list[Note], chords: list[list[Note]], tpb: int) -> list[Note]:
    """Where the melody holds one note past its first beat while the hand keeps playing under it, the hand's top
    notes take over after that beat and run until the next melody note: the held note is cut there. Without this,
    Moonlight's level 1 was one long G#4 for several seconds while the triplets that make the piece moved underneath
    (Jeremy, 2026-10-26: "missing parts ... just plays the high note"). Holds of a beat or less stay as they are."""
    if not line:
        return line
    tops = [c[-1] for c in chords]
    out: list[Note] = []
    for i, m in enumerate(line):
        nxt = line[i + 1].start if i + 1 < len(line) else max(n.end for n in tops)
        fill = [n for n in tops if m.start + tpb <= n.start < nxt]
        if m.end > m.start + tpb and len(fill) >= 2:
            out.append(replace(m, end=min(m.end, fill[0].start)))
            out.extend(replace(n, end=min(n.end, nxt)) for n in fill)
        else:
            out.append(m)
    return out


def bass_grid(meta, tpb: int):
    """[(tick, slot ticks)]: the time signature's beat (features.beat_grid: a dotted quarter in 6/8), widened when
    it is shorter than a quarter to the fewest beats that are a quarter or more and divide the bar (two in 4/8),
    else to the whole bar (3/8: one bass note per bar, as a fast 3/8 is felt)."""
    bars = {t: m.numerator * tpb * 4 // m.denominator for t, m in meta if m.type == "time_signature"}
    grid = []
    for tick, beat in features.beat_grid(meta, tpb):
        bar = bars.get(tick, beat)
        slot = beat
        if beat < tpb:
            slot = next((k * beat for k in range(2, bar // beat) if k * beat >= tpb and bar % (k * beat) == 0), bar)
        grid.append((tick, slot))
    return grid


def one_per_beat(notes: list[Note], meta, tpb: int) -> list[Note]:
    """The first note in each slot of bass_grid."""
    grid = bass_grid(meta, tpb)
    out, slots = [], set()
    for n in sorted(notes, key=lambda n: n.start):
        slot = features.beat_index(n.start, grid)
        if slot not in slots:
            slots.add(slot)
            out.append(n)
    return out


def make_level(notes: list[Note], level: int, ticks_per_beat: int, meta=()) -> list[Note]:
    """Level `level` of the full arrangement `notes` (hands set). `meta` is the song's meta events (for the time
    signature's beat; without one, a beat is a quarter)."""
    if level not in LEVELS:
        raise ValueError(f"unknown level {level}")
    if level == 4:
        return list(notes)
    tpb = ticks_per_beat
    if level in (1, 2):
        melody = outer_notes(notes, "R", tpb, MELODY_MIN_GAP * tpb // 8)
        if level == 1:
            return _quantize(melody, tpb // 2)
        bass = one_per_beat(outer_notes(notes, "L", tpb, MELODY_MIN_GAP * tpb // 8), meta, tpb)
        return _quantize(melody + bass, tpb // 2)
    rh = _per_chord(notes, "R", lambda c: c[-2:], tpb)
    lh = _per_chord(notes, "L", lambda c: c[:1] + (c[-1:] if len(c) > 1 else []), tpb)
    return _quantize(rh + lh, tpb // 4)


def make_levels(notes: list[Note], ticks_per_beat: int, meta=()) -> dict[int, list[Note]]:
    return {lvl: make_level(notes, lvl, ticks_per_beat, meta) for lvl in LEVELS}


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
