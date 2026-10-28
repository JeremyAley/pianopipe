"""Rule-based difficulty levels (the first step of milestone M4).

Each level is a full song the app can switch to with the difficulty slider:
  1 Melody          the right hand's outer line: the top note of each onset, skipping notes under a melody note that
                    is still held, except that a note held past its first beat over moving notes gives way to them
                    after that beat (_fill_held); a trill becomes one held note, a grace note gives way to its main
                    note, and broken-chord figuration a sixteenth or faster becomes the top note of each beat
                    (_figuration). In bars whose tune is in the left hand (left_tune_bars), the left hand's top line
                    instead (melody)
  2 Melody + bass   adds the other hand's outer line, thinned to the first onset in each beat of the time signature
                    (at least a quarter long, see bass_grid): the left hand's bottom line, or the right hand's top
                    line over a left-hand tune
  3 Simplified      right hand keeps top two notes, left hand keeps root and top
  4 Full            the original arrangement
Levels 1 and 2 are snapped to the eighth grid and level 3 to the sixteenth grid, except notes that would land on the
same grid point as another note of their hand: those keep their own timing (see _quantize).
The Nakamura and Yoshii statistical reduction can replace levels 1 to 3 later.
difficulty_score() rates any level on a fixed 1-10 scale for the song browser.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import replace

from . import features, notation
from .hands import group_chords
from .model import Note

LEVELS = {1: "Melody", 2: "Melody + bass", 3: "Simplified", 4: "Full"}
# Melody (and bass line) onsets closer than this many 32nds compete, and one is dropped (outer_notes). A sixteenth
# lost real notes: the short note of each dotted pair in Dvorak's Humoresque is a 32nd before the next one.
MELODY_MIN_GAP = 1
TRILL_NOTES = 4  # this many notes or more, alternating between two pitches a step apart a 32nd or less apart: a trill
FIGURE_NOTES = 4  # this many notes or more a sixteenth or less apart can be a broken-chord figure (_figuration)


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
    """Drop duplicate (start, pitch) notes, one key struck twice at once (a level 2 tune meeting the other hand's line
    on a unison, say): the longer stays, in the left hand below middle C when the hands tie, as musicxml.merge_unisons.
    Then trim overlaps of the same pitch in one hand."""
    seen: dict[tuple, Note] = {}
    for n in sorted(notes, key=lambda n: (n.start, n.pitch, -n.duration, n.hand != ("L" if n.pitch < 60 else "R"))):
        seen.setdefault((n.start, n.pitch), n)
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


def _line(notes: list[Note], hand: str, top: bool, w: int) -> tuple[list[Note], list[list[Note]]]:
    """The hand's top (or bottom) line, one note per onset (features.outer_line), and the hand's chords."""
    chords = group_chords([n for n in notes if n.hand == hand], window=w)
    return [c[-1] if top else c[0] for c in features.outer_line(chords, "R" if top else "L", w)], chords


def outer_notes(notes: list[Note], hand: str, tpb: int, min_gap: int, meta=(), top: bool | None = None) -> list[Note]:
    """One note per onset of the hand's outer line (features.outer_line): the top note (right hand) or bottom note
    (left hand) of each chord not under (over) a note still held; `top` picks the side for either hand (a tune in
    the left hand is its top line). Trills become one held note (_trills), broken-chord figuration one note a beat
    (_figuration), and an onset less than `min_gap` after the last kept one competes with it (_stronger), so a grace
    note gives way to its main note."""
    top = hand == "R" if top is None else top
    w = max(1, tpb // 32)
    line, chords = _line(notes, hand, top, w)
    if top:
        line = _fill_held(line, chords, tpb)  # before trills merge into one long note, which isn't a real hold
    line = _trills(line, tpb)
    line = _figuration(line, notes, tpb, meta, top)
    out: list[Note] = []
    for n in line:
        if out and n.start - out[-1].start < min_gap:
            if _stronger(out[-1], n, tpb):
                out[-1] = n
            continue
        out.append(n)
    return out


def _recurs(line: list[Note]) -> float:
    """The share of notes (from the third) that repeat a pitch of the four before: a cycling pattern, not a tune.
    A broken chord going round (G C E G C E) is near 1; a tune, even one that outlines a chord (C F A C, F Bb D F),
    stays low."""
    if len(line) < 3:
        return 0.0
    hits = sum(any(p.pitch == n.pitch for p in line[max(0, k - 4):k]) for k, n in enumerate(line) if k >= 2)
    return hits / (len(line) - 2)


def _figure_beat(run: list[Note], a: int, b: int) -> bool:
    """Whether run[a:b] (one beat of a fast run) is broken-chord figuration: two notes or more that mostly leap (two
    moves in three a minor third or more, counting the move into the beat) round the same few pitches (half of them
    repeat a pitch within four notes either side). The Bach C major prelude's G C E G C E is; a fast tune moving by
    step (Für Elise's E D# E D# E B) or not coming back (its C E A B) isn't. Judged a beat at a time, so a tune
    straight after a figure stays, and a figure that changes chord (C G E G | D A F A) still counts."""
    if b - a < 2:
        return False
    moves = [abs(run[k + 1].pitch - run[k].pitch) for k in range(max(0, a - 1), b - 1)]
    again = sum(_comes_round(run, k) for k in range(a, b))
    return 3 * sum(m >= 3 for m in moves) >= 2 * len(moves) and 2 * again >= b - a


def _comes_round(run: list[Note], k: int) -> bool:
    """run[k]'s pitch is played again within four notes either side."""
    return any(run[j].pitch == run[k].pitch for j in range(max(0, k - 4), min(len(run), k + 5)) if j != k)


def _pedal(group: list[Note]) -> int | None:
    """The pitch on every other note of a beat (C G E G: G), if there is one."""
    for first in (0, 1):
        if len(group) >= first + 3 and all(n.pitch == group[first].pitch for n in group[first::2]):
            return group[first].pitch
    return None


def _moves_by_step(groups: list[list[Note]], g: int) -> bool:
    """Whether, round a beat that alternates with a repeated note (_pedal), the other notes mostly move by step
    (round the beat and its neighbours): a tune under or over a pedal, like Für Elise's G4 A4 B4 C5 D5 under a
    repeated G5, not an Alberti figure (C E | D F)."""
    others = [n for k in range(max(0, g - 1), min(len(groups), g + 2)) for n in groups[k] if n.pitch != _pedal(groups[k])]
    moves = [abs(b.pitch - a.pitch) for a, b in zip(others, others[1:])]
    return 2 * sum(m <= 2 for m in moves) > len(moves)


def _figuration(line: list[Note], notes: list[Note], tpb: int, meta, top: bool) -> list[Note]:
    """In runs of FIGURE_NOTES or more notes a sixteenth or less apart, each beat (the time signature's,
    features.beat_grid) that is broken-chord figuration (_figure_beat) becomes one note: the beat's top note (bottom,
    for a bass line), from the beat if the piece has a note there, to the end of the beat's last note. The Bach C
    major prelude's level 1 was its right hand's sixteenths, unsimplified and half snapped to eighths; now it's the
    top of each beat, C5 E5 C5 E5 | D5 F5 D5 F5. Triplet eighths (Moonlight) aren't fast enough to count."""
    w = max(1, tpb // 32)
    fast = tpb // 4 + w
    grid = features.beat_grid(meta, tpb)
    starts = sorted({n.start for n in notes})
    out, i = [], 0
    while i < len(line):
        j = i + 1
        while j < len(line) and line[j].start - line[j - 1].start <= fast:
            j += 1
        run = line[i:j]
        i = j
        if len(run) < FIGURE_NOTES:
            out.extend(run)
            continue
        spans, a = [], 0
        while a < len(run):
            beat_of = features.beat_index(run[a].start, grid)
            b = a + 1
            while b < len(run) and features.beat_index(run[b].start, grid) == beat_of:
                b += 1
            spans.append((a, b, beat_of))
            a = b
        groups = [run[a:b] for a, b, _ in spans]
        for g, (a, b, beat_of) in enumerate(spans):
            group = groups[g]
            if not _figure_beat(run, a, b) or _pedal(group) is not None and _moves_by_step(groups, g):
                out.extend(group)
            else:
                # A tune that starts mid-beat after the figure (notes that don't come round again) keeps its notes.
                last = max(k for k in range(a, b) if _comes_round(run, k))
                tail, group = run[last + 1:b], run[a:last + 1]
                pick = max(group, key=lambda n: n.pitch) if top else min(group, key=lambda n: n.pitch)
                seg, k = beat_of
                beat = grid[seg][0] + k * grid[seg][1]
                near = bisect_left(starts, beat - w)
                on_beat = near < len(starts) and starts[near] <= beat + w and (not out or out[-1].end <= beat + w)
                start = beat if on_beat else group[0].start
                out.append(replace(pick, start=start, end=max(start + 1, max(n.end for n in group))))
                out.extend(tail)
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


def _repeats(line: list[Note]) -> float:
    """The share of moves in a line that stay on the same pitch."""
    if len(line) < 2:
        return 0.0
    return sum(a.pitch == b.pitch for a, b in zip(line, line[1:])) / (len(line) - 1)


def _tune_like(line: list[Note], least: int, span: int, tpb: int) -> bool:
    """At least `least` notes that move (few repeated pitches) without going round a figure (_recurs) or spelling out
    one chord (six notes or more on three pitch classes or fewer: the Fantaisie-Impromptu's left-hand arpeggios), and
    no more than three a quarter on average over `span` ticks (sixteenth runs, like the Revolutionary Etude's, aren't
    a level 1 tune)."""
    if not least <= len(line) <= 3 * span / tpb or _repeats(line) >= 0.25 or _recurs(line) >= 0.5:
        return False
    return len(line) < 6 or len({n.pitch % 12 for n in line}) > 3


def left_tune_bars(notes: list[Note], tpb: int, meta=()) -> set[int]:
    """Bars (notation.bar_grid numbers) whose tune is in the left hand, for two bars or more in a row:
      - the right hand is silent while the left plays a line (_tune_like) in single notes (three onsets in four or
        more; not an accompaniment's bass and chords), or
      - the right hand only keeps time over a left hand moving in single notes: mostly chords (two onsets in five or
        more), their top note and the note under it mostly repeating (half their moves or more; an inner voice moving
        under a repeated top note, as in Joplin's Solace, is the tune), and no longer than the left hand's notes
        (half as long again at most; a held melody over a moving left hand, as in Chopin's nocturnes, is still the
        tune). Chopin's B minor prelude, Op. 28 No. 6; Schumann's Happy Farmer. A hymn's soprano repeating a note
        over a moving tenor isn't one: the left hand plays two parts.
    A single bar between two such bars, with a right hand that's silent or repeats itself a third of the time, joins
    them, and a single bar alone doesn't count, so level 1 doesn't hop between hands for one bar. Without this,
    level 1 was the right hand's repeated chord tops (B4 B4 B4 ...) and the tune was lost."""
    if not any(n.hand == "L" for n in notes):
        return set()
    w = max(1, tpb // 32)
    right, right_chords = _line(notes, "R", True, w)
    left, left_chords = _line(notes, "L", True, w)
    bars = notation.bar_grid(meta, tpb, max(n.end for n in notes))
    rh = sorted((n for n in notes if n.hand == "R"), key=lambda n: n.start)
    starts = {id(line): [n.start for n in line] for line in (right, left, rh)}
    longest = max((n.duration for n in rh), default=0)

    def within(line: list[Note], bar) -> list[Note]:
        s = starts[id(line)]
        return line[bisect_left(s, bar.start):bisect_left(s, bar.end)]

    def silent(bar) -> bool:
        s = starts[id(rh)]
        return not any(n.end > bar.start + w for n in rh[bisect_left(s, bar.start - longest):bisect_left(s, bar.end)])

    def mean_duration(line: list[Note]) -> float:
        return sum(n.duration for n in line) / len(line)

    found, quiet = set(), set()
    for bar in bars:
        lh = within(left, bar)
        span = bar.end - bar.start
        single = [len(c) == 1 for c in left_chords if bar.start <= c[0].start < bar.end]
        melodic = 4 * sum(single) >= 3 * len(single)
        if silent(bar):
            quiet.add(bar.number)
            if melodic and _tune_like(lh, 2, span, tpb):
                found.add(bar.number)
            continue
        top = within(right, bar)
        chords = [c for c in right_chords if bar.start <= c[0].start < bar.end]
        inner = [c[-2] for c in chords if len(c) > 1]
        keeps_time = (len(top) >= 3 and _repeats(top) >= 0.5 and 5 * len(inner) >= 2 * len(chords)
                      and _repeats(inner) >= 0.5)
        if keeps_time and melodic and _tune_like(lh, 3, span, tpb) and mean_duration(top) <= 1.5 * mean_duration(lh):
            found.add(bar.number)
    for bar in bars:
        if bar.number - 1 in found and bar.number + 1 in found and bar.number not in found:
            if bar.number in quiet or _repeats(within(right, bar)) >= 1 / 3:
                found.add(bar.number)
    return {b for b in found if b - 1 in found or b + 1 in found}


def _bar_of(notes: list[Note], tpb: int, meta):
    """A function from a tick to its bar number (notation.bar_grid)."""
    bars = notation.bar_grid(meta, tpb, max(n.end for n in notes))
    starts = [b.start for b in bars]
    return lambda tick: bars[max(0, bisect_right(starts, tick) - 1)].number


def _melody(notes: list[Note], tpb: int, meta) -> tuple[list[Note], set[int], object]:
    """(melody, left_tune_bars, bar_of): see melody."""
    gap = MELODY_MIN_GAP * tpb // 8
    right = outer_notes(notes, "R", tpb, gap, meta)
    left_bars = left_tune_bars(notes, tpb, meta)
    if not left_bars:
        return right, left_bars, None
    bar_of = _bar_of(notes, tpb, meta)
    left = outer_notes(notes, "L", tpb, gap, meta, top=True)
    line = [n for n in right if bar_of(n.start) not in left_bars] + [n for n in left if bar_of(n.start) in left_bars]
    return sorted(line, key=lambda n: (n.start, n.pitch)), left_bars, bar_of


def melody(notes: list[Note], tpb: int, meta=()) -> list[Note]:
    """Level 1's line: the right hand's top line (outer_notes), except in the bars whose tune is in the left hand
    (left_tune_bars), which take the left hand's top line instead (its notes stay in the left hand)."""
    return _melody(notes, tpb, meta)[0]


def make_level(notes: list[Note], level: int, ticks_per_beat: int, meta=()) -> list[Note]:
    """Level `level` of the full arrangement `notes` (hands set). `meta` is the song's meta events (for the time
    signature's beat and bars; without one, a beat is a quarter in 4/4)."""
    if level not in LEVELS:
        raise ValueError(f"unknown level {level}")
    if level == 4:
        return list(notes)
    tpb = ticks_per_beat
    if level in (1, 2):
        tune, left_bars, bar_of = _melody(notes, tpb, meta)
        if level == 1:
            return _quantize(tune, tpb // 2)
        # The other hand's outer line, one note a beat: the bass under a right-hand tune, the right hand's top line
        # over a left-hand one.
        gap = MELODY_MIN_GAP * tpb // 8
        other = outer_notes(notes, "L", tpb, gap, meta)
        if left_bars:
            over = outer_notes(notes, "R", tpb, gap, meta)
            other = [n for n in other if bar_of(n.start) not in left_bars] + [n for n in over if bar_of(n.start) in left_bars]
        return _quantize(tune + one_per_beat(other, meta, tpb), tpb // 2)
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
