"""A song's melody (a voice or melody instrument part) merged into the piano's right hand as its top voice, so that a
piano/vocal score plays as solo piano: docs/catalog-expansion.md §2.6. musicxml.load picks the parts and the bars to
merge (`load(melody=...)`); this module works on notes.

- The melody line (top_line): the highest note sounding at each onset of the melody parts, as one line. A note
  starting under a higher one still held is left out, and a held note is cut where a higher one starts (a duet's
  parts taking turns or harmonising).
- The right hand under it (merge): accompaniment notes above the melody note sounding at their onset are dropped, so
  are the melody's unisons and its octave doublings struck with it; a note held into a melody note at or below it is
  cut there. A right-hand note more than a tenth (hands.HARD_SPAN) under the melody goes to the left hand where it
  fits there (a tenth and five notes), else it's dropped; so is the lowest note of a chord that would have more than
  five. A left-hand note at or above the melody is dropped too. It runs on the hands as repaired
  (hands.repair_hands), so a cross-staff note is already with the hand that plays it.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, replace

from .hands import HARD_SPAN, MAX_NOTES_PER_HAND
from .model import Note

ACCENT = 10  # velocity added to the melody, so it sings over the accompaniment


@dataclass
class MergeStats:
    dropped: int = 0  # accompaniment notes left out (above, doubling, or past a tenth with no room in the left hand)
    moved: int = 0  # right-hand notes given to the left hand
    cut: int = 0  # accompaniment notes shortened where a melody note at or below them starts


def top_line(notes: list[Note], tpb: int) -> list[Note]:
    """The highest note sounding at each onset of `notes` (the melody parts' notes), as one line: a note starting
    while a higher (or the same) note is still held is left out, even one cut from the line, and a held note is cut
    where a higher one starts (left out if that leaves less than a 32nd note)."""
    shortest = max(1, tpb // 8)
    out: list[Note] = []
    sounding: list[Note] = []  # every part's notes still held, cut from the line or not
    for n in sorted(notes, key=lambda n: (n.start, -n.pitch, -n.end)):
        sounding = [s for s in sounding if s.end > n.start]
        under = any(s.pitch >= n.pitch for s in sounding)
        sounding.append(n)
        if under:
            continue
        if out and out[-1].end > n.start:
            cur = out[-1]
            if n.start - cur.start < shortest:
                out.pop()
            else:
                out[-1] = replace(cur, end=n.start)
        out.append(n)
    return out


def merge(notes: list[Note], line: list[Note], tpb: int) -> tuple[list[Note], MergeStats]:
    """The piano's notes (hands set) with `line` (top_line's melody) as the right hand's top voice."""
    stats = MergeStats()
    if not line:
        return list(notes), stats
    window = max(1, tpb // 32)  # notes this close start together (hands.group_chords' window)
    shortest = max(1, tpb // 8)
    line = sorted(line, key=lambda n: n.start)
    starts = [m.start for m in line]

    def melody_at(t: int) -> Note | None:
        """The melody note struck with a note starting at t (within the chord window), else the one sounding at t."""
        i = bisect_left(starts, t - window)
        if i < len(line) and line[i].start <= t + window:
            return line[i]
        i = bisect_right(starts, t) - 1
        return line[i] if i >= 0 and line[i].end > t else None

    def release(n: Note, too_far: bool) -> Note | None:
        """n, released where a later melody note at or below it starts (or, `too_far`, more than a tenth above it);
        None if that leaves less than a 32nd note."""
        end = n.end
        for k in range(bisect_right(starts, n.start + window), bisect_left(starts, n.end)):
            later = line[k]
            if later.pitch <= n.pitch or (too_far and later.pitch - n.pitch > HARD_SPAN):
                end = later.start
                break
        if end - n.start < shortest:
            stats.dropped += 1
            return None
        if end != n.end:
            stats.cut += 1
            n = replace(n, end=end)
        return n

    # The left hand never reaches the melody: a note at or above it (a cross-staff note, or a low tune) is left out,
    # so its key isn't struck twice and the hands don't cross over it.
    left: list[Note] = []
    for n in notes:
        if n.hand == "R":
            continue
        m = melody_at(n.start)
        if m is not None and n.pitch >= m.pitch:
            stats.dropped += 1
        elif (n := release(n, False)) is not None:
            left.append(n)
    kept: list[Note] = []
    for n in notes:
        if n.hand != "R":
            continue
        m = melody_at(n.start)
        if m is not None and (n.pitch >= m.pitch or
                              ((m.pitch - n.pitch) % 12 == 0 and abs(n.start - m.start) <= window)):
            stats.dropped += 1
            continue
        # Held into a later melody note at or below it, or more than a tenth above it: released there.
        n = release(n, True)
        if n is None:
            continue
        if m is not None and m.pitch - n.pitch > HARD_SPAN:
            if _fits(left, n, window):
                left.append(replace(n, hand="L", track=1, channel=1, finger=None))
                stats.moved += 1
            else:
                stats.dropped += 1
            continue
        kept.append(n)

    # No more than five right-hand keys down with the melody, at any onset (the melody's or the accompaniment's):
    # the lowest accompaniment notes past five go, a note struck there to the left hand (or out), a held one
    # released there.
    melody = [replace(m, hand="R", track=0, channel=0, velocity=min(127, m.velocity + ACCENT), finger=None) for m in line]
    for t in sorted({n.start for n in kept} | set(starts)):
        if melody_at(t) is None:
            continue
        down = sorted((n for n in kept if n.start <= t + window and n.end > t + window), key=lambda n: n.pitch)
        for n in down[:max(0, len(down) + 1 - MAX_NOTES_PER_HAND)]:
            i = kept.index(n)
            if n.start >= t - window:
                kept.pop(i)
                if _fits(left, n, window):
                    left.append(replace(n, hand="L", track=1, channel=1, finger=None))
                    stats.moved += 1
                else:
                    stats.dropped += 1
            else:
                kept[i] = replace(n, end=t)
                stats.cut += 1
    return sorted(kept + melody + left, key=lambda n: (n.start, n.pitch)), stats


def _fits(left: list[Note], n: Note, window: int) -> bool:
    """Whether the left hand can also take n: its notes sounding at n's start (held or struck with it) and n within
    a tenth and five notes, and no left-hand note on n's key then."""
    held = [x.pitch for x in left if x.start - window <= n.start < x.end or abs(x.start - n.start) <= window]
    if n.pitch in held or len(held) + 1 > MAX_NOTES_PER_HAND:
        return False
    pitches = held + [n.pitch]
    return max(pitches) - min(pitches) <= HARD_SPAN
