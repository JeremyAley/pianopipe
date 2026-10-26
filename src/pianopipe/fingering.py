"""Automatic piano fingering: a finger (1 thumb to 5 little finger) for every note of each hand.

A Viterbi search over each hand's chords picks the cheapest fingering under a hand-set cost model after
Parncutt, Sloboda, Clarke, Raekallio and Desain (1997), "An ergonomic model of keyboard fingering for melodic
fragments": a table of comfortable and practical spans per finger pair, and rules for stretches, thumb passing,
weak fingers and black keys. Chords and held notes follow Al Kasimi, Nichols and Raphael (2007): a state is the
fingering of every sounding note, so a held note keeps its finger and the next notes must fit around it. No
parameters are learned from data (see docs/decisions.md #12 for why PIG-trained models aren't used).

The left hand uses the right hand's model on mirrored pitches: the keyboard is symmetric around D (and G#), so
mirroring keeps black keys black and turns the left hand's thumb side into the right hand's.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from itertools import combinations

from .hands import group_chords
from .model import Note

FINGERS = (1, 2, 3, 4, 5)
MIRROR = 2 * 62  # left-hand pitch p is treated as right-hand pitch MIRROR - p (mirror around D4)

# Parncutt et al. (1997), table 1: spans in semitones from the lower to the higher finger of a pair, right hand.
# (MinPrac, MinComf, MinRel, MaxRel, MaxComf, MaxPrac). A negative span is the thumb passing under (or a finger
# over the thumb). Changed from the paper's: the negative limits for 1-2, 1-3 and 1-4 are wider, so the thumb can
# pass under a fourth in arpeggios, and the pairs without the thumb reach a semitone further (MaxRel and up), so
# 2-3 can take the major third of an arpeggio.
SPANS = {
    (1, 2): (-7, -5, 1, 5, 8, 10),
    (1, 3): (-7, -5, 3, 7, 10, 12),
    (1, 4): (-5, -3, 5, 9, 12, 14),
    (1, 5): (-1, 1, 7, 10, 13, 15),
    (2, 3): (1, 1, 1, 3, 4, 6),
    (2, 4): (1, 1, 3, 5, 6, 8),
    (2, 5): (2, 2, 5, 7, 9, 11),
    (3, 4): (1, 1, 1, 2, 3, 5),
    (3, 5): (1, 1, 3, 5, 6, 8),
    (4, 5): (1, 1, 1, 2, 3, 5),
}
# Where each finger rests above the thumb in a five-finger position (C D E F G): in semitones, to judge chord
# shapes, and in keyboard units (_key_x, where every white key is two units wide), to judge how far the hand moves.
OFFSET = {1: 0.0, 2: 2.0, 3: 4.0, 4: 5.5, 5: 7.0}
KEY_OFFSET = {1: 0, 2: 2, 3: 4, 4: 6, 5: 8}
_KEY_X = (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12)  # C C# D D# E F F# G G# A A# B; the next C is at 14
# Passing the thumb under a finger (or that finger over the thumb), on top of the span costs: a flat cost per
# crossing, so a scale crosses as rarely as it can, plus W_CROSS_SPAN per semitone crossed, so it crosses on
# semitones (E-F, B-C in C major) when it can.
CROSS = {2: 5.0, 3: 4.0, 4: 5.0, 5: 12.0}
W_CROSS_SPAN = 0.5

W_IMPOSSIBLE = 10.0  # per semitone beyond MinPrac/MaxPrac
W_POSITION = 0.4  # per keyboard unit (half a white key) the hand's position moves between onsets
W_FIT = 1.5  # per semitone a chord's notes spread away from the five-finger shape
W_REUSE = 0.5  # per semitone a finger moves between two chords
WEAK = 0.5  # finger 4 or 5 on a single note (Parncutt's weak-finger rule, at half weight)
THUMB_BLACK = 3.0  # thumb on a black key, for each white key just before or after it (Parncutt: 2)
REPEAT_CHANGE = 3.0  # changing fingers on a repeated key
SAME_FINGER = 12.0  # one finger playing two different keys in a row, legato
SHIFT = 24.0  # lifting the hand to a new place in legato playing (caps every other transition cost)
LEAP_FREE = 8  # keyboard units (a fifth) a melody can leap before a shift gets cheaper ...
W_LEAP = 2.0  # ... by this much per unit, since nobody keeps a wide leap legato
SHIFT_MIN = 4.0
SHIFT_DETACHED = 4.0  # the same after a rest, when the hand has time to move
W_SHIFT_DISTANCE = 0.02  # per keyboard unit of a shift, so nearer landings win ties
DETACHED_SCALE = 0.4  # how much of the span costs apply after a rest
MAX_HAND = 5
RELEASE = 4.0  # letting a held note go before its end
HOLD_REACH = 12  # semitones: a held note further than this from the next notes is treated as released


def _black(pitch: int) -> bool:
    return pitch % 12 in (1, 3, 6, 8, 10)


def _key_x(pitch: int) -> int:
    return 14 * (pitch // 12) + _KEY_X[pitch % 12]


@lru_cache(maxsize=None)
def _span_cost(f1: int, p1: int, f2: int, p2: int) -> float:
    """Cost of fingers f1 and f2 on pitches p1 and p2 (right-hand pitches), played together or one after the other."""
    (lo_f, lo_p), (hi_f, hi_p) = sorted(((f1, p1), (f2, p2)))
    d = hi_p - lo_p
    min_prac, min_comf, min_rel, max_rel, max_comf, max_prac = SPANS[(lo_f, hi_f)]
    thumb = lo_f == 1
    cost = 2.0 * (max(0, min_comf - d) + max(0, d - max_comf))  # stretch rule
    cost += W_IMPOSSIBLE * (max(0, min_prac - d) + max(0, d - max_prac))
    if d >= 0 or not thumb:  # a thumb crossing is charged by CROSS instead
        cost += (2.0 if thumb else 1.0) * max(0, min_rel - d)  # small-span rule
    cost += (1.0 if thumb else 2.0) * max(0, d - max_rel)  # large-span rule
    return cost


@lru_cache(maxsize=None)
def _step_cost(fa: int, pa: int, fb: int, pb: int) -> float:
    """Legato cost of finger fa on pa followed by finger fb on pb (right-hand pitches)."""
    if fa == fb:
        return 0.0 if pa == pb else SAME_FINGER + 0.5 * abs(pb - pa)
    if pa == pb:
        return REPEAT_CHANGE
    cost = _span_cost(fa, pa, fb, pb)
    lo_f, hi_f = sorted((fa, fb))
    d = (pb - pa) if fb > fa else (pa - pb)  # span from the lower finger to the higher one
    if lo_f == 1 and d < 0:  # thumb passing under, or a finger crossing over the thumb
        cost += CROSS[hi_f] + W_CROSS_SPAN * -d
        thumb_p, other_p = (pa, pb) if fa == 1 else (pb, pa)
        if _black(thumb_p) == _black(other_p):
            cost += 1.0
        elif _black(thumb_p):  # thumb on the upper black key, finger on the lower white key
            cost += 3.0
    # Black-key rules: thumb or fifth finger on a black key next to a white one.
    if fb == 1 and _black(pb) and not _black(pa):
        cost += THUMB_BLACK
    if fa == 1 and _black(pa) and not _black(pb):
        cost += THUMB_BLACK
    if fb == 5 and _black(pb) and not _black(pa):
        cost += 2.0
    if fa == 5 and _black(pa) and not _black(pb):
        cost += 2.0
    if {fa, fb} == {3, 4}:  # 3 and 4 in a row are weakly independent (half of Parncutt's point)
        cost += 0.5
        if fa == 3 and not _black(pa) and _black(pb):
            cost += 1.0
    return cost


class _State:
    __slots__ = ("fingers", "unary", "position", "cost", "back")

    def __init__(self, fingers: dict[int, int], unary: float, position: float):
        self.fingers = fingers  # note index -> finger
        self.unary = unary
        self.position = position  # where the thumb would rest, from the onset notes
        self.cost = 0.0
        self.back = -1


def _vertical(pitches: list[int], fingers: tuple[int, ...], onsets: list[bool]) -> float:
    """Cost of one hand shape: the sounding notes (ascending right-hand pitches) with ascending fingers."""
    cost = 0.0
    for p, f, onset in zip(pitches, fingers, onsets):
        if f == 1 and _black(p):
            cost += 1.0  # thumb on a black key
        if onset and len(pitches) == 1 and f in (4, 5):
            cost += WEAK  # single notes only
    if len(pitches) > 1:
        rests = [p - OFFSET[f] for p, f in zip(pitches, fingers)]
        cost += W_FIT * (max(rests) - min(rests))
        for i in range(len(pitches) - 1):
            if fingers[i] != fingers[i + 1]:  # equal only in a chord of more than five notes (_spread)
                cost += _span_cost(fingers[i], pitches[i], fingers[i + 1], pitches[i + 1])
    return cost


def _spread(n: int) -> tuple[int, ...]:
    """Fingers for a chord of more than five notes: spread 1-5 over them (some fingers repeat)."""
    return tuple(1 + round(4 * i / (n - 1)) for i in range(n))


def finger_hand(notes: list[Note], hand: str, ticks_per_beat: int = 480) -> dict[int, int]:
    """Choose a finger for every note of one hand. Returns note index (into `notes`) -> finger."""
    return search(notes, hand, ticks_per_beat)[0]


def search(
    notes: list[Note], hand: str, ticks_per_beat: int = 480, forced: dict[int, int] | None = None
) -> tuple[dict[int, int], float]:
    """Viterbi search over one hand's notes (sorted by start). Returns (note index -> finger, total cost).

    `forced` pins some notes' fingers, which prices a given fingering (used by the evaluation to explain choices).
    """
    if not notes:
        return {}, 0.0
    forced = forced or {}
    rh_pitch = (lambda p: p) if hand == "R" else (lambda p: MIRROR - p)
    window = max(1, ticks_per_beat // 32)
    hold = max(1, ticks_per_beat // 4)  # a note still sounding this long after the next onset is held
    rest = max(1, ticks_per_beat // 4)  # a gap this long before an onset lets the hand move freely
    index = {id(n): i for i, n in enumerate(notes)}
    chords = [[index[id(n)] for n in c] for c in group_chords(notes, window)]

    trellis: list[list[_State]] = []
    onset_sets: list[list[int]] = []
    sounding: list[int] = []
    prev_end = None
    for chord in chords:
        start = notes[chord[0]].start
        sounding = [i for i in sounding if notes[i].end - start >= hold and i not in chord]
        # A held note more than an octave from the new notes is left to the pedal (or the score's notation).
        low, high = min(notes[i].pitch for i in chord), max(notes[i].pitch for i in chord)
        sounding = [i for i in sounding if max(high, notes[i].pitch) - min(low, notes[i].pitch) <= HOLD_REACH]
        while sounding and len(sounding) + len(chord) > MAX_HAND:
            sounding.remove(min(sounding, key=lambda i: notes[i].end))  # drop the note released first
        # Each held note is either kept under its finger or let go early (for RELEASE each; the pedal or a
        # player's judgement covers it), so one long note can't lock the hand.
        states = []
        for members in ([sounding + chord, chord] if sounding else [chord]):
            members = sorted(members, key=lambda i: (rh_pitch(notes[i].pitch), i))
            pitches = [rh_pitch(notes[i].pitch) for i in members]
            onsets = [i in chord for i in members]
            many = len(members) > MAX_HAND
            for fingers in [_spread(len(members))] if many else combinations(FINGERS, len(members)):
                fmap = dict(zip(members, fingers))
                if any(fmap[i] != forced[i] for i in chord if i in forced):
                    continue
                rests = [_key_x(rh_pitch(notes[i].pitch)) - KEY_OFFSET[fmap[i]] for i in chord]
                states.append(_State(fmap, _vertical(pitches, fingers, onsets), sum(rests) / len(rests)))
        if not states:
            raise ValueError(f"forced fingers {[forced.get(i) for i in chord]} don't fit the notes at tick {start}")

        if trellis:
            prev_chord = onset_sets[-1]
            detached = start - prev_end >= rest
            for s in states:
                held = [i for i in s.fingers if i not in chord]
                best, back = float("inf"), -1
                for j, p in enumerate(trellis[-1]):
                    if any(p.fingers.get(i) != s.fingers[i] for i in held):
                        continue  # a held note keeps its finger, and a note let go stays released
                    released = sum(i not in s.fingers for i in sounding if i in p.fingers)
                    total = p.cost + _move(p, s, prev_chord, chord, notes, rh_pitch, detached) + RELEASE * released
                    if total < best:
                        best, back = total, j
                if back < 0:  # every previous state used a held note's finger differently: start afresh
                    j = min(range(len(trellis[-1])), key=lambda k: trellis[-1][k].cost)
                    best, back = trellis[-1][j].cost + SHIFT, j
                s.cost, s.back = best + s.unary, back
        else:
            for s in states:
                s.cost = s.unary
        trellis.append(states)
        onset_sets.append(chord)
        sounding = sounding + chord
        prev_end = max(notes[i].end for i in chord)

    k = min(range(len(trellis[-1])), key=lambda i: trellis[-1][i].cost)
    total = trellis[-1][k].cost
    out: dict[int, int] = {}
    for c in range(len(trellis) - 1, -1, -1):
        state = trellis[c][k]
        for i in onset_sets[c]:
            out[i] = state.fingers[i]
        k = state.back
    return out, total


def _move(p: _State, s: _State, prev_chord: list[int], chord: list[int], notes: list[Note], rh_pitch, detached: bool) -> float:
    """Cost of going from the previous onset set (in state p) to the next (in state s)."""
    if len(prev_chord) == 1 and len(chord) == 1:
        a, b = prev_chord[0], chord[0]
        step = _step_cost(p.fingers[a], rh_pitch(notes[a].pitch), s.fingers[b], rh_pitch(notes[b].pitch))
    else:
        step = _chord_move(p, s, prev_chord, chord, notes, rh_pitch)
    distance = abs(s.position - p.position)
    cost = step + W_POSITION * distance
    if detached:
        cost *= DETACHED_SCALE
        shift = SHIFT_DETACHED
    else:
        leap = abs(_center(chord, notes, rh_pitch) - _center(prev_chord, notes, rh_pitch))
        shift = max(SHIFT_MIN, SHIFT - W_LEAP * max(0.0, leap - LEAP_FREE))
    # Lifting the hand to a new place costs the same whatever the fingers; it caps everything else.
    return min(cost, shift + W_SHIFT_DISTANCE * distance)


def _center(chord: list[int], notes: list[Note], rh_pitch) -> float:
    return sum(_key_x(rh_pitch(notes[i].pitch)) for i in chord) / len(chord)


def _chord_move(p: _State, s: _State, prev_chord: list[int], chord: list[int], notes: list[Note], rh_pitch) -> float:
    """Cost of moving from one onset set to the next when either has more than one note."""
    cost = 0.0
    prev_by_finger = {p.fingers[i]: rh_pitch(notes[i].pitch) for i in prev_chord}
    melodic = len(prev_chord) == 1 or len(chord) == 1  # from a chord to one note or back: fingers play legato
    for i in chord:
        f = s.fingers[i]
        if f in prev_by_finger and prev_by_finger[f] != rh_pitch(notes[i].pitch):
            cost += W_REUSE * abs(rh_pitch(notes[i].pitch) - prev_by_finger[f]) + (SAME_FINGER if melodic else 0.0)
    # Outer voices move like a melody (top to top, bottom to bottom), at half weight, without the same-finger rule.
    for pick in (min, max):
        a = pick(prev_chord, key=lambda i: rh_pitch(notes[i].pitch))
        b = pick(chord, key=lambda i: rh_pitch(notes[i].pitch))
        fa, fb = p.fingers[a], s.fingers[b]
        if fa != fb:
            cost += 0.5 * _step_cost(fa, rh_pitch(notes[a].pitch), fb, rh_pitch(notes[b].pitch))
    return cost


def assign_fingering(notes: list[Note], ticks_per_beat: int = 480) -> list[Note]:
    """Return the notes (same order) with `finger` set for every note that has a hand."""
    out = list(notes)
    for hand in ("R", "L"):
        idx = [i for i, n in enumerate(notes) if n.hand == hand]
        hand_notes = sorted((notes[i] for i in idx), key=lambda n: (n.start, n.pitch))
        order = {id(n): k for k, n in enumerate(hand_notes)}
        fingers = finger_hand(hand_notes, hand, ticks_per_beat)
        for i in idx:
            out[i] = replace(notes[i], finger=fingers[order[id(notes[i])]])
    return out
