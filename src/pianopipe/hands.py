"""Cost-based left/right hand separation for single-stream piano MIDI.

This is the fallback from the brief: notes are grouped into onset chords, and
a Viterbi pass picks, for each chord, the split point between the left and
right hand. Costs penalize hand jumps, more than 5 notes or more than about an
octave in one hand, and hands crossing. Penner's statistical models can
replace this later behind the same split_hands() interface.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import Note

MAX_NOTES_PER_HAND = 5
COMFORT_SPAN = 12  # semitones a hand covers without penalty
HARD_SPAN = 16  # beyond this a hand cannot reach
RH_HOME, LH_HOME = 72, 48  # C5, C3

W_MOVE = 0.3  # per semitone a hand's center moves between chords
W_SPAN = 4.0  # per semitone beyond COMFORT_SPAN
W_PRIOR = 0.05  # per semitone away from the hand's home position
W_CROSS = 6.0  # flat cost when the left hand ends up above the right
INF = 1e9


@dataclass
class _State:
    cost: float
    back: int  # split chosen for the previous chord
    rh: float  # right hand center after this chord
    lh: float


def group_chords(notes: list[Note], window: int) -> list[list[Note]]:
    """Group notes whose onsets fall within `window` ticks of the chord's first note."""
    chords: list[list[Note]] = []
    for n in sorted(notes, key=lambda n: (n.start, n.pitch)):
        if chords and n.start - chords[-1][0].start <= window:
            chords[-1].append(n)
        else:
            chords.append([n])
    return [sorted(c, key=lambda n: n.pitch) for c in chords]


def _hand_cost(pitches: list[int], home: int) -> float:
    if not pitches:
        return 0.0
    if len(pitches) > MAX_NOTES_PER_HAND:
        return INF
    span = pitches[-1] - pitches[0]
    if span > HARD_SPAN:
        return INF
    cost = max(0, span - COMFORT_SPAN) * W_SPAN
    center = (pitches[0] + pitches[-1]) / 2
    return cost + abs(center - home) * W_PRIOR


def split_hands(notes: list[Note], ticks_per_beat: int = 480) -> list[Note]:
    """Return the notes with hand set to "R" or "L"."""
    if not notes:
        return []
    chords = group_chords(notes, window=max(1, ticks_per_beat // 32))
    trellis: list[list[_State]] = []
    prev: list[_State] = [_State(0.0, -1, RH_HOME, LH_HOME)]
    for chord in chords:
        pitches = [n.pitch for n in chord]
        states = []
        for k in range(len(pitches) + 1):
            lh, rh = pitches[:k], pitches[k:]
            local = _hand_cost(lh, LH_HOME) + _hand_cost(rh, RH_HOME)
            best = _State(INF * 10, 0, RH_HOME, LH_HOME)
            for j, p in enumerate(prev):
                rc = (rh[0] + rh[-1]) / 2 if rh else p.rh
                lc = (lh[0] + lh[-1]) / 2 if lh else p.lh
                move = (abs(rc - p.rh) if rh else 0) + (abs(lc - p.lh) if lh else 0)
                cross = W_CROSS + (lc - rc) if lc > rc else 0.0
                total = p.cost + local + move * W_MOVE + cross
                if total < best.cost:
                    best = _State(total, j, rc, lc)
            states.append(best)
        trellis.append(states)
        prev = states

    # Backtrack the cheapest path.
    k = min(range(len(prev)), key=lambda i: prev[i].cost)
    splits = [0] * len(chords)
    for i in range(len(chords) - 1, -1, -1):
        splits[i] = k
        k = trellis[i][k].back

    out: list[Note] = []
    for chord, k in zip(chords, splits):
        out.extend(n.with_hand("L" if idx < k else "R") for idx, n in enumerate(chord))
    return out


def hand_accuracy(predicted: list[Note], truth: list[Note]) -> float:
    """Share of notes whose predicted hand matches the labeled hand (matched by start and pitch)."""
    labels = {(n.start, n.pitch): n.hand for n in truth}
    scored = [(labels.get((n.start, n.pitch)), n.hand) for n in predicted]
    scored = [(t, p) for t, p in scored if t is not None]
    if not scored:
        return 0.0
    return sum(t == p for t, p in scored) / len(scored)


def playability_flags(notes: list[Note], ticks_per_beat: int = 480) -> list[tuple[int, str]]:
    """(tick, description) for each chord that one hand cannot play."""
    flags = []
    for chord in group_chords(notes, window=max(1, ticks_per_beat // 32)):
        for hand in ("R", "L"):
            pitches = [n.pitch for n in chord if n.hand == hand]
            if len(pitches) > MAX_NOTES_PER_HAND:
                flags.append((chord[0].start, f"{hand}H has {len(pitches)} notes"))
            elif pitches and max(pitches) - min(pitches) > HARD_SPAN:
                flags.append((chord[0].start, f"{hand}H spans {max(pitches) - min(pitches)} semitones"))
    return flags


def playability_issues(notes: list[Note], ticks_per_beat: int = 480) -> list[str]:
    """Describe chords that one hand cannot play, for flagging or skipping a file."""
    return [f"tick {tick}: {text}" for tick, text in playability_flags(notes, ticks_per_beat)]
