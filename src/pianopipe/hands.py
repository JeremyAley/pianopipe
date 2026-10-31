"""Cost-based left/right hand separation for single-stream piano MIDI.

This is the fallback from the brief: notes are grouped into onset chords, and
a Viterbi pass picks, for each chord, the split point between the left and
right hand. Costs penalize hand jumps, more than 5 notes or more than about an
octave in one hand, and hands crossing. Penner's statistical models can
replace this later behind the same split_hands() interface.

The same costs repair hands read from a file's structure (repair_hands): chords one hand can't play, and a score's
cross-staff notes (docs/catalog-expansion.md §2.2).

Both end with a playability pass over what each hand holds as well as strikes (playable_hands): a hand stretching
past a tenth to keep a key down, or the left hand inside the right; the given hands stay wherever they're playable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

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
    return playable_hands(out, ticks_per_beat)  # chords alone don't show a hand holding a key the other plays under


W_STAFF = 2.0  # a score's cross-staff note given to the hand of the staff it's printed on instead of its voice's
W_REPAIR = 3.0  # any other note moved off the hand the file gives it, in a chord nothing else makes playable
MAX_FREE = 5  # cross-staff notes per chord tried in every combination (2**5 candidates); more stay with their voice


def _chord_candidates(chord: list[Note], free: list[bool]) -> list[tuple[tuple[str, ...], float, bool]]:
    """(hands, cost of the moves, whether a note the file placed moved) for one chord: the file's hands and every
    combination of moving its cross-staff notes, and, when none of those is playable, every split into left (the
    lowest notes) and right."""
    given = tuple(n.hand for n in chord)
    other = {"R": "L", "L": "R"}
    movable = [i for i, f in enumerate(free) if f][:MAX_FREE]
    out = []
    for mask in range(1 << len(movable)):
        hands = list(given)
        for b, i in enumerate(movable):
            if mask >> b & 1:
                hands[i] = other[given[i]]
        out.append((tuple(hands), W_STAFF * bin(mask).count("1"), False))

    def playable(hands):
        return all(_hand_cost([n.pitch for n, h in zip(chord, hands) if h == x], 0) < INF for x in "RL")

    if not any(playable(h) for h, _, _ in out):
        for k in range(len(chord) + 1):
            hands = tuple("L" if i < k else "R" for i in range(len(chord)))
            moved = [i for i in range(len(chord)) if hands[i] != given[i]]
            if moved and playable(hands):
                cost = sum(W_STAFF if free[i] else W_REPAIR for i in moved)
                out.append((hands, cost, any(not free[i] for i in moved)))
    return out


def repair_hands(notes: list[Note], ticks_per_beat: int = 480, cross_staff: list[bool] | None = None,
                 repaired: list[int] | None = None) -> list[Note]:
    """Notes with hands from the file's structure (staves, voices, tracks), with chords one hand can't play (more
    than MAX_NOTES_PER_HAND notes or HARD_SPAN semitones) split between the hands, and a score's cross-staff notes
    given to the hand that plays them.

    The same Viterbi as split_hands (spans, hand movement, crossing) chooses among each chord's candidates: the file's
    hands; each combination of moving its `cross_staff` notes (W_STAFF each), so that a note printed in the treble
    staff in a bass-staff voice goes to the right hand when the left would stretch past a tenth or leap away to reach
    it; and, only in a chord none of those makes playable, each playable split into a lower left and an upper right
    hand (W_REPAIR for each other note moved). A chord no split makes playable keeps the file's hands (the gate counts
    it). `repaired` gets the start tick of each chord where a note the file placed was moved. A moved note loses its
    printed finger, which was for the other hand. Then playable_hands, for what the hands hold as well as strike
    (its moves aren't `repaired`: the gate's count is of chords one hand can't strike)."""
    if not notes:
        return []
    free_of = {id(n): bool(f) for n, f in zip(notes, cross_staff or [False] * len(notes))}
    chords = group_chords(notes, window=max(1, ticks_per_beat // 32))
    cands = [_chord_candidates(c, [free_of[id(n)] for n in c]) for c in chords]
    if all(len(c) == 1 for c in cands):
        return playable_hands(list(notes), ticks_per_beat)

    trellis: list[list[_State]] = []
    prev = [_State(0.0, -1, RH_HOME, LH_HOME)]
    for chord, options in zip(chords, cands):
        states = []
        for hands, moved, _ in options:
            rh = [n.pitch for n, h in zip(chord, hands) if h == "R"]
            lh = [n.pitch for n, h in zip(chord, hands) if h == "L"]
            local = _hand_cost(lh, LH_HOME) + _hand_cost(rh, RH_HOME) + moved
            best = _State(float("inf"), 0, RH_HOME, LH_HOME)
            for j, p in enumerate(prev):
                rc = (min(rh) + max(rh)) / 2 if rh else p.rh
                lc = (min(lh) + max(lh)) / 2 if lh else p.lh
                move = (abs(rc - p.rh) if rh else 0) + (abs(lc - p.lh) if lh else 0)
                cross = W_CROSS + (lc - rc) if lc > rc else 0.0
                total = p.cost + local + move * W_MOVE + cross
                if total < best.cost:
                    best = _State(total, j, rc, lc)
            states.append(best)
        trellis.append(states)
        prev = states

    k = min(range(len(prev)), key=lambda i: prev[i].cost)
    picks = [0] * len(chords)
    for i in range(len(chords) - 1, -1, -1):
        picks[i] = k
        k = trellis[i][k].back
    hand_of = {}
    for chord, options, k in zip(chords, cands, picks):
        hands, _, structural = options[k]
        if structural and repaired is not None:
            repaired.append(chord[0].start)
        for n, h in zip(chord, hands):
            hand_of[id(n)] = h
    notes = [n if hand_of[id(n)] == n.hand else replace(n, hand=hand_of[id(n)], finger=None) for n in notes]
    return playable_hands(notes, ticks_per_beat)


# playable_hands
W_HELD_REACH = 2.0  # each onset a hand's reach (keys struck and held from just before) spans past HARD_SPAN
W_HELD_SEMI = 0.1  # and per semitone past it (a tie-break: a move that only narrows the stretch doesn't pay)
W_CROWD = 2.0  # per key past MAX_NOTES_PER_HAND in a hand's reach
W_CROSSED = 3.0  # each onset the left hand's extent reaches above the right's by more than CROSS_SLACK
W_CROSSED_SEMI = 0.25  # per semitone of that overlap
CROSS_SLACK = 4  # semitones the hands may overlap: interlocked chords, the left thumb up to a third over the right
W_MOVED = 2.5  # each note moved off its given hand
W_SWAPPED = 1.0  # each note of a passage whose hands are swapped as a whole (one decision, not one per note)
W_STRUCK = INF  # a chord struck at once that one hand can't play: a move never makes one


def playable_hands(notes: list[Note], ticks_per_beat: int = 480, moved: list[int] | None = None) -> list[Note]:
    """Notes with the hands they're given where those are playable, and moved to the other hand where they aren't,
    counting the notes each hand still holds (repair_hands and split_hands look only at what's struck together).

    At each onset, per hand: its reach is what it strikes plus what it struck a 16th or less before and still holds
    (a grace note into its chord, a quick roll); a key held longer can be let go under the pedal (a bass note under
    the hand's arpeggio, a rolled chord under its run). A reach past a tenth (HARD_SPAN) costs W_HELD_REACH, past five
    keys W_CROWD a key. Its extent is the reach plus, while it strikes, every key it still holds, as the highway draws
    them, since a hand that keeps playing is still over them: where the left hand's extent
    reaches more than CROSS_SLACK above the right's lowest note, W_CROSSED (the left hand inside the right, or over
    it while the right is playing there; OMR files left-hand notes on the treble staff that way).
    A local search moves notes, each for its whole length, W_MOVED each: at each onset that costs something it
    tries every single note sounding, every split of them into a lower left and an upper right hand, and swapping
    the hands through the run of costly onsets around it (a passage the left hand plays over the right, one note
    each in turn), and keeps the move that saves the most, until none saves anything. So the given hands win
    wherever they're playable, and where they aren't, the fewest notes move. `moved` gets each moved note's start
    tick. A moved note loses its printed finger, which was for the other hand."""
    if not notes:
        return []
    tpb = ticks_per_beat
    grace = max(1, tpb // 8)  # a note let go within a 32nd of the next onset is legato, not held
    together = tpb // 4  # notes struck within a 16th are one handful: a grace note into its chord, a quick roll
    order = sorted(range(len(notes)), key=lambda i: (notes[i].start, notes[i].pitch))
    window = max(1, tpb // 32)
    onsets: list[list[int]] = []  # note indices struck at each onset (group_chords' grouping)
    for i in order:
        if onsets and notes[i].start - notes[onsets[-1][0]].start <= window:
            onsets[-1].append(i)
        else:
            onsets.append([i])
    times = [notes[o[0]].start for o in onsets]
    first = {i: k for k, o in enumerate(onsets) for i in o}
    held: list[list[int]] = [[] for _ in onsets]  # notes struck at an earlier onset and still down at this one
    counts_at: list[list[int]] = [[] for _ in notes]  # the onsets each note sounds at
    for i in range(len(notes)):
        counts_at[i].append(first[i])
        k = first[i] + 1
        while k < len(onsets) and notes[i].end - times[k] > grace:
            held[k].append(i)
            counts_at[i].append(k)
            k += 1
    hand = [n.hand for n in notes]
    given = list(hand)

    def position(k: int, h: str) -> tuple[list[int], list[int], list[int]]:
        """(struck, reach: and the keys held since just before, extent: and every key held while the hand plays)."""
        struck = [notes[i].pitch for i in onsets[k] if hand[i] == h]
        mine = [i for i in held[k] if hand[i] == h]
        reach = struck + [notes[i].pitch for i in mine if times[k] - notes[i].start <= together]
        return struck, reach, (struck + [notes[i].pitch for i in mine]) if struck else reach

    def cost(k: int) -> float:
        total = 0.0
        pos = {}
        for h in "RL":
            struck, p, pos[h] = position(k, h)
            if not p:
                continue
            if len(struck) > MAX_NOTES_PER_HAND or max(struck, default=0) - min(struck, default=0) > HARD_SPAN:
                total += W_STRUCK
            if max(p) - min(p) > HARD_SPAN:
                total += W_HELD_REACH + (max(p) - min(p) - HARD_SPAN) * W_HELD_SEMI
            total += max(0, len(set(p)) - MAX_NOTES_PER_HAND) * W_CROWD
        if pos["R"] and pos["L"] and max(pos["L"]) - min(pos["R"]) > CROSS_SLACK:
            total += W_CROSSED + (max(pos["L"]) - min(pos["R"])) * W_CROSSED_SEMI
        return total

    costs = [cost(k) for k in range(len(onsets))]
    other = {"R": "L", "L": "R"}
    price = [W_MOVED] * len(notes)  # what each note costs off its given hand: W_SWAPPED once a passage swap moved it

    def try_move(flip: tuple[int, ...], each: float | None = None) -> float:
        """Saving (positive: better) from moving the `flip` notes to the other hand, at `each` a note if given (a
        passage swap); they're moved back after."""
        affected = {k for i in flip for k in counts_at[i]}
        before = sum(costs[k] for k in affected) + sum(price[i] for i in flip if hand[i] != given[i])
        for i in flip:
            hand[i] = other[hand[i]]
        after = sum(cost(k) for k in affected) + sum(each or price[i] for i in flip if hand[i] != given[i])
        for i in flip:
            hand[i] = other[hand[i]]
        return before - after

    todo = [k for k, c in enumerate(costs) if c > 0]
    budget = 20 * len(onsets)
    while todo and budget > 0:
        budget -= 1
        k = todo.pop()
        if costs[k] <= 0:
            continue
        sounding = sorted(set(onsets[k] + held[k]), key=lambda i: notes[i].pitch)
        candidates: dict[tuple[int, ...], float | None] = {(i,): None for i in sounding}
        for s in range(len(sounding) + 1):
            flip = tuple(i for j, i in enumerate(sounding) if hand[i] != ("L" if j < s else "R"))
            if flip:
                candidates[flip] = None
        a, b = k, k  # the run of costly onsets around this one, a beat or less apart, its hands swapped as a whole
        j = k - 1
        while j >= 0 and times[a] - times[j] <= tpb:
            a = j if costs[j] > 0 else a
            j -= 1
        j = k + 1
        while j < len(onsets) and times[j] - times[b] <= tpb:
            b = j if costs[j] > 0 else b
            j += 1
        if b > a:
            candidates[tuple(i for o in onsets[a:b + 1] for i in o)] = W_SWAPPED
        best, saving = None, 1e-6
        for flip, each in candidates.items():
            gain = try_move(flip, each)
            if gain > saving:
                best, saving = flip, gain
        if best is None:
            continue
        for i in best:
            hand[i] = other[hand[i]]
            price[i] = candidates[best] or price[i]
        for a in {k2 for i in best for k2 in counts_at[i]}:
            costs[a] = cost(a)
            if costs[a] > 0:
                todo.append(a)

    if moved is not None:
        moved += [n.start for n, h, g in zip(notes, hand, given) if h != g]
    return [n if h == g else replace(n, hand=h, finger=None) for n, h, g in zip(notes, hand, given)]


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
