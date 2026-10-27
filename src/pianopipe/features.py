"""Cheap musical features of a level, from its notes and the song's meta events (no pedal: CC64 isn't loaded).

level_features() feeds the grade model (grading.py); skills(), character() and tempo_bpm() fill the catalog's
optional fields of the same names (docs/formats.md describes each rule). All positions are in ticks; a "beat" is
the time signature's beat (a quarter in 4/4, a dotted quarter in 6/8), a "quarter" is ticks_per_beat.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from statistics import median

from .hands import group_chords
from .model import Note

HANDS = ("R", "L")
NAMES_MAJOR = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
NAMES_MINOR = ("Cm", "C#m", "Dm", "Ebm", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "Bbm", "Bm")
# Krumhansl-Kessler key profiles, for telling a major key from its relative minor.
KK_MAJOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
KK_MINOR = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)
MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
POSITION_SPAN = 9  # semitones one hand position covers (a sixth); beyond it the hand moves
FIVE_FINGER_SPAN = 7  # five white keys, C to G
AGILE_IOI_S = 0.16  # a hand's onsets this close count as running notes
AGILE_SHARE = 0.4  # share of all onsets that are running notes for "agile"
LYRICAL_MELODY_S = 0.4  # median time between melody onsets for "lyrical"
FEATURES = (
    "difficulty", "log_notes", "pitch_range", "outside_key", "rhythm_values", "syncopation", "independence",
    "polyphony", "span", "shifts",
)


def _window(tpb: int) -> int:
    return max(1, tpb // 32)


def _chords(notes: list[Note], tpb: int) -> dict[str, list[list[Note]]]:
    return {h: group_chords([n for n in notes if n.hand == h], _window(tpb)) for h in HANDS}


def beat_grid(meta, tpb: int):
    """[(tick, beat ticks)] from the time signatures: compound meters (6/8, 9/8, 12/8...) beat in dotted notes."""
    grid = []
    for tick, msg in meta:
        if msg.type != "time_signature":
            continue
        unit = tpb * 4 // msg.denominator
        beat = unit * 3 if msg.denominator >= 8 and msg.numerator % 3 == 0 and msg.numerator > 3 else unit
        if grid and grid[-1][0] == tick:
            grid[-1] = (tick, beat)
        elif not grid or grid[-1][1] != beat:
            grid.append((tick, beat))
    return grid or [(0, tpb)]


def _beat_pos(tick: int, grid) -> tuple[int, int]:
    """(ticks since the last beat, beat length) at `tick`."""
    start, beat = grid[0]
    for t, b in grid:
        if t > tick:
            break
        start, beat = t, b
    return (tick - start) % beat, beat


def beat_index(tick: int, grid) -> tuple[int, int]:
    """(time signature segment, beat number within it) of the beat `tick` falls in."""
    i = max(0, bisect_right([t for t, _ in grid], tick) - 1)
    start, beat = grid[i]
    return i, (tick - start) // beat


def _sounding_after(notes: list[Note]):
    """f(t, margin): whether a note that started before t (by more than margin) is still sounding after t + margin."""
    notes = sorted(notes, key=lambda n: n.start)
    starts = [n.start for n in notes]
    reach, best = [], -1
    for n in notes:
        best = max(best, n.end)
        reach.append(best)

    def f(t: int, before: int, after: int) -> bool:
        i = bisect_left(starts, t - before)
        return i > 0 and reach[i - 1] > t + after

    return f


def pitch_class_weights(notes: list[Note]) -> list[float]:
    w = [0.0] * 12
    for n in notes:
        w[n.pitch % 12] += n.duration
    return w


def _corr(a, b) -> float:
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else 0.0


def _mode_for(weights, major: str, minor: str) -> str:
    """The major key or its relative minor, whichever Krumhansl-Kessler profile fits the pitch-class weights better."""
    rot = lambda t: [weights[(t + i) % 12] for i in range(12)]  # noqa: E731
    return minor if _corr(rot(_pc(minor)), KK_MINOR) > _corr(rot(_pc(major)), KK_MAJOR) else major


def best_diatonic(notes: list[Note]) -> tuple[int, float]:
    """(major tonic of the diatonic set that holds the most notes, share of notes outside it)."""
    counts = [0] * 12
    for n in notes:
        counts[n.pitch % 12] += 1
    total = sum(counts) or 1
    inside = [sum(counts[(t + s) % 12] for s in MAJOR_SCALE) for t in range(12)]
    tonic = max(range(12), key=lambda t: (inside[t], -t))
    return tonic, 1 - inside[tonic] / total


# Each key signature's major key and its relative minor, spelled as mido spells them.
RELATIVE_MINOR = {
    "C": "Am", "G": "Em", "D": "Bm", "A": "F#m", "E": "C#m", "B": "G#m", "F#": "D#m", "C#": "A#m",
    "F": "Dm", "Bb": "Gm", "Eb": "Cm", "Ab": "Fm", "Db": "Bbm", "Gb": "Ebm", "Cb": "Abm",
}
RELATIVE_MAJOR = {v: k for k, v in RELATIVE_MINOR.items()}


def _pc(name: str) -> int:
    return ({"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[name[0]] + name.count("#") - name.count("b")) % 12


def keys(notes: list[Note], meta, end: int | None = None) -> list[str]:
    """Keys in order of appearance: each key signature's sharps/flats, with major or relative minor chosen from the
    notes it covers (files often write A minor as C). Without key signatures, the best-fitting diatonic key."""
    sigs = [(t, m.key) for t, m in meta if m.type == "key_signature" and (end is None or t < end)]
    if not sigs:
        if not notes:
            return []
        tonic = best_diatonic(notes)[0]
        return [_mode_for(pitch_class_weights(notes), NAMES_MAJOR[tonic], NAMES_MINOR[(tonic + 9) % 12])]
    out = []
    for i, (t, key) in enumerate(sigs):
        stop = sigs[i + 1][0] if i + 1 < len(sigs) else None
        part = [n for n in notes if n.start >= t and (stop is None or n.start < stop)]
        if not part:
            continue
        major = RELATIVE_MAJOR.get(key, key)
        if major not in RELATIVE_MINOR:
            continue  # a spelling mido doesn't produce
        name = _mode_for(pitch_class_weights(part), major, RELATIVE_MINOR[major])
        if name not in out:
            out.append(name)
    return out


def time_signatures(meta, end: int | None = None, tpb: int | None = None) -> list[str]:
    """Every time signature in order, without repeats. With tpb, a partial bar's signature is left out: one that lasts at
    most its own bar between two bars of another meter, the first bar's before that meter or the last bar's after it
    (a pickup and the bar that completes it, musicxml.partial_meter)."""
    sigs = sorted(((t, (m.numerator, m.denominator)) for t, m in meta if m.type == "time_signature"), key=lambda e: e[0])
    out = []
    for i, (t, m) in enumerate(sigs):
        if tpb:
            nt = sigs[i + 1][0] if i + 1 < len(sigs) else end
            after = sigs[i + 1][1] if i + 1 < len(sigs) else None
            before = sigs[i - 1][1] if i > 0 else None
            around = {x for x in (before, after) if x is not None}
            if nt is not None and nt - t <= tpb * 4 * m[0] / m[1] and len(around) == 1 and m not in around:
                continue
        if end is None or t < end:
            name = f"{m[0]}/{m[1]}"
            if name not in out:
                out.append(name)
    return out


def tempo_bpm(meta, end: int, tpb: int) -> int:
    """Median tempo over the piece weighted by time, in quarter notes per minute."""
    changes = [(0, 500000)]
    for t, m in meta:
        if m.type == "set_tempo" and t < end:
            if t == changes[-1][0]:
                changes[-1] = (t, m.tempo)
            else:
                changes.append((t, m.tempo))
    spans = []  # (bpm, seconds)
    for i, (t, tempo) in enumerate(changes):
        stop = changes[i + 1][0] if i + 1 < len(changes) else end
        if stop > t:
            spans.append((60e6 / tempo, (stop - t) * tempo / 1e6 / tpb))
    if not spans:
        return round(60e6 / changes[-1][1])
    spans.sort()
    half, acc = sum(s for _, s in spans) / 2, 0.0
    for bpm, s in spans:
        acc += s
        if acc >= half:
            return round(bpm)
    return round(spans[-1][0])


def outer_line(chords: list[list[Note]], hand: str, w: int) -> list[list[Note]]:
    """The hand's outer voice: chords whose top note (right hand) or bottom note (left hand) is not under (over) a
    note of the same hand that is still held, so an inner voice under a held melody note isn't counted."""
    out, held = [], []  # held: notes of earlier chords
    for c in chords:
        t = c[0].start
        held = [n for n in held if n.end > t + w]
        if hand == "R":
            ok = all(n.pitch <= c[-1].pitch for n in held)
        else:
            ok = all(n.pitch >= c[0].pitch for n in held)
        if ok:
            out.append(c)
        held += c
    return out


def _iois(chords: list[list[Note]]) -> list[int]:
    return [b[0].start - a[0].start for a, b in zip(chords, chords[1:])]


def _positions(chords: list[list[Note]], span: int) -> list[int]:
    """Greedy hand positions: how many chords each position lasts before a chord falls outside `span` semitones."""
    lengths, lo, hi, count = [], None, None, 0
    for c in chords:
        a, b = c[0].pitch, c[-1].pitch
        if lo is not None and max(hi, b) - min(lo, a) <= span:
            lo, hi, count = min(lo, a), max(hi, b), count + 1
        else:
            if count:
                lengths.append(count)
            lo, hi, count = a, b, 1
    if count:
        lengths.append(count)
    return lengths


def syncopation(chords: dict[str, list[list[Note]]], grid, tpb: int) -> float:
    """Share of outer-voice onsets that start off the beat and are held across the next beat (a held inner note
    in a broken chord, like the left hand of Bach's C major prelude, doesn't count)."""
    tol, w = max(1, tpb // 24), _window(tpb)
    sync = total = 0
    for h in HANDS:
        for c in outer_line(chords[h], h, w):
            total += 1
            pos, beat = _beat_pos(c[0].start, grid)
            if tol < pos < beat - tol and max(n.end for n in c) > c[0].start - pos + beat + tol:
                sync += 1
    return sync / max(1, total)


def _held_ratios(chords: list[list[Note]], tpb: int) -> list[float]:
    """For gaps of an eighth or more between a hand's onsets: how much of the gap the first chord holds."""
    out = []
    for c, d in zip(chords, chords[1:]):
        gap = d[0].start - c[0].start
        if gap >= tpb // 2:
            out.append((max(n.end for n in c) - c[0].start) / gap)
    return out


def level_features(notes: list[Note], seconds, tpb: int, meta=()) -> dict[str, float]:
    """Features of one level (see the list in docs/formats.md, `grade`). `seconds(tick)` uses the tempo map."""
    if not notes:
        return {k: 0.0 for k in FEATURES}
    grid = beat_grid(meta, tpb)
    chords = _chords(notes, tpb)
    onsets = sum(len(chords[h]) for h in HANDS)
    f: dict[str, float] = {}
    from .difficulty import difficulty_score  # here: difficulty.py builds its levels from this module

    f["difficulty"] = difficulty_score(notes, seconds, tpb)
    f["log_notes"] = math.log(len(notes))
    f["pitch_range"] = float(max(n.pitch for n in notes) - min(n.pitch for n in notes))
    f["outside_key"] = best_diatonic(notes)[1]

    # Distinct rhythm values: onset-to-onset times in each hand, on a 1/24-quarter grid, kept when 3% or more use them.
    values: dict[int, int] = {}
    for h in HANDS:
        for d in _iois(chords[h]):
            q = round(d * 24 / tpb)
            if 0 < q <= 96:
                values[q] = values.get(q, 0) + 1
    total = sum(values.values()) or 1
    f["rhythm_values"] = float(sum(1 for c in values.values() if c / total >= 0.03))

    f["syncopation"] = syncopation(chords, grid, tpb)

    # Hand independence: onsets where one hand starts while the other is still holding a note.
    sounding = {h: _sounding_after([n for n in notes if n.hand == h]) for h in HANDS}
    starts = {h: [c[0].start for c in chords[h]] for h in HANDS}
    times = sorted(set(starts["R"]) | set(starts["L"]))
    w = _window(tpb)
    indep = 0
    for t in times:
        hits = [h for h in HANDS if _near(starts[h], t, w)]
        if len(hits) == 1:
            other = "L" if hits[0] == "R" else "R"
            indep += sounding[other](t, w, w)
    f["independence"] = indep / max(1, len(times))

    # Polyphony within a hand: a chord starts while an earlier note of the same hand still sounds a sixteenth on.
    poly = 0
    for h in HANDS:
        for c in chords[h][1:]:
            poly += sounding[h](c[0].start, w, tpb // 4)
    f["polyphony"] = poly / max(1, onsets)

    # Hand span (90th percentile of chord spans, wider hand) and position changes per chord.
    spans = []
    for h in HANDS:
        s = sorted(c[-1].pitch - c[0].pitch for c in chords[h])
        if s:
            spans.append(s[min(len(s) - 1, int(0.9 * len(s)))])
    f["span"] = float(max(spans or [0]))
    moves = sum(max(0, len(_positions(chords[h], POSITION_SPAN)) - 1) for h in HANDS)
    f["shifts"] = moves / max(1, onsets)
    return f


def _near(sorted_ticks: list[int], t: int, w: int) -> bool:
    i = bisect_left(sorted_ticks, t - w)
    return i < len(sorted_ticks) and sorted_ticks[i] <= t + w


def character(notes: list[Note], seconds, tpb: int) -> str:
    """agile, lyrical or other, from the full level (rule in docs/formats.md, `character`)."""
    chords = _chords(notes, tpb)
    fast = total = 0
    for h in HANDS:
        for c, d in zip(chords[h], chords[h][1:]):
            total += 1
            fast += seconds(d[0].start) - seconds(c[0].start) <= AGILE_IOI_S
    if total and fast / total >= AGILE_SHARE:
        return "agile"
    melody = outer_line(chords["R"], "R", _window(tpb))
    gaps = [seconds(b[0].start) - seconds(a[0].start) for a, b in zip(melody, melody[1:])]
    held = _held_ratios(melody, tpb)
    if gaps and chords["L"] and median(gaps) >= LYRICAL_MELODY_S and held and median(held) >= 0.9:
        return "lyrical"
    return "other"


def skills(notes: list[Note], seconds, tpb: int, meta=()) -> list[str]:
    """Skill tags a level practises (rules in docs/formats.md, `skills`); pedal is left out (CC64 isn't loaded)."""
    if not notes:
        return []
    grid = beat_grid(meta, tpb)
    chords = _chords(notes, tpb)
    w = _window(tpb)
    tags = []
    by_hand = {h: [n for n in notes if n.hand == h] for h in HANDS}
    played = [h for h in HANDS if chords[h]]
    onsets = sum(len(chords[h]) for h in HANDS)

    # five-finger: every hand stays in each five-key position (C to G) for 24 or more onsets on average.
    if all(len(chords[h]) / len(_positions(chords[h], FIVE_FINGER_SPAN)) >= 24 for h in played):
        tags.append("five-finger")

    # hands-together: both hands play, and at a quarter or more of the onsets the other hand sounds too.
    if len(played) == 2:
        sounding = {h: _sounding_after(by_hand[h]) for h in HANDS}
        starts = {h: [c[0].start for c in chords[h]] for h in HANDS}
        together = 0
        for h in HANDS:
            other = "L" if h == "R" else "R"
            for t in starts[h]:
                together += _near(starts[other], t, w) or sounding[other](t, w, w)
        if together / onsets >= 0.25:
            tags.append("hands-together")

    # staccato / legato: how much of the gap to the hand's next onset a note holds (gaps of an eighth or more).
    held = [r for h in HANDS for r in _held_ratios(chords[h], tpb)]
    if held:
        if sum(r <= 0.6 for r in held) / len(held) >= 0.2:
            tags.append("staccato")
        if sum(r >= 0.9 for r in held) / len(held) >= 0.6:
            tags.append("legato")

    # dotted: a long-short pair in a 3:1 ratio lasting an eighth, a quarter, a half or a whole (dotted eighth and
    # sixteenth, dotted quarter and eighth...) that starts on a multiple of its own length, 6 or more times.
    dotted = triplets = 0
    for h in HANDS:
        cs = chords[h]
        for a, b, c in zip(cs, cs[1:], cs[2:]):
            long, short = b[0].start - a[0].start, c[0].start - b[0].start
            pair = long + short
            lengths = (tpb // 2, tpb, 2 * tpb, 4 * tpb)
            if short >= tpb // 8 and abs(long - 3 * short) <= w and any(abs(pair - k) <= w for k in lengths):
                pos = a[0].start % pair
                dotted += min(pos, pair - pos) <= w
        # triplets: onsets a third or two thirds into a quarter (not a compound meter's eighths).
        for c in cs:
            pos = c[0].start % tpb
            off16 = pos % (tpb // 4)  # away from the sixteenth grid
            if any(abs(pos - k * tpb // 3) <= w for k in (1, 2)) and w < off16 < tpb // 4 - w:
                triplets += 1
    if dotted >= 6:
        tags.append("dotted")
    if triplets >= max(6, 0.05 * onsets):
        tags.append("triplets")

    if syncopation(chords, grid, tpb) >= 0.05:
        tags.append("syncopation")

    # chords: three or more notes at once in one hand, at 10% or more of that hand's onsets.
    if any(chords[h] and sum(len(c) >= 3 for c in chords[h]) / len(chords[h]) >= 0.1 for h in played):
        tags.append("chords")

    # ornaments: trills (4+ alternating notes a step apart, an eighth of a quarter apart or less) and grace notes
    # (a note a 32nd or shorter before a note a step away held 4 times as long), 2 or more.
    orn = 0
    for h in HANDS:
        line = [c[-1] if h == "R" else c[0] for c in chords[h]]
        run = 0
        for a, b, c in zip(line, line[1:], line[2:]):
            if b.start - a.start <= tpb // 8 and abs(a.pitch - b.pitch) <= 2 and a.pitch == c.pitch:
                run += 1
                if run == 2:
                    orn += 1
            else:
                run = 0
        for a, b in zip(line, line[1:]):
            if 0 < b.start - a.start <= tpb // 8 and abs(a.pitch - b.pitch) <= 2 and b.duration >= 4 * (b.start - a.start) \
                    and seconds(b.start) - seconds(a.start) <= 0.12:
                orn += 1
    if orn >= 2:
        tags.append("ornaments")

    # octaves: a hand plays two notes exactly an octave apart, 8 or more times and at 5% or more of its onsets.
    for h in played:
        octs = sum(len(c) >= 2 and c[-1].pitch - c[0].pitch == 12 for c in chords[h])
        if octs >= 8 and octs / len(chords[h]) >= 0.05:
            tags.append("octaves")
            break

    end = max(n.end for n in notes)
    tags += [f"key:{k}" for k in keys(notes, meta, end)]
    tags += [f"time:{t}" for t in time_signatures(meta, end, tpb)]
    return tags
