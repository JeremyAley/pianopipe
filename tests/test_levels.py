"""Levels 1 and 2 (difficulty.make_level): the melody is the right hand's outer line, the bass sits on the beat."""

import mido

from pianopipe.difficulty import make_level
from pianopipe.model import Note

TPB = 480


def _r(start, dur, pitch):
    return Note(start=start, end=start + dur, pitch=pitch, hand="R")


def _l(start, dur, pitch):
    return Note(start=start, end=start + dur, pitch=pitch, hand="L")


def _time_sig(num, den):
    return [(0, mido.MetaMessage("time_signature", numerator=num, denominator=den))]


def test_held_melody_over_moving_inner_notes():
    """Moonlight-style: a melody over triplets in the same hand. The melody notes are all there; a note held past its
    first beat gives way to the triplets for the rest of its length, so level 1 never stands still on one long note
    while the piece moves (Jeremy's listen, 2026-10-26)."""
    melody = [_r(0, 3 * TPB, 68), _r(3 * TPB, TPB, 69), _r(4 * TPB, 4 * TPB, 68)]  # G#4 (dotted half), A4, G#4
    triplets = [_r(b * TPB + k * TPB // 3, TPB // 3, p) for b in range(8) for k, p in enumerate((56, 61, 64))]
    out = make_level(melody + triplets, 1, TPB)
    starts = {(n.start, n.pitch) for n in out}
    assert {(0, 68), (3 * TPB, 69), (4 * TPB, 68)} <= starts  # every melody note
    assert all(n.pitch == 68 for n in out if n.start < TPB)  # its first beat is the melody alone
    assert {n.pitch for n in out if TPB <= n.start < 3 * TPB} == {56, 61, 64}  # then the triplets until A4
    assert all(n.pitch == 69 for n in out if 3 * TPB <= n.start < 4 * TPB)  # a one-beat note isn't filled
    held = next(n for n in out if n.start == 0)
    assert held.end <= TPB + TPB // 2  # the held G#4 stops where the triplets take over


def test_long_trill_stays_one_held_note():
    """A two-beat trill becomes one held note, and isn't then mistaken for a held melody note and filled back in."""
    trill = [_r(k * TPB // 8, TPB // 8, 72 if k % 2 == 0 else 74) for k in range(16)]
    out = make_level(trill, 1, TPB)
    assert [n.pitch for n in out] == [72] and out[0].end >= 2 * TPB - TPB // 8


def test_short_held_melody_note_isnt_filled():
    """A melody note held one beat over moving inner notes stays the melody (Für Elise, Humoresque)."""
    melody = [_r(0, TPB, 76), _r(TPB, TPB, 75), _r(2 * TPB, TPB, 76)]
    inner = [_r(k * TPB // 2, TPB // 2, 60) for k in range(6)]
    out = make_level(melody + inner, 1, TPB)
    assert [n.pitch for n in out] == [76, 75, 76]


def test_inner_notes_count_once_the_melody_is_released():
    """With no melody note held, the right hand's top line is all there is, and level 1 keeps it."""
    triplets = [_r(b * TPB + k * TPB // 3, TPB // 3, p) for b in range(2) for k, p in enumerate((56, 61, 64))]
    out = make_level(triplets, 1, TPB)
    assert [n.pitch for n in out] == [56, 61, 64] * 2


def test_top_note_of_each_chord():
    chords = [_r(b * TPB, TPB, p) for b in range(4) for p in (60, 64, 67 + b)]
    assert [n.pitch for n in make_level(chords, 1, TPB)] == [67, 68, 69, 70]


def test_grace_note_gives_way_to_its_main_note():
    grace, main = _r(TPB - TPB // 16, TPB // 16, 74), _r(TPB, TPB, 72)
    out = make_level([_r(0, TPB, 71), grace, main], 1, TPB)
    assert [(n.start, n.pitch) for n in out] == [(0, 71), (TPB, 72)]


def test_dotted_rhythm_kept():
    """Dvorak's Humoresque: dotted 16th + 32nd pairs. The short note is part of the tune, not an ornament."""
    pitches = [67, 69, 67, 69, 71, 74, 76, 76]
    notes = [_r(i // 2 * TPB // 2 + (i % 2) * 3 * TPB // 8, 3 * TPB // 8 if i % 2 == 0 else TPB // 8, p)
             for i, p in enumerate(pitches)]
    assert [n.pitch for n in make_level(notes, 1, TPB)] == pitches


def test_trill_becomes_one_held_note():
    trill = [_r(TPB + i * TPB // 8, TPB // 8, 74 if i % 2 == 0 else 76) for i in range(8)]
    out = make_level([_r(0, TPB, 72)] + trill + [_r(2 * TPB, TPB, 72)], 1, TPB)
    assert [(n.start, n.pitch) for n in out] == [(0, 72), (TPB, 74), (2 * TPB, 72)]
    assert out[1].end == 2 * TPB


def test_bass_sits_on_the_beat_grid():
    """An Alberti bass (C G E G in eighths): one bass note per beat, on the beat, never counted from the last one."""
    melody = [_r(b * TPB, TPB, 72) for b in range(4)]
    bass = [_l(i * TPB // 2, TPB // 2, p) for i, p in enumerate((48, 55, 52, 55) * 2)]
    late = [_l(4 * TPB + TPB // 4, TPB // 2, 43)]  # a bass note a sixteenth after the beat still starts beat 5
    out = [n for n in make_level(melody + bass + late, 2, TPB, _time_sig(4, 4)) if n.hand == "L"]
    assert [(n.start, n.pitch) for n in out] == [(0, 48), (TPB, 52), (2 * TPB, 48), (3 * TPB, 52), (4 * TPB, 43)]


def test_bass_in_compound_and_fast_meters():
    bass = [_l(i * TPB // 2, TPB // 2, 48 + i % 3) for i in range(12)]  # eighths
    melody = [_r(0, 6 * TPB, 72)]
    in_6_8 = [n.start for n in make_level(melody + bass, 2, TPB, _time_sig(6, 8)) if n.hand == "L"]
    assert in_6_8 == [0, 3 * TPB // 2, 3 * TPB, 9 * TPB // 2]  # dotted quarters
    in_3_8 = [n.start for n in make_level(melody + bass, 2, TPB, _time_sig(3, 8)) if n.hand == "L"]
    assert in_3_8 == in_6_8  # one per bar, not one per eighth
    in_4_8 = [n.start for n in make_level(melody + bass, 2, TPB, _time_sig(4, 8)) if n.hand == "L"]
    assert in_4_8 == [k * TPB for k in range(6)]  # quarters


def test_bass_skips_inner_notes_over_a_held_bass_note():
    melody = [_r(b * TPB, TPB, 72) for b in range(4)]
    lh = [_l(0, 4 * TPB, 36)] + [_l(b * TPB, TPB, 55) for b in range(1, 4)]  # a held C2 under chords
    out = [n for n in make_level(melody + lh, 2, TPB, _time_sig(4, 4)) if n.hand == "L"]
    assert [n.pitch for n in out] == [36]
