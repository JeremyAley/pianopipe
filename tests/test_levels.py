"""Levels 1 and 2 (difficulty.make_level): the melody is the right hand's outer line (or the left hand's where the tune
is there), broken-chord figuration one note a beat, the bass on the beat."""

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


S = TPB // 4  # a sixteenth


def _prelude(bars):
    """The Bach C major prelude's figure: two left-hand sixteenths held under G4 C5 E5 G4 C5 E5 in the right hand,
    twice a bar; `bars` gives each bar's (bass, tenor, right-hand pitches)."""
    notes = []
    for b, (bass, tenor, rh) in enumerate(bars):
        for half in range(2):
            t = b * 4 * TPB + half * 2 * TPB
            notes += [_l(t, 2 * TPB, bass), _l(t + S, 2 * TPB - S, tenor)]
            notes += [_r(t + (2 + k) * S, S, rh[k % 3]) for k in range(6)]
    return notes


def test_broken_chords_become_the_top_note_of_each_beat():
    """The Bach C major prelude's level 1 was all its right-hand sixteenths, half snapped to eighths. Now: the top note
    of each beat, on the beat, held through it."""
    notes = _prelude([(48, 52, (67, 72, 76)), (48, 50, (69, 74, 77))])
    out = make_level(notes, 1, TPB, _time_sig(4, 4))
    assert [(n.start, n.pitch) for n in out] == [(k * TPB, p) for k, p in enumerate([72, 76, 72, 76, 74, 77, 74, 77])]
    assert all(n.end - n.start == TPB for n in out) and all(n.hand == "R" for n in out)


def test_figure_in_the_bass_line_keeps_its_lowest_note():
    """Level 2's bass under a right-hand tune: an Alberti figure in sixteenths reduces to its bottom note."""
    melody = [_r(b * TPB, TPB, 72 + b) for b in range(4)]
    alberti = [_l(i * S, S, p) for i, p in enumerate((48, 55, 52, 55) * 4)]
    out = [n for n in make_level(melody + alberti, 2, TPB, _time_sig(4, 4)) if n.hand == "L"]
    assert [(n.start, n.pitch) for n in out] == [(b * TPB, 48) for b in range(4)]


def test_fast_tunes_are_not_figuration():
    """Für Elise's E D# E D# E B D C (steps) and its C E A B (leaps that don't come back) stay note for note."""
    tune = [76, 75, 76, 75, 76, 71, 74, 72, 69]
    notes = [_r(i * S, S, p) for i, p in enumerate(tune)]
    arpeggio = [_r(12 * S + i * S, S, p) for i, p in enumerate((60, 64, 69, 71))]
    out = make_level(notes + arpeggio, 1, TPB, _time_sig(3, 8))
    assert [n.pitch for n in out] == tune + [60, 64, 69, 71]


def _farmer(bars=4):
    """Schumann's Happy Farmer: the tune in the left hand (C3 F3 A3 C4 ...), the right hand off-beat chords, each
    twice."""
    tune = [(0, 48), (1, 53), (2, 57), (3, 60), (4, 53), (5, 58), (6, 62), (7, 65), (8, 62), (9, 60), (10, 57),
            (11, 58), (12, 55), (13, 48), (14, 58), (15, 57)]
    lh = [_l(k * TPB, TPB, p) for k, p in tune if k < 4 * bars]
    chords = [((65, 69, 72), (65, 70, 74)), ((64, 67, 72), (65, 69, 72))]
    rh = [_r(b * 4 * TPB + beat * TPB + TPB // 2, TPB // 2, p)
          for b in range(bars) for beat in range(4) for p in chords[b % 2][beat // 2]]
    return lh + rh


def test_tune_in_the_left_hand():
    """Where the right hand only repeats chords over a left hand that moves in single notes, level 1 is the left hand's
    line (it was the chords' top notes, F5 F5 D5 D5 ...)."""
    out = make_level(_farmer(), 1, TPB, _time_sig(4, 4))
    assert all(n.hand == "L" for n in out)
    assert [n.pitch for n in out][:8] == [48, 53, 57, 60, 53, 58, 62, 65]


def test_level_2_over_a_left_hand_tune_adds_the_right_hand():
    """Level 2 is level 1's left-hand tune, whole, with the right hand's top line one note a beat."""
    out = make_level(_farmer(), 2, TPB, _time_sig(4, 4))
    left, right = [n for n in out if n.hand == "L"], [n for n in out if n.hand == "R"]
    tune = make_level(_farmer(), 1, TPB, _time_sig(4, 4))
    assert [(n.start, n.pitch, n.hand) for n in left] == [(n.start, n.pitch, n.hand) for n in tune]
    assert right and {n.pitch for n in right} <= {72, 74}  # the chords' tops
    assert len({n.start // TPB for n in right}) == len(right)  # one a beat at most


def test_left_hand_answers_while_the_right_rests():
    """Two bars of the right hand silent while the left plays a tune: level 1 keeps playing it."""
    rh = [_r(b * TPB, TPB, 72 + b % 3) for b in range(4)]
    answer = [_l(4 * TPB + k * TPB // 2, TPB // 2, p) for k, p in enumerate((55, 57, 59, 60, 62, 60, 59, 57, 55, 53, 52, 50, 48, 50, 52, 53))]
    out = make_level(rh + answer, 1, TPB, _time_sig(4, 4))
    assert [n.hand for n in out] == ["R"] * 4 + ["L"] * 16


def test_one_bar_alone_stays_in_the_right_hand():
    """A single bar of left-hand answer doesn't move level 1 to the other hand and back."""
    rh = [_r(b * TPB, TPB, 72 + b % 3) for b in range(4)] + [_r(8 * TPB + b * TPB, TPB, 72) for b in range(4)]
    answer = [_l(4 * TPB + k * TPB // 2, TPB // 2, p) for k, p in enumerate((55, 57, 59, 60, 62, 60, 59, 57))]
    out = make_level(rh + answer, 1, TPB, _time_sig(4, 4))
    assert all(n.hand == "R" for n in out)


def test_hymn_soprano_stays_the_tune():
    """A soprano repeating its note (Holy, holy, holy) over a moving tenor: the left hand plays two parts, so the tune
    stays in the right hand."""
    sop = [74, 74, 77, 77, 81, 81, 81, 81]
    alto = [69, 69, 72, 72, 74, 74, 72, 72]
    tenor = [62, 64, 65, 67, 69, 67, 65, 64]
    bass = [50, 48, 46, 45, 43, 45, 46, 48]
    notes = [_r(k * TPB, TPB, p) for k, p in enumerate(sop)] + [_r(k * TPB, TPB, p) for k, p in enumerate(alto)]
    notes += [_l(k * TPB, TPB, p) for k, p in enumerate(tenor)] + [_l(k * TPB, TPB, p) for k, p in enumerate(bass)]
    assert all(n.hand == "R" for n in make_level(notes, 1, TPB, _time_sig(4, 4)))


def test_held_melody_over_left_hand_arpeggios_stays_the_tune():
    """A nocturne: the right hand's melody holds its notes over left-hand triplet arpeggios that move."""
    rh = []
    for b, top in enumerate((71, 71, 73, 74)):
        rh += [_r(b * 4 * TPB + k * TPB, TPB, p) for k in range(4) for p in (62, top if k < 3 else top + 2)]
    arps = (43, 50, 59, 55, 64, 62, 42, 54, 61, 64, 61, 54)
    lh = [_l(b * 4 * TPB + k * TPB // 3, TPB // 3, p + b) for b in range(4) for k, p in enumerate(arps)]
    assert all(n.hand == "R" for n in make_level(rh + lh, 1, TPB, _time_sig(4, 4)))


def test_tune_straight_after_a_figure_stays():
    """Figuration is judged a beat at a time: a tune that follows a broken chord in the same run of sixteenths keeps
    its notes (it collapsed to its top note when the whole run was judged at once)."""
    notes = [_r(k * S, S, p) for k, p in enumerate([67, 72, 76] * 4 + [60, 64, 69, 71])]
    out = make_level(notes, 1, TPB, _time_sig(4, 4))
    assert [n.pitch for n in out if n.start < 3 * TPB] == [76, 76, 76]
    assert [n.pitch for n in out if n.start >= 3 * TPB] == [60, 64, 69, 71]
    # The tune starting halfway through a beat: the figure's half becomes one note, the tune's notes stay.
    notes = [_r(k * S, S, p) for k, p in enumerate([67, 72, 76] * 4 + [67, 72, 60, 64, 69, 71])]
    out = make_level(notes, 1, TPB, _time_sig(4, 4))
    assert [n.pitch for n in out if n.start >= 3 * TPB] == [72, 60, 64, 69, 71]


def test_figure_that_changes_chord_still_counts():
    """An Alberti figure in the right hand moving from C to D minor: one note a beat."""
    notes = [_r(k * S, S, p) for k, p in enumerate([60, 67, 64, 67, 62, 69, 65, 69])]
    assert [(n.start, n.pitch) for n in make_level(notes, 1, TPB, _time_sig(4, 4))] == [(0, 67), (TPB, 69)]


def test_tune_under_a_repeated_note_stays():
    """Für Elise's 32nds: C5 G5 G4 G5 A4 G5 B4 G5 C5 G5 D5 G5, a rising line under a repeated G5. Not figuration:
    the notes between the G5s move by step."""
    pitches = [72, 79, 67, 79, 69, 79, 71, 79, 72, 79, 74, 79]
    notes = [_r(k * TPB // 8, TPB // 8, p) for k, p in enumerate(pitches)]
    out = make_level(notes, 1, TPB, _time_sig(3, 8))
    assert [n.pitch for n in out] == pitches


def test_level_two_strikes_a_key_once():
    """A tune that comes down onto the bass's note (Für Elise, a Bach invention): level 2 sounds the key once, the
    longer note, rather than both hands striking it at once."""
    tune = [_r(0, TPB, 64), _r(TPB, 2 * TPB, 57), _r(3 * TPB, TPB, 60)]
    bass = [_l(0, TPB, 45), _l(TPB, TPB, 57), _l(2 * TPB, 2 * TPB, 52)]
    out = make_level(tune + bass, 2, TPB)
    assert [(n.start, n.end, n.hand) for n in out if n.pitch == 57] == [(TPB, 3 * TPB, "R")]
