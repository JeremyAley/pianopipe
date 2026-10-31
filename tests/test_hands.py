"""The playability pass (hands.playable_hands): what each hand holds as well as strikes."""

from pianopipe.hands import playable_hands, repair_hands
from pianopipe.model import Note
from pianopipe.process import load_notes

from helpers import TPB, make_midi


def notes_of(hand, spec):
    """(beat, beats, pitch, finger) or (beat, beats, pitch) -> Notes in `hand`."""
    return [Note(round(s[0] * TPB), round((s[0] + s[1]) * TPB), s[2], hand=hand, finger=s[3] if len(s) > 3 else None)
            for s in spec]


def hands_of(notes):
    return {(n.start / TPB, n.pitch): n.hand for n in notes}


def test_a_held_left_hand_chord_the_right_hand_plays_under_goes_right():
    # Golden (a MuseScore file): the bass staff holds C4-E4 for the bar over C3 on every beat, while the treble staff's
    # triplets climb G3-A3-B3 underneath. The right hand takes the held chord; the left keeps its C3s.
    lh = notes_of("L", [(0, 4, 60), (0, 4, 64)] + [(b, 1, 48) for b in range(4)])
    rh = notes_of("R", [(b + t / 3, 1 / 3, p) for b in (1, 3) for t, p in enumerate((55, 57, 59))])
    moved = []
    out = playable_hands(lh + rh, TPB, moved)
    h = hands_of(out)
    assert h[0, 60] == h[0, 64] == "R"
    assert all(h[b, 48] == "L" for b in range(4)) and all(n.hand == "R" for n in out if n.pitch in (55, 57, 59))
    assert moved == [0, 0]


def test_a_pedal_bass_under_the_hands_arpeggios_stays():
    # Liszt's Consolation 3: the left hand holds C#2 for bars under its own triplets up to F4, over two octaves away.
    # The bass is pedalled, so nothing moves.
    bass = notes_of("L", [(0, 8, 37)])
    arps = notes_of("L", [(b + t / 3, 1 / 3, p) for b in range(8) for t, p in enumerate((53, 61, 65))])
    tune = notes_of("R", [(b, 1, 77) for b in range(2, 8)])
    notes = bass + arps + tune
    assert playable_hands(notes, TPB) == notes


def test_a_rolled_chord_the_right_hand_plays_under_later_stays():
    # Colors of the Wind bar 10, as printed: the left hand rolls G3-D4-G4 for the bar, and three beats on the right
    # hand's tune comes down to D4 under its G4. The left hand has let go under the pedal by then.
    lh = notes_of("L", [(0, 4, 55), (0, 4, 62), (0, 4, 67)])
    rh = notes_of("R", [(0, 1, 76), (1, 2, 71), (3, 0.5, 62), (3.5, 0.5, 66)])
    notes = lh + rh
    assert playable_hands(notes, TPB) == notes


def test_a_passage_the_left_hand_plays_over_the_right_swaps_whole():
    # I See the Light (a scan): the left hand plays over the right's tune, with it on the beat and alone between.
    # All of it swaps, not some.
    tune = [72, 67, 72, 76, 74, 79, 77, 76]
    sparkle = [95, 93, 88, 89, 91, 86, 93, 87]
    rh = notes_of("R", [(i, 1, p) for i, p in enumerate(tune)])
    lh = notes_of("L", [(i + j / 2, 0.5, p + 2 * j) for i, p in enumerate(sparkle) for j in (0, 1) if i + j / 2 <= 7])
    out = playable_hands(rh + lh, TPB)
    assert [n.hand for n in out[:len(rh)]] == ["L"] * len(rh)
    assert [n.hand for n in out[len(rh):]] == ["R"] * len(lh)


def test_interlocked_chords_within_a_third_stay():
    # Rachmaninoff's C# minor prelude: the left thumb on Eb4 a minor third over the right thumb on C4, struck together.
    lh = notes_of("L", [(b, 0.5, p) for b in range(4) for p in (51, 54, 63)])
    rh = notes_of("R", [(b, 0.5, p) for b in range(4) for p in (60, 66, 72)])
    notes = lh + rh
    assert playable_hands(notes, TPB) == notes


def test_a_grace_note_a_hand_cant_hold_into_its_chord_goes_to_the_other_hand():
    # The right hand's low C4 a 32nd before its chord G5-B5-D6, held through it: over two octaves at once. The left
    # hand, holding only C3, takes the C4; the chord keeps its printed fingers.
    lh = notes_of("L", [(0, 2, 48)])
    rh = notes_of("R", [(0.875, 1.125, 60, 1)] + [(1, 1, p, f) for p, f in ((79, 1), (83, 3), (86, 5))])
    out = playable_hands(lh + rh, TPB)
    h = hands_of(out)
    assert h[0.875, 60] == "L" and all(h[1, p] == "R" for p in (79, 83, 86))
    assert next(n for n in out if n.pitch == 60).finger is None and next(n for n in out if n.pitch == 86).finger == 5


def test_a_rolled_chord_held_under_the_hands_run_stays():
    # Colors of the Wind bar 28, as printed: the right hand rolls A3-B3-D4-E4 for the bar and a beat later runs up
    # to G5, over a tenth from A3. The chord is pedalled, so nothing moves.
    lh = notes_of("L", [(0, 3, 47), (0, 3, 54)])
    rh = notes_of("R", [(0, 4, p) for p in (57, 59, 62, 64)] + [(1 + i / 4, 0.25, p) for i, p in enumerate((74, 76, 78, 79))])
    notes = lh + rh
    assert playable_hands(notes, TPB) == notes


def test_a_move_never_makes_a_chord_one_hand_cant_strike():
    # The left hand holds C5 over its repeated C4s, the right repeats F4 under it: moving C5 right would end the
    # crossing but make the right hand's first chord C5 + F#6, over a tenth struck at once.
    lh = notes_of("L", [(0, 4, 72)] + [(i / 4, 0.25, 60) for i in range(16)])
    rh = notes_of("R", [(0, 0.5, 90)] + [(0.5 + i / 4, 0.25, 65) for i in range(14)])
    out = repair_hands(lh + rh, TPB)
    assert hands_of(out)[0, 72] == "L"


def test_playable_hands_change_nothing():
    lh = notes_of("L", [(b, 1, p) for b in range(4) for p in (48, 55)])
    rh = notes_of("R", [(b, 1, p) for b in range(4) for p in (64, 67, 72)])
    notes = lh + rh
    moved = []
    assert playable_hands(notes, TPB, moved) == notes and moved == []


def test_track_hands_get_the_pass_without_counting_as_unplayable(tmp_path):
    # Hands from a MIDI file's tracks go through repair_hands, which ends with the pass; its moves aren't chords one
    # hand can't strike, so the gate's `repaired` stays empty.
    lh = [(0, 4 * TPB, 60), (0, 4 * TPB, 64)] + [(b * TPB, TPB, 48) for b in range(4)]
    rh = [(b * TPB + t * TPB // 3, TPB // 3, p) for b in (1, 3) for t, p in enumerate((55, 57, 59))]
    path = make_midi(tmp_path / "a.mid", [{"name": "Right", "notes": rh}, {"name": "Left", "notes": lh, "channel": 1}])
    repaired = []
    _, notes, hand_source, _ = load_notes(path, repaired=repaired)
    assert hand_source == "track_names" and repaired == []
    assert {n.pitch for n in notes if n.hand == "L"} == {48}
    direct = repair_hands(notes_of("L", [(0, 4, 60), (0, 4, 64)] + [(b, 1, 48) for b in range(4)]), TPB)
    assert {n.hand for n in direct if n.pitch in (60, 64)} == {"L"}  # the left hand alone holds them playably
