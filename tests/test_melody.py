"""A piano/vocal score's melody merged into the right hand (musicxml.load(melody=...), melody.py; catalog-expansion §2.6)."""

from __future__ import annotations

import mido

from pianopipe import musicxml
from pianopipe.difficulty import make_levels
from pianopipe.hands import playability_flags
from pianopipe.melody import merge, top_line
from pianopipe.model import Note
from pianopipe.process import process_file

from test_musicxml import TPB, backup, bar, note, rest, score, write

Q = TPB  # a quarter note in ticks
C3, C4, E4, G4, A4, B4, C5, D5, E5, G5, C6 = 48, 60, 64, 67, 69, 71, 72, 74, 76, 79, 84


def song(tmp_path, voices: list[list[str]], piano: list[str], names=None, melody=True, **kw):
    """A score with voice parts (one staff each) over a piano; returns load(melody=...)."""
    names = names or [f"Voice {i + 1}" if len(voices) > 1 else "Voice" for i in range(len(voices))]
    parts = [(n, ms, 1) for n, ms in zip(names, voices)] + [("Piano", piano)]
    return musicxml.load(write(tmp_path, score(parts)), melody=melody, **kw)


def right(sc) -> list[tuple[float, int, float]]:
    """(beat from the start, pitch, length in beats) of the right hand's notes."""
    return [(n.start / Q, n.pitch, (n.end - n.start) / Q) for n in sc.notes if n.hand == "R"]


def left(sc) -> list[int]:
    return [n.pitch for n in sc.notes if n.hand == "L"]


LH = note("C", 3, 16, 5, 2)


def test_off_by_default_the_voice_is_left_out(tmp_path):
    sc = song(tmp_path, [[note("E", 5, 16)]], [bar(note("C", 4, 16), LH)], melody=False)
    assert right(sc) == [(0, C4, 4)]
    assert sc.melody == {}
    assert any("left out Voice" in w for w in sc.warnings)


def test_melody_on_top_drops_notes_above_unisons_and_octaves(tmp_path):
    # The voice holds E5; the right hand strikes G5 (above), E4 (its octave), C5 with it, then E5 again (its key) and
    # a G4 under it.
    rh = (note("G", 5, 4) + backup(4) + note("E", 4, 4, 2) + backup(4) + note("C", 5, 4, 3)
          + note("E", 5, 4) + note("G", 4, 8))
    sc = song(tmp_path, [[note("E", 5, 16)]], [bar(rh + backup(16), LH)])
    assert right(sc) == [(0, C5, 1), (0, E5, 4), (2, G4, 2)]
    assert left(sc) == [C3]
    assert sc.melody["parts"] == ["Voice"]
    assert sc.melody["bars"]["Voice"] == {"merged": [1], "skipped": []}
    assert sc.melody["dropped"] == 3
    assert any(w.startswith("melody: merged Voice into the right hand in 1 of the 1 bars") for w in sc.warnings)
    assert any("with the melody of Voice" in w for w in sc.warnings)
    melody = next(n for n in sc.notes if n.pitch == E5)
    assert melody.velocity > next(n for n in sc.notes if n.pitch == C5).velocity
    # Level 1 is the voice: the E5 held over the moving right hand stays one held note.
    assert [(n.start, n.end, n.pitch) for n in lv_notes(sc, 1)] == [(0, 4 * Q, E5)]


def test_a_rest_in_the_melody_keeps_the_piano_and_a_held_note_is_cut_under_it(tmp_path):
    # The voice rests for two beats, then sings G4: the piano's A4, held from the start, stops there.
    sc = song(tmp_path, [[rest(8) + note("G", 4, 8)]], [bar(note("A", 4, 16), LH)])
    assert right(sc) == [(0, A4, 2), (2, G4, 2)]
    assert sc.melody["cut"] == 1


def test_a_low_right_hand_note_moves_to_the_left_hand_where_it_fits(tmp_path):
    # Under a C6, the right hand's D4 is a 14th down: the left hand (on C3) takes it. The second bar's left hand is
    # on C2, too far below, so the D4 there is left out.
    voice = [note("C", 6, 16), note("C", 6, 16)]
    piano = [bar(note("D", 4, 16), note("C", 3, 16, 5, 2)), bar(note("D", 4, 16), note("C", 2, 16, 5, 2))]
    sc = song(tmp_path, [voice], piano)
    assert right(sc) == [(0, C6, 4), (4, C6, 4)]
    assert sorted(left(sc)) == [36, C3, 62]
    assert (sc.melody["moved"], sc.melody["dropped"]) == (1, 1)


def test_cross_staff_notes_are_merged_with_the_hand_that_plays_them(tmp_path):
    # Found/Tonight bar 33: a bass-staff voice's Bb3 + Eb4 printed on the treble staff go to the right hand (the left
    # is on C2 + C3). Under the voice's F#5 the Bb3 is past a tenth and the left hand can't take it either: left out.
    lh = note("C", 2, 4, 5, 2) + note("C", 3, 4, 5, 2, "<chord/>") + note("G", 2, 4, 5, 2) + note("A", 2, 4, 5, 2) + note("B", 2, 4, 5, 2)
    xs = note("B", 3, 4, 5, 1, alter=-1) + note("E", 4, 4, 5, 1, "<chord/>", alter=-1)
    piano = [rest(16) + backup() + lh + backup() + xs]
    plain = song(tmp_path, [[note("F", 5, 4, alter=1) + rest(12)]], piano, melody=False)
    assert [p for _, p, _ in right(plain)] == [58, 63]  # the repair gave them to the right hand
    sc = song(tmp_path, [[note("F", 5, 4, alter=1) + rest(12)]], piano)
    assert [p for _, p, _ in right(sc)] == [63, 78]
    assert sorted(left(sc)) == [36, 43, 45, 47, C3]
    assert not playability_flags(sc.notes, TPB)


def test_the_left_hand_never_reaches_the_melody(tmp_path):
    # A low tune (G4) over a left hand that climbs to A4 and holds a G4 into the tune's next note: the A4 is left
    # out, the held G4 is released where the tune strikes F4 under it.
    lh = note("C", 3, 4, 5, 2) + note("A", 4, 4, 5, 2) + note("G", 4, 8, 5, 2)
    sc = song(tmp_path, [[note("G", 4, 12) + note("F", 4, 4)]], [bar(rest(16), lh)])
    assert [(b, p, d) for b, p, d in right(sc)] == [(0, G4, 3), (3, 65, 1)]
    assert sorted((n.start / Q, n.pitch, (n.end - n.start) / Q) for n in sc.notes if n.hand == "L") == [(0, C3, 1)]


def test_no_right_hand_chord_of_more_than_five_notes(tmp_path):
    # Five notes D4-A4 under a new B4: the lowest goes to the left hand.
    chord = note("D", 4, 16) + "".join(note(s, 4, 16, 1, 1, "<chord/>") for s in "EFGA")
    sc = song(tmp_path, [[note("B", 4, 16)]], [bar(chord, LH)])
    assert [p for _, p, _ in right(sc)] == [E4, 65, G4, A4, B4]
    assert sorted(left(sc)) == [C3, 62]


def test_a_bar_whose_voice_doesnt_add_up_keeps_the_piano(tmp_path):
    # Bar 2's voice has three beats in 4/4 (OMR lost a note or a dot): it isn't merged, the piano's E4 stays and the
    # warning names the bar.
    voice = [note("C", 5, 16), note("D", 5, 8) + note("E", 5, 4), note("C", 5, 16)]
    piano = [bar(note("E", 4, 16), LH), bar(note("E", 4, 16), LH), bar(note("E", 4, 16), LH)]
    sc = song(tmp_path, [voice], piano)
    assert right(sc) == [(0, E4, 4), (0, C5, 4), (4, E4, 4), (8, E4, 4), (8, C5, 4)]
    assert sc.melody["bars"]["Voice"] == {"merged": [1, 3], "skipped": [2]}
    assert any("in 2 of the 3 bars" in w and "skipped bars 2," in w for w in sc.warnings)


def test_a_tie_across_a_skipped_bar_is_broken(tmp_path):
    # A C5 tied through bar 2, which doesn't add up: bar 2 is the piano's (its E5 stays), bar 3's C5 is struck again.
    start, both, stop = '<tie type="start"/>', ('<tie type="stop"/>', '<tie type="start"/>'), '<tie type="stop"/>'
    voice = [note("C", 5, 16, 1, 1, start), note("C", 5, 16, 1, 1, *both) + note("D", 5, 4), note("C", 5, 16, 1, 1, stop)]
    piano = [bar(rest(16), LH), bar(note("E", 5, 16), LH), bar(rest(16), LH)]
    sc = song(tmp_path, [voice], piano)
    assert right(sc) == [(0, C5, 4), (4, E5, 4), (8, C5, 4)]


def test_a_note_moved_to_the_left_hand_doesnt_overlap_its_key(tmp_path):
    # Under a C6 the right hand's D4 (four beats) goes to the left hand, which strikes D4 itself on beat 2: one key,
    # so the moved note stops there.
    lh = note("C", 3, 4, 5, 2) + note("D", 4, 4, 5, 2) + note("C", 3, 8, 5, 2)
    sc = song(tmp_path, [[note("C", 6, 16)]], [bar(note("D", 4, 16), lh)])
    d4 = sorted((n.start / Q, n.end / Q) for n in sc.notes if n.pitch == 62)
    assert d4 == [(0, 1), (1, 2)]


def test_a_chord_under_a_held_melody_note_keeps_five_notes(tmp_path):
    # The voice holds G5; on beat 2 the right hand strikes A4 C5 D5 E5 F5: six keys at once, so the A4 goes (the
    # left hand on C3 can't reach it).
    chord = note("A", 4, 4) + "".join(note(s, 5, 4, 1, 1, "<chord/>") for s in "CDEF")
    sc = song(tmp_path, [[note("G", 5, 16)]], [bar(rest(4) + chord + rest(8), LH)])
    assert [p for b, p, _ in right(sc) if b == 1] == [C5, D5, E5, 77]
    assert sc.melody["dropped"] == 1


def test_a_melody_note_entering_over_five_held_notes_releases_the_lowest(tmp_path):
    chord = note("D", 4, 16) + "".join(note(s, 4, 16, 1, 1, "<chord/>") for s in "EFGA")
    sc = song(tmp_path, [[rest(4) + note("B", 4, 12)]], [bar(chord, LH)])
    d4 = next(n for n in sc.notes if n.pitch == 62)
    assert (d4.start, d4.end) == (0, Q)
    assert len([n for n in sc.notes if n.hand == "R" and n.start <= Q < n.end]) == 5


def test_held_and_struck_notes_count_together_under_the_melody(tmp_path):
    # E4 F4 G4 A4 held from beat 1; D4 struck on beat 2 with the voice's B4: six keys, so the D4 goes (the left
    # hand on C3 could take it: a ninth).
    held = note("E", 4, 16) + "".join(note(s, 4, 16, 1, 1, "<chord/>") for s in "FGA")
    struck = rest(4) + note("D", 4, 12)
    piano = [held + backup() + struck.replace("<voice>1</voice>", "<voice>2</voice>") + backup() + LH]
    sc = song(tmp_path, [[rest(4) + note("B", 4, 12)]], piano)
    assert len([n for n in sc.notes if n.hand == "R" and n.start <= Q < n.end]) == 5
    assert 62 in left(sc)


def test_a_voice_bar_read_too_long_doesnt_stretch_the_piano(tmp_path):
    # OMR read bar 1's voice as five beats: the bar lines come from the piano, with or without the merge.
    voice = [note("C", 5, 16) + note("D", 5, 4), note("C", 5, 16)]
    piano = [bar(note("E", 4, 16), LH), bar(note("G", 4, 16), LH)]
    for melody in (False, True):
        sc = song(tmp_path, [voice], piano, melody=melody)
        assert sc.bars == [0, 4 * Q, 8 * Q]
    assert right(sc) == [(0, E4, 4), (4, G4, 4), (4, C5, 4)]
    assert sc.melody["bars"]["Voice"]["skipped"] == [1]


def test_a_duet_takes_the_highest_voice_and_whoever_sings(tmp_path):
    # Bar 1: voice 1 then voice 2 (turns). Bar 2: voice 1's G5 over voice 2's E5s (harmony). Bar 3: voice 1 holds G4
    # and voice 2 comes in above it on beat 2.
    v1 = [note("C", 5, 8) + rest(8), note("G", 5, 16), note("G", 4, 8) + rest(8)]
    v2 = [rest(8) + note("E", 5, 8), note("E", 5, 4) * 4, rest(4) + note("C", 5, 4) + rest(8)]
    piano = [bar(rest(16), LH)] * 3
    sc = song(tmp_path, [v1, v2], piano)
    assert right(sc) == [(0, C5, 2), (2, E5, 2), (4, G5, 4), (8, G4, 1), (9, C5, 1)]
    assert sc.melody["parts"] == ["Voice 1", "Voice 2"]


def test_a_tied_melody_note_and_a_grace_note(tmp_path):
    # The voice ties a C5 over the bar line, where the piano strikes C5 again (dropped: the key is held), then sings
    # an acciaccatura D5 before an E5.
    voice = [note("C", 5, 16, 1, 1, '<tie type="start"/>'),
             note("C", 5, 8, 1, 1, '<tie type="stop"/>')
             + '<note><grace slash="yes"/><pitch><step>D</step><octave>5</octave></pitch><voice>1</voice></note>'
             + note("E", 5, 8)]
    piano = [bar(note("G", 4, 16), LH), bar(note("C", 5, 8) + note("G", 4, 8), LH)]
    sc = song(tmp_path, [voice], piano)
    # The grace note takes a 32nd note before the beat (the main note stays on it), cutting the C5 there.
    assert right(sc) == [(0, G4, 4), (0, C5, 5.875), (5.875, D5, 0.125), (6, G4, 2), (6, E5, 2)]


def test_a_tenor_in_an_octave_clef_sounds_where_its_printed(tmp_path):
    clef = "<attributes><clef><sign>G</sign><line>2</line><clef-octave-change>-1</clef-octave-change></clef></attributes>"
    sc = song(tmp_path, [[clef + note("E", 4, 16)]], [bar(note("C", 4, 16), LH)], names=["Tenor"])
    assert right(sc) == [(0, C4, 4), (0, E5, 4)]
    assert any("raised 1 octave" in w for w in sc.warnings)


def test_naming_the_melody_part(tmp_path):
    voices = [[note("C", 5, 16)], [note("E", 5, 16)]]
    piano = [bar(note("G", 4, 16), LH)]
    assert [p for _, p, _ in right(song(tmp_path, voices, piano, melody="Voice 1"))] == [G4, C5]
    assert [p for _, p, _ in right(song(tmp_path, voices, piano, melody=["P1"]))] == [G4, E5]
    sc = song(tmp_path, voices, piano, melody="Flute")
    assert right(sc) == [(0, G4, 4)]
    assert "melody: no voice or melody part to merge" in sc.warnings


def test_a_guitar_part_isnt_a_melody(tmp_path):
    sc = song(tmp_path, [[note("C", 5, 16)], [note("E", 5, 16)]], [bar(note("C", 4, 16), LH)], names=["Guitar", "Vocal"])
    assert sc.melody["parts"] == ["Vocal"]


def test_top_line_cuts_a_held_note_where_a_higher_one_starts():
    notes = [Note(0, 4 * Q, C5), Note(0, Q, E4), Note(Q, 2 * Q, E5), Note(Q + 10, 3 * Q, D5), Note(3 * Q, 4 * Q, G4)]
    assert [(n.start, n.end, n.pitch) for n in top_line(notes, TPB)] == [(0, Q, C5), (Q, 2 * Q, E5)]


def test_merge_without_a_melody_changes_nothing():
    notes = [Note(0, Q, C5, hand="R"), Note(0, Q, C3, hand="L")]
    out, stats = merge(notes, [], TPB)
    assert out == notes and stats.dropped == stats.moved == stats.cut == 0


def test_process_file_merges_the_melody_and_level_1_is_the_tune(tmp_path):
    voice = [note("E", 5, 4) + note("D", 5, 4) + note("C", 5, 8)] * 4
    rh = note("G", 4, 8) + note("E", 4, 8)
    lh = note("C", 3, 16, 5, 2)
    path = write(tmp_path, score([("Voice", voice, 1), ("Piano", [bar(rh, lh)] * 4)]))
    entry = process_file(path, tmp_path / "out", {"id": "s", "melody": True}, grade_model=None)
    assert entry.warnings[0].startswith("melody: merged Voice into the right hand in 4 of the 4 bars")
    level1 = mido.MidiFile(tmp_path / "out" / "s" / "level1.mid")
    played = [m.note for t in level1.tracks for m in t if m.type == "note_on" and m.velocity > 0 and m.note > 60]
    assert played == [E5, D5, C5] * 4
    plain = process_file(path, tmp_path / "plain", {"id": "s"}, grade_model=None)
    assert not any(w.startswith("melody: merged") for w in plain.warnings)
    assert "melody: levels 1 and 2 take the tune from Voice in 4 of 4 bars, the piano's top line in the rest" in plain.warnings


# Levels 1 and 2 from the voice part, merged or not.


def levels(sc):
    lv = make_levels(sc.notes, TPB, sc.song.meta, (sc.tune, sc.tune_bars))
    return {k: [(n.start / Q, n.pitch, n.hand) for n in sorted(v, key=lambda n: (n.start, n.pitch))] for k, v in lv.items()}


TUNE = note("E", 5, 4) + note("D", 5, 4) + note("C", 5, 8)
CHORDS = note("E", 4, 8) + note("G", 4, 8, 1, 1, "<chord/>") + note("F", 4, 8) + note("A", 4, 8, 1, 1, "<chord/>")


def test_levels_1_and_2_follow_the_voice_not_the_right_hands_top(tmp_path):
    # The right hand plays G4 / A4 chords; the voice sings E5 D5 C5. The fold is off: level 4 is the piano alone.
    sc = song(tmp_path, [[TUNE] * 2], [bar(CHORDS, LH)] * 2, melody=False)
    lv = levels(sc)
    tune = [(b, p, "R") for b, p in ((0, E5), (1, D5), (2, C5), (4, E5), (5, D5), (6, C5))]
    assert lv[1] == tune
    assert lv[2] == sorted(tune + [(0, C3, "L"), (4, C3, "L")])
    assert not any(n.pitch in (E5, D5) for n in lv_notes(sc, 4))
    assert sc.tune_bars == [(0, 8 * Q)]


def lv_notes(sc, level):
    return make_levels(sc.notes, TPB, sc.song.meta, (sc.tune, sc.tune_bars))[level]


def test_an_intro_without_the_voice_keeps_the_pianos_line(tmp_path):
    # Two bars of piano intro (the voice rests a whole bar each), then the voice comes in: level 1 plays the right
    # hand's top line (A4 / G4) first, then the tune.
    whole_rest = '<note><rest measure="yes"/><duration>16</duration><voice>1</voice></note>'
    intro = note("A", 4, 8) + note("G", 4, 8)
    sc = song(tmp_path, [[whole_rest, whole_rest, TUNE]], [bar(intro, LH), bar(intro, LH), bar(CHORDS, LH)])
    assert [(b, p) for b, p, _ in levels(sc)[1]] == [(0, A4), (2, G4), (4, A4), (6, G4), (8, E5), (9, D5), (10, C5)]
    assert sc.tune_bars == [(8 * Q, 12 * Q)]


def test_the_piano_plays_until_a_pickup_and_through_a_long_rest(tmp_path):
    # Bar 1 is all piano until the voice's pickup on beat 4; in bar 3 the voice rests from beat 2 (three beats), and
    # the piano's A4, struck with the voice's E5, plays on from there.
    intro = note("A", 4, 8) + note("G", 4, 8)
    voice = [rest(12) + note("C", 5, 4), TUNE, note("E", 5, 4) + rest(12)]
    sc = song(tmp_path, [voice], [bar(intro, LH)] * 3, melody=False)
    assert [(b, p) for b, p, _ in levels(sc)[1]] == [(0, A4), (2, G4), (3, C5), (4, E5), (5, D5), (6, C5),
                                                     (8, E5), (9, A4), (10, G4)]


def test_a_figure_intro_is_one_held_note_a_bar_not_hammered(tmp_path):
    # Go The Distance's intro: D6 D5 A5 D5 in sixteenths for two bars before the voice comes in. Level 1 holds the
    # D6 a bar at a time (it struck it on every beat); the merged arrangement's intro is the piano's, unchanged.
    whole_rest = '<note><rest measure="yes"/><duration>16</duration><voice>1</voice></note>'
    figure = "".join(note(s, o, 1) for s, o in (("D", 6), ("D", 5), ("A", 5), ("D", 5))) * 4
    piano = [bar(figure, LH), bar(figure, LH), bar(CHORDS, LH)]
    sc = song(tmp_path, [[whole_rest, whole_rest, TUNE]], piano, melody=False)
    assert [(n.start / Q, n.end / Q, n.pitch) for n in sorted(lv_notes(sc, 1), key=lambda n: n.start)][:3] == \
        [(0, 4, 86), (4, 8, 86), (8, 9, E5)]
    merged = song(tmp_path, [[whole_rest, whole_rest, TUNE]], piano)
    intro = lambda s: [(n.start, n.end, n.pitch, n.hand) for n in s.notes if n.start < 8 * Q]  # noqa: E731
    assert intro(merged) == intro(sc)


def test_held_figure_counts_from_the_previous_strike_and_never_crosses_the_voice():
    from pianopipe.difficulty import _held_figure
    s = Q // 4
    # Beat 1 a sixteenth figure under D6, then D6 chords on beats 2-4 with nothing between them: only beat 2's
    # D6 (three strikes after beat 1's) joins the hold, beats 3 and 4 are struck.
    figure = [Note(i * s, (i + 1) * s, p, hand="R") for i, p in enumerate((86, 74, 81, 74))]
    chords = [Note(b * Q, (b + 1) * Q, p, hand="R") for b in (1, 2, 3) for p in (86, 78)]
    line = [Note(0, Q, 86, hand="R")] + [Note(b * Q, (b + 1) * Q, 86, hand="R") for b in (1, 2, 3)]
    held = _held_figure(line, figure + chords, TPB, (), [])
    assert [(n.start, n.end) for n in held] == [(0, 2 * Q), (2 * Q, 3 * Q), (3 * Q, 4 * Q)]
    # A sung span between two D6s of a figure: they stay apart.
    fig2 = figure + [Note(2 * Q + i * s, 2 * Q + (i + 1) * s, p, hand="R") for i, p in enumerate((86, 74, 81, 74))]
    line2 = [Note(0, Q, 86, hand="R"), Note(2 * Q, 3 * Q, 86, hand="R")]
    assert len(_held_figure(line2, fig2 + [Note(Q, 2 * Q, 70, hand="R")], TPB, (), [(Q, 2 * Q)])) == 2


def test_a_misread_voice_bar_keeps_the_pianos_line_in_level_1(tmp_path):
    bad = note("E", 5, 4) + note("D", 5, 4)  # two beats in 4/4
    sc = song(tmp_path, [[TUNE, bad]], [bar(CHORDS, LH)] * 2, melody=False)
    assert [(b, p) for b, p, _ in levels(sc)[1]] == [(0, E5), (1, D5), (2, C5), (4, G4), (6, A4)]


def test_a_piano_note_held_into_a_bar_without_the_voice_plays_on_there(tmp_path):
    # The piano's G4 is tied from bar 1 (sung) into bar 2 (the voice misread): level 1 has it from bar 2's start.
    bad = note("E", 5, 4) + note("D", 5, 4)
    piano = [bar(note("G", 4, 16, 1, 1, '<tie type="start"/>'), LH), bar(note("G", 4, 16, 1, 1, '<tie type="stop"/>'), LH)]
    sc = song(tmp_path, [[TUNE, bad]], piano, melody=False)
    assert [(n.start / Q, n.end / Q, n.pitch) for n in sorted(lv_notes(sc, 1), key=lambda n: n.start)][-1] == (4, 8, G4)


def test_a_piano_note_around_a_sung_stretch_keeps_both_ends(tmp_path):
    # The piano holds G4 for two bars; the voice sings only on bar 1's beats 3-4 (bar 2 misread): level 1 has the G4
    # before the voice, the voice, then the G4 again through bar 2.
    piano = [bar(note("G", 4, 16, 1, 1, '<tie type="start"/>'), LH), bar(note("G", 4, 16, 1, 1, '<tie type="stop"/>'), LH)]
    sc = song(tmp_path, [[rest(8) + note("E", 5, 8), note("E", 5, 4)]], piano, melody=False)
    got = [(n.start / Q, n.end / Q, n.pitch) for n in sorted(lv_notes(sc, 1), key=lambda n: n.start)]
    assert got == [(0, 2, G4), (2, 4, E5), (4, 8, G4)]


def test_a_baritone_in_the_bass_clef_is_raised_to_the_right_hand(tmp_path):
    clef = "<attributes><clef><sign>F</sign><line>4</line></clef></attributes>"
    low = clef + note("E", 3, 4) + note("D", 3, 4) + note("C", 3, 8)
    sc = song(tmp_path, [[low]], [bar(CHORDS, LH)], names=["Baritone"], melody=False)
    assert [p for _, p, _ in levels(sc)[1]] == [E4, 62, C4]
