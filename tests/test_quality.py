"""The quality gate's level checks (quality.level_faults): a reduced level against the full arrangement."""

from dataclasses import replace

import pytest

from pianopipe.difficulty import make_levels
from pianopipe.model import Note
from pianopipe.process import process_file
from pianopipe.quality import OFFICIAL, GatePolicy, QualityError, check, level_faults

from helpers import TPB, make_midi

T = TPB // 3  # a triplet eighth
S = TPB // 4  # a sixteenth


def _r(start, dur, pitch):
    return Note(start=start, end=start + dur, pitch=pitch, hand="R")


def _l(start, dur, pitch):
    return Note(start=start, end=start + dur, pitch=pitch, hand="L")


def _triplet_piece(beats=48):
    """Moonlight-like: the right hand's triplets over a left hand that holds each beat."""
    rh = [_r(b * TPB + k * T, T, p) for b in range(beats) for k, p in enumerate((68, 73, 76))]
    return rh + [_l(b * TPB, TPB, 49 - b % 2) for b in range(beats)]


def _kinds(faults):
    return sorted((level, count, text.split(": ")[-1].split(" ", 1)[-1]) for level, count, text in faults)


def test_levels_made_today_have_no_faults():
    for notes in (_triplet_piece(),
                  [_r(i * S, S, p) for i, p in enumerate([76, 75, 76, 75, 76, 71, 74, 72, 69] * 4)] + [_l(0, 9 * TPB, 45)]):
        assert level_faults(notes, make_levels(notes, TPB), TPB) == []


def test_triplets_written_as_sixteenths_are_moved_notes():
    """What level 3 used to do: triplets at 0, 1/3 and 2/3 of the beat on a sixteenth grid, at 0, 1/4 and 3/4."""
    notes = _triplet_piece()
    levels = make_levels(notes, TPB)
    wrong = {0: 0, T: S, 2 * T: 3 * S}
    levels[3] = [replace(n, start=n.start - n.start % TPB + wrong[n.start % TPB]) if n.hand == "R" else n for n in levels[3]]
    assert _kinds(level_faults(notes, levels, TPB)) == [(3, 96, "notes moved off the rhythm")]


def test_first_triplet_stretched_to_an_eighth_is_held_into_the_next():
    """What levels 1 and 2 used to do: each beat's first triplet as long as an eighth, over the second."""
    notes = _triplet_piece()
    levels = make_levels(notes, TPB)
    for level in (1, 2):
        levels[level] = [replace(n, end=n.start + TPB // 2) if n.hand == "R" and n.start % TPB == 0 else n for n in levels[level]]
    assert _kinds(level_faults(notes, levels, TPB)) == [(1, 48, "notes held into the next note"),
                                                       (2, 48, "notes held into the next note")]


def test_holds_and_figures_that_are_the_piece_s_own_are_not_faults():
    """A key the piece really holds under the next note, and a broken chord's top note moved to where the figure
    starts, are what the level is meant to be."""
    held = [_r(0, 4 * TPB, 60)] + [_r(k * TPB, TPB, 72 + k) for k in range(1, 4)] + [_l(0, 4 * TPB, 36)]
    assert level_faults(held, make_levels(held, TPB), TPB) == []
    figure = [_r(k * S, S, p) for k, p in enumerate([60, 67, 64, 67, 62, 69, 65, 69] * 2)] + [_l(0, 4 * TPB, 36)]
    levels = make_levels(figure, TPB)
    assert [n.start for n in levels[1]] == [0, TPB, 2 * TPB, 3 * TPB]  # the 67s and 69s, each from a sixteenth later
    assert level_faults(figure, levels, TPB) == []


def test_a_note_off_the_grid_where_nothing_starts_is_moved():
    notes = [_r(b * TPB, TPB, 72) for b in range(4)] + [_l(0, 4 * TPB, 36)]
    levels = make_levels(notes, TPB)
    levels[1] = [replace(n, start=n.start + 7, end=n.end + 7) if n.start == TPB else n for n in levels[1]]
    assert _kinds(level_faults(notes, levels, TPB)) == [(1, 1, "notes moved off the rhythm")]


def test_a_level_with_fewer_notes_than_the_one_below():
    notes = _triplet_piece()
    levels = make_levels(notes, TPB)
    levels[3] = levels[3][:len(levels[2]) - 5]
    faults = [f for f in level_faults(notes, levels, TPB) if "fewer" in f[2]]
    assert [(level, count) for level, count, _ in faults] == [(2, 5)]
    assert faults[0][2] == f"level 3 has fewer notes than level 2 ({len(levels[3])}, {len(levels[2])})"


def test_a_voice_part_s_tune_counts_as_the_piece():
    """Levels 1 and 2 of a song play its voice part, which the piano's notes don't have."""
    piano = [_r(b * TPB, TPB, 60) for b in range(8)] + [_l(b * TPB, TPB, 36) for b in range(8)]
    tune = [_r(b * TPB + 7, TPB // 2, 72 + b) for b in range(8)]  # sung a little late: off every grid
    levels = {1: tune, 2: tune + [n for n in piano if n.hand == "L"], 3: piano, 4: piano}
    assert [text for _, _, text in level_faults(piano, levels, TPB)] == ["level 1: 8 notes moved off the rhythm",
                                                                        "level 2: 8 notes moved off the rhythm"]
    assert level_faults(piano, levels, TPB, tune) == []
    # A busier tune than the piano part under it: levels 1-2 have more notes than 3-4, which is the arrangement.
    busy = [_r(k * TPB // 2, TPB // 2, 72 + k % 5) for k in range(16)]
    levels = {1: busy, 2: busy + [n for n in piano if n.hand == "L"], 3: piano, 4: piano}
    assert level_faults(piano, levels, TPB, busy) == []
    assert [text for _, _, text in level_faults(piano, levels, TPB)][-1] == "level 3 has fewer notes than level 2 (16, 24)"


def test_faults_warn_by_default_and_skip_the_song_in_the_official_catalog():
    notes = _triplet_piece()
    levels = make_levels(notes, TPB)
    levels[1] = [replace(n, end=n.start + TPB // 2) if n.start % TPB == 0 else n for n in levels[1]]
    text = "level 1: 48 notes held into the next note"
    plain = check(notes, levels, "two_tracks", 60.0, TPB)
    assert plain.ok and text in plain.warnings
    official = check(notes, levels, "two_tracks", 60.0, TPB, policy=OFFICIAL)
    assert official.reasons == [text] and text not in official.warnings
    lenient = check(notes, levels, "two_tracks", 60.0, TPB, policy=GatePolicy(max_level_faults=0.5))
    assert lenient.ok and text in lenient.warnings  # 48 of 144 notes


def test_official_build_of_a_triplet_piece_passes(tmp_path):
    """End to end: the levels process_file makes of a triplet piece pass the official gate, and a level file's notes are
    where the piece has them."""
    notes = _triplet_piece()
    tracks = [{"name": "Right", "notes": [(n.start, n.end - n.start, n.pitch) for n in notes if n.hand == "R"]},
              {"name": "Left", "channel": 1, "notes": [(n.start, n.end - n.start, n.pitch) for n in notes if n.hand == "L"]}]
    entry = process_file(make_midi(tmp_path / "triplets.mid", tracks), tmp_path / "out", policy=OFFICIAL)
    assert not [w for w in entry.warnings if w.startswith("level ")]
    with pytest.raises(QualityError):
        process_file(make_midi(tmp_path / "short.mid", tracks[:1]), tmp_path / "out", policy=OFFICIAL)
