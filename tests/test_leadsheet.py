"""House arrangements: the ABC lead-sheet subset and the generated left hand."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from pianopipe import leadsheet
from pianopipe.leadsheet import LeadSheetError, parse, parse_chord
from pianopipe.process import process_file
from pianopipe.quality import GatePolicy

HOUSE = Path(__file__).resolve().parents[2] / "data" / "house"


def sheet(body: str, header: str = "M:4/4\nL:1/4\nK:C") -> leadsheet.LeadSheet:
    return parse(f"X:1\nT:Test\n{header}\n{body}")


def melody(s) -> list[tuple[float, float, int]]:
    return [(float(a), float(b), p) for a, b, p in s.melody]


def test_notes_lengths_octaves_and_rests():
    s = sheet('"C"C D/ E/ F2 | z c C, C\'')
    assert melody(s) == [(0, 1, 60), (1, 0.5, 62), (1.5, 0.5, 64), (2, 2, 65), (5, 1, 72), (6, 1, 48), (7, 1, 72)]
    assert s.length == 8


def test_key_signature_and_accidentals_last_to_the_bar():
    s = sheet('"G"F ^F F =F | F G _B B', header="M:4/4\nL:1/4\nK:G")
    assert [p for _, _, p in s.melody] == [66, 66, 66, 65, 66, 67, 70, 70]
    s = sheet('"Am"B c d e', header="M:4/4\nL:1/4\nK:Dm")  # D minor = F major: one flat
    assert s.melody[0][2] == 70 and s.key_minor


def test_ties_triplets_and_dotted_lengths():
    s = sheet('"C"C3/2 D/ E2- | E (3FGA B', header="M:4/4\nL:1/4\nK:C")
    assert melody(s)[:3] == [(0, 1.5, 60), (1.5, 0.5, 62), (2, 3, 64)]
    starts = [round(a, 3) for a, _, _ in melody(s)[3:6]]
    assert starts == [5.0, round(5 + 2 / 3, 3), round(5 + 4 / 3, 3)]


def test_repeats_and_endings_unroll():
    s = sheet('"C"C4 |: D4 |[1 E4 :|[2 F4 |]')
    assert [p for _, _, p in s.melody] == [60, 62, 64, 62, 65]
    assert len(s.bars) == 5


def test_chord_symbols():
    c = parse_chord("F#m7", Fraction(0))
    assert c.root == 6 and c.tones == (6, 9, 1, 4)
    assert parse_chord("C/E", Fraction(0)).bass == 4
    with pytest.raises(LeadSheetError):
        parse_chord("H7", Fraction(0))


def test_left_hand_patterns_and_pickup():
    s = sheet('C | "C"E E E | "G7"D D D | "C"C3 |]', header="M:3/4\nL:1/4\nK:C")
    song = leadsheet.arrange(s)
    lh = [n for n in song.notes if n.hand == "L"]
    assert min(n.start for n in lh) == 480  # nothing under the pickup
    bar1 = [n for n in lh if 480 <= n.start < 480 * 4]
    roots = [n for n in bar1 if n.start == 480]
    assert len(roots) == 1 and roots[0].pitch % 12 == 0  # oom...
    assert len([n for n in bar1 if n.start == 960]) == 3  # ...pah (a triad)
    bar2 = [n for n in lh if 480 * 4 <= n.start < 480 * 7]
    assert {n.pitch % 12 for n in bar2} <= {7, 11, 2, 5}  # G7 tones only
    rh = [n for n in song.notes if n.hand == "R"]
    for n in lh:  # the accompaniment stays under the melody note sounding with it
        above = [m.pitch for m in rh if m.start < n.end and m.end > n.start]
        assert not above or n.pitch < min(above)


def test_needs_chords():
    with pytest.raises(LeadSheetError):
        leadsheet.arrange(sheet("C D E F"))


@pytest.mark.skipif(not HOUSE.is_dir(), reason="the house lead sheets live in the Quest Piano Trainer repo (data/house)")
def test_house_files_build(tmp_path):
    files = sorted(HOUSE.glob("*.abc"))
    assert files
    for f in files:
        entry = process_file(f, tmp_path, {"title": f.stem}, policy=GatePolicy(min_duration_s=15, min_notes=60))
        assert entry.hand_source == "leadsheet"
        assert all(lv.fingering for lv in entry.levels)
