import json

import mido
import pytest

from pianopipe import midi_io
from pianopipe.catalog import validate
from pianopipe.cli import main
from pianopipe.fingering import assign_fingering
from pianopipe.fingering_eval import CASES, evaluate, pitch, run_case
from pianopipe.model import Note
from pianopipe.process import process_file

from helpers import TPB, make_midi, two_hand_piece

# Cases the model must get exactly right; the rest are scored together below.
MUST_PASS = [
    "C major RH up, 2 oct", "C major RH down, 2 oct", "C major RH up and down", "C major LH up and down",
    "G major RH up", "G major LH up", "D major RH up", "D major LH up", "F major RH up", "F major LH up",
    "C position RH", "C position LH", "G position RH skips", "Ode to Joy RH", "Mary had a little lamb RH",
    "C major arpeggio RH up", "C major arpeggio RH down", "C major arpeggio LH up", "A minor arpeggio RH up",
    "C broken chord RH", "Alberti bass C LH", "C triad inversions RH", "C triad inversions LH", "I IV V I RH",
    "Octaves RH", "Jump up then down RH", "Jump down then up RH", "Minuet in G LH bars 1-8",
]


@pytest.mark.parametrize("name", MUST_PASS)
def test_textbook_fingerings(name):
    result = run_case(next(c for c in CASES if c.name == name))
    assert result.agree == result.total, f"got {result.predicted}"


def test_overall_agreement_with_hand_written_cases():
    results = evaluate()
    agree = sum(r.agree for r in results) / sum(r.total for r in results)
    assert agree >= 0.89


def test_every_note_gets_a_finger_and_chords_are_ordered():
    rh, lh = two_hand_piece(4)
    notes = [Note(s, s + d, p, hand="R") for s, d, p in rh] + [Note(s, s + d, p, hand="L") for s, d, p in lh]
    notes.append(Note(0, TPB, 30))  # no hand: left alone
    out = assign_fingering(notes, TPB)
    assert [(n.start, n.pitch, n.hand) for n in out] == [(n.start, n.pitch, n.hand) for n in notes]
    assert out[-1].finger is None
    assert all(n.finger in (1, 2, 3, 4, 5) for n in out[:-1])
    for hand, sign in (("R", 1), ("L", -1)):
        by_start = {}
        for n in out:
            if n.hand == hand:
                by_start.setdefault(n.start, []).append(n)
        for chord in by_start.values():
            fingers = [n.finger for n in sorted(chord, key=lambda n: sign * n.pitch)]
            assert fingers == sorted(set(fingers)), "a chord's fingers rise with pitch (RH) or fall (LH)"


def test_held_note_keeps_its_finger_and_the_others_fit_around_it():
    # C4 held for four beats while D4 E4 F4 move above it: C stays under one finger, the others above it.
    notes = [Note(0, 4 * TPB, pitch("C4"), hand="R")] + [
        Note(i * TPB, (i + 1) * TPB, pitch(name), hand="R") for i, name in enumerate(["E4", "F4", "G4", "A4"])
    ]
    out = assign_fingering(notes, TPB)
    held = out[0].finger
    assert held == 1
    assert all(n.finger > held for n in out[1:])
    assert len({n.finger for n in out}) == 5


def test_huge_chords_and_empty_hands_do_not_fail():
    cluster = [Note(0, TPB, p, hand="R") for p in range(60, 68)]
    out = assign_fingering(cluster, TPB)
    assert [n.finger for n in out][0] == 1 and [n.finger for n in out][-1] == 5
    assert assign_fingering([], TPB) == []


def test_fingers_round_trip_through_app_midi(tmp_path):
    notes = [
        Note(0, TPB, 60, hand="R", finger=1), Note(0, TPB, 64, hand="R", finger=3), Note(TPB, 2 * TPB, 62, hand="R", finger=2),
        Note(0, 2 * TPB, 48, hand="L", finger=5), Note(TPB, 2 * TPB, 48, hand="L", finger=4),
    ]
    path = tmp_path / "f.mid"
    midi_io.write_app_midi(path, TPB, [], notes)
    # Each finger text sits right before its own note-on, at the same tick.
    for track in mido.MidiFile(str(path)).tracks[1:]:
        msgs = [m for m in track if m.type in ("text", "note_on")]
        assert [m.type for m in msgs] == ["text", "note_on"] * (len(msgs) // 2)
        assert all(on.time == 0 for on in msgs[1::2])
    song = midi_io.load(path)
    got = sorted((n.start, n.pitch, n.finger) for t in song.tracks for n in t.notes)
    assert got == sorted((n.start, n.pitch, n.finger) for n in notes)


def test_unfingered_notes_write_no_text_events(tmp_path):
    path = tmp_path / "plain.mid"
    midi_io.write_app_midi(path, TPB, [], [Note(0, TPB, 60, hand="R")])
    assert not any(m.type == "text" for t in mido.MidiFile(str(path)).tracks for m in t)
    assert midi_io.read_finger("f6") is None and midi_io.read_finger("fingering") is None


def test_process_fingers_every_level_and_flags_it(tmp_path):
    rh, lh = two_hand_piece(10)
    src = make_midi(tmp_path / "piece.mid", [{"name": "RH", "notes": rh}, {"name": "LH", "notes": lh, "channel": 1}])
    entry = process_file(src, tmp_path / "out")
    assert [lv.fingering for lv in entry.levels] == [True] * 4
    for lv in entry.levels:
        song = midi_io.load(tmp_path / "out" / lv.url)
        notes = [n for t in song.tracks for n in t.notes]
        assert len(notes) == lv.notes and all(n.finger in (1, 2, 3, 4, 5) for n in notes)
    plain = process_file(src, tmp_path / "plain", fingering=False)
    assert not any(lv.fingering for lv in plain.levels)


def test_catalog_json_fingering_flag(tmp_path):
    level = {"level": 1, "url": "a.mid", "sha256": "0" * 64}
    data = {"name": "x", "version": 1, "songs": [{"id": "a", "title": "A", "levels": [dict(level, fingering=True)]}]}
    validate(data)
    data["songs"][0]["levels"][0]["fingering"] = "yes"
    with pytest.raises(ValueError, match="fingering"):
        validate(data)


def test_built_catalog_marks_fingered_levels(tmp_path):
    rh, lh = two_hand_piece(10)
    src = tmp_path / "src"
    src.mkdir()
    make_midi(src / "piece.mid", [{"name": "RH", "notes": rh}, {"name": "LH", "notes": lh, "channel": 1}])
    assert main(["build", str(src), str(tmp_path / "cat")]) == 0
    data = json.loads((tmp_path / "cat" / "catalog.json").read_text(encoding="utf-8"))
    assert all(lv["fingering"] is True for s in data["songs"] for lv in s["levels"])


def test_cli_eval_fingering(capsys):
    assert main(["eval-fingering"]) == 0
    out = capsys.readouterr().out
    assert "C major RH up, 2 oct" in out and "of notes" in out


def _scale(fingers):
    return [Note(i * TPB, (i + 1) * TPB, p, hand="R", finger=f) for i, (p, f) in enumerate(zip((60, 62, 64, 65, 67), fingers))]


def test_printed_fingering_is_kept():
    """A score's fingering (MusicXML <fingering>) pins those notes; the search fills in the others around it."""
    free = [n.finger for n in assign_fingering(_scale([None] * 5), TPB)]
    printed = [2, None, 4, None, None]  # not what the search picks on its own
    assert free[0] != 2
    out = [n.finger for n in assign_fingering(_scale(printed), TPB)]
    assert out[0] == 2 and out[2] == 4 and all(out)


def test_printed_fingering_that_doesnt_fit_is_dropped_there_only():
    chord = [Note(0, TPB, 60, hand="R", finger=1), Note(0, TPB, 64, hand="R", finger=1)]  # one finger on two keys
    tune = [Note((i + 1) * TPB, (i + 2) * TPB, p, hand="R", finger=f) for i, (p, f) in enumerate(((67, 3), (65, 2)))]
    out = assign_fingering(chord + tune, TPB)
    assert out[0].finger != out[1].finger
    assert [n.finger for n in out[2:]] == [3, 2]
