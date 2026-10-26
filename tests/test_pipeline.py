import json

import mido

from pianopipe import midi_io
from pianopipe.catalog import Catalog, validate
from pianopipe.cli import main
from pianopipe.difficulty import make_level
from pianopipe.hands import hand_accuracy, playability_issues, split_hands
from pianopipe.process import build_catalog, process_file
from pianopipe.tracks import pick_piano

from helpers import TPB, make_midi, two_hand_piece


def test_load_pairs_notes_and_keeps_tempo(tmp_path):
    rh, lh = two_hand_piece(1)
    path = make_midi(tmp_path / "a.mid", [{"name": "Piano", "notes": rh + lh}])
    song = midi_io.load(path)
    assert len(song.notes) == len(rh) + len(lh)
    assert any(m.type == "set_tempo" and m.tempo == 600000 for _, m in song.meta)


def test_hands_from_track_names(tmp_path):
    rh, lh = two_hand_piece(1)
    path = make_midi(tmp_path / "a.mid", [{"name": "Right Hand", "notes": rh}, {"name": "Left Hand", "notes": lh, "channel": 1}])
    part = pick_piano(midi_io.load(path))
    assert part.hand_source == "track_names"
    assert {n.hand for n in part.notes if n.pitch >= 60} == {"R"}
    assert {n.hand for n in part.notes if n.pitch < 60} == {"L"}


def test_two_unnamed_tracks_split_by_pitch(tmp_path):
    rh, lh = two_hand_piece(1)
    path = make_midi(tmp_path / "a.mid", [{"name": "Track A", "notes": lh}, {"name": "Track B", "notes": rh, "channel": 1}])
    part = pick_piano(midi_io.load(path))
    assert part.hand_source == "two_tracks"
    assert all(n.hand == ("R" if n.pitch >= 60 else "L") for n in part.notes)


def test_drums_and_non_piano_dropped(tmp_path):
    rh, lh = two_hand_piece(1)
    path = make_midi(
        tmp_path / "band.mid",
        [
            {"name": "Piano", "notes": rh + lh},
            {"name": "Drums", "channel": 9, "notes": [(0, 100, 36)]},
            {"name": "Strings", "channel": 2, "program": 48, "notes": [(0, 100, 60)]},
        ],
    )
    part = pick_piano(midi_io.load(path))
    assert len(part.notes) == len(rh) + len(lh)


def test_split_hands_recovers_simple_piece(tmp_path):
    rh, lh = two_hand_piece(4)
    path = make_midi(tmp_path / "a.mid", [{"name": "Right", "notes": rh}, {"name": "Left", "notes": lh, "channel": 1}])
    truth = pick_piano(midi_io.load(path)).notes
    predicted = split_hands([n.with_hand(None) for n in truth], TPB)
    assert hand_accuracy(predicted, truth) == 1.0


def test_split_hands_follows_a_hand_that_moves(tmp_path):
    # Right hand walks down into the left hand's home range while the left hand stays low.
    rh = [(i * TPB, TPB, 72 - i * 2) for i in range(10)]  # 72 down to 54
    lh = [(i * TPB, TPB, 36 + (i % 2) * 3) for i in range(10)]
    path = make_midi(tmp_path / "a.mid", [{"name": "Right", "notes": rh}, {"name": "Left", "notes": lh, "channel": 1}])
    truth = pick_piano(midi_io.load(path)).notes
    predicted = split_hands([n.with_hand(None) for n in truth], TPB)
    assert hand_accuracy(predicted, truth) == 1.0


def test_playability_flags_big_chords():
    from pianopipe.model import Note

    notes = [Note(0, 100, p, hand="R") for p in (60, 62, 64, 65, 67, 69)]
    assert playability_issues(notes, TPB)


def test_levels_get_simpler(tmp_path):
    rh, lh = two_hand_piece(4)
    path = make_midi(tmp_path / "a.mid", [{"name": "Right", "notes": rh}, {"name": "Left", "notes": lh, "channel": 1}])
    notes = pick_piano(midi_io.load(path)).notes
    counts = [len(make_level(notes, lvl, TPB)) for lvl in (1, 2, 3, 4)]
    assert counts == sorted(counts)
    assert counts[0] == 16  # one melody note per beat
    assert all(n.hand == "R" for n in make_level(notes, 1, TPB))


def test_process_writes_app_ready_files(tmp_path):
    rh, lh = two_hand_piece(2)
    src = make_midi(tmp_path / "Minuet_in_G.mid", [{"name": "Piano", "notes": rh + lh}])
    entry = process_file(src, tmp_path / "out", {"composer": "Petzold", "license": "Public domain"})
    assert entry.id == "minuet-in-g" and entry.hand_source == "auto"
    assert [lv.level for lv in entry.levels] == [1, 2, 3, 4]
    assert entry.duration_s == 2 * 4 * 0.6
    mid = mido.MidiFile(str(tmp_path / "out" / "minuet-in-g" / "level4.mid"))
    names = [next((m.name for m in t if m.type == "track_name"), None) for t in mid.tracks]
    assert names == ["Conductor", "RH", "LH"]


def test_build_catalog_roundtrip_and_cli(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    rh, lh = two_hand_piece(10)  # long enough for the quality gate
    make_midi(src / "one.mid", [{"name": "Piano", "notes": rh + lh}])
    (src / "broken.mid").write_bytes(b"not midi")
    (tmp_path / "sources.json").write_text(json.dumps({"one.mid": {"title": "Piece One", "license": "CC BY-SA"}}))
    assert main(["build", str(src), str(tmp_path / "cat"), "--name", "Test", "--sources", str(tmp_path / "sources.json")]) == 0
    data = json.loads((tmp_path / "cat" / "catalog.json").read_text())
    validate(data)
    assert [s["title"] for s in data["songs"]] == ["Piece One"]
    assert data["songs"][0]["levels"][0]["url"] == "songs/one/level1.mid"
    assert Catalog.load(tmp_path / "cat" / "catalog.json").songs[0].license == "CC BY-SA"


def test_quantize_never_turns_a_fast_melody_into_chords():
    """Für Elise's sixteenth-note opening (E D# E D# E B D C A) on an eighth grid must stay one note at a time."""
    from pianopipe.difficulty import make_level
    from pianopipe.model import Note
    tpb = 480
    pitches = [76, 75, 76, 75, 76, 71, 74, 72, 69]
    notes = [Note(pitch=p, start=i * tpb // 4, end=(i + 1) * tpb // 4, velocity=80, channel=0, hand="R") for i, p in enumerate(pitches)]
    for level in (1, 2, 3):
        out = sorted(make_level(notes, level, tpb), key=lambda n: n.start)
        starts = [n.start for n in out if n.hand == "R"]
        assert len(starts) == len(set(starts)), f"level {level}: notes merged into chords"
        assert [n.pitch for n in out if n.hand == "R"] == pitches, f"level {level}: melody changed"
