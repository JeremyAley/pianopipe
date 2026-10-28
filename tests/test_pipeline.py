import json

import mido

from pianopipe import midi_io
from pianopipe.catalog import Catalog, validate
from pianopipe.cli import main
from pianopipe.difficulty import make_level
from pianopipe.hands import hand_accuracy, playability_issues, split_hands
from pianopipe.process import build_catalog, load_notes, process_file
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


def test_unplayable_chord_from_the_tracks_is_split(tmp_path):
    # The right track's chord holds a low G2 (piano-midi.de's Moonlight Sonata): over two octaves in one press. The
    # left hand, which plays C#2 there, takes it; the playable chords stay where the tracks put them.
    rh = [(0, TPB, p) for p in (43, 64, 73)] + [(TPB, TPB, p) for p in (64, 68, 73)]
    lh = [(0, 2 * TPB, 37), (TPB, TPB, 44)]
    path = make_midi(tmp_path / "a.mid", [{"name": "Right", "notes": rh}, {"name": "Left", "notes": lh, "channel": 1}])
    repaired = []
    _, notes, hand_source, _ = load_notes(path, repaired=repaired)
    assert hand_source == "track_names" and repaired == [0]
    assert [(n.start, n.pitch) for n in notes if n.hand == "L"] == [(0, 37), (0, 43), (TPB, 44)]
    assert playability_issues(notes, TPB) == []


def test_a_key_struck_twice_at_once_is_one_note(tmp_path):
    # Two voice tracks on a unison (Mutopia's Schubert impromptus D899/3 and 4): the key sounds once, the longer note.
    rh = [(0, TPB, 72), (0, 2 * TPB, 67)]
    inner = [(0, TPB, 67), (TPB, TPB, 64)]
    lh = [(0, 2 * TPB, 48)]
    path = make_midi(tmp_path / "a.mid", [{"name": "upA:", "notes": rh}, {"name": "upB:", "notes": inner, "channel": 1},
                                          {"name": "down:", "notes": lh, "channel": 2}])
    _, notes, _, _ = load_notes(path)
    assert sorted((n.start, n.end, n.pitch) for n in notes if n.pitch == 67) == [(0, 2 * TPB, 67)]


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


def test_printed_fingers_from_the_manifest_are_kept(tmp_path):
    # A Mutopia MIDI file has no fingering; its LilyPond source's (tools/mutopia_fingering.py) comes in the manifest.
    # The search alone gives the melody's C5 a 3 and the bass's G3 a 1; the source says 4 and 2.
    rh, lh = two_hand_piece(2)
    src = make_midi(tmp_path / "a.mid", [{"name": "Right", "notes": rh}, {"name": "Left", "notes": lh, "channel": 1}])
    process_file(src, tmp_path / "out", {"id": "a", "fingers": [[TPB // 4, 55, 2], [0, 72, 4]]})
    song = midi_io.load(tmp_path / "out" / "a" / "level4.mid")
    fingers = {(n.start, n.pitch): n.finger for n in song.notes}
    assert fingers[TPB // 4, 55] == 2 and fingers[0, 72] == 4


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


def test_version_has_one_source(capsys):
    """pyproject.toml holds the version; the package reads it from the install and the README installs that tag."""
    import re
    from pathlib import Path

    import pytest

    from pianopipe import __version__

    root = Path(__file__).resolve().parents[1]
    declared = re.search(r'^version = "([^"]+)"', (root / "pyproject.toml").read_text(encoding="utf-8"), re.M).group(1)
    assert f"pianopipe@v{declared}" in (root / "README.md").read_text(encoding="utf-8")
    assert not re.search(r'__version__ = "\d+\.\d', (root / "src" / "pianopipe" / "__init__.py").read_text(encoding="utf-8"))
    with pytest.raises(SystemExit):
        main(["--version"])
    assert capsys.readouterr().out.strip() == f"pianopipe {__version__}"


def test_tempo_map_matches_summing_the_spans():
    from pianopipe.model import Song
    from pianopipe.process import seconds_at, tempo_map

    meta = [(0, mido.MetaMessage("set_tempo", tempo=600000)), (960, mido.MetaMessage("set_tempo", tempo=300000)),
            (960, mido.MetaMessage("set_tempo", tempo=1000000)), (1440, mido.MetaMessage("time_signature"))]
    song = Song(ticks_per_beat=480, meta=meta)
    seconds = tempo_map(song)
    assert seconds(0) == 0.0 and seconds(480) == 0.6 and seconds(960) == 1.2  # two beats at 0.6 s
    assert seconds(1440) == 1.2 + 1.0  # the last tempo event at a tick wins
    assert seconds(240) == seconds_at(song, 240) == 0.3
    assert tempo_map(Song(ticks_per_beat=480))(480) == 0.5  # no tempo: 120 a minute


def _band(tmp_path, extra=()):
    rh, lh = two_hand_piece(2)
    tracks = [
        {"name": "Melody", "channel": 0, "program": 73, "notes": rh},  # flute
        {"name": "Bass", "channel": 1, "program": 33, "notes": lh},  # electric bass (finger)
        {"name": "Drums", "channel": 9, "program": 0, "notes": [(i * TPB, TPB // 2, 36) for i in range(8)]},
        *extra,
    ]
    return midi_io.load(make_midi(tmp_path / "band.mid", tracks))


def test_band_melody_and_bass_are_not_two_hands(tmp_path):
    """A flute "Melody" and a bass guitar "Bass" are a band, not a pianist's hands: no track_names."""
    part = pick_piano(_band(tmp_path))
    assert part.hand_source == "none"
    assert all(n.channel != 9 for n in part.notes)


def test_band_with_a_piano_track_uses_only_the_piano(tmp_path):
    piano = [(i * TPB, TPB, p) for i, p in enumerate((60, 64, 67, 72, 48, 52, 55, 60) * 2)]
    part = pick_piano(_band(tmp_path, [{"name": "Piano", "channel": 2, "program": 0, "notes": piano}]))
    assert {n.channel for n in part.notes} == {2} and len(part.notes) == len(piano)


def test_band_with_two_named_piano_tracks_gets_track_names(tmp_path):
    rh, lh = two_hand_piece(2)
    extra = [{"name": "Piano RH", "channel": 3, "program": 0, "notes": rh},
             {"name": "Piano LH", "channel": 4, "program": 1, "notes": lh}]
    part = pick_piano(_band(tmp_path, extra))
    assert part.hand_source == "track_names"
    assert {n.channel for n in part.notes} == {3, 4}
    assert {n.hand for n in part.notes if n.channel == 3} == {"R"} and {n.hand for n in part.notes if n.channel == 4} == {"L"}


def test_gm_drum_channel_never_counts(tmp_path):
    """Drums on channel 10 are dropped even when their track has a hand-like name and the default program."""
    rh, lh = two_hand_piece(2)
    tracks = [{"name": "Right Hand", "notes": rh}, {"name": "Left Hand", "channel": 1, "notes": lh},
              {"name": "Bass Drum", "channel": 9, "notes": [(i * TPB, TPB // 2, 36) for i in range(8)]}]
    part = pick_piano(midi_io.load(make_midi(tmp_path / "d.mid", tracks)))
    assert part.hand_source == "track_names" and len(part.notes) == len(rh) + len(lh)


def test_named_instruments_without_program_changes(tmp_path):
    """Every track at the default program: a track named after another instrument isn't the piano."""
    rh, lh = two_hand_piece(2)
    tracks = [{"name": "Melody", "notes": rh}, {"name": "Bass", "channel": 1, "notes": lh},
              {"name": "Guitar", "channel": 2, "notes": [(i * TPB, TPB, 64) for i in range(8)]}]
    part = pick_piano(midi_io.load(make_midi(tmp_path / "g.mid", tracks)))
    assert part.hand_source == "track_names" and {n.channel for n in part.notes} == {0, 1}
