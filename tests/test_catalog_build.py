import json

import pytest

from pianopipe.catalog import Catalog, sha256_file, validate, verify
from pianopipe.cli import main
from pianopipe.difficulty import difficulty_score
from pianopipe.model import Note
from pianopipe.process import build_catalog
from pianopipe.quality import OFFICIAL, GatePolicy, QualityError

from helpers import TPB, make_midi, two_hand_piece


def two_track(path, bars=10, **kw):
    rh, lh = two_hand_piece(bars)
    return make_midi(path, [{"name": "Piano right", "notes": rh}, {"name": "Piano left", "notes": lh, "channel": 1}], **kw)


@pytest.fixture
def sources(tmp_path):
    """A raw folder with one good file per hand source and one file per gate failure, plus its manifest."""
    raw = tmp_path / "raw"
    (raw / "set").mkdir(parents=True)
    rh, lh = two_hand_piece(10)
    two_track(raw / "set" / "good.mid")
    make_midi(raw / "set" / "one_track.mid", [{"name": "Piano", "notes": rh + lh}])
    two_track(raw / "set" / "short.mid", bars=2)
    # Hands swapped: the track named right plays the bass.
    make_midi(raw / "set" / "swapped.mid", [{"name": "Right", "notes": lh}, {"name": "Left", "notes": rh, "channel": 1}])
    # Every beat an eight-note right-hand chord: no hand can play it.
    clusters = [(b * TPB, TPB, p) for b in range(40) for p in range(72, 80)]
    make_midi(raw / "set" / "clusters.mid", [{"name": "Right", "notes": clusters}, {"name": "Left", "notes": lh, "channel": 1}])
    (raw / "set" / "broken.mid").write_bytes(b"not midi")
    names = ["good", "one_track", "short", "swapped", "clusters", "broken", "missing"]
    manifest = [
        {"id": n.replace("_", "-"), "file": f"set/{n}.mid", "title": n.title(), "composer": "Test", "license": "Public Domain",
         "attribution": "Fixture", "source_url": "https://example.org", "era": "Classical", "tags": ["test"]}
        for n in names
    ]
    return raw, manifest


def test_official_build_gates_files_and_reports_why(sources, tmp_path):
    raw, manifest = sources
    result = build_catalog(raw, tmp_path / "cat", "Test", manifest, OFFICIAL)
    assert [s.id for s in result.catalog.songs] == ["good"]
    reasons = dict(result.skipped)
    assert set(reasons) == {f"set/{n}.mid" for n in ("one_track", "short", "swapped", "clusters", "broken", "missing")}
    assert "hands not in the file's structure" in reasons["set/one_track.mid"]
    assert "too few notes" in reasons["set/short.mid"] and "too short" in reasons["set/short.mid"]
    assert "hands swapped" in reasons["set/swapped.mid"]
    assert "can't be played by one hand" in reasons["set/clusters.mid"]
    assert reasons["set/broken.mid"].startswith("error:")
    assert "file missing" in reasons["set/missing.mid"]
    report = json.loads((tmp_path / "cat" / "build-report.json").read_text(encoding="utf-8"))
    assert report["songs"] == 1 and report["levels"] == 4 and len(report["skipped"]) == 6


def test_duplicate_ids_are_skipped_before_overwriting(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    two_track(src / "Same Name.mid")
    two_track(src / "same_name.mid", bars=12)
    result = build_catalog(src, tmp_path / "cat", "Test")
    assert [s.id for s in result.catalog.songs] == ["same-name"]
    assert [r for _, r in result.skipped] == ["duplicate id same-name"]
    assert verify(tmp_path / "cat" / "catalog.json") == []


def test_default_gate_accepts_automatic_hands(sources, tmp_path):
    raw, manifest = sources
    result = build_catalog(raw, tmp_path / "cat", "Test", manifest)
    one = next(s for s in result.catalog.songs if s.id == "one-track")
    assert one.hand_source == "auto"


def test_catalog_validates_and_sha256_matches_files(sources, tmp_path):
    raw, manifest = sources
    build_catalog(raw, tmp_path / "cat", "Test", manifest, OFFICIAL)
    path = tmp_path / "cat" / "catalog.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    validate(data)
    assert verify(path) == []
    song = data["songs"][0]
    assert song["era"] == "Classical" and song["tags"][-1] == "test" and song["tags"][0] in ("beginner", "intermediate")
    assert song["duration_s"] == 10 * 4 * 0.6
    for lv in song["levels"]:
        assert lv["url"] == f"songs/good/level{lv['level']}.mid"
        assert lv["sha256"] == sha256_file(tmp_path / "cat" / lv["url"])
        assert 1 <= lv["difficulty"] <= 10
    # A changed file no longer matches.
    (tmp_path / "cat" / "songs" / "good" / "level2.mid").write_bytes(b"changed")
    assert verify(path) == ["good level 2: sha256 does not match songs/good/level2.mid"]
    (tmp_path / "cat" / "songs" / "good" / "level3.mid").unlink()
    assert "good level 3: songs/good/level3.mid is missing" in verify(path)


def test_full_title_kept_and_songs_ordered_easiest_first(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rh, lh = two_hand_piece(20)  # at double speed still over the gate's 20 s minimum
    # Same music, three tempos: the faster it goes, the harder it rates.
    for name, tempo in (("medium", 600000), ("fast", 300000), ("slow", 1200000)):
        make_midi(raw / f"{name}.mid", [{"name": "Right", "notes": rh}, {"name": "Left", "notes": lh, "channel": 1}], tempo=tempo)
    manifest = [{"id": n, "file": f"{n}.mid", "title": n.title()} for n in ("medium", "fast", "slow")]
    manifest[0]["full_title"] = "Medium (a much longer title with the opus number)"
    in_order = build_catalog(raw, tmp_path / "a", "T", manifest, OFFICIAL)
    assert [s.id for s in in_order.catalog.songs] == ["medium", "fast", "slow"]
    easiest = build_catalog(raw, tmp_path / "b", "T", manifest, OFFICIAL, easiest_first=True)
    assert [s.id for s in easiest.catalog.songs] == ["slow", "medium", "fast"]
    data = json.loads((tmp_path / "b" / "catalog.json").read_text(encoding="utf-8"))
    validate(data)
    by_id = {s["id"]: s for s in data["songs"]}
    assert by_id["medium"]["full_title"].startswith("Medium (")
    assert "full_title" not in by_id["fast"]
    assert Catalog.load(tmp_path / "b" / "catalog.json").songs[1].full_title == manifest[0]["full_title"]


def test_rebuild_is_byte_identical(sources, tmp_path):
    """Stable ids and deterministic files: an unchanged song keeps its sha256, so the app's cache stays valid."""
    raw, manifest = sources
    build_catalog(raw, tmp_path / "a", "Test", manifest, OFFICIAL)
    build_catalog(raw, tmp_path / "b", "Test", manifest, OFFICIAL)
    assert (tmp_path / "a" / "catalog.json").read_bytes() == (tmp_path / "b" / "catalog.json").read_bytes()


def _catalog(**level):
    lv = {"level": 1, "label": "Melody", "url": "songs/a/level1.mid", "sha256": "0" * 64, "notes": 3, **level}
    return {"name": "T", "version": 1, "songs": [{"id": "a", "title": "A", "levels": [lv]}]}


@pytest.mark.parametrize(
    "bad",
    [
        {"sha256": "abc"},
        {"difficulty": 0},
        {"difficulty": 11},
        {"url": "http://example.org/level1.mid"},
        {"url": "file:///etc/passwd"},
        {"url": "//example.org/level1.mid"},
        {"url": ""},
    ],
)
def test_validate_rejects_bad_levels(bad):
    validate(_catalog())
    validate(_catalog(url="https://example.org/level1.mid"))
    with pytest.raises(ValueError):
        validate(_catalog(**bad))


def test_validate_rejects_bad_songs():
    data = _catalog()
    data["songs"].append(dict(data["songs"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        validate(data)
    data = _catalog()
    data["songs"][0]["id"] = "Has Spaces"
    with pytest.raises(ValueError, match="bad song id"):
        validate(data)


def test_load_ignores_unknown_fields_and_accepts_missing_optional_ones(tmp_path):
    data = _catalog()
    data["songs"][0]["future_field"] = {"x": 1}
    data["songs"][0]["levels"][0]["future_level_field"] = True
    (tmp_path / "catalog.json").write_text(json.dumps(data))
    cat = Catalog.load(tmp_path / "catalog.json")
    assert cat.songs[0].tags == [] and cat.songs[0].levels[0].difficulty is None
    assert "difficulty" not in cat.to_json()


def test_difficulty_rises_with_speed_and_chords():
    def notes(step, chord, beats=64):
        out = []
        for i in range(beats):
            for k in range(chord):
                out.append(Note(i * step, i * step + step, 60 + 2 * (i % 5) + 4 * k, hand="R"))
                out.append(Note(i * step, i * step + step, 48 - 3 * k, hand="L"))
        return out

    def score(n):
        return difficulty_score(n, lambda tick: tick / TPB * 0.5, TPB)

    slow, fast, chords = score(notes(TPB, 1)), score(notes(TPB // 4, 1)), score(notes(TPB, 3))
    assert 1 <= slow < fast <= 10
    assert slow < chords


def test_process_without_policy_keeps_small_files(tmp_path):
    from pianopipe.process import process_file

    entry = process_file(two_track(tmp_path / "tiny.mid", bars=1), tmp_path / "out")
    assert entry.levels
    with pytest.raises(QualityError):
        process_file(tmp_path / "tiny.mid", tmp_path / "out", policy=GatePolicy())


def test_cli_build_official_and_validate(sources, tmp_path, capsys):
    raw, manifest = sources
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    out = tmp_path / "cat"
    assert main(["build", str(raw), str(out), "--name", "T", "--sources", str(tmp_path / "manifest.json"), "--official"]) == 0
    assert "skipped set/one_track.mid: hands not in the file's structure" in capsys.readouterr().out
    assert main(["validate", str(out / "catalog.json")]) == 0
    (out / "songs" / "good" / "level1.mid").write_bytes(b"x")
    assert main(["validate", str(out / "catalog.json")]) == 1
