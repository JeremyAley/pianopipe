import json
from pathlib import Path

import mido
import pytest

from pianopipe import features, grading
from pianopipe.catalog import Catalog, Level, SongEntry, validate
from pianopipe.cli import main
from pianopipe.model import Note
from pianopipe.process import build_catalog, process_file
from pianopipe.quality import OFFICIAL

from helpers import TPB, make_midi, two_hand_piece

ROOT = Path(__file__).resolve().parents[2]


def secs(tick):
    return tick / TPB * 0.5  # 120 quarter notes per minute


def meta(*msgs):
    return [(t, m) for t, m in msgs]


# ---- statistics ----

def test_ranks_and_spearman():
    assert grading.ranks([10, 20, 20, 5]) == [2, 3.5, 3.5, 1]
    assert grading.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1)
    assert grading.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1)


def test_isotonic_pools_violators_and_interpolates():
    knots = grading.isotonic_fit([1, 2, 3, 4], [1, 3, 2, 4])
    assert knots == [(1, 1), (2.5, 2.5), (4, 4)]
    assert grading.isotonic_predict(knots, 0) == 1  # flat beyond the ends
    assert grading.isotonic_predict(knots, 9) == 4
    assert grading.isotonic_predict(knots, 1.75) == pytest.approx(1.75)
    # Tied x values become one point.
    assert grading.isotonic_fit([1, 1, 2], [2, 4, 5]) == [(1, 3), (2, 5)]


def test_ridge_recovers_a_line_and_selection_skips_noise():
    rows = [{"a": float(i), "b": float((i * 7) % 5)} for i in range(20)]
    ys = [1 + 0.4 * r["a"] for r in rows]
    m = grading.ridge_fit(rows, ys, ["a"], 0.0)
    assert m.raw({"a": 5.0}) == pytest.approx(3.0)
    names, lam, err = grading.select_features(rows, ys, candidates=("a", "b"))
    assert names == ["a"] and err < 0.1


def test_model_clamps_and_round_trips(tmp_path):
    m = grading.GradeModel(["a"], [0.0], [1.0], [2.0], 5.0, 1.0)
    assert m.predict({"a": 10}) == 10.0 and m.predict({"a": -10}) == 0.0 and m.predict({"a": 0.26}) == 5.5
    m.save(tmp_path / "m.json")
    assert grading.GradeModel.load(tmp_path / "m.json") == m


def test_length_stops_counting_at_the_cap():
    m = grading.GradeModel(["log_notes"], [6.0], [1.0], [1.0], 5.0, 1.0, caps={"log_notes": 7.0})
    assert m.predict({"log_notes": 7.0}) == m.predict({"log_notes": 9.0}) == 6.0
    model = grading.default_model()
    assert model.caps.get("log_notes") == pytest.approx(grading.CAPS["log_notes"])


def test_features_are_clipped_to_the_anchors_range():
    rows = [{"a": float(x)} for x in range(5)]
    m = grading.ridge_fit(rows, [float(x) for x in range(5)], ["a"], 0.0)
    assert m.ranges == {"a": [0.0, 4.0]}
    assert m.predict({"a": 40.0}) == m.predict({"a": 4.0}) == 4.0


# ---- anchors and the checked-in model ----

def test_anchor_rule():
    assert grading.anchor_grade(None, None, 3) == 2
    assert grading.anchor_grade(4, None, 4) == 4  # ABRSM wins over RCM
    assert grading.anchor_grade(6, 7, None) == 6.5
    assert grading.anchor_grade(None, None, None) is None


def test_anchor_csv_follows_the_rule_and_names_catalog_pieces():
    anchors = grading.load_anchors()
    assert len(anchors) >= 26
    sources = ROOT / "data" / "catalog" / "sources.json"
    if not sources.exists():
        pytest.skip("the catalog manifest is only in the Quest Piano Trainer repo")
    manifest = {e["id"] for e in json.loads(sources.read_text(encoding="utf-8"))}
    assert set(anchors) <= manifest
    for a in anchors.values():
        assert 0 <= a.grade <= 10 and a.sources
    assert anchors["burgmuller-ballade"].grade == 4 and anchors["deb-clai"].grade == 8


def test_anchor_csv_rejects_a_wrong_grade(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("# c\nid,title,grade,abrsm,trinity,rcm,henle,sources,notes\nx,X,5,,,3,,R,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="rule"):
        grading.load_anchors(path)


def test_checked_in_model_is_small_and_positive():
    m = grading.GradeModel.load()
    assert 1 <= len(m.features) <= grading.MAX_FEATURES
    assert set(m.features) <= set(features.FEATURES)
    assert all(c > 0 for c in m.coef)
    assert m.fit["anchors"] >= 26 and m.fit["step2"]["nested_loo_mae"] < m.fit["step1"]["loo_mae"]
    report = grading.REPORT_MD.read_text(encoding="utf-8")
    assert str(m.fit["step2"]["nested_loo_mae"]) in report and "PSyllabus" in report


def test_fit_reports_both_steps():
    ids = [f"p{i}" for i in range(12)]
    rows = [{f: 0.0 for f in features.FEATURES} | {"difficulty": float(i % 4), "log_notes": float(i)} for i in range(12)]
    grades = [0.5 * i for i in range(12)]
    res = grading.fit(ids, ids, grades, rows)
    assert res.step2["features"] == ["log_notes"]
    assert res.step2["loo_mae"] < res.step1["loo_mae"]
    assert res.step1["misplaced"] and not res.step2["misplaced"]
    text = grading.report(res)
    assert "## Step 1" in text and "## Step 2" in text and "`log_notes`" in text


# ---- features ----

def line(pitches, step=TPB, hand="R", dur=None, start=0):
    return [Note(start + i * step, start + i * step + (dur or step), p, hand=hand) for i, p in enumerate(pitches)]


def test_keys_fit_from_notes_and_from_signatures():
    c_major = line([60, 62, 64, 65, 67, 69, 71, 72, 67, 64, 60] * 3)
    assert features.keys(c_major, []) == ["C"]
    a_minor = line([69, 71, 72, 74, 76, 77, 68, 69, 64, 69, 57, 69] * 3)
    assert features.keys(a_minor, []) == ["Am"]
    # A file that writes A minor as "C": the signature gives the accidentals, the notes the mode.
    sig = [(0, mido.MetaMessage("key_signature", key="C"))]
    assert features.keys(a_minor, sig) == ["Am"]
    flat = [(0, mido.MetaMessage("key_signature", key="Bb"))]
    assert features.keys(line([70, 72, 74, 75, 77, 79, 81, 82, 77, 70] * 3), flat) == ["Bb"]


def test_time_signatures_and_tempo():
    m = [
        (0, mido.MetaMessage("time_signature", numerator=3, denominator=4)),
        (0, mido.MetaMessage("set_tempo", tempo=500000)),  # 120 for 30 quarters = 15 s
        (30 * TPB, mido.MetaMessage("set_tempo", tempo=1000000)),  # 60 for 10 quarters = 10 s
        (30 * TPB, mido.MetaMessage("time_signature", numerator=6, denominator=8)),
        (30 * TPB, mido.MetaMessage("time_signature", numerator=3, denominator=4)),
    ]
    assert features.time_signatures(m) == ["3/4", "6/8"]
    assert features.tempo_bpm(m, 40 * TPB, TPB) == 120
    assert features.tempo_bpm(m, 50 * TPB, TPB) == 60  # 15 s at 120, 20 s at 60
    assert features.beat_grid([(0, mido.MetaMessage("time_signature", numerator=6, denominator=8))], TPB) == [(0, 3 * TPB // 2)]


def skills_of(notes, m=()):
    return [s for s in features.skills(notes, secs, TPB, list(m)) if ":" not in s]


def test_skills_tags():
    rh = line([60, 62, 64, 65, 67, 65, 64, 62] * 4)  # legato quarters in a five-finger position
    lh = line([48, 52] * 16, hand="L")
    assert skills_of(rh) == ["five-finger", "legato"]
    assert skills_of(rh + lh) == ["five-finger", "hands-together", "legato"]
    short = line([60, 64, 67, 72] * 8, dur=TPB // 3)
    assert "staccato" in skills_of(short) and "five-finger" not in skills_of(short)
    # Dotted quarter + eighth, eight times.
    dotted = [Note(t, t + d, 60, hand="R") for k in range(8) for t, d in ((2 * k * TPB, 3 * TPB // 2), (2 * k * TPB + 3 * TPB // 2, TPB // 2))]
    assert "dotted" in skills_of(dotted)
    triplets = line([60, 62, 64] * 12, step=TPB // 3)
    assert "triplets" in skills_of(triplets)
    # Off-beat notes held across the beat.
    sync = [Note(k * TPB + TPB // 2, k * TPB + 3 * TPB // 2, 60 + k % 3, hand="R") for k in range(20)]
    assert "syncopation" in skills_of(sync)
    chords = [n for i in range(16) for n in (Note(i * TPB, (i + 1) * TPB, 60, hand="R"), Note(i * TPB, (i + 1) * TPB, 64, hand="R"),
                                             Note(i * TPB, (i + 1) * TPB, 67, hand="R"))]
    assert "chords" in skills_of(chords)
    octaves = [n for i in range(16) for n in (Note(i * TPB, (i + 1) * TPB, 48, hand="L"), Note(i * TPB, (i + 1) * TPB, 60, hand="L"))]
    assert "octaves" in skills_of(octaves)
    trill = line([72, 74] * 8 + [72], step=TPB // 8) + line([60] * 4, start=4 * TPB)
    trill += line([72, 74] * 8 + [72], step=TPB // 8, start=8 * TPB)
    assert "ornaments" in skills_of(trill)
    tagged = features.skills(rh, secs, TPB, [(0, mido.MetaMessage("time_signature", numerator=2, denominator=4))])
    assert tagged[-2:] == ["key:C", "time:2/4"]


def test_character():
    lh = line([48, 55, 52, 55] * 16, hand="L")
    running = line([72, 74, 76, 77] * 64, step=TPB // 4)  # sixteenths at 120: 8 notes a second
    assert features.character(running + lh, secs, TPB) == "agile"
    melody = line([72, 74, 76, 77] * 8, step=2 * TPB)  # legato half notes over the bass
    assert features.character(melody + lh, secs, TPB) == "lyrical"
    assert features.character(line([72, 74] * 16, step=2 * TPB, dur=TPB // 2) + lh, secs, TPB) == "other"  # detached


def test_level_features_on_a_two_hand_piece():
    rh, lh = two_hand_piece(8)
    notes = [Note(s, s + d, p, hand="R") for s, d, p in rh] + [Note(s, s + d, p, hand="L") for s, d, p in lh]
    f = features.level_features(notes, secs, TPB)
    assert set(f) == set(features.FEATURES)
    assert 0 < f["outside_key"] < 0.5 and f["pitch_range"] == 79 - 48  # the lower melody line adds accidentals
    assert f["independence"] > 0  # the bass moves while the right hand holds
    assert f["polyphony"] == 0


# ---- catalog fields and the build ----

def test_new_fields_written_only_when_set_and_validated():
    lv = Level(1, "Melody", "songs/a/level1.mid", "0" * 64, 3)
    cat = Catalog("T", [SongEntry("a", "A", levels=[lv])])
    text = cat.to_json()
    for key in ("grade", "grade_source", "tempo_bpm", "skills", "character"):
        assert f'"{key}"' not in text
    lv.grade, lv.grade_source, lv.tempo_bpm, lv.skills = 2.5, "estimated", 96, ["legato", "key:G"]
    cat.songs[0].character = "lyrical"
    data = json.loads(cat.to_json())
    validate(data)
    assert data["songs"][0]["character"] == "lyrical" and data["songs"][0]["levels"][0]["skills"] == ["legato", "key:G"]


@pytest.mark.parametrize(
    "where,bad",
    [
        ("level", {"grade": 10.5}),
        ("level", {"grade": -1}),
        ("level", {"grade": 3, "grade_source": "guess"}),
        ("level", {"grade_source": "estimated"}),
        ("level", {"tempo_bpm": 0}),
        ("level", {"skills": "legato"}),
        ("song", {"character": "fast"}),
    ],
)
def test_validate_rejects_bad_new_fields(where, bad):
    data = {"name": "T", "version": 1, "songs": [{"id": "a", "title": "A", "levels": [
        {"level": 1, "label": "Melody", "url": "x.mid", "sha256": "0" * 64, "notes": 3}]}]}
    validate(data)
    (data["songs"][0]["levels"][0] if where == "level" else data["songs"][0]).update(bad)
    with pytest.raises(ValueError):
        validate(data)


def two_track(path, bars=10, **kw):
    rh, lh = two_hand_piece(bars)
    return make_midi(path, [{"name": "Piano right", "notes": rh}, {"name": "Piano left", "notes": lh, "channel": 1}], **kw)


def test_build_grades_every_level_and_uses_reference_for_anchors(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    two_track(raw / "a.mid")
    two_track(raw / "b.mid")
    manifest = [{"id": "a", "file": "a.mid", "title": "A"}, {"id": "b", "file": "b.mid", "title": "B", "character": "agile"}]
    result = build_catalog(raw, tmp_path / "cat", "T", manifest, OFFICIAL)
    data = json.loads((tmp_path / "cat" / "catalog.json").read_text(encoding="utf-8"))
    validate(data)
    a, b = data["songs"]
    for lv in a["levels"]:
        assert lv["grade_source"] == "estimated" and 0 <= lv["grade"] <= 10
        assert lv["tempo_bpm"] == 100  # the helper's 600000 microseconds a quarter
        assert "hands-together" in lv["skills"] or lv["level"] == 1
        assert sum(s.startswith("key:") for s in lv["skills"]) == 1 and not any(s.startswith("time:") for s in lv["skills"])
    grades = [lv["grade"] for lv in a["levels"]]
    assert grades == sorted(grades)
    assert a["character"] in ("agile", "lyrical", "other") and b["character"] == "agile"
    assert result.report()["built"][0]["grade"] == grades

    anchors = {"a": grading.Anchor("a", "A", 1.5, rcm=2)}
    entry = process_file(raw / "a.mid", tmp_path / "out", {"id": "a"}, anchors=anchors)
    assert [(lv.grade_source, lv.grade) for lv in entry.levels][-1] == ("reference", 1.5)
    assert all(lv.grade_source == "estimated" and lv.grade <= 1.5 for lv in entry.levels[:-1])
    plain = process_file(raw / "a.mid", tmp_path / "out2", {"id": "a"}, grade_model=None, anchors={})
    assert all(lv.grade is None and lv.grade_source == "" for lv in plain.levels)


def test_cli_grade_fit_writes_model_and_report(tmp_path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows, manifest = [], []
    for i in range(6):
        two_track(raw / f"p{i}.mid", bars=6 + 6 * i)  # longer pieces get higher grades
        manifest.append({"id": f"p{i}", "file": f"p{i}.mid", "title": f"P{i}"})
        rows.append(f"p{i},P{i},{i + 1},{i + 1},,,,A25,")
    (tmp_path / "sources.json").write_text(json.dumps(manifest))
    (tmp_path / "anchors.csv").write_text("# test\nid,title,grade,abrsm,trinity,rcm,henle,sources,notes\n" + "\n".join(rows) + "\n")
    args = [str(raw), "--sources", str(tmp_path / "sources.json"), "--anchors", str(tmp_path / "anchors.csv"),
            "--model", str(tmp_path / "m.json"), "--report", str(tmp_path / "r.md")]
    assert main(["grade-fit", *args]) == 0
    model = grading.GradeModel.load(tmp_path / "m.json")
    assert "log_notes" in model.features
    assert "grade fit on 6 anchors" in capsys.readouterr().out
    assert "# Grade fit" in (tmp_path / "r.md").read_text(encoding="utf-8")
