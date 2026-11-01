"""Sight-reading pieces (sightread.py): each grade's parameters (docs/learning-mode.md §12.2, §1.2), checked on the
generated notes and on the notation the sheet shows, the engraved page, and the reading collection's catalog."""

from __future__ import annotations

import json
import re
from collections import Counter

import pytest

from pianopipe import engrave, notation, sightread
from pianopipe.catalog import Catalog, validate, verify
from pianopipe.sightread import GRADES, generate

needs_engraver = pytest.mark.skipif(not engrave.available(), reason="pip install pianopipe[sheet]")

SEEDS = range(1, 41)  # pieces checked per grade
PIECES = {g: [generate(g, s) for s in SEEDS] for g in GRADES}
WHITE = [0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6]


def white(pitch: int) -> int:
    return 7 * (pitch // 12) + WHITE[pitch % 12]


def written_values(piece) -> set[tuple[str, int]]:
    return {(t, len(d) // 6) for t, d in re.findall(r"<type>(\w+)</type>((?:<dot/>)?)", sightread.written(piece))}


def test_the_grades_follow_the_boards_table():
    """§12.2: Initial C major and D minor, 4/4 or 2/4, 4 or 6 bars, hands separately in five-finger position; Grade 1
    adds 3/4 and G, F, A minor; Grade 2 hands together; Grade 3 3/8 and out of position, up to 8 bars; Grade 4 6/8 and
    an anacrusis; Grade 5 8-12 bars, syncopation. Each grade keeps everything the one before had."""
    g = GRADES
    assert set(g[0].keys) == {"C", "Dm"}
    assert {(n, d) for n, d, _ in g[0].metres} == {(4, 4), (2, 4)} and g[0].bars == {(4, 4): (4,), (2, 4): (6,)}
    assert set(g[1].keys) - set(g[0].keys) == {"G", "F", "Am"}
    assert (3, 4) in g[1].bars and (3, 8) not in g[2].bars and (3, 8) in g[3].bars
    assert (6, 8) not in g[3].bars and (6, 8) in g[4].bars
    assert [x.hands_together for x in g.values()] == [False, False, True, True, True, True]
    assert [x.five_finger for x in g.values()] == [True, True, True, False, False, False]
    assert max(b for bs in g[3].bars.values() for b in bs) == 8
    assert {b for bs in g[5].bars.values() for b in bs} <= set(range(8, 13))
    assert not g[3].upbeat and g[4].upbeat and not g[4].syncopation and g[5].syncopation
    # Keys to two sharps or flats at Grade 3, three at 4, four at 5.
    sig = {k: abs(notation.key_fifths(k)) for k in g[5].keys}
    for grade, most in ((2, 2), (3, 2), (4, 3), (5, 4)):
        assert max(sig[k] for k in g[grade].keys) == most
    for lower, higher in zip(list(g.values()), list(g.values())[1:]):
        assert set(lower.keys) <= set(higher.keys)
        assert set(lower.bars) <= set(higher.bars)
        assert lower.values <= higher.values
    assert ("16th", 0) not in g[3].values and ("16th", 0) in g[4].values
    assert ("eighth", 0) not in g[0].values and ("eighth", 0) in g[1].values


@pytest.mark.parametrize("grade", list(GRADES))
def test_every_piece_is_inside_its_grade(grade):
    spec = GRADES[grade]
    for p in PIECES[grade]:
        where = f"grade {grade} seed {p.seed}"
        assert p.key in spec.keys, where
        assert (p.num, p.den) in spec.bars and p.bars in spec.bars[(p.num, p.den)], where
        assert not p.upbeat or spec.upbeat, where
        values = written_values(p)
        assert values <= spec.values, f"{where}: {values - spec.values}"
        if not spec.ties:
            assert "<tie " not in sightread.written(p), where
        for hand, (lo, hi) in (("R", spec.rh), ("L", spec.lh)):
            ps = [n.pitch for n in p.notes if n.hand == hand]
            if ps:
                assert lo <= min(ps) and max(ps) <= hi, f"{where} {hand}: {min(ps)}-{max(ps)}"
                assert max(ps) - min(ps) <= spec.span, where
                if spec.five_finger:  # one position: five neighbouring keys, a finger each
                    keys = sorted({white(x) for x in ps})
                    assert keys[-1] - keys[0] <= 4, f"{where} {hand} spans {keys}"
        assert all(n.finger in (1, 2, 3, 4, 5) for n in p.notes), where
        keys = Counter((n.hand, n.start, n.pitch) for n in p.notes)
        assert max(keys.values()) == 1, f"{where}: a key struck twice at once (one head on the page)"
        # The tune moves by at most the grade's leap (Initial: steps and thirds).
        for hand in ("R",) if spec.hands_together else ("R", "L"):
            tops = {}
            for n in p.notes:
                if n.hand == hand:
                    tops[n.start] = max(tops.get(n.start, 0), n.pitch)
            line = [white(tops[t]) for t in sorted(tops)]
            assert all(abs(a - b) <= spec.leap for a, b in zip(line, line[1:])), f"{where} {hand}: {line}"
        end = max(n.start for n in p.notes)
        top = max(n.pitch for n in p.notes if n.end > end and (n.hand == "R" or not spec.hands_together))
        assert (top - sightread.PC[p.key.rstrip("m")]) % 12 == 0, f"{where}: ends off the tonic"


@pytest.mark.parametrize("grade", [0, 1])
def test_hands_separately_never_overlap(grade):
    for p in PIECES[grade]:
        rh = [(n.start, n.end) for n in p.notes if n.hand == "R"]
        for n in p.notes:
            if n.hand == "L":
                assert not any(a < n.end and n.start < b for a, b in rh), f"seed {p.seed}"


@pytest.mark.parametrize("grade", [2, 3, 4, 5])
def test_hands_together_from_grade_2(grade):
    for p in PIECES[grade]:
        assert p.hands == "together"
        assert {n.hand for n in p.notes} == {"R", "L"}
        bars = {(n.start - int(p.upbeat * 480)) // int(480 * 4 * p.num / p.den) for n in p.notes if n.hand == "L"}
        assert len(bars) == p.bars, f"seed {p.seed}: the left hand plays in every bar"


def test_five_finger_fingers_follow_the_keys():
    """In one position the fingers go up with the right hand's keys and down with the left's."""
    for grade in (0, 1, 2):
        for p in PIECES[grade]:
            for hand in "RL":
                by_key = {}
                for n in p.notes:
                    if n.hand == hand:
                        by_key.setdefault(n.pitch, set()).add(n.finger)
                assert all(len(f) == 1 for f in by_key.values()), f"grade {grade} seed {p.seed}"
                order = [next(iter(by_key[k])) for k in sorted(by_key)]
                assert order == sorted(order, reverse=hand == "L"), f"grade {grade} seed {p.seed}"


def test_the_bank_covers_each_grade():
    """Across 150 pieces every key and metre of a grade comes up, and the later grades' features appear."""
    for grade in GRADES:
        spec = GRADES[grade]
        pieces = PIECES[grade] + [generate(grade, s) for s in range(SEEDS.stop, 151)]
        assert {p.key for p in pieces} == set(spec.keys), grade
        assert {(p.num, p.den) for p in pieces} == set(spec.bars), grade
    assert any(p.upbeat for p in PIECES[4])
    assert any(("16th", 0) in written_values(p) for p in PIECES[4])
    assert any("<tie " in sightread.written(p) for p in PIECES[3])
    assert any("<staccato/>" in sightread.written(p) for p in PIECES[3])
    assert any(Counter((n.hand, n.start) for n in p.notes if n.hand == "L").most_common(1)[0][1] == 2 for p in PIECES[4])
    # Grade 5's chromatic notes: a note outside the key that isn't a minor key's raised sixth or seventh.
    def chromatic(p):
        scale = sightread.MINOR if p.minor else sightread.MAJOR
        tonic = sightread.PC[p.key.rstrip("m")]
        raised = {(tonic + 9) % 12, (tonic + 11) % 12} if p.minor else set()
        return any((n.pitch - tonic) % 12 not in scale and n.pitch % 12 not in raised for n in p.notes)
    assert sum(chromatic(p) for p in PIECES[5]) >= 5
    assert not any(chromatic(p) for g in (0, 1, 2, 3, 4) for p in PIECES[g])


def test_same_seed_same_piece():
    assert generate(3, 11).fingerprint() == generate(3, 11).fingerprint()
    assert generate(3, 11).fingerprint() != generate(3, 12).fingerprint()
    with pytest.raises(ValueError):
        generate(6, 1)


def test_printed_fingers_mark_the_entries():
    """The sheet prints a hand's first finger (and where it comes back or moves), not every one."""
    for grade in GRADES:
        for p in PIECES[grade]:
            for hand in "RL":
                notes = sorted((n for n in p.notes if n.hand == hand), key=lambda n: (n.start, -n.pitch))
                if notes:
                    assert notes[0].printed, f"grade {grade} seed {p.seed}"
        shown = sum(n.printed for p in PIECES[grade] for n in p.notes) / sum(len(p.notes) for p in PIECES[grade])
        assert shown < 0.4, f"grade {grade}: {shown:.0%} of fingers printed"


def test_reading_collection_validates():
    data = {"name": "r", "version": 1, "collection": "reading", "songs": []}
    validate(data)
    assert Catalog.from_dict(data).collection == "reading"
    assert '"collection": "reading"' in Catalog.from_dict(data).to_json()
    assert "collection" not in Catalog(name="x").to_json()
    with pytest.raises(ValueError, match="collection"):
        validate({**data, "collection": "drills"})


@needs_engraver
def test_a_piece_engraves_one_page_with_its_bars(tmp_path):
    entry = sightread.build_piece(4, 2, 1, tmp_path)
    piece = generate(4, 2)
    lv = entry["levels"][0]
    assert entry["id"] == "sr-g4-001" and lv["grade"] == 4.0 and lv["grade_source"] == "reference" and lv["fingering"]
    assert entry["license"] == "CC0-1.0" and "--grade 4 --seed 2" in entry["attribution"]
    sheet = json.loads((tmp_path / lv["sheet"]["url"]).read_text(encoding="utf-8"))
    assert lv["sheet"]["pages"] == 1 and sheet["level"] == 1
    assert sheet["bars"] == piece.bars + (1 if piece.upbeat else 0)
    heads = [sheet["measures"][i]["notes"][k] for i in range(sheet["bars"]) for k in range(1, len(sheet["measures"][i]["notes"]), 4)]
    # A head for every key struck (a tie's far end is the same key held), so the app can mark each one.
    assert sorted(int(h) for h in heads) == sorted(n.pitch for n in piece.notes)
    assert f"<key><fifths>{notation.key_fifths(piece.key)}</fifths></key>" in sightread.written(piece)


@needs_engraver
def test_dynamics_reach_the_notation(tmp_path):
    p = next(p for p in PIECES[2] if len(p.dynamics) > 1)
    p.write(tmp_path, "level1")
    # engrave_level adds the markings to the MusicXML it engraves; check them there.
    from pianopipe import midi_io

    song = midi_io.load(tmp_path / "level1.mid")
    xml = sightread._with_dynamics(notation.to_musicxml(notation.build(song, 1)), p.dynamics, 2)
    measures = re.findall(r"<measure\b.*?</measure>", xml, flags=re.S)
    for i, mark in p.dynamics:
        assert f"<dynamics><{mark}/></dynamics>" in measures[i]
    assert xml.count("<dynamics>") == len(p.dynamics)


@needs_engraver
def test_a_small_bank_is_a_valid_reading_catalog(tmp_path):
    cat = sightread.build_bank(tmp_path, count=2, grades=[0, 5], jobs=2, log=lambda *_: None)
    assert cat.collection == "reading"
    assert [s.id for s in cat.songs] == ["sr-g0-001", "sr-g0-002", "sr-g5-001", "sr-g5-002"]
    assert verify(tmp_path / "catalog.json") == []
    assert Catalog.load(tmp_path / "catalog.json").collection == "reading"
    assert all(s.levels[0].sheet is not None for s in cat.songs)
