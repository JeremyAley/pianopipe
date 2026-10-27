"""Sheet music: notation from app-ready levels (notation.py), engraving (engrave.py), the catalog's `sheet` field, and
pickup bars from MusicXML lining up with the app's bar grid (docs/sheet-music.md)."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mido
import pytest

from pianopipe import catalog, engrave, features, midi_io, musicxml, notation
from pianopipe.catalog import Catalog, validate, verify, verify_sheet
from pianopipe.model import Note
from pianopipe.process import process_file

from helpers import TPB, make_midi, two_hand_piece

needs_engraver = pytest.mark.skipif(not engrave.available(), reason="pip install pianopipe[sheet]")


def app_midi(path: Path, rh: list, lh: list, meta: list | None = None) -> Path:
    """An app-ready level: (start, dur, pitch[, finger]) per hand, with conductor meta events [(tick, MetaMessage)]."""
    def notes(spec, hand):
        return [Note(s, s + d, p, 80, hand, finger=(f[0] if f else None)) for s, d, p, *f in spec]
    midi_io.write_app_midi(path, TPB, meta or [], notes(rh, "R") + notes(lh, "L"))
    return path


def ts(tick: int, num: int, den: int):
    return (tick, mido.MetaMessage("time_signature", numerator=num, denominator=den))


def ks(tick: int, key: str):
    return (tick, mido.MetaMessage("key_signature", key=key))


def score_xml(tmp_path: Path, rh: list, lh: list, meta=None, level: int = 4, fingering: bool = True):
    song = midi_io.load(app_midi(tmp_path / "level.mid", rh, lh, meta))
    sc = notation.build(song, level)
    return sc, notation.to_musicxml(sc, fingering)


def reread(tmp_path: Path, xml: str):
    """Our own MusicXML reader on the written notation: (rh, lh) as (start, pitch) in quarters, and the score."""
    path = tmp_path / "out.musicxml"
    path.write_text(xml, encoding="utf-8")
    sc = musicxml.load(path)
    q = musicxml.TPB

    def hand(h):
        return sorted((n.start / q, n.pitch) for n in sc.notes if n.hand == h)
    return hand("R"), hand("L"), sc


# --- the bar grid -------------------------------------------------------------------------------------------------


def test_bar_grid_matches_the_app():
    # 4/4 until the first time signature; 3/4 from bar 2; a 2/4 that starts in the middle of a 3/4 bar ends that bar.
    meta = [ts(4 * TPB, 3, 4), ts(4 * TPB + 3 * TPB + TPB, 2, 4)]
    bars = notation.bar_grid(meta, TPB, 4 * TPB + 3 * TPB + TPB + 2 * TPB + 10)
    assert [(b.start, b.end, b.num, b.den) for b in bars] == [
        (0, 4 * TPB, 4, 4), (4 * TPB, 7 * TPB, 3, 4), (7 * TPB, 8 * TPB, 3, 4), (8 * TPB, 10 * TPB, 2, 4),
        (10 * TPB, 12 * TPB, 2, 4)]
    assert [b.number for b in bars] == [1, 2, 3, 4, 5]


def test_last_note_ending_on_a_bar_line_adds_no_bar():
    bars = notation.bar_grid([], TPB, 8 * TPB)
    assert len(bars) == 2 and bars[-1].end == 8 * TPB


# --- spelling ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("pitch,fifths,minor,step_to,expected", [
    (61, 0, False, 0, ("C", 1, 4)),   # C# in C
    (63, 0, False, 0, ("E", -1, 4)),  # Eb in C
    (63, 0, False, 1, ("D", 1, 4)),   # D# stepping up to E (Für Elise)
    (63, 0, True, 0, ("D", 1, 4)),    # A minor: D# (E major's third)
    (68, 0, True, 0, ("G", 1, 4)),    # A minor's leading tone
    (70, 0, True, 0, ("B", -1, 4)),   # but Bb stays flat in A minor
    (70, -1, False, 0, ("B", -1, 4)), # F major's Bb is in the key
    (66, 2, False, 0, ("F", 1, 4)),   # D major's F#
    (71, -7, False, 0, ("C", -1, 5)), # Cb in Cb major (octave from the written note)
    (68, 0, False, -1, ("A", -1, 4)), # Ab falling to G
])
def test_spelling(pitch, fifths, minor, step_to, expected):
    assert notation.spell(pitch, fifths, minor, step_to) == expected


def test_key_signature_and_accidentals_last_the_bar(tmp_path):
    rh = [(0, TPB, 66), (TPB, TPB, 66), (2 * TPB, TPB, 65), (3 * TPB, TPB, 66), (4 * TPB, TPB, 66)]
    _, xml = score_xml(tmp_path, rh, [], [ks(0, "C")])
    root = ET.fromstring(xml)
    measures = root.findall(".//measure")
    acc = [[n.findtext("accidental") for n in m.findall("note")] for m in measures]
    # F# shown once in bar 1, the natural on F, F# again; and again at the start of bar 2.
    assert acc[0] == ["sharp", None, "natural", "sharp"]
    assert acc[1][0] == "sharp"


# --- rhythm, voices, round trip ----------------------------------------------------------------------------------


def test_round_trip_keeps_every_onset_and_pitch(tmp_path):
    rh, lh = two_hand_piece(4)
    rh = [(s, d, p) for s, d, p in rh]
    sc, xml = score_xml(tmp_path, rh, lh, [ts(0, 4, 4)])
    got_rh, got_lh, _ = reread(tmp_path, xml)
    assert got_rh == sorted((s / TPB, p) for s, _, p in rh)
    assert got_lh == sorted((s / TPB, p) for s, _, p in lh)
    assert len(ET.fromstring(xml).findall(".//measure")) == 4


def test_notes_across_bar_lines_and_beats_are_tied(tmp_path):
    # A note from beat 4 lasting two beats; an off-beat quarter from 1.5 to 2.5.
    rh = [(3 * TPB, 2 * TPB, 72), (5 * TPB + TPB // 2, TPB, 74)]
    _, xml = score_xml(tmp_path, rh, [], [ts(0, 4, 4)])
    root = ET.fromstring(xml)
    ties = [t.get("type") for t in root.iter("tie")]
    assert ties.count("start") == 2 and ties.count("stop") == 2
    got, _, sc = reread(tmp_path, xml)
    assert got == [(3, 72), (5.5, 74)]
    durs = sorted((n.start / musicxml.TPB, (n.end - n.start) / musicxml.TPB) for n in sc.notes)
    assert durs == [(3, 2), (5.5, 1)]  # the ties join up again


def test_rests_start_on_their_own_length(tmp_path):
    # Beat 1 played, then three beats of rest: a quarter rest and a half rest, not a dotted half rest.
    _, xml = score_xml(tmp_path, [(0, TPB, 72), (4 * TPB, 4 * TPB, 72)], [], [ts(0, 4, 4)])
    first = ET.fromstring(xml).find(".//measure")
    rests = [(n.findtext("type"), n.find("dot") is not None) for n in first.findall("note") if n.find("rest") is not None]
    assert rests == [("quarter", False), ("half", False)]


def test_triplets_are_written_as_tuplets(tmp_path):
    third = TPB // 3
    rh = [(i * third, third, 72 + i % 3) for i in range(6)] + [(2 * TPB, 2 * TPB, 79)]
    _, xml = score_xml(tmp_path, rh, [], [ts(0, 4, 4)])
    root = ET.fromstring(xml)
    mods = root.findall(".//time-modification")
    assert len(mods) == 6
    assert [t.get("type") for t in root.iter("tuplet")] == ["start", "stop", "start", "stop"]
    got, _, _ = reread(tmp_path, xml)
    assert [round(s * 3) for s, _ in got] == [0, 1, 2, 3, 4, 5, 6]


def test_a_played_short_quarter_is_a_staccato_quarter(tmp_path):
    rh = [(0, TPB * 3 // 8, 67), (TPB, TPB * 3 // 8, 67), (2 * TPB, 2 * TPB, 72)]
    _, xml = score_xml(tmp_path, rh, [], [ts(0, 4, 4)])
    first = ET.fromstring(xml).find(".//measure")
    notes = [(n.findtext("type"), n.find(".//staccato") is not None) for n in first.findall("note") if n.find("rest") is None]
    assert notes == [("quarter", True), ("quarter", True), ("half", False)]


def test_held_melody_over_moving_notes_gets_a_second_voice(tmp_path):
    # Level 4: a half note held over four eighths in the same hand: two voices, stems up and down.
    rh = [(0, 2 * TPB, 79)] + [(i * TPB // 2, TPB // 2, 64 + i) for i in range(4)]
    _, xml = score_xml(tmp_path, rh, [], [ts(0, 4, 4)])
    first = ET.fromstring(xml).find(".//measure")
    voices = {n.findtext("voice") for n in first.findall("note")}
    assert voices == {"1", "2"}
    assert {n.findtext("stem") for n in first.findall("note") if n.find("rest") is None} == {"up", "down"}
    got, _, _ = reread(tmp_path, xml)
    assert sorted(p for _, p in got) == sorted([79, 64, 65, 66, 67])


def test_levels_one_and_two_are_one_line_per_hand(tmp_path):
    rh = [(0, 2 * TPB, 79), (TPB, TPB, 77)]  # overlapping: level 1 cuts the first note where the second starts
    _, xml = score_xml(tmp_path, rh, [], [ts(0, 4, 4)], level=1)
    first = ET.fromstring(xml).find(".//measure")
    assert {n.findtext("voice") for n in first.findall("note")} == {"1"}
    assert [n.findtext("type") for n in first.findall("note") if n.find("rest") is None] == ["quarter", "quarter"]


def test_one_hand_gets_one_staff_and_fingers_follow_the_level(tmp_path):
    rh = [(0, TPB, 60, 1), (TPB, TPB, 62, 2), (2 * TPB, TPB, 64, 3), (3 * TPB, TPB, 65, 4)]
    _, xml = score_xml(tmp_path, rh, [], [ts(0, 4, 4)])
    root = ET.fromstring(xml)
    assert root.find(".//staves") is None and len(root.findall(".//clef")) == 1
    assert [f.text for f in root.iter("fingering")] == ["1", "2", "3", "4"]
    _, plain = score_xml(tmp_path, rh, [], [ts(0, 4, 4)], fingering=False)
    assert ET.fromstring(plain).find(".//fingering") is None
    assert engrave.fingering_for(1) and engrave.fingering_for(3) and not engrave.fingering_for(4)


def test_left_hand_crossing_high_changes_clef_for_those_bars(tmp_path):
    lh = [(0, 4 * TPB, 43), (4 * TPB, TPB, 76), (5 * TPB, TPB, 79), (6 * TPB, 2 * TPB, 84), (8 * TPB, 4 * TPB, 43)]
    sc, xml = score_xml(tmp_path, [(0, 12 * TPB, 72)], lh, [ts(0, 4, 4)])
    assert sc.staves[1].clefs == ["F", "G", "F"]
    signs = [(c.get("number"), c.findtext("sign")) for c in ET.fromstring(xml).iter("clef")]
    assert signs == [("1", "G"), ("2", "F"), ("2", "G"), ("2", "F")]


def test_a_short_first_bar_is_a_pickup(tmp_path):
    meta = [ts(0, 1, 4), ts(TPB, 3, 4)]
    rh = [(0, TPB, 67)] + [(TPB + i * TPB, TPB, 72) for i in range(6)]
    sc, xml = score_xml(tmp_path, rh, [], meta)
    assert sc.pickup and len(sc.bars) == 3
    root = ET.fromstring(xml)
    first = root.find(".//measure")
    assert first.get("implicit") == "yes"
    assert [(t.findtext("beats"), t.findtext("beat-type")) for t in root.iter("time")] == [("3", "4")]


# --- MusicXML pickups line up with the app's bars ---------------------------------------------------------------


def test_musicxml_pickup_gets_its_own_time_signature(tmp_path):
    def n(step, octave, dur):
        return f"<note><pitch><step>{step}</step><octave>{octave}</octave></pitch><duration>{dur}</duration><voice>1</voice></note>"
    xml = ('<?xml version="1.0"?><score-partwise version="3.1"><part-list><score-part id="P1"><part-name>Piano</part-name>'
           '</score-part></part-list><part id="P1">'
           '<measure number="0" implicit="yes"><attributes><divisions>4</divisions><key><fifths>0</fifths></key>'
           '<time><beats>3</beats><beat-type>4</beat-type></time></attributes>' + n("G", 4, 4) + '</measure>'
           '<measure number="1">' + n("C", 5, 12) + '</measure>'
           '<measure number="2">' + n("E", 5, 8) + '</measure></part></score-partwise>')
    path = tmp_path / "pickup.musicxml"
    path.write_text(xml, encoding="utf-8")
    sc = musicxml.load(path)
    q = musicxml.TPB
    sigs = [(t, m.numerator, m.denominator) for t, m in sc.song.meta if m.type == "time_signature"]
    assert sigs == [(0, 1, 4), (q, 3, 4), (4 * q, 2, 4)]  # the pickup, the meter, the last bar's two beats
    bars = notation.bar_grid(sc.song.meta, q, max(nn.end for nn in sc.notes))
    assert [b.start for b in bars] == [0, q, 4 * q]  # bar lines where the score has them
    assert features.time_signatures(sc.song.meta, max(nn.end for nn in sc.notes), q) == ["3/4"]


# --- engraving and the catalog -----------------------------------------------------------------------------------


@needs_engraver
def test_engrave_writes_pages_and_a_bar_map(tmp_path):
    rh, lh = two_hand_piece(12)
    level = app_midi(tmp_path / "level4.mid", rh, lh, [ts(0, 4, 4), ks(0, "C")])
    sheet = engrave.engrave(level, tmp_path / "out", "level4", 4)
    data = json.loads(sheet.json_path.read_text(encoding="utf-8"))
    assert data["version"] == 1 and data["bars"] == 12 and data["level"] == 4
    assert sheet.pages == len(data["pages"]) >= 1
    assert [m["bar"] for m in data["measures"]] == list(range(1, 13))
    for page in data["pages"]:
        png = (tmp_path / "out" / page["file"]).read_bytes()
        assert png[:8] == b"\x89PNG\r\n\x1a\n" and png[25] == 0  # 8-bit greyscale, what the app's decoder reads
        assert catalog.hashlib.sha256(png).hexdigest() == page["sha256"]
    for m in data["measures"]:
        sys_ = data["systems"][m["system"]]
        assert 0 < m["left"] < m["right"] < 1 and 0 < sys_["top"] < sys_["bottom"] < 1
        fr = m["onsets"][0::2]
        assert fr == sorted(fr) and {0, 0.25, 0.5, 0.75} <= set(fr)  # every beat of the bar has a start
        assert all(m["left"] < x < m["right"] for x in m["onsets"][1::2])
    # Systems go down the page, bars along a system.
    first = [m for m in data["measures"] if m["system"] == 0]
    assert [m["left"] for m in first] == sorted(m["left"] for m in first)
    # The same level engraves to the same files (the app keeps pages whose sha256 hasn't changed).
    again = engrave.engrave(level, tmp_path / "out2", "level4", 4)
    assert again.sha256 == sheet.sha256


@needs_engraver
def test_process_file_with_sheets_and_the_catalog_checks_them(tmp_path):
    rh, lh = two_hand_piece(8)
    src = make_midi(tmp_path / "piece.mid", [{"name": "RH", "channel": 0, "notes": rh}, {"name": "LH", "channel": 1, "notes": lh}])
    entry = process_file(src, tmp_path / "songs", {"id": "piece", "title": "Piece"}, "songs/", sheets=True)
    assert [lv.sheet.url for lv in entry.levels] == [f"songs/piece/level{n}.sheet.json" for n in (1, 2, 3, 4)]
    cat = Catalog(name="test", songs=[entry])
    cat.save(tmp_path / "catalog.json")
    assert verify(tmp_path / "catalog.json") == []
    again = Catalog.load(tmp_path / "catalog.json")
    assert again.songs[0].levels[0].sheet.pages == entry.levels[0].sheet.pages
    # A changed page is caught.
    page = tmp_path / "songs" / "piece" / "level1-p1.png"
    page.write_bytes(page.read_bytes() + b"x")
    assert any("level1-p1.png" in p for p in verify(tmp_path / "catalog.json"))


def test_catalog_without_sheets_stays_as_it_was(tmp_path):
    rh, lh = two_hand_piece(4)
    src = make_midi(tmp_path / "piece.mid", [{"name": "RH", "channel": 0, "notes": rh}, {"name": "LH", "channel": 1, "notes": lh}])
    entry = process_file(src, tmp_path / "songs", {"id": "piece", "title": "Piece"}, "songs/")
    text = Catalog(name="test", songs=[entry]).to_json()
    assert '"sheet"' not in text


@pytest.mark.parametrize("sheet,error", [
    ({"url": "http://x/level1.sheet.json", "sha256": "a" * 64, "pages": 2}, "sheet needs a url"),
    ({"url": "songs/x/level1.sheet.json", "sha256": "nope", "pages": 2}, "sheet sha256"),
    ({"url": "songs/x/level1.sheet.json", "sha256": "a" * 64, "pages": 0}, "sheet pages"),
    ({"url": "songs/x/level1.sheet.json", "sha256": "a" * 64, "pages": 65}, "sheet pages"),
])
def test_bad_sheet_entries_are_refused(sheet, error):
    data = {"name": "c", "version": 1, "songs": [{"id": "x", "title": "X", "levels": [
        {"level": 1, "url": "songs/x/level1.mid", "sha256": "b" * 64, "sheet": sheet}]}]}
    with pytest.raises(ValueError, match=error):
        validate(data)


def test_verify_sheet_refuses_page_paths(tmp_path):
    sheet = {"version": 1, "pages": [{"file": "../evil.png", "sha256": "c" * 64}], "systems": [], "measures": []}
    path = tmp_path / "level1.sheet.json"
    path.write_text(json.dumps(sheet), encoding="utf-8")
    ref = {"url": "level1.sheet.json", "sha256": catalog.sha256_file(path), "pages": 1}
    assert any("must be a .png name" in p for p in verify_sheet(path, ref))
