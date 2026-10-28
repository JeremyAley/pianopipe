"""Sheet music: notation from app-ready levels (notation.py), engraving (engrave.py), the catalog's `sheet` field, and
pickup bars from MusicXML lining up with the app's bar grid (docs/sheet-music.md)."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import Counter
from fractions import Fraction as Q
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


def test_a_hand_crossing_inside_a_bar_changes_clef_at_that_beat(tmp_path):
    # Left hand: bar 1 low; bar 2 two low beats, then two well inside the treble clef; bar 3 high; bar 4 low again.
    lh = [(0, 4 * TPB, 43), (4 * TPB, TPB, 43), (5 * TPB, TPB, 47), (6 * TPB, TPB, 72), (7 * TPB, TPB, 76),
          (8 * TPB, 4 * TPB, 79), (12 * TPB, 4 * TPB, 43)]
    sc, xml = score_xml(tmp_path, [(0, 16 * TPB, 84)], lh, [ts(0, 4, 4)])
    st = sc.staves[1]
    assert st.clefs == ["F", "F", "G", "F"] and st.clef_changes == {1: [(Q(2), "G")]}
    # In bar 2's stream the treble clef comes between the low beats and the high ones; bar 3 needs none of its own.
    measures = ET.fromstring(xml).findall(".//measure")
    stream = []
    for el in measures[1]:
        if el.tag == "attributes":
            stream += [f"clef {c.findtext('sign')}" for c in el.iter("clef")]
        elif el.tag == "note" and el.findtext("staff") == "2":
            stream.append(el.findtext("pitch/step"))
    assert stream == ["G", "B", "clef G", "C", "E"]
    assert [c.findtext("sign") for c in measures[2].iter("clef")] == []
    assert [c.findtext("sign") for c in measures[3].iter("clef")] == ["F"]
    # A crossing played a tick late changes clef on the quantised beat, and every piece still has a written length.
    late = [(n[0] + 1, n[1] - 1, n[2]) if n[0] == 6 * TPB else n for n in lh]
    sc5, xml5 = score_xml(tmp_path, [(0, 16 * TPB, 84)], late, [ts(0, 4, 4)])
    assert sc5.staves[1].clef_changes == {1: [(Q(2), "G")]}
    for m in ET.fromstring(xml5).findall(".//measure"):
        for staff in ("1", "2"):
            voices = {}
            for n in m.findall("note"):
                if n.findtext("staff") == staff and n.find("chord") is None and n.find("grace") is None:
                    assert int(n.findtext("duration")) > 0
                    voices[n.findtext("voice")] = voices.get(n.findtext("voice"), 0) + int(n.findtext("duration"))
            assert all(v == 4 * notation.DIVISIONS for v in voices.values())
    # A whole-bar bass note held under the crossing: no change inside the bar (from there the bass note would be drawn in
    # the treble clef, eight ledger lines down); the crossing keeps its ledger lines.
    lh3 = [(0, 4 * TPB, 36), (2 * TPB, TPB, 72), (3 * TPB, TPB, 76), (4 * TPB, 4 * TPB, 43)]
    sc3, _ = score_xml(tmp_path, [(0, 8 * TPB, 84)], lh3, [ts(0, 4, 4)])
    assert sc3.staves[1].clefs == ["F", "F"] and sc3.staves[1].clef_changes == {}
    # Where the first voice rests across the change, the rest is split there so the clef sits at its place.
    lh4 = [(0, TPB, 43), (2 * TPB, TPB, 72), (3 * TPB, TPB, 76), (4 * TPB, 4 * TPB, 43)]
    pieces = notation.measure_pieces(score_xml(tmp_path, [], lh4, [ts(0, 4, 4)])[0].staves[0].voices[0],
                                     notation.Bar(1, 0, 4 * TPB, 4, 4), TPB, set(), Q(0), (Q(3, 2),))
    assert [(p.at, bool(p.pitches)) for p in pieces][:3] == [(0, True), (1, False), (Q(3, 2), False)]
    # A bar that is low but for its last beat stays in its clef (the crossing is too short to change for).
    lh2 = [(0, 3 * TPB, 43), (3 * TPB, TPB, 76), (4 * TPB, 4 * TPB, 43)]
    sc2, _ = score_xml(tmp_path, [(0, 8 * TPB, 84)], lh2, [ts(0, 4, 4)])
    assert sc2.staves[1].clefs == ["F", "F"] and sc2.staves[1].clef_changes == {}


G64 = TPB // 16  # a 64th


def test_short_notes_before_a_longer_one_are_grace_notes(tmp_path):
    # Bar 1: a 64th just before beat 2's note (an acciaccatura) and two before beat 3's (a slide). Bar 2: a written-out
    # run of eight 64ths, which stays in the rhythm.
    rh = [(0, TPB - G64, 72), (TPB - G64, G64, 76), (TPB, TPB - 2 * G64, 74),
          (2 * TPB - 2 * G64, G64, 79), (2 * TPB - G64, G64, 77), (2 * TPB, 2 * TPB, 76)]
    rh += [(4 * TPB + i * G64, G64, 72 + i % 2) for i in range(8)] + [(4 * TPB + TPB // 2, 3 * TPB, 76)]
    sc, xml = score_xml(tmp_path, rh, [], [ts(0, 4, 4)])
    graces = {c.on: c.graces for v in sc.staves[0].voices for c in v if c.graces}
    assert graces == {Q(1): [[76]], Q(2): [[79], [77]]}
    measures = ET.fromstring(xml).findall(".//measure")
    written = [(n.find("grace").get("slash"), n.findtext("type"), n.findtext("pitch/step"), n.findtext("beam"))
               for n in measures[0].findall("note") if n.find("grace") is not None]
    assert written == [("yes", "eighth", "E", None), ("yes", "16th", "G", "begin"), ("yes", "16th", "F", "end")]
    # Grace notes come right before the note they lead to, and take no time.
    notes = [n for n in measures[0].findall("note") if n.find("rest") is None]
    steps = [("g" if n.find("grace") is not None else "") + n.findtext("pitch/step") for n in notes]
    assert steps == ["C", "gE", "D", "gG", "gF", "E"]
    assert all(n.find("duration") is None for n in notes if n.find("grace") is not None)
    assert not any(n.find("grace") is not None for n in measures[1].findall("note"))


def test_split_graces_leaves_runs_and_short_main_notes(tmp_path):
    def notes(spec):
        return [Note(s, s + d, p, 80, "R") for s, d, p in spec]
    # Five 64ths before a quarter: a run, not grace notes.
    run = notes([(i * G64, G64, 72 + i) for i in range(5)] + [(5 * G64, TPB, 79)])
    assert notation.split_graces(run, TPB)[1] == {}
    # A 64th before another short note (a 32nd): not a grace note either.
    short = notes([(0, G64, 72), (G64, TPB // 8, 74), (TPB, TPB, 76)])
    assert notation.split_graces(short, TPB)[1] == {}
    # Four before a quarter are.
    four = notes([(i * G64, G64, 72 + i) for i in range(4)] + [(4 * G64, TPB, 79)])
    rest, graces = notation.split_graces(four, TPB)
    assert graces == {4 * G64: [[72], [73], [74], [75]]} and [n.pitch for n in rest] == [79]


def test_a_high_passage_goes_under_an_8va_line(tmp_path):
    # Right hand: a bar of C5 quarters; a bar climbing from G5 to C7 in eighths; C5 again. Left hand: a bar reaching
    # down to F1 (8vb), then middle range.
    high = [79, 81, 84, 86, 88, 91, 93, 96]
    rh = [(i * TPB, TPB, 72) for i in range(4)] + [(4 * TPB + i * TPB // 2, TPB // 2, p) for i, p in enumerate(high)]
    rh += [(8 * TPB, 4 * TPB, 72)]
    lh = [(i * TPB, TPB, p) for i, p in enumerate([36, 33, 31, 29])] + [(4 * TPB, 8 * TPB, 48)]
    sc, xml = score_xml(tmp_path, rh, lh, [ts(0, 4, 4)])
    assert sc.staves[0].ottavas == [(Q(4), Q(8), 1)]
    assert sc.staves[1].ottavas == [(Q(0), Q(4), -1)]
    root = ET.fromstring(xml)
    shifts = [(o.get("type"), o.get("number")) for o in root.iter("octave-shift")]
    assert shifts == [("up", "2"), ("stop", "2"), ("down", "1"), ("stop", "1")]
    # The 8va starts before bar 2's first note and stops after its last (in the first voice's stream).
    bar2 = [el.tag if el.tag != "note" else el.findtext("pitch/step") + el.findtext("pitch/octave")
            for el in root.findall(".//measure")[1] if el.tag in ("note", "direction")]
    assert bar2[:2] == ["direction", "G5"] and bar2[8:10] == ["C7", "direction"]
    # Pitches stay as they sound (the round trip reads them back unchanged).
    got, _, _ = reread(tmp_path, xml)
    assert [p for _, p in got] == [p for _, _, p in sorted(rh)]
    # A lone A6 eighth among lower notes is left on its ledger lines; a lone C7 gets a short 8va of its own.
    lone = [(0, TPB, 72), (TPB, TPB // 2, 93), (TPB + TPB // 2, TPB // 2, 76), (2 * TPB, TPB // 2, 96), (2 * TPB + TPB // 2, TPB // 2, 76)]
    sc3, _ = score_xml(tmp_path, lone, [], [ts(0, 4, 4)])
    assert sc3.staves[0].ottavas == [(Q(2), Q(5, 2), 1)]


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
def test_bar_map_lists_every_struck_note_with_its_head(tmp_path):
    """Each note the level plays has its notehead in its bar's `notes` (fraction, pitch, x, y): one per key struck, so
    the far end of a tie (the same key held over the bar line) has none."""
    q = TPB
    # Bar 1: a C-E chord, D, then E held over the bar line into bar 2; bar 2: F, a half note G; the left hand in quarters.
    rh = [(0, q, 72), (0, q, 76), (q, q, 74), (2 * q, 3 * q, 76), (5 * q, q, 77), (6 * q, 2 * q, 79)]
    lh = [(i * q, q, 48 + (i % 3) * 2) for i in range(8)]
    level = app_midi(tmp_path / "level4.mid", rh, lh, [ts(0, 4, 4), ks(0, "C")])
    data = json.loads(engrave.engrave(level, tmp_path / "out", "level4", 4).json_path.read_text(encoding="utf-8"))
    bars = []
    for m in data["measures"]:
        n = m["notes"]
        assert len(n) % 4 == 0
        bars.append([tuple(n[i:i + 4]) for i in range(0, len(n), 4)])
    played = [sorted((round((s % (4 * q)) / (4 * q), 4), p) for s, _, p in rh + lh if s // (4 * q) == b) for b in (0, 1)]
    assert [sorted((f, p) for f, p, _, _ in heads) for heads in bars] == played  # no head for the tied E in bar 2
    for m, heads in zip(data["measures"], bars):
        sys_ = data["systems"][m["system"]]
        for f, p, x, y in heads:
            assert m["left"] < x < m["right"] and sys_["top"] - 0.02 < y < sys_["bottom"] + 0.02
            assert any(abs(of - f) < 1e-6 and ox < x < ox + 0.03 for of, ox in zip(m["onsets"][0::2], m["onsets"][1::2]))
    # Higher notes are drawn higher up the page; a chord's notes share one x.
    chord = {p: (x, y) for f, p, x, y in bars[0] if f == 0}
    assert chord[76][1] < chord[72][1] < chord[48][1] and chord[76][0] == chord[72][0]


def heads_by_pitch(data: dict) -> Counter:
    """Each pitch's noteheads in a bar map, once per bar its measure is played as."""
    out = Counter()
    for m in data["measures"]:
        for i in range(0, len(m["notes"]), 4):
            out[int(m["notes"][i + 1])] += len(m.get("plays", [m["bar"]]))
    return out


@needs_engraver
def test_grace_notes_8va_and_clef_changes_keep_every_head(tmp_path):
    """Grace notes have heads of their own, notes under an 8va line keep their sounding pitch (and are drawn an octave
    lower), and a clef change inside a bar moves nothing: every note the level plays still has its head."""
    rh = [(0, TPB - G64, 72), (TPB - G64, G64, 76), (TPB, 3 * TPB, 74)]
    rh += [(4 * TPB + i * TPB // 2, TPB // 2, p) for i, p in enumerate([79, 81, 84, 86, 88, 91, 93, 96])]
    lh = [(0, 4 * TPB, 43), (4 * TPB, TPB, 43), (5 * TPB, TPB, 47), (6 * TPB, TPB, 72), (7 * TPB, TPB, 76)]
    level = app_midi(tmp_path / "level4.mid", rh, lh, [ts(0, 4, 4), ks(0, "C")])
    data = engrave.engrave(level, tmp_path / "out", "level4", 4, keep_musicxml=True).data
    xml = (tmp_path / "out" / "level4.musicxml").read_text(encoding="utf-8")
    assert "<grace" in xml and "octave-shift" in xml and xml.count("<clef") == 3
    assert heads_by_pitch(data) == Counter(p for _, _, p in rh + lh)
    # The grace note's head is drawn left of the note it leads to; C7 under the 8va sits two ledger lines up, not five.
    bar1 = {p: (f, x, y) for f, p, x, y in zip(*[iter(data["measures"][0]["notes"])] * 4)}
    assert bar1[76][1] < bar1[74][1]
    bar2 = {p: (f, x, y) for f, p, x, y in zip(*[iter(data["measures"][1]["notes"])] * 4)}
    top = data["systems"][data["measures"][1]["system"]]["top"]
    space = 16 / engrave.PAGE_H
    assert 1.5 * space < top - bar2[96][2] < 3.5 * space


SCORE_HEAD = ('<?xml version="1.0"?><score-partwise version="4.0"><work><work-title>Test</work-title></work>'
              '<part-list><score-part id="P1"><part-name>Piano</part-name></score-part>'
              '<score-part id="P2"><part-name>Voice</part-name></score-part></part-list>')


def xnote(step, octave, dur, staff=1, voice=None, extra="", typ="quarter", pre=""):
    voice = voice or (1 if staff == 1 else 5)
    return (f"<note>{pre}<pitch><step>{step}</step><octave>{octave}</octave></pitch><duration>{dur}</duration>"
            f"<voice>{voice}</voice><type>{typ}</type><staff>{staff}</staff>{extra}</note>")


def score_with_repeats(transpose: bool = False) -> str:
    """Piano (two staves) and a voice part: measure 1; measures 2-3 repeated with a first ending at 3 and a second at
    4; measure 5. A grace note, a slur and a dynamic in measure 1."""
    attrs = ('<attributes><divisions>2</divisions><key><fifths>0</fifths></key><time><beats>2</beats><beat-type>4</beat-type>'
             '</time><staves>2</staves>' + ('<transpose><chromatic>-2</chromatic></transpose>' if transpose else '') +
             '<clef number="1"><sign>G</sign><line>2</line></clef><clef number="2"><sign>F</sign><line>4</line></clef></attributes>')
    back = "<backup><duration>4</duration></backup>"

    def bass(step):
        return back + xnote(step, 3, 4, 2, typ="half")
    m1 = ('<measure number="1">' + attrs + '<direction placement="below"><direction-type><dynamics><p/></dynamics>'
          '</direction-type><staff>1</staff></direction>'
          '<direction placement="above"><direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>96'
          '</per-minute></metronome></direction-type><staff>1</staff><sound tempo="96"/></direction>'
          + xnote("D", 5, 0, pre='<grace slash="yes"/>', typ="eighth").replace("<duration>0</duration>", "")
          + xnote("C", 5, 2, extra='<notations><slur type="start" number="1"/></notations>')
          + xnote("E", 5, 2, extra='<notations><slur type="stop" number="1"/></notations>') + bass("C") + '</measure>')
    m2 = ('<measure number="2"><barline location="left"><repeat direction="forward"/></barline>'
          + xnote("F", 5, 2) + xnote("G", 5, 2) + bass("D") + '</measure>')
    m3 = ('<measure number="3"><barline location="left"><ending number="1" type="start"/></barline>'
          + xnote("A", 5, 4, typ="half") + bass("E")
          + '<barline location="right"><ending number="1" type="stop"/><repeat direction="backward"/></barline></measure>')
    m4 = ('<measure number="4"><barline location="left"><ending number="2" type="start"/></barline>'
          + xnote("B", 5, 4, typ="half") + bass("F")
          + '<barline location="right"><ending number="2" type="discontinue"/></barline></measure>')
    m5 = '<measure number="5">' + xnote("C", 6, 4, typ="half") + bass("G") + '</measure>'
    voice = ''.join(f'<measure number="{i}">' + ('<attributes><divisions>2</divisions></attributes>' if i == 1 else '')
                    + xnote("C", 4, 4, typ="half") + '</measure>' for i in range(1, 6))
    return SCORE_HEAD + '<part id="P1">' + m1 + m2 + m3 + m4 + m5 + '</part><part id="P2">' + voice + '</part></score-partwise>'


@needs_engraver
def test_full_level_of_a_musicxml_score_shows_its_own_notation(tmp_path):
    """Level 4 of a MusicXML score is engraved from the score's keyboard part as written: its grace note, slur and
    dynamic, the repeat and endings written once, and a bar map whose measures list the bars they are played as
    (version 2). Levels 1-3 are engraved from their level files, a measure per bar."""
    src = tmp_path / "piece.musicxml"
    src.write_text(score_with_repeats(), encoding="utf-8")
    entry = process_file(src, tmp_path / "songs", {"id": "piece", "title": "Piece"}, "songs/", sheets=True)
    assert entry.hand_source == "staves" and not any("not the score" in w for w in entry.warnings)
    data = json.loads((tmp_path / "songs" / "piece" / "level4.sheet.json").read_text(encoding="utf-8"))
    # Played: 1, 2, 3 (first ending), 2, 4 (second ending), 5.
    assert data["version"] == 2 and data["bars"] == 6 and len(data["measures"]) == 5
    assert [m["plays"] for m in data["measures"]] == [[1], [2, 4], [3], [5], [6]]
    level4 = midi_io.load(tmp_path / "songs" / "piece" / "level4.mid")
    assert heads_by_pitch(data) == Counter(n.pitch for n in level4.notes)  # the grace note's head included
    assert 74 in heads_by_pitch(data)
    for lv in (1, 2, 3):
        low = json.loads((tmp_path / "songs" / "piece" / f"level{lv}.sheet.json").read_text(encoding="utf-8"))
        assert low["version"] == 1 and len(low["measures"]) == low["bars"] == 6
    cat = Catalog(name="test", songs=[entry])
    cat.save(tmp_path / "catalog.json")
    assert verify(tmp_path / "catalog.json") == []
    # The notation is the score's: the slur, the dynamic and the repeat signs; the voice part and the title are gone.
    sheet = engrave.engrave(tmp_path / "songs" / "piece" / "level4.mid", tmp_path / "x", "level4", 4, source=src,
                            keep_musicxml=True)
    assert sheet.source and sheet.note == ""
    xml = (tmp_path / "x" / "level4.musicxml").read_text(encoding="utf-8")
    for kept in ("<slur", "<dynamics", '<repeat direction="backward"', '<ending number="2"', "<grace"):
        assert kept in xml
    assert 'id="P2"' not in xml and "work-title" not in xml
    assert "metronome" not in xml  # its note glyph is text in the music font, which resvg can't draw
    # A bar map with plays that miss a bar is refused.
    broken = dict(data, measures=[dict(m, plays=[1]) if i == 1 else m for i, m in enumerate(data["measures"])])
    path = tmp_path / "broken.sheet.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    ref = {"url": "broken.sheet.json", "sha256": catalog.sha256_file(path), "pages": len(data["pages"])}
    problems = verify_sheet(path, ref)
    assert any("plays" in p for p in problems)


@needs_engraver
def test_a_score_that_cant_stand_for_its_level_falls_back_to_the_level(tmp_path):
    """A transposing keyboard part (its written notes aren't what the level plays) is engraved from the level file,
    unrolled, with the reason kept."""
    src = tmp_path / "piece.musicxml"
    src.write_text(score_with_repeats(transpose=True), encoding="utf-8")
    entry = process_file(src, tmp_path / "songs", {"id": "piece", "title": "Piece"}, "songs/", sheets=True)
    data = json.loads((tmp_path / "songs" / "piece" / "level4.sheet.json").read_text(encoding="utf-8"))
    assert data["version"] == 1 and len(data["measures"]) == data["bars"] == 6
    assert any("level 4: sheet music from the level, not the score (a transposing keyboard part)" in w for w in entry.warnings)


@needs_engraver
def test_a_score_that_leaves_struck_notes_without_heads_falls_back(tmp_path):
    """A tie into a repeated section: the first playing holds the note, the second strikes it, but the score draws one
    head for both (the tie's end). Too many of the level's notes would go unmarked, so the level's notation is used."""
    attrs = ('<attributes><divisions>2</divisions><time><beats>2</beats><beat-type>4</beat-type></time><staves>2</staves>'
             '<clef number="1"><sign>G</sign><line>2</line></clef><clef number="2"><sign>F</sign><line>4</line></clef></attributes>')
    back = "<backup><duration>4</duration></backup>" + xnote("C", 3, 4, 2, typ="half")
    xml = (SCORE_HEAD.replace('<score-part id="P2"><part-name>Voice</part-name></score-part>', '') + '<part id="P1">'
           '<measure number="1">' + attrs + xnote("E", 5, 4, typ="half", extra='<tie type="start"/>') + back + '</measure>'
           '<measure number="2"><barline location="left"><repeat direction="forward"/></barline>'
           + xnote("E", 5, 4, typ="half", extra='<tie type="stop"/>') + back + '</measure>'
           '<measure number="3">' + xnote("G", 5, 4, typ="half") + back
           + '<barline location="right"><repeat direction="backward"/></barline></measure>'
           '<measure number="4">' + xnote("C", 6, 4, typ="half") + back + '</measure></part></score-partwise>')
    xml = xml.replace('<tie type="start"/></note>', '<tie type="start"/><notations><tied type="start"/></notations></note>')
    xml = xml.replace('<tie type="stop"/></note>', '<tie type="stop"/><notations><tied type="stop"/></notations></note>')
    src = tmp_path / "tied.musicxml"
    src.write_text(xml, encoding="utf-8")
    entry = process_file(src, tmp_path / "songs", {"id": "tied", "title": "Tied"}, "songs/", sheets=True)
    data = json.loads((tmp_path / "songs" / "tied" / "level4.sheet.json").read_text(encoding="utf-8"))
    level4 = midi_io.load(tmp_path / "songs" / "tied" / "level4.mid")
    assert sum(n.pitch == 76 for n in level4.notes) == 2  # held over the bar line, then struck again on the repeat
    assert data["version"] == 1 and heads_by_pitch(data) == Counter(n.pitch for n in level4.notes)
    assert any("have no notehead in the score" in w for w in entry.warnings)


def test_musicxml_long_measures_get_their_own_time_signature(tmp_path):
    """A measure longer than its time signature (an unmarked 5/4 bar in 4/4) gets a time signature of its own length,
    so the app's later bar lines stay on the score's; one a hair over (tuplet durations rounded) counts as full."""
    def n(step, dur):
        return f"<note><pitch><step>{step}</step><octave>5</octave></pitch><duration>{dur}</duration><voice>1</voice></note>"
    xml = ('<?xml version="1.0"?><score-partwise version="3.1"><part-list><score-part id="P1"><part-name>Piano</part-name>'
           '</score-part></part-list><part id="P1">'
           '<measure number="1"><attributes><divisions>480</divisions><time><beats>4</beats><beat-type>4</beat-type>'
           '</time></attributes>' + n("C", 1920) + '</measure>'
           '<measure number="2">' + n("D", 1920) + n("E", 480) + '</measure>'
           '<measure number="3">' + n("F", 1921) + '</measure>'
           '<measure number="4">' + n("G", 1921) + '</measure></part></score-partwise>')
    path = tmp_path / "long.musicxml"
    path.write_text(xml, encoding="utf-8")
    sc = musicxml.load(path)
    q = musicxml.TPB
    sigs = [(t, m.numerator, m.denominator) for t, m in sc.song.meta if m.type == "time_signature"]
    assert sigs == [(0, 4, 4), (4 * q, 5, 4), (9 * q, 4, 4)]
    assert sc.bars == [0, 4 * q, 9 * q, 13 * q, 17 * q]
    assert max(nn.end for nn in sc.notes) == 17 * q  # the last note ends with its snapped measure: no extra bar
    grid = notation.bar_grid(sc.song.meta, q, max(nn.end for nn in sc.notes))
    assert [b.start for b in grid] == sc.bars[:-1]


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


@needs_engraver
def test_a_level_too_long_to_engrave_keeps_the_song(tmp_path, monkeypatch):
    """A level whose sheet can't be made (here: over the page limit) plays without sheet music; the song stays."""
    rh, lh = two_hand_piece(8)
    src = make_midi(tmp_path / "piece.mid", [{"name": "RH", "channel": 0, "notes": rh}, {"name": "LH", "channel": 1, "notes": lh}])
    monkeypatch.setattr(engrave, "MAX_PAGES", 0)
    entry = process_file(src, tmp_path / "songs", {"id": "piece", "title": "Piece"}, "songs/", sheets=True)
    assert len(entry.levels) == 4 and all(lv.sheet is None for lv in entry.levels)
    assert sum("no sheet music" in w for w in entry.warnings) == 4


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


def test_verify_sheet_refuses_notes_not_in_fours(tmp_path):
    sheet = {"version": 1, "pages": [], "systems": [{"page": 1, "top": 0.1, "bottom": 0.2}],
             "measures": [{"bar": 1, "system": 0, "left": 0.1, "right": 0.5, "onsets": [0, 0.12], "notes": [0, 60, 0.12]}]}
    path = tmp_path / "level1.sheet.json"
    path.write_text(json.dumps(sheet), encoding="utf-8")
    ref = {"url": "level1.sheet.json", "sha256": catalog.sha256_file(path), "pages": 0}
    assert any("notes in fours" in p for p in verify_sheet(path, ref))
