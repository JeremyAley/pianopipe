"""Two readings of a scan compared, merged and fixed by hand (scan_check), on small MusicXML exports."""

from __future__ import annotations

from fractions import Fraction

import pytest

from pianopipe import musicxml, scan_check

from omr_builder import backup, export_xml, xnote

Q = 4  # divisions per quarter note in the exports


def bar(rh: list[str], lh: str = "C3", dur: int = Q) -> str:
    """A 4/4 bar: the right hand's notes of `dur` each, a whole note in the left."""
    return "".join(xnote(n, dur) for n in rh) + backup(len(rh) * dur) + xnote(lh, 4 * Q, voice=5, staff=2)


def root_of(xml: str):
    import xml.etree.ElementTree as ET

    return ET.fromstring(xml)


def piano_texts(root) -> list[str]:
    part = scan_check.piano_part(root)
    return [scan_check.bar_text(b.element, b.divisions) for b in scan_check.read_bars(part)]


def test_the_bar_text_reads_back_as_written():
    root = root_of(export_xml([bar(["C5", "D5", "E5", "F5"])]))
    m = scan_check.piano_part(root).find("measure")
    text = "RH 0:A3/1,C4/1/2 1/2:Bb4/1/2~ 1:Bb4/1,gC#5 2:F#4/2 | LH 0:F2/4"
    scan_check.fine(scan_check.piano_part(root))
    scan_check.set_bar(m, text, scan_check.FINE, Fraction(4))
    assert scan_check.bar_text(m, scan_check.FINE) == "time=4/4 RH 0:A3/1,C4/1/2 1/2:Bb4/1/2~ 1:gC#5,Bb4/1 2:F#4/2 | LH 0:F2/4"
    with pytest.raises(ValueError, match="runs past"):
        scan_check.set_bar(m, "RH 3:C5/2", scan_check.FINE, Fraction(4))
    with pytest.raises(ValueError, match="not a note"):
        scan_check.set_bar(m, "RH 0:H5/1", scan_check.FINE, Fraction(4))
    scan_check.set_bar(m, "time=2/4 RH 0:C5/2 | LH 0:C3/2", scan_check.FINE, Fraction(4))
    assert scan_check.bar_text(m, scan_check.FINE).startswith("time=2/4 RH 0:C5/2")
    scan_check.set_bar(m, "RH 0:gG4,gEb4,Bb3/2 | LH 0:C3/2", scan_check.FINE, Fraction(2))  # grace notes keep their order
    assert scan_check.bar_text(m, scan_check.FINE) == "time=2/4 RH 0:gG4,gEb4,Bb3/2 | LH 0:C3/2"


def test_a_bar_one_reader_split_in_two_is_joined_and_taken_from_the_other():
    """copista read bar 2 as two bars (a stray mark taken for a barline): its first beat, then the rest. Audiveris
    read bar 2 whole but bar 3 with a wrong note; copista's bar 3 stays (flagged), Audiveris's bar 2 is taken."""
    cop = export_xml([bar(["C5", "D5", "E5", "F5"]),
                      xnote("G5", Q) + "<forward><duration>12</duration></forward>" + backup(4 * Q) + xnote("E3", Q, voice=5, staff=2),
                      "".join(xnote(n, Q) for n in ["A5", "G5", "F5"]) + backup(3 * Q)
                      + xnote("F3", 3 * Q, voice=5, staff=2),
                      bar(["E5", "D5", "C5", "B4"], "G3")],
                     vocal=[xnote("C5", 4 * Q)] * 4)
    aud = export_xml([bar(["C5", "D5", "E5", "F5"]),
                      bar(["G5", "A5", "G5", "F5"], "E3"),
                      bar(["E5", "D5", "C5", "A4"], "G3")])
    merged, report = scan_check.merge(root_of(cop), root_of(aud))
    assert [(b.status, b.source, b.copista, b.audiveris) for b in report.bars] == [
        ("same", "copista", [1], [1]), ("split", "audiveris", [2, 3], [2]), ("differs", "copista", [4], [3])]
    assert report.needs_review == [2, 3]
    assert "right hand: copista has B4, Audiveris A4" in report.bars[2].reasons
    texts = piano_texts(merged)
    assert texts[1] == "RH 0:G5/1 1:A5/1 2:G5/1 3:F5/1 | LH 0:E3/4"
    assert texts[2] == "RH 0:E5/1 1:D5/1 2:C5/1 3:B4/1 | LH 0:G3/4"
    voice = merged.findall("part")[0].findall("measure")
    assert len(voice) == 3  # the voice follows the piano's bars


def test_a_split_whose_second_piece_keeps_its_beats_is_joined_too():
    """When She Loved Me bar 9: copista's first piece holds beat 1, its second starts with a gap of one beat and
    then has beats 2-4 where they belong; laid over each other they are the bar."""
    first = xnote("C5", Q) + "<forward><duration>12</duration></forward>" + backup(4 * Q) + xnote("C3", Q, voice=5, staff=2)
    rest = ("<forward><duration>4</duration></forward>" + "".join(xnote(n, Q) for n in ["D5", "E5", "F5"]) + backup(4 * Q)
            + "<forward><duration>4</duration></forward>" + xnote("C3", 3 * Q, voice=5, staff=2))
    cop = export_xml([bar(["C5", "D5", "E5", "F5"]), first, rest, bar(["G5", "A5", "G5", "F5"])])
    aud = export_xml([bar(["C5", "D5", "E5", "F5"]),
                      bar(["C5", "D5", "E5", "F5"])[: -len(xnote("C3", 4 * Q, voice=5, staff=2))]
                      + xnote("C3", Q, voice=5, staff=2) + xnote("C3", 3 * Q, voice=5, staff=2),
                      bar(["G5", "A5", "G5", "F5"])])
    merged, report = scan_check.merge(root_of(cop), root_of(aud))
    assert [(b.status, b.copista, b.audiveris) for b in report.bars] == [
        ("same", [1], [1]), ("split", [2, 3], [2]), ("same", [4], [3])]
    assert report.bars[1].reasons == ["copista read it as 2 bars"]


def test_a_staff_copista_left_empty_is_taken_from_audiveris():
    cop = export_xml([xnote("rest", 4 * Q) + backup(4 * Q) + xnote("C3", 4 * Q, voice=5, staff=2)])
    aud = export_xml([bar(["C5", "D5", "E5", "F5"])])
    merged, report = scan_check.merge(root_of(cop), root_of(aud))
    assert (report.bars[0].status, report.bars[0].source) == ("differs", "audiveris")
    assert piano_texts(merged)[0] == "time=4/4 RH 0:C5/1 1:D5/1 2:E5/1 3:F5/1 | LH 0:C3/4"


def test_notes_copista_lost_into_its_rests_are_taken_from_audiveris():
    """Part of Your World: copista dropped a triplet's second note and left a rest; Audiveris has it and agrees on
    the rest of the bar. A note Audiveris adds over a copista note (pencil read as a note) isn't taken."""
    lost = xnote("C5", Q) + xnote("rest", Q) + xnote("E5", Q) + xnote("F5", Q) + backup(4 * Q) + xnote("C3", 4 * Q, voice=5, staff=2)
    extra = bar(["C5", "D5", "E5", "F5"]) + backup(4 * Q) + xnote("F2", 4 * Q, voice=6, staff=2)
    cop = export_xml([lost, bar(["C5", "D5", "E5", "F5"])])
    aud = export_xml([bar(["C5", "D5", "E5", "F5"]), extra])
    merged, report = scan_check.merge(root_of(cop), root_of(aud))
    assert [(b.status, b.source) for b in report.bars] == [("filled", "audiveris"), ("differs", "copista")]
    assert report.needs_review == [2]
    assert piano_texts(merged)[0] == "time=4/4 RH 0:C5/1 1:D5/1 2:E5/1 3:F5/1 | LH 0:C3/4"
    # a tie copista had on a shared note, or a bar of another length: not filled unchecked
    tied = lost.replace(xnote("C3", 4 * Q, voice=5, staff=2), xnote("C3", 4 * Q, voice=5, staff=2, tie="start"))
    merged, report = scan_check.merge(root_of(export_xml([tied, bar(["C5", "D5", "E5", "F5"])])), root_of(aud))
    assert report.bars[0].status == "differs"
    short = xnote("C5", Q) + xnote("rest", Q) + backup(2 * Q) + xnote("C3", 2 * Q, voice=5, staff=2)
    merged, report = scan_check.merge(root_of(export_xml([short], time=(2, 4))),
                                      root_of(export_xml([bar(["C5", "D5", "E5", "F5"])])))
    assert report.bars[0].status == "differs"


def test_vocal_bars_are_kept_only_where_they_can_be_trusted():
    """Bar 1: copista read a chord box as an A5 in the vocal line, Audiveris read it clean and the piano doubles
    it: Audiveris's bar is taken. Bar 2: both readers read the voice alike: kept, though the piano plays other
    notes. Bar 3: the readers differ and neither is doubled: rests, so the tune comes from the piano there."""
    piano = [bar(["C5", "D5", "E5", "F5"]), bar(["C5", "C5", "C5", "C5"]), bar(["G5", "G5", "G5", "G5"])]
    cop = export_xml(piano, vocal=["".join(xnote(n, Q) for n in ["C5", "D5", "E5", "F5"]) + backup(Q) + xnote("A5", Q),
                                   "".join(xnote(n, Q) for n in ["E5", "E5", "D5", "D5"]),
                                   "".join(xnote(n, Q) for n in ["A4", "B4", "A4", "B4"])])
    aud = export_xml(piano, vocal=["".join(xnote(n, Q) for n in ["C5", "D5", "E5", "F5"]),
                                   "".join(xnote(n, Q) for n in ["E5", "E5", "D5", "D5"]),
                                   "".join(xnote(n, Q) for n in ["A4", "B4", "A4", "C5"])])
    merged, report = scan_check.merge(root_of(cop), root_of(aud))
    notes = scan_check.check_voices(merged, root_of(aud), report)
    assert notes == ["Voice: 1 bars as Audiveris read them (the piano doubles them): 1",
                     "Voice: 1 bars left to the piano's top line (the readers differ and the piano doesn't double them): 3"]
    voice = scan_check.read_bars(merged.findall("part")[0])
    assert [sorted(n.midi for n in b.notes) for b in voice] == [[72, 74, 76, 77], [74, 74, 76, 76], []]


def test_hand_fixes_replace_delete_and_add_bars_and_ties_follow(tmp_path):
    root = root_of(export_xml([bar(["C5", "D5", "E5", "F5"]), bar(["G5", "A5", "G5", "F5"]), bar(["E5", "D5", "C5", "D5"])],
                              vocal=[xnote("C5", 4 * Q)] * 3))
    done = scan_check.apply_fixes(root, {"1": "RH 0:C5/3 3:G5/1~ | LH 0:C3/4", "2": "delete",
                                         "3+": "RH 0:C5/4 | LH 0:C3/4"})
    assert done == ["bar 1", "a bar after 3", "bar 2 deleted"]
    texts = piano_texts(root)
    assert texts == ["time=4/4 RH 0:C5/3 3:G5/1~ | LH 0:C3/4", "RH 0:E5/1 1:D5/1 2:C5/1 3:D5/1 | LH 0:C3/4",
                     "RH 0:C5/4 | LH 0:C3/4"]
    assert len(root.findall("part")[0].findall("measure")) == 3
    # the tie from bar 1's G5 goes nowhere (bar 2 starts on E5): no stop is invented; tie it to a G5 and it stops there
    scan_check.apply_fixes(root, {"2": "RH 0:G5/1 1:D5/1 2:C5/1 3:D5/1 | LH 0:C3/4"})
    stops = [n for n in scan_check.piano_part(root).findall("measure")[1].iter("note")
             if any(t.get("type") == "stop" for t in n.findall("tie"))]
    assert [s.findtext("pitch/step") for s in stops] == ["G"]
    path = tmp_path / "fixed.musicxml"
    import xml.etree.ElementTree as ET

    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    score = musicxml.load(path)
    g5 = [n for n in score.notes if n.pitch == 79]
    assert len(g5) == 1 and g5[0].end - g5[0].start == 2 * musicxml.TPB  # tied into one note of two beats
    with pytest.raises(ValueError, match="has 3 bars"):
        scan_check.apply_fixes(root, {"9": "delete"})


def test_several_bars_added_after_one_in_order():
    """A list after "<bar>+" writes out a passage ("Repeat as needed"): the bars go in after it, in order."""
    root = root_of(export_xml([bar(["C5", "D5", "E5", "F5"]), bar(["G5", "A5", "G5", "F5"])]))
    done = scan_check.apply_fixes(root, {"1+": ["RH 0:A5/4 | LH 0:C3/4", "RH 0:B5/4 | LH 0:C3/4"]})
    assert done == ["2 bars after 1"]
    assert [t.split(" | ")[0] for t in piano_texts(root)] == [
        "time=4/4 RH 0:C5/1 1:D5/1 2:E5/1 3:F5/1", "RH 0:A5/4", "RH 0:B5/4", "RH 0:G5/1 1:A5/1 2:G5/1 3:F5/1"]


def test_repeats_and_voltas_as_printed(tmp_path):
    """I See the Light: a repeat from bar 2 to 4 with a first ending over 3-4 and a second over 5; the readers lost
    the first ending. set_repeats replaces the parts' own signs; the song plays 1-4, 2, 5."""
    import xml.etree.ElementTree as ET

    body = bar(["C5", "D5", "E5", "F5"])
    stray = '<barline location="right"><repeat direction="backward"/></barline>'
    root = root_of(export_xml([body, body, body, body + stray, body], vocal=[xnote("C5", 4 * Q)] * 5))
    done = scan_check.set_repeats(root, {"2": "|:", "3-4": "1. :|", "5": "2."})
    assert done == ["2 |:", "3-4 1. :|", "5 2."]
    path = tmp_path / "r.musicxml"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    assert musicxml.load(path).order == [0, 1, 2, 3, 1, 4]
    assert sum(1 for _ in root.iter("repeat")) == 2  # the vocal part's stray sign is gone
    with pytest.raises(ValueError, match="has 5 bars"):
        scan_check.set_repeats(root, {"4-9": "1."})
    # a volta 1,2 then a third ending; and overlapping entries on one barline
    root = root_of(export_xml([body] * 5))
    scan_check.set_repeats(root, {"2": "|:", "3-4": "1,2. :|", "5": "3."})
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    assert musicxml.load(path).order == [0, 1, 2, 3, 1, 2, 3, 1, 4]
    root = root_of(export_xml([body] * 5))
    scan_check.set_repeats(root, {"2-4": "|: :|", "3-4": "1.", "5": "2."})
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    assert musicxml.load(path).order == [0, 1, 2, 3, 1, 4]
    root = root_of(export_xml([body] * 5))
    scan_check.set_repeats(root, {"2-4": "|: :|", "3-4": "1,2.", "5": "3."})
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    assert musicxml.load(path).order == [0, 1, 2, 3, 1, 2, 3, 1, 4]
    final = '<barline location="right"><bar-style>light-heavy</bar-style></barline>'
    root = root_of(export_xml([body] * 4 + [body + final]))
    scan_check.set_repeats(root, {"1-2": ":| x3"})  # Touch the Sky: "play 3 times"; the final barline stays
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    assert musicxml.load(path).order == [0, 1, 0, 1, 0, 1, 2, 3, 4]
    assert root.find("part").findall("measure")[4].find("barline/bar-style").text == "light-heavy"


def test_a_deleted_bar_passes_its_key_change_on():
    """Look Through My Eyes: copista made a phantom empty bar at the change to E major; deleting it keeps the key."""
    key = "<attributes><key><fifths>4</fifths></key></attributes>"
    root = root_of(export_xml([bar(["C5", "D5", "E5", "F5"]), key + xnote("rest", 4 * Q), bar(["E5", "E5", "E5", "E5"])]))
    scan_check.apply_fixes(root, {"2": "delete"})
    ms = scan_check.piano_part(root).findall("measure")
    assert len(ms) == 2 and ms[1].findtext("attributes/key/fifths") == "4"
    root = root_of(export_xml([bar(["C5", "D5", "E5", "F5"]), key + xnote("rest", 4 * Q), xnote("rest", 4 * Q),
                               bar(["E5", "E5", "E5", "E5"])]))
    scan_check.apply_fixes(root, {"2": "delete", "3": "delete"})
    ms = scan_check.piano_part(root).findall("measure")
    assert len(ms) == 2 and ms[1].findtext("attributes/key/fifths") == "4"
