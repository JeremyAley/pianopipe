"""Reading a publisher's vector PDF (pdf_score) and checking an OMR export against it (pdf_check), on small
synthetic PDFs drawn with pymupdf (tests/pdf_builder.py). Needs the [omr] extra."""

from __future__ import annotations

import json
from fractions import Fraction

import pytest

pytest.importorskip("pymupdf")

from pianopipe import cli, musicxml, omr_repair, pdf_check, pdf_score  # noqa: E402

from omr_builder import backup, export_xml, xnote  # noqa: E402
from pdf_builder import FONT_NAME, SP, Sheet  # noqa: E402

Q = 4  # divisions per quarter note in the exports


@pytest.fixture(autouse=True)
def stand_in_font(monkeypatch):
    """The builder draws the symbols with Liberation Serif: read it as a Sonata-layout music font."""
    monkeypatch.setattr(pdf_score, "SONATA_FONTS", pdf_score.SONATA_FONTS + [FONT_NAME])


def notes(reading: pdf_check.Reading, staff: int) -> list[tuple[Fraction, str, Fraction]]:
    return sorted((n.onset, pdf_check.name(n.midi), n.duration) for n in reading.notes if n.staff == staff)


def placed(path, bar: int = 1) -> list[tuple[float, int, str, Fraction]]:
    """(onset, staff, pitch name, duration) of one bar of a written file, as the importer reads it."""
    root = musicxml.read_root(path)
    part = musicxml._parse_part(omr_repair._piano_part(root), {})
    names = "C C# D D# E F F# G G# A A# B".split()
    return sorted((float(n.pos), n.staff, f"{names[n.pitch % 12]}{n.pitch // 12 - 1}", n.dur)
                  for n in part.measures[bar - 1].notes)


def simple_bar(s: Sheet, bar: int, treble: list[str], bass: list[str], system: int = 0) -> None:
    """Four quarter notes in the right hand over a whole note (or chord) in the left."""
    for k, n in enumerate(treble):
        s.chord(bar, 1 + k, 1, [n], "q", system=system)
    s.chord(bar, 1, 2, bass, "w", system=system)


def simple_xml(treble: list[str], bass: list[str]) -> str:
    return ("".join(xnote(n, Q) for n in treble) + backup(4 * Q)
            + "".join(xnote(n, 4 * Q, voice=5, staff=2, chord=k > 0) for k, n in enumerate(bass)))


# ---------------------------------------------------------------- reading the PDF

def test_symbols_from_sonata_opus_and_smufl_fonts():
    assert pdf_score.symbol("sonata", "œ") == ("head", Fraction(1))
    assert pdf_score.symbol("sonata", "") == ("head", Fraction(1))  # Opus: U+F000 + the Mac Roman code
    assert pdf_score.symbol("sonata", "") == ("head", Fraction(2))
    assert pdf_score.symbol("sonata-special", "") == ("dot", None)
    assert pdf_score.symbol("sonata-special", "b") is None  # a companion font's letters aren't flats
    assert pdf_score.symbol("smufl", "") == ("head", Fraction(1))
    assert pdf_score.symbol("smufl", "") == ("clef", "BASS")
    assert pdf_score._family("ABCDEF+Opus") == "sonata"
    assert pdf_score._family("OpusSpecialStd") == "sonata-special"
    assert pdf_score._family("OpusText") is None
    assert pdf_score._family("Leland") == "smufl"


def test_reader_finds_the_piano_under_a_vocal_staff_and_its_bars(tmp_path):
    s = Sheet((3, 2), vocal=True)
    for system, bars in enumerate((3, 2)):
        for bar in range(1, bars + 1):
            simple_bar(s, bar, ["C5", "D5", "E5", "F5"], ["C3"], system)
    score = pdf_score.read(s.save(tmp_path / "song.pdf"))
    assert len(score.bars) == 5
    assert score.systems == [0, 3]
    assert score.bars[0].time == (4, 4)
    assert all(pdf_check.read_bar(b, Fraction(4)).placed for b in score.bars)


def test_reader_pitches_from_clef_key_accidentals_ledgers_and_8va(tmp_path):
    s = Sheet((2,), key=2)  # F# and C#
    s.chord(1, 1, 1, ["F5"], "q")  # F#5 from the key
    s.chord(1, 2, 1, ["C5"], "q", accidentals={"C5": "n"})  # C natural
    s.chord(1, 3, 1, ["C5"], "q")  # the natural carries through the bar
    s.chord(1, 4, 1, ["B4"], "q", accidentals={"B4": "b"})  # Bb4
    s.chord(1, 1, 2, ["D3"], "h")
    s.clef_change(1, 3, 2, "treble")  # the left hand goes into the treble clef mid-bar
    s.chord(1, 3, 2, ["E4"], "h")  # E4 on the bottom line of the lower staff, now in treble
    s.chord(2, 1, 1, ["C5"], "q")  # a new bar: C#5 from the key again
    s.chord(2, 2, 1, ["G5"], "q")  # under an 8va line: G6
    s.octave_line(2, 2, 2.5)
    s.chord(2, 3, 1, ["A5"], "h")  # past it
    s.clef_change(2, 1, 2, "bass")
    # back in the bass clef: a C5 nearer the treble staff than the bass, hung on the bass staff's ledger lines
    c5 = s.chord(2, 1, 2, ["C#5"], "w")
    for n in ("C4", "E4", "G4", "B4"):
        s.ledger(c5[0].x, s.y(n, 2))
    score = pdf_score.read(s.save(tmp_path / "song.pdf"))
    first, second = (pdf_check.read_bar(b, Fraction(4)) for b in score.bars)
    assert [p for _, p, _ in notes(first, 1)] == ["F#5", "C5", "C5", "Bb4"]
    assert [p for _, p, _ in notes(first, 2)] == ["D3", "E4"]
    assert [p for _, p, _ in notes(second, 1)] == ["C#5", "G6", "A5"]
    assert [p for _, p, _ in notes(second, 2)] == ["C#5"]


def test_key_signature_on_a_left_hand_that_starts_in_the_treble_clef(tmp_path):
    """The key's sharps sit at treble places on a lower staff in the treble clef (Go the Distance, bar 2)."""
    s = Sheet((1,), key=2, clefs=("treble", "treble"))
    s.chord(1, 1, 1, ["D5"], "w")
    s.chord(1, 1, 2, ["C4", "F4"], "w")  # C#4 and F#4 from the key
    reading = pdf_check.read_bar(pdf_score.read(s.save(tmp_path / "song.pdf")).bars[0], Fraction(4))
    assert [p for _, p, _ in notes(reading, 2)] == ["C#4", "F#4"]


@pytest.mark.parametrize("glyphs", [False, True])
def test_octave_signs_as_text_or_glyphs_move_the_notes_under_them(tmp_path, glyphs):
    """8va over the treble and 8vb under the bass, written as words or as Opus Special's "8" + "va"/"vb" glyphs
    (Go the Distance): the file's pitch is the sounding one."""
    s = Sheet((1,))
    s.chord(1, 1, 1, ["E5"], "h")
    s.chord(1, 3, 1, ["E5"], "h")
    s.octave_line(1, 3, 3.5, glyphs=glyphs)
    s.chord(1, 1, 2, ["D2"], "h")
    s.octave_line(1, 1, 1.5, staff=2, down=True, glyphs=glyphs)
    s.chord(1, 3, 2, ["D2"], "h")
    reading = pdf_check.read_bar(pdf_score.read(s.save(tmp_path / "song.pdf")).bars[0], Fraction(4))
    assert [p for _, p, _ in notes(reading, 1)] == ["E5", "E6"]
    assert [p for _, p, _ in notes(reading, 2)] == ["D1", "D2"]


def test_reader_durations_from_heads_beams_flags_dots_rests_and_triplets(tmp_path):
    s = Sheet((1,))
    s.chord(1, 1, 1, ["C5"], "q", dots=1)
    s.chord(1, 2.5, 1, ["D5"], "8")  # a flag
    trip = [s.chord(1, 3 + k / 3, 1, [n], "8", beamed=True) for k, n in enumerate(["E5", "F5", "G5"])]
    s.beam(trip)
    s.text("3", s.x(1, 3.33), s.tops[0][1] - 4 * SP)
    sixteenths = [s.chord(1, 4 + k / 4, 1, [n], "16", beamed=True) for k, n in enumerate(["A5", "G5", "F5", "E5"])]
    s.beam(sixteenths, count=2)
    s.rest(1, 1, 2, "h")
    s.chord(1, 3, 2, ["C3", "G3"], "h", up=False)
    reading = pdf_check.read_bar(pdf_score.read(s.save(tmp_path / "song.pdf")).bars[0], Fraction(4))
    assert reading.why == ""
    assert notes(reading, 1) == [
        (Fraction(0), "C5", Fraction(3, 2)), (Fraction(3, 2), "D5", Fraction(1, 2)),
        (Fraction(2), "E5", Fraction(1, 3)), (Fraction(7, 3), "F5", Fraction(1, 3)), (Fraction(8, 3), "G5", Fraction(1, 3)),
        (Fraction(3), "A5", Fraction(1, 4)), (Fraction(13, 4), "G5", Fraction(1, 4)),
        (Fraction(7, 2), "F5", Fraction(1, 4)), (Fraction(15, 4), "E5", Fraction(1, 4))]
    assert notes(reading, 2) == [(Fraction(2), "C3", Fraction(2)), (Fraction(2), "G3", Fraction(2))]


def test_reader_ties_carry_the_note_and_its_accidental_over_the_barline(tmp_path):
    s = Sheet((2,))
    a = s.chord(1, 4, 1, ["F5"], "q", accidentals={"F5": "#"})
    b = s.chord(2, 1, 1, ["F5"], "q")
    s.tie(a[0], b[0])
    for beat, n in ((1, "C5"), (2, "D5"), (3, "E5")):
        s.chord(1, beat, 1, [n], "q")
    for beat, n in ((2, "G5"), (3, "A5"), (4, "B5")):
        s.chord(2, beat, 1, [n], "q")
    for bar in (1, 2):
        s.chord(bar, 1, 2, ["C3"], "w")
    score = pdf_score.read(s.save(tmp_path / "song.pdf"))
    last = next(h for c in score.bars[0].measure.chords for h in c.heads if h.diatonic == 38)
    first = next(h for c in score.bars[1].measure.chords for h in c.heads if h.diatonic == 38)
    assert last.ties == {"start"} and first.ties == {"stop"}
    assert first.midi == 78  # F#5: the tie keeps the sharp


def test_a_cross_staff_stem_is_not_a_barline(tmp_path):
    s = Sheet((1,))
    for beat in (1, 2, 3, 4):  # each run starts on the bass staff and goes on in the treble
        low = s.chord(1, beat, 2, ["B3"], "16", beamed=True)
        x, _ = low[0].stem  # type: ignore[attr-defined]
        tip = s.tops[0][1] - 2 * SP
        s.line(x, s.tops[0][2] - SP, x, tip)  # the long stem up to the beam over the treble
        low[0].stem = (x, tip)  # type: ignore[attr-defined]
        run = [low] + [s.chord(1, beat + k / 4, 1, [n], "16", beamed=True) for k, n in enumerate(["D5", "F5", "Bb5"], 1)]
        for h in run[1:]:
            h[0].stem = (h[0].stem[0], tip)  # type: ignore[attr-defined]
        s.beam(run, count=2)
    score = pdf_score.read(s.save(tmp_path / "song.pdf"))
    assert len(score.bars) == 1


# ---------------------------------------------------------------- the audit

def song(tmp_path, s: Sheet, measures: list[str], **kw):
    pdf = s.save(tmp_path / "song.pdf")
    xml = tmp_path / "song.musicxml"
    xml.write_text(export_xml(measures, **kw), encoding="utf-8")
    return pdf, xml


def test_a_bar_that_matches_is_ok(tmp_path):
    s = Sheet((1,))
    simple_bar(s, 1, ["C5", "D5", "E5", "F5"], ["C3", "G3"])
    pdf, xml = song(tmp_path, s, [simple_xml(["C5", "D5", "E5", "F5"], ["C3", "G3"])])
    report = pdf_check.check(pdf, xml)
    assert [b.status for b in report.bars] == ["ok"]
    assert report.to_json()["status"] == "ok"


def test_bass_notes_filed_on_the_treble_staff_are_found_and_rewritten(tmp_path):
    """Audiveris read the left hand's E2+B2 on the treble staff, in the treble clef: C#4+G4. The pitch classes
    nearly add up; the per-staff comparison doesn't."""
    s = Sheet((1,), key=2)
    simple_bar(s, 1, ["F5", "E5", "D5", "C5"], ["E2", "B2"])
    wrong = ("".join(xnote(n, Q) for n in ["F#5", "E5", "D5", "C#5"]) + backup(4 * Q)
             + xnote("C#4", 4 * Q, voice=2) + xnote("G4", 4 * Q, voice=2, chord=True))
    pdf, xml = song(tmp_path, s, [wrong])
    out = tmp_path / "out.musicxml"
    report = pdf_check.check(pdf, xml, out)
    bar = report.bars[0]
    assert bar.status == "filled"
    assert "right hand: the PDF has nothing more, the file C#4 G4" in bar.reasons[0]
    assert "left hand: the PDF has E2 B2" in bar.reasons[1]
    assert [(p[1], p[2]) for p in placed(out) if p[1] == 2] == [(2, "B2"), (2, "E2")]
    assert pdf_check.check(pdf, out).found == []


def test_a_voice_shifted_early_is_found_by_its_onsets(tmp_path):
    """A leading rest dropped from an inner voice: the same notes, a beat early."""
    s = Sheet((1,))
    simple_bar(s, 1, ["C5", "D5", "E5", "F5"], ["C3"])
    s.rest(1, 1, 2, "q")
    s.chord(1, 2, 2, ["G3"], "h", up=True, dots=1)
    xml_bar = (simple_xml(["C5", "D5", "E5", "F5"], ["C3"]) + backup(4 * Q)
               + xnote("G3", 3 * Q, voice=6, staff=2) + f"<forward><duration>{Q}</duration><voice>6</voice>"
               "<staff>2</staff></forward>")
    pdf, xml = song(tmp_path, s, [xml_bar])
    report = pdf_check.check(pdf, xml, tmp_path / "out.musicxml")
    assert report.bars[0].status == "filled"
    assert "left hand: notes start at other times (G3 at beat 1 in the file; the PDF has beat 2)" in report.bars[0].reasons
    assert (1.0, 2, "G3", 3) in [(p[0], p[1], p[2], p[3]) for p in placed(tmp_path / "out.musicxml")]


def test_a_bar_whose_timing_doesnt_read_is_left_for_review(tmp_path):
    s = Sheet((1,))
    s.chord(1, 1, 1, ["C5"], "q")
    s.glyph("œ", s.x(1, 2), s.y("D5", 1))  # a black head with no stem: its value can't be read
    s.chord(1, 3, 1, ["E5"], "h")
    s.chord(1, 1, 2, ["C3"], "w")
    pdf, xml = song(tmp_path, s, [simple_xml(["C5", "E5"], ["C3"])])
    report = pdf_check.check(pdf, xml, tmp_path / "out.musicxml", )
    assert report.bars[0].status == "differs"
    assert "has no stem" in report.bars[0].reasons[-1]
    assert report.needs_review == [1]


def test_omr_repair_flags_that_match_the_pdf_are_cleared(tmp_path):
    s = Sheet((2,))
    for bar in (1, 2):
        simple_bar(s, bar, ["C5", "D5", "E5", "F5"], ["C3"])
    pdf, xml = song(tmp_path, s, [simple_xml(["C5", "D5", "E5", "F5"], ["C3"])] * 2)
    flags = tmp_path / "repair.json"
    flags.write_text(json.dumps({"needs_review": [2]}), encoding="utf-8")
    report = pdf_check.check(pdf, xml, repair_report=flags)
    assert report.needs_review == []


def test_a_bar_the_export_lost_is_put_back(tmp_path):
    s = Sheet((3,))
    tunes = [["C5", "D5", "E5", "F5"], ["G5", "A5", "B5", "C6"], ["E5", "D5", "C5", "D5"]]
    for bar, t in enumerate(tunes, 1):
        simple_bar(s, bar, t, ["C3"])
    pdf, xml = song(tmp_path, s, [simple_xml(tunes[0], ["C3"]), simple_xml(tunes[2], ["C3"])], vocal=["", ""])
    out = tmp_path / "out.musicxml"
    report = pdf_check.check(pdf, xml, out)
    assert [(b.number, b.status) for b in report.bars] == [(1, "ok"), (2, "inserted"), (3, "ok")]
    assert [p[2] for p in placed(out, 2) if p[1] == 1] == ["G5", "A5", "B5", "C6"]
    root = musicxml.read_root(out)
    assert [len(p.findall("measure")) for p in root.findall("part")] == [3, 3]  # the vocal part got a bar too
    assert [m.get("number") for m in root.find("part").findall("measure")] == ["1", "2", "3"]


def test_the_pdfs_time_signature_wins(tmp_path):
    s = Sheet((1,), time=(3, 4))
    for beat, n in ((1, "C5"), (2, "D5"), (3, "E5")):
        s.chord(1, beat, 1, [n], "q")
    s.chord(1, 1, 2, ["C3"], "h", dots=1)
    xml_bar = "".join(xnote(n, Q) for n in ["C5", "D5", "E5"]) + backup(3 * Q) + xnote("C3", 3 * Q, voice=5, staff=2)
    pdf, xml = song(tmp_path, s, [xml_bar])  # the export says 4/4
    out = tmp_path / "out.musicxml"
    report = pdf_check.check(pdf, xml, out)
    assert report.bars[0].reasons[0] == "time signature: 3/4 in the PDF, 4/4 in the file"
    assert report.bars[0].status == "filled"
    t = musicxml.read_root(out).find("part/measure/attributes/time")
    assert (t.findtext("beats"), t.findtext("beat-type")) == ("3", "4")


def test_the_pdfs_time_signature_keeps_its_beat(tmp_path):
    """6/8 lasts as long as 3/4, but it isn't 3/4."""
    s = Sheet((1,), time=(6, 8))
    s.chord(1, 1, 1, ["C5"], "q", dots=1)
    s.chord(1, 2.5, 1, ["D5"], "q", dots=1)
    s.chord(1, 1, 2, ["C3"], "h", dots=1)
    xml_bar = xnote("C5", 6) + xnote("D5", 6) + backup(3 * Q) + xnote("C3", 3 * Q, voice=5, staff=2)
    pdf, xml = song(tmp_path, s, [xml_bar], time=(3, 4))
    out = tmp_path / "out.musicxml"
    report = pdf_check.check(pdf, xml, out)
    assert report.bars[0].reasons == ["time signature: 6/8 in the PDF, 3/4 in the file"]
    t = musicxml.read_root(out).find("part/measure/attributes/time")
    assert (t.findtext("beats"), t.findtext("beat-type")) == ("6", "8")


def test_grace_notes_go_before_their_notes_in_order(tmp_path):
    s = Sheet((1,))
    for beat, n, v in ((1, "C5", "q"), (2, "D5", "q"), (3, "E5", "h")):
        s.chord(1, beat, 1, [n], v)
    for beat, n in ((1.7, "B4"), (2.55, "C5"), (2.75, "D5")):  # one before D5, two before E5
        s.chord(1, beat, 1, [n], "8", small=True)
    s.chord(1, 1, 2, ["C3"], "w")
    pdf, xml = song(tmp_path, s, [simple_xml(["C5", "D5", "E5"], ["C3"]).replace(
        xnote("E5", Q), xnote("E5", 2 * Q))])
    out = tmp_path / "out.musicxml"
    report = pdf_check.check(pdf, xml, out)
    assert report.bars[0].status == "filled"
    written = [n.findtext("pitch/step") + n.findtext("pitch/octave") + ("g" if n.find("grace") is not None else "")
               for n in musicxml.read_root(out).find("part/measure").findall("note") if n.findtext("staff") == "1"]
    assert written == ["C5", "B4g", "D5", "C5g", "D5g", "E5"]


def test_cli_writes_the_checked_file_and_a_report(tmp_path, capsys):
    s = Sheet((1,))
    simple_bar(s, 1, ["C5", "D5", "E5", "F5"], ["C3"])
    pdf, xml = song(tmp_path, s, [simple_xml(["C5", "D5", "E5", "G5"], ["C3"])])
    out, rep = tmp_path / "out.musicxml", tmp_path / "report.json"
    assert cli.main(["pdf-check", str(pdf), str(xml), "-o", str(out), "--json", str(rep)]) == 0
    assert "1 wrong: 1 rewritten from the PDF" in capsys.readouterr().out
    data = json.loads(rep.read_text(encoding="utf-8"))
    assert data["found"] == [1] and data["filled"] == [1] and data["status"] == "ok"
    assert [p[2] for p in placed(out) if p[1] == 1] == ["C5", "D5", "E5", "F5"]
