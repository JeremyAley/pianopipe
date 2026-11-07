"""The Audiveris .omr reader and the OMR repair, on small synthetic projects (tests/omr_builder.py)."""

from __future__ import annotations

import json
from fractions import Fraction

from pianopipe import cli, musicxml, omr, omr_repair

from omr_builder import N, System, backup, bass, export_xml, omr_bytes, treble, write_project, xnote

Q = 4  # divisions per quarter note in the exports


def sixteenths(x0: float, names: list[str], group: str, beams: int = 2, **kw) -> list[N]:
    return [N(1, x0 + 30 * k, (treble(n),), beam=group, beams=beams, **kw) for k, n in enumerate(names)]


def placed(path, bar: int = 1) -> list[tuple[float, int, str, Fraction]]:
    """(onset, staff, pitch name, duration) of one bar of a repaired file, as the importer reads it."""
    root = musicxml.read_root(path)
    part = musicxml._parse_part(omr_repair._piano_part(root), {})
    names = "C C# D D# E F F# G G# A A# B".split()
    return sorted((float(n.pos), n.staff, f"{names[n.pitch % 12]}{n.pitch // 12 - 1}", n.dur)
                  for n in part.measures[bar - 1].notes)


def repair(tmp_path, systems, measures, image=False, **kw):
    xml, project = write_project(tmp_path, systems, measures, image, **kw)
    out = tmp_path / "out.musicxml"
    return omr_repair.repair(xml, project, out), out


# ---------------------------------------------------------------- reading the project

def test_reader_durations_from_heads_beams_flags_dots_and_tuplets(tmp_path):
    bar = [
        N(1, 20, (treble("C5"),), beam="a", beams=2), N(1, 50, (treble("D5"),), beam="a", beams=2),
        N(1, 90, (treble("E5"), treble("G5")), shape="NOTEHEAD_VOID", dots=1),  # dotted half: one dot per head
        N(1, 200, (treble("F5"),), flags=1),
        N(1, 260, (treble("A4"),), beam="t", beams=1, tuplet="t"), N(1, 290, (treble("B4"),), beam="t", beams=1, tuplet="t"),
        N(2, 20, (bass("C3"),), shape="WHOLE_NOTE", stem=None),
        N(2, 300, (), rest="QUARTER_REST"),
    ]
    _, project = write_project(tmp_path, [System([bar])], [""])
    m = omr.read(project).measures[0]
    values = [(c.staff, c.rest or c.heads[0].midi, c.value) for c in m.chords]
    assert (1, 72, Fraction(1, 4)) in values
    assert (1, 76, Fraction(3)) in values
    assert (1, 77, Fraction(1, 2)) in values
    assert (1, 69, Fraction(1, 3)) in values
    assert (2, 48, Fraction(4)) in values
    assert (2, "QUARTER_REST", Fraction(1)) in values


def test_reader_pitches_from_clef_key_and_accidentals(tmp_path):
    bar = [
        N(1, 20, (treble("F5"),)),  # F#5 from the key
        N(1, 120, (treble("C5"),), alters={treble("C5"): "NATURAL"}),  # C natural, though the key has C#
        N(1, 220, (treble("C5"),)),  # the natural carries through the bar
        N(2, 20, (bass("B2"),), alters={bass("B2"): "FLAT"}),
        N(2, 120, (bass("B2"),)),
    ]
    _, project = write_project(tmp_path, [System([bar], fifths=2)], [""])
    m = omr.read(project).measures[0]
    assert [h.midi for c in sorted(m.chords, key=lambda c: (c.staff, c.x)) for h in c.heads] == [78, 72, 72, 46, 46]


def test_reader_takes_a_chord_without_a_staff_from_its_heads(tmp_path):
    """Audiveris 5.11 leaves some head chords with no staff attribute (seen on a Verovio page's left hand); read on
    the top staff, the left hand came out in treble pitches and omr-repair 'fixed' right bars into wrong ones."""
    import re
    import zipfile

    bar = [N(1, 20, (treble("C5"),), shape="WHOLE_NOTE", stem=None), N(2, 20, (bass("C3"),), shape="WHOLE_NOTE", stem=None)]
    _, project = write_project(tmp_path, [System([bar])], [""])
    with zipfile.ZipFile(project) as z:
        files = {n: z.read(n) for n in z.namelist()}
    xml = files["sheet#1/sheet#1.xml"].decode()
    files["sheet#1/sheet#1.xml"] = re.sub(r'(<head-chord [^>]*?) staff="\d+"', r"\1", xml).encode()
    with zipfile.ZipFile(project, "w") as z:
        for n, data in files.items():
            z.writestr(n, data)
    m = omr.read(project).measures[0]
    assert sorted((c.staff, c.heads[0].midi) for c in m.chords) == [(1, 72), (2, 48)]


def test_reader_lines_up_with_the_export_when_audiveris_split_a_system(tmp_path):
    systems = [System([[]], piano=False, vocal=True), System([[N(1, 20, (treble("C5"),), shape="WHOLE_NOTE", stem=None)]])]
    _, project = write_project(tmp_path, systems, ["", ""])
    ms = omr.read(project).measures
    assert [(m.absent, m.alone) for m in ms] == [(True, False), (False, True)]


# ---------------------------------------------------------------- repairs

def test_a_missed_beam_is_put_back_and_the_bar_retimed(tmp_path):
    # four groups of sixteenths; Audiveris saw one beam on the second group and exported eighths
    names = ["D6", "D5", "A5", "D5"]
    bar = (sixteenths(20, names, "a") + sixteenths(150, names, "b", beams=1) + sixteenths(280, names, "c")
           + sixteenths(410, names, "d") + [N(2, 20, (bass("D3"),), shape="WHOLE_NOTE", stem=None, voice=5)])
    rh = "".join(xnote(n, 1, x=20 + 30 * k) for k, n in enumerate(names))
    rh += "".join(xnote(n, 2, x=150 + 30 * k) for k, n in enumerate(names))
    rh += "".join(xnote(n, 1, x=280 + 30 * k) for k in range(2) for n in names)
    export = rh + backup(20) + xnote("D3", 16, voice=5, staff=2, x=20)
    report, out = repair(tmp_path, [System([bar])], [export])
    assert report.repaired == [1] and not report.needs_review
    assert "added a beam Audiveris missed to 4 beamed notes" in report.bars[0].reasons
    rh_times = [(t, d) for t, s, _, d in placed(out) if s == 1]
    assert rh_times == [(k / 4, Fraction(1, 4)) for k in range(16)]
    assert musicxml.load(out).song.notes  # the importer reads it


def test_two_voices_on_the_same_notes_keep_their_place(tmp_path):
    # quarter, quarter, quarter rest, eighth rest, two sixteenths printed with stems both ways (melody and inner
    # voice); Audiveris put the sixteenths in voices of their own and left the melody short
    up = [N(1, 410 + 40 * k, (treble(n),), beam="u", beams=2, voice=2) for k, n in enumerate(["D4", "F4"])]
    down = [N(1, 410 + 40 * k, (treble(n),), stem="down", beam="d", beams=2, voice=3) for k, n in enumerate(["D4", "F4"])]
    bar = [N(1, 20, (treble("B4"),)), N(1, 150, (treble("C5"),)), N(1, 280, (), rest="QUARTER_REST"),
           N(1, 350, (), rest="EIGHTH_REST"),
           N(2, 20, (bass("G2"),), shape="NOTEHEAD_VOID", voice=5), N(2, 280, (bass("A2"),), shape="NOTEHEAD_VOID", voice=5)] + up + down
    export = (xnote("B4", 4, x=20) + xnote("C5", 4, x=150) + xnote("rest", 4) + xnote("rest", 2) + backup(14)
              + xnote("D4", 1, voice=2, x=410) + xnote("F4", 1, voice=2, x=450) + backup(2)
              + xnote("D4", 1, voice=3, x=410) + xnote("F4", 1, voice=3, x=450) + backup(2)
              + xnote("G2", 8, voice=5, staff=2, x=20) + xnote("A2", 8, voice=5, staff=2, x=280))
    report, out = repair(tmp_path, [System([bar])], [export])
    assert report.repaired == [1]
    rh = [(t, n) for t, s, n, _ in placed(out) if s == 1]
    assert rh == [(0.0, "B4"), (1.0, "C5"), (3.5, "D4"), (3.5, "D4"), (3.75, "F4"), (3.75, "F4")]
    song = musicxml.load(out).song
    assert len([n for n in song.notes if n.pitch == 62]) == 1  # one key, one note


def test_chords_left_out_of_every_voice_come_back(tmp_path):
    bar = [N(1, 20, (treble("C5"),)), N(1, 150, (treble("D5"),)), N(1, 280, (treble("E5"),), voice=None),
           N(1, 410, (treble("F5"),), voice=None), N(2, 20, (bass("C3"),), shape="WHOLE_NOTE", stem=None, voice=5)]
    export = xnote("C5", 4, x=20) + xnote("D5", 4, x=150) + backup(8) + xnote("C3", 16, voice=5, staff=2, x=20)
    report, out = repair(tmp_path, [System([bar])], [export])
    assert report.repaired == [1]
    assert "2 chords were left out of every voice" in report.bars[0].reasons
    assert [(t, n) for t, s, n, _ in placed(out) if s == 1] == [(0.0, "C5"), (1.0, "D5"), (2.0, "E5"), (3.0, "F5")]


def test_a_bar_that_was_fine_is_left_alone(tmp_path):
    bar = [N(1, 20 + 130 * k, (treble(n),)) for k, n in enumerate(["C5", "D5", "E5", "F5"])]
    bar.append(N(2, 20, (bass("C3"),), shape="WHOLE_NOTE", stem=None, voice=5))
    export = ("".join(xnote(n, 4, x=20 + 130 * k) for k, n in enumerate(["C5", "D5", "E5", "F5"])) + backup(16)
              + xnote("C3", 16, voice=5, staff=2, x=20))
    xml, project = write_project(tmp_path, [System([bar])], [export])
    before = musicxml.read_root(xml)
    report = omr_repair.repair_tree(before, omr.read(project))
    assert [b.status for b in report.bars] == ["ok"] and report.to_json()["status"] == "ok"


def test_a_missing_note_is_timed_around_and_listed_for_review(tmp_path):
    # quarter, quarter, (a quarter Audiveris didn't see), quarter
    bar = [N(1, 20, (treble("C5"),)), N(1, 150, (treble("D5"),)), N(1, 410, (treble("F5"),)),
           N(2, 20, (bass("C3"),), shape="WHOLE_NOTE", stem=None, voice=5)]
    export = (xnote("C5", 4, x=20) + xnote("D5", 4, x=150) + xnote("F5", 4, x=410) + backup(12)
              + xnote("C3", 16, voice=5, staff=2, x=20))
    report, out = repair(tmp_path, [System([bar])], [export])
    assert report.needs_review == [1] and report.bars[0].status == "partial"
    assert [(t, n) for t, s, n, _ in placed(out) if s == 1] == [(0.0, "C5"), (1.0, "D5"), (3.0, "F5")]
    assert report.to_json()["status"] == "needs_review"


def test_beams_audiveris_missed_are_found_in_the_page_image(tmp_path):
    # three quarters and four sixteenths... whose beams are on the page but not in the project
    bar = [N(1, 20, (treble("C5"),)), N(1, 150, (treble("D5"),)), N(1, 280, (treble("E5"),))]
    bar += [N(1, 410 + 40 * k, (treble(n),), beam="b", beams=2, beam_inter=False) for k, n in enumerate(["F5", "G5", "A5", "B5"])]
    bar += [N(2, 20, (bass("C3"),), shape="NOTEHEAD_VOID", stem=None, voice=5),
            N(2, 280, (bass("G2"),), shape="NOTEHEAD_VOID", stem=None, voice=5)]
    export = (xnote("C5", 4, x=20) + xnote("D5", 4, x=150) + xnote("E5", 4, x=280)
              + "".join(xnote(n, 4, x=410 + 40 * k) for k, n in enumerate(["F5", "G5", "A5", "B5"])) + backup(28)
              + xnote("C3", 8, voice=5, staff=2, x=20)
              + xnote("G2", 8, voice=5, staff=2, x=280))
    report, out = repair(tmp_path, [System([bar])], [export], image=True)
    assert report.repaired == [1]
    assert "found beams on 4 notes in the page image that Audiveris missed" in report.bars[0].reasons
    assert [(t, d) for t, s, _, d in placed(out) if s == 1][-4:] == [(3 + k / 4, Fraction(1, 4)) for k in range(4)]


def test_no_image_no_beam_evidence(tmp_path):
    bar = [N(1, 20, (treble("C5"),), beam="b", beams=2, beam_inter=False), N(1, 50, (treble("D5"),), beam="b", beams=2, beam_inter=False)]
    _, project = write_project(tmp_path, [System([bar])], [""])
    assert all(c.seen_beams == 0 for c in omr.read(project).measures[0].chords)


def test_an_unread_note_head_on_the_page_flags_the_bar(tmp_path):
    bar = [N(1, 20, (treble("C5"),), voice=None), N(1, 150, (treble("D5"),)), N(1, 280, (treble("E5"),)),
           N(1, 410, (treble("F5"),)), N(2, 20, (bass("C3"),), shape="WHOLE_NOTE", stem=None, voice=5)]
    system = System([bar], free_heads=[(0, 1, 300, treble("A4"))])
    export = (xnote("D5", 4, x=150) + xnote("E5", 4, x=280) + xnote("F5", 4, x=410) + backup(12)
              + xnote("C3", 16, voice=5, staff=2, x=20))
    report, _ = repair(tmp_path, [system], [export], image=True)
    assert report.bars[0].status == "partial"
    assert "1 marks on the staves look like note heads Audiveris didn't read" in report.bars[0].reasons


def test_a_split_system_is_merged_back(tmp_path):
    # Audiveris made the vocal staff one system and the piano the next: the export has an empty piano bar, then
    # a bar where the voice rests and the piano plays
    systems = [System([[]], piano=False, vocal=True),
               System([[N(1, 20, (treble("C5"),), shape="WHOLE_NOTE", stem=None), N(2, 20, (bass("C3"),), shape="WHOLE_NOTE", stem=None, voice=5)]])]
    piano = ["<note><rest/><duration>16</duration><voice>1</voice><staff>1</staff></note>",
             xnote("C5", 16, x=20) + backup(16) + xnote("C3", 16, voice=5, staff=2, x=20)]
    vocal = [xnote("E5", 16), "<note><rest/><duration>16</duration><voice>1</voice></note>"]
    report, out = repair(tmp_path, systems, piano, vocal=vocal)
    assert report.removed == [2] and [b.number for b in report.bars] == [1]
    root = musicxml.read_root(out)
    assert all(len(p.findall("measure")) == 1 for p in root.findall("part"))
    assert [(t, n) for t, _, n, _ in placed(out)] == [(0.0, "C5"), (0.0, "C3")]


def test_a_missed_time_signature_is_put_in(tmp_path):
    # 2/4, 4/4, then a 2/4 bar whose time signature Audiveris missed (the export still says 4/4)
    half = lambda: [N(1, 20, (treble("C5"),), shape="NOTEHEAD_VOID"), N(2, 20, (bass("C3"),), shape="NOTEHEAD_VOID", voice=5)]
    four = [N(1, 20 + 130 * k, (treble("D5"),)) for k in range(4)] + [N(2, 20, (bass("C3"),), shape="WHOLE_NOTE", stem=None, voice=5)]
    two = [N(1, 20, (treble("E5"),)), N(1, 300, (treble("F5"),)), N(2, 20, (bass("C3"),), shape="NOTEHEAD_VOID", voice=5)]
    measures = [xnote("C5", 8, x=20) + backup(8) + xnote("C3", 8, voice=5, staff=2, x=20),
                "".join(xnote("D5", 4, x=20 + 130 * k) for k in range(4)) + backup(16) + xnote("C3", 16, voice=5, staff=2, x=20),
                xnote("E5", 4, x=20) + xnote("F5", 4, x=300) + backup(8) + xnote("C3", 8, voice=5, staff=2, x=20)]
    report, out = repair(tmp_path, [System([half(), four, two])], measures, time=(2, 4), times={1: (4, 4)})
    assert report.bars[2].status == "repaired"
    assert "it's in 2/4: Audiveris missed the time signature" in report.bars[2].reasons
    m3 = omr_repair._piano_part(musicxml.read_root(out)).findall("measure")[2]
    assert m3.findtext("attributes/time/beats") == "2"
    assert musicxml.load(out).bars[-1] - musicxml.load(out).bars[-2] == 2 * musicxml.TPB


def test_pitches_follow_the_page_when_the_export_disagrees(tmp_path):
    # Audiveris's export put this bar an octave up (it misread the clef), the heads say otherwise
    bar = [N(1, 20 + 130 * k, (treble(n),)) for k, n in enumerate(["C5", "D5", "E5", "F5"])]
    bar.append(N(2, 20, (bass("C3"),), shape="WHOLE_NOTE", stem=None, voice=5))
    export = ("".join(xnote(n, 4, x=20 + 130 * k) for k, n in enumerate(["C6", "D6", "E6", "F6"])) + backup(16)
              + xnote("C3", 16, voice=5, staff=2, x=20))
    report, out = repair(tmp_path, [System([bar])], [export])
    assert "the export's pitches differ from the page's clefs" in report.bars[0].reasons
    assert [n for _, s, n, _ in placed(out) if s == 1] == ["C5", "D5", "E5", "F5"]


def test_ties_come_from_the_project(tmp_path):
    bar = [N(1, 20, (treble("C5"),), shape="NOTEHEAD_VOID", ties={treble("C5"): "LEFT"}),
           N(1, 280, (treble("C5"),), shape="NOTEHEAD_VOID", voice=None, ties={treble("C5"): "RIGHT"}),
           N(2, 20, (bass("C3"),), shape="WHOLE_NOTE", stem=None, voice=5)]
    export = xnote("C5", 8, x=20) + backup(8) + xnote("C3", 16, voice=5, staff=2, x=20)
    _, out = repair(tmp_path, [System([bar])], [export])
    c5 = [n for n in musicxml.load(out).song.notes if n.pitch == 72]
    assert len(c5) == 1 and c5[0].end - c5[0].start == 4 * musicxml.TPB  # tied into one whole-bar note


def test_measures_that_dont_line_up_are_left_alone(tmp_path):
    bar = [N(1, 20, (treble("C5"),), shape="WHOLE_NOTE", stem=None)]
    report, out = repair(tmp_path, [System([bar])], [xnote("C5", 16, x=20), xnote("D5", 16, x=20)])
    assert report.needs_review == [1, 2] and report.warnings
    assert len(omr_repair._piano_part(musicxml.read_root(out)).findall("measure")) == 2


def test_cli_writes_the_repaired_file_and_a_report(tmp_path, capsys):
    bar = [N(1, 20, (treble("C5"),)), N(1, 150, (treble("D5"),)), N(1, 280, (treble("E5"),), voice=None),
           N(1, 410, (treble("F5"),), voice=None), N(2, 20, (bass("C3"),), shape="WHOLE_NOTE", stem=None, voice=5)]
    export = xnote("C5", 4, x=20) + xnote("D5", 4, x=150) + backup(8) + xnote("C3", 16, voice=5, staff=2, x=20)
    xml, project = write_project(tmp_path, [System([bar])], [export])
    out, rep = tmp_path / "fixed.musicxml", tmp_path / "report.json"
    assert cli.main(["omr-repair", str(xml), str(project), "-o", str(out), "--json", str(rep)]) == 0
    assert "1 repaired" in capsys.readouterr().out
    data = json.loads(rep.read_text(encoding="utf-8"))
    assert data["status"] == "ok" and data["repaired"] == [1] and data["needs_review"] == []
    assert len(musicxml.load(out).song.notes) == 5


def _scale(names: list[str]) -> tuple[list[N], str]:
    """A bar of four quarter notes over a whole note, as printed and as exported."""
    printed = [N(1, 20 + 130 * k, (treble(n),)) for k, n in enumerate(names)]
    printed.append(N(2, 20, (bass("C3"),), shape="WHOLE_NOTE", stem=None, voice=5))
    exported = ("".join(xnote(n, 4, x=20 + 130 * k) for k, n in enumerate(names)) + backup(16)
                + xnote("C3", 16, voice=5, staff=2, x=20))
    return printed, exported


def test_movement_exports_are_joined(tmp_path):
    # Audiveris made the indented coda system a movement of its own: song.mvt1.mxl, song.mvt2.mxl
    bars = [_scale(["C5", "D5", "E5", "F5"]), _scale(["G5", "F5", "E5", "D5"]), _scale(["C5", "E5", "G5", "C6"])]
    (tmp_path / "song.omr").write_bytes(omr_bytes([System([b[0] for b in bars[:2]]), System([bars[2][0]])]))
    (tmp_path / "song.mvt1.mxl").write_text(export_xml([b[1] for b in bars[:2]]), encoding="utf-8")
    (tmp_path / "song.mvt2.mxl").write_text(export_xml([bars[2][1]]), encoding="utf-8")
    out = tmp_path / "out.musicxml"
    report = omr_repair.repair([tmp_path / "song.mvt1.mxl", tmp_path / "song.mvt2.mxl"], tmp_path / "song.omr", out)
    assert report.movements == 2 and not report.warnings and [b.status for b in report.bars] == ["ok"] * 3
    measures = omr_repair._piano_part(musicxml.read_root(out)).findall("measure")
    assert [m.get("number") for m in measures] == ["1", "2", "3"]
    assert measures[2].find("print").get("new-system") == "yes"  # where the importer looks for a coda
    assert [n for _, s, n, _ in placed(out, 3) if s == 1] == ["C5", "E5", "G5", "C6"]
    # the command line takes the folder (or a glob), movements in order
    assert omr_repair.export_paths([str(tmp_path)], tmp_path / "song.omr") == [tmp_path / "song.mvt1.mxl",
                                                                                tmp_path / "song.mvt2.mxl"]
    assert omr_repair.export_paths([str(tmp_path / "song.mvt*.mxl")], tmp_path / "song.omr")[1].name == "song.mvt2.mxl"
    assert cli.main(["omr-repair", str(tmp_path), str(tmp_path / "song.omr"), "-o", str(out)]) == 0
    assert len(omr_repair._piano_part(musicxml.read_root(out)).findall("measure")) == 3


def test_a_measure_the_exporter_dropped_is_rebuilt_and_flagged(tmp_path):
    # "Error visiting Measure{#9}": the export goes 1, 3; the project has all three
    bars = [_scale(["C5", "D5", "E5", "F5"]), _scale(["G5", "F5", "E5", "D5"]), _scale(["C5", "E5", "G5", "C6"])]
    xml, project = write_project(tmp_path, [System([b[0] for b in bars])], [bars[0][1], bars[2][1]])
    xml.write_text(xml.read_text(encoding="utf-8").replace('<measure number="2">', '<measure number="3">'), encoding="utf-8")
    out = tmp_path / "out.musicxml"
    report = omr_repair.repair(xml, project, out)
    assert [b.status for b in report.bars] == ["ok", "partial", "ok"]
    assert "Audiveris's export dropped this measure (it had no measure 2)" in report.bars[1].reasons
    assert report.needs_review == [2] and any("has no measure 2" in w for w in report.warnings)
    assert [n for _, s, n, _ in placed(out, 2) if s == 1] == ["G5", "F5", "E5", "D5"]
    assert omr_repair.FILLED not in musicxml.read_root(out).find("part").find("measure[2]").attrib


def test_spacing_prefers_the_reading_that_fits_the_page():
    # bar 28 of the test song: eighth, eighth, quarter, dotted quarter, two notes Audiveris read as eighths. A beam
    # added to the last two (sixteenths) fits the spacing; one added to the first two doesn't
    xs = [0, 95, 190, 330, 560, 626]
    good = omr_repair.spacing([[_chord(x)] for x in xs], [Fraction(t) for t in ("0", "1/2", "1", "2", "7/2", "15/4")], 700, Fraction(4))
    bad = omr_repair.spacing([[_chord(x)] for x in xs], [Fraction(t) for t in ("0", "1/4", "1/2", "3/2", "3", "7/2")], 700, Fraction(4))
    assert good[0] < bad[0]


def _chord(x: float) -> omr.Chord:
    return omr.Chord(0, 1, omr.Box(x, 0, 24, 20))
