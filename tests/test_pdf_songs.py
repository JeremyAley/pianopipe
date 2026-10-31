"""Reading a bought sheet-music PDF for import (pdf_songs: the songs in it, tempo, jumps; pdf_score's signs and words
per bar) and writing the marks an OMR export lacks (score_marks), on small synthetic PDFs. Needs the [omr] extra."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

pytest.importorskip("pymupdf")

import pymupdf  # noqa: E402

from pianopipe import musicxml, pdf_score, pdf_songs, score_marks  # noqa: E402

from omr_builder import backup, export_xml, xnote  # noqa: E402
from pdf_builder import FONT, FONT_NAME, SP, Sheet  # noqa: E402

Q = 4  # divisions per quarter note in the exports


@pytest.fixture(autouse=True)
def stand_in_font(monkeypatch):
    """The builders draw the symbols with Liberation Serif: read it as a Sonata-layout music font."""
    monkeypatch.setattr(pdf_score, "SONATA_FONTS", pdf_score.SONATA_FONTS + [FONT_NAME])


# ---------------------------------------------------------------- the songs in a PDF

def notation_page(doc, title: str = "", lines: tuple[str, ...] = (), footer: str = "", copyright: bool = False,
                  number: int | None = None):
    """A page of vector notation (a few music-font heads on a staff), with a title block when given."""
    page = doc.new_page(width=612, height=792)
    if title:
        page.insert_text((150, 50), title, fontsize=28, fontname="hebo")
    for k, text in enumerate(lines):
        page.insert_text((200 if text.lower().startswith("from") else 420, 70 + 12 * k), text, fontsize=10, fontname="helv")
    if number is not None:
        page.insert_text((560, 30), str(number), fontsize=9, fontname="helv")
    for i in range(5):
        page.draw_line((40, 160 + 5 * i), (570, 160 + 5 * i), width=0.5)
    for k in range(4):
        page.insert_text((100 + 40 * k, 170), "œ", fontsize=9, fontname="music", fontfile=str(FONT))
    page.insert_text((60, 500), "a line of lyrics under the staff", fontsize=10, fontname="tiro")
    if copyright:
        page.insert_text((200, 720), "© 1992 Some Music Company  All Rights Reserved", fontsize=6, fontname="helv")
    if footer:
        page.insert_text((250, 770), footer, fontsize=7, fontname="helv")
    return page


def test_a_combined_order_splits_at_each_title_page(tmp_path):
    doc = pymupdf.open()
    notation_page(doc, "A WHOLE NEW WORLD", ("from Walt Disney's ALADDIN", "Music by ALAN MENKEN"), copyright=True)
    notation_page(doc, number=2)
    notation_page(doc, "ROCKIN' ROBIN", ("Words and Music by", "J. THOMAS"), copyright=True)
    notation_page(doc, number=2)
    notation_page(doc, number=3)
    notation_page(doc, "FROM THE START", ("By LAUFEY, SPENCER STEWART",), copyright=True)
    notation_page(doc, footer="Go the Distance - 2 - 1")  # no title text (drawn as outlines): the footer says
    notation_page(doc, footer="Go the Distance - 2 - 2")
    doc.save(str(tmp_path / "order.pdf"))
    songs = pdf_songs.propose(tmp_path / "order.pdf")
    assert [(s.title, s.artist, s.first, s.last, s.kind) for s in songs] == [
        ("A Whole New World", "Aladdin", 1, 2, "vector"),
        ("Rockin' Robin", "J. Thomas", 3, 5, "vector"),
        ("From the Start", "Laufey", 6, 6, "vector"),
        ("Go the Distance", "", 7, 8, "vector"),
    ]
    part = pdf_songs.split(tmp_path / "order.pdf", 3, 5, tmp_path / "robin.pdf")
    with pymupdf.open(str(part)) as d:
        assert len(d) == 3
        assert "ROBIN" in d[0].get_text()


def test_a_title_drawn_as_outlines_starts_a_song(tmp_path):
    doc = pymupdf.open()
    notation_page(doc, "FIRST SONG")
    notation_page(doc, number=2)
    page = notation_page(doc)
    for k in range(8):  # letters as filled outlines
        page.draw_rect(pymupdf.Rect(150 + 40 * k, 30, 180 + 40 * k, 58), color=None, fill=(0, 0, 0))
    doc.save(str(tmp_path / "order.pdf"))
    songs = pdf_songs.propose(tmp_path / "order.pdf")
    assert [(s.title, s.first, s.last) for s in songs] == [("First Song", 1, 2), ("Order 2", 3, 3)]


def test_a_scan_is_one_song_named_after_its_file(tmp_path):
    doc = pymupdf.open()
    for _ in range(3):
        page = doc.new_page(width=612, height=792)
        pix = pymupdf.Pixmap(pymupdf.csGRAY, pymupdf.IRect(0, 0, 200, 260), False)
        pix.clear_with(255)
        page.insert_image(page.rect, pixmap=pix)
    doc.save(str(tmp_path / "go-the-distance-scan.pdf"))
    songs = pdf_songs.propose(tmp_path / "go-the-distance-scan.pdf")
    assert [(s.title, s.first, s.last, s.kind) for s in songs] == [("Go the distance scan", 1, 3, "scan")]


def test_titles_in_capitals_come_out_in_title_case():
    assert pdf_songs.nice_case("I SEE THE LIGHT") == "I See the Light"
    assert pdf_songs.nice_case("ROCKIN’ ROBIN") == "Rockin' Robin"
    assert pdf_songs.nice_case("PART OF YOUR WORLD") == "Part of Your World"
    assert pdf_songs.nice_case("TOY STORY 2") == "Toy Story 2"
    assert pdf_songs.nice_case("Married Life") == "Married Life"  # already cased: left alone
    assert pdf_songs._show('from "HERCULES"') == "Hercules"
    assert pdf_songs._show("from Walt Disney Pictures’ TANGLED") == "Tangled"
    assert pdf_songs._show("from THE LITTLE MERMAID") == "The Little Mermaid"


# ---------------------------------------------------------------- tempo and jumps from the PDF

def jump_sheet() -> Sheet:
    """Six bars, two a system: a segno over bar 2, "To Coda" and its sign at the end of bar 3, "D.S. al Coda" at the
    end of bar 5, and the coda (its sign under the heading "CODA") at bar 6."""
    s = Sheet((2, 2, 2))
    for system in range(3):
        for bar in (1, 2):
            for k, n in enumerate(["C5", "D5", "E5", "F5"]):
                s.chord(bar, 1 + k, 1, [n], "q", system=system)
            s.chord(bar, 1, 2, ["C3"], "w", system=system)
    top = s.tops[0][1]
    s.text("Moderately fast", 50, top - 20, italic=False)
    s.glyph("%", s.x(2, 1) - 4, top - 8)
    top = s.tops[1][1]
    s.text("To Coda", s.lefts[1][1] - 45, top - 10, italic=False)
    s.glyph("Þ", s.lefts[1][1] - 12, top - 10)
    top = s.tops[2][1]
    s.text("D.S. al Coda", s.lefts[2][1] - 50, top - 10, italic=False)
    s.text("CODA", s.x(2, 1, 2) - 4, top - 22, italic=False)
    s.glyph("Þ", s.x(2, 1, 2), top - 9)
    return s


def test_signs_and_jump_words_are_read_per_bar(tmp_path):
    score = pdf_score.read(jump_sheet().save(tmp_path / "song.pdf"))
    assert len(score.bars) == 6
    assert [b.signs for b in score.bars] == [[], ["segno"], [], [], [], ["coda"]]
    assert "To Coda Coda" in score.bars[2].words  # the sign set in its line of text
    assert "D.S. al Coda" in score.bars[4].words
    assert pdf_songs.jumps(score) == ({"segno": 2, "tocoda": 3, "dalsegno": 5, "coda": 6}, [])
    assert pdf_songs.tempo(score) == ("Moderately fast", 126.0)


def test_a_ds_with_no_segno_is_incomplete(tmp_path):
    s = Sheet((2,))
    for bar in (1, 2):
        s.chord(bar, 1, 1, ["C5"], "w")
        s.chord(bar, 1, 2, ["C3"], "w")
    s.text("D.S. al Fine", s.lefts[0][1] + 100, s.tops[0][1] - 10, italic=False)
    found, missing = pdf_songs.jumps(pdf_score.read(s.save(tmp_path / "song.pdf")))
    assert found == {"dalsegno": 2}
    assert missing == ["D.S. but no segno", "al Fine but no Fine"]


def test_al_coda_with_no_to_coda_is_incomplete(tmp_path):
    """Left alone: writing it as a plain D.S. would lose the coda the words ask for."""
    s = Sheet((2,))
    for bar in (1, 2):
        s.chord(bar, 1, 1, ["C5"], "w")
        s.chord(bar, 1, 2, ["C3"], "w")
    s.glyph("%", s.x(1, 1) - 4, s.tops[0][1] - 8)
    s.text("D.S. al Coda", s.lefts[0][1] + 100, s.tops[0][1] - 10, italic=False)
    found, missing = pdf_songs.jumps(pdf_score.read(s.save(tmp_path / "song.pdf")))
    assert found == {"segno": 1, "dalsegno": 2}
    assert missing == ["al Coda but no To Coda"]


def test_signs_beside_text_and_signs_in_text_fonts():
    """A segno in a text font (MuseScore's Bravura Text) is a sign; a coda sign beside a bar number isn't text."""
    sign = pdf_score.Glyph("sign", "coda", 100, 110, 50, 16)
    marks = pdf_score._marks([sign], [("", 40, 30, 50, 46, False), ("54", 84, 38, 98, 48, False),
                                      ("D.", 200, 38, 210, 48, False)])
    assert ("segno", True) in [(m[0], m[4]) for m in marks]
    assert ("coda", True) in [(m[0], m[4]) for m in marks]
    assert ("54", False) in [(m[0], m[4]) for m in marks]


def test_a_metronome_mark_wins_and_the_title_is_not_a_tempo(tmp_path):
    bar = pdf_score.Bar(1, 1, 0, None, words=["SLOW HANDS", "from SOMETHING", "Spirited (q = ca. 132)"])
    assert pdf_songs.tempo(pdf_score.PdfScore([bar], [0]), "Slow Hands") == ("Spirited (q = ca. 132)", 132.0)
    bar.words = ["SLOW HANDS"]
    assert pdf_songs.tempo(pdf_score.PdfScore([bar], [0]), "Slow Hands") is None


# ---------------------------------------------------------------- writing the marks

def five_bars(extra: dict[int, str] | None = None) -> ET.Element:
    body = "".join(xnote(n, Q) for n in ["C5", "D5", "E5", "F5"]) + backup(4 * Q) + xnote("C3", 4 * Q, voice=5, staff=2)
    ms = [body + (extra or {}).get(k, "") for k in range(5)]
    return ET.fromstring(export_xml(ms))


def played(root: ET.Element, tmp_path) -> musicxml.Score:
    path = tmp_path / "song.musicxml"
    path.write_bytes(ET.tostring(root, encoding="utf-8"))
    return musicxml.load(path)


def test_jumps_from_the_pdf_replace_the_exports_guesses(tmp_path):
    """The OCR'd export has "D.S. al Coda" in bar 4 and a stray "To Coda" in bar 1: the importer guesses. The PDF's
    marks replace them, and the song plays 1 2 3 4 | 2 3 | 5."""
    words = '<direction placement="above"><direction-type><words>{}</words></direction-type></direction>'
    root = five_bars({0: words.format("To Coda"), 3: words.format("D.S. al Coda")})
    before = played(root, tmp_path)
    assert any("no segno" in w for w in before.warnings)
    done = score_marks.set_jumps(root, {"segno": 2, "tocoda": 3, "dalsegno": 4, "coda": 5})
    assert done == ["segno 2", "tocoda 3", "dalsegno 4", "coda 5"]
    after = played(root, tmp_path)
    assert after.order == [0, 1, 2, 3, 1, 2, 4]
    assert not [w for w in after.warnings if "segno" in w or "oda" in w]
    assert root.find(".//measure[@number='1']/direction") is None  # the stray To Coda is gone


def test_tempo_and_title(tmp_path):
    root = five_bars()
    assert any(w.startswith("no tempo marking") for w in played(root, tmp_path).warnings)
    score_marks.set_tempo(root, 152, "Bright Rock")
    score_marks.set_title(root, "Rockin' Robin", "Bobby Day")
    score = played(root, tmp_path)
    assert not any(w.startswith("no tempo marking") for w in score.warnings)
    tempos = [msg.tempo for tick, msg in score.song.meta if msg.type == "set_tempo"]
    assert tempos[0] == pytest.approx(60_000_000 / 152, abs=1)
    assert (score.title, score.composer) == ("Rockin' Robin", "Bobby Day")
    first = root.find("part/measure")
    assert first.find("direction/sound").get("tempo") == "152"
    assert list(first).index(first.find("direction")) < list(first).index(first.find("note"))
