"""The OMR benchmark's scoring (omr_bench.py) and pages (omr_pages.py)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from pianopipe import omr_bench

# a bar: list of (staff, step+octave, quarters, chord?) in order; quarters 0 for a grace note
C_BAR = [(1, "C5", 1, False), (1, "E5", 1, True), (1, "D5", 1, False), (1, "E5", 2, False), (2, "C3", 4, False)]


def score_xml(bars: list[list[tuple]], parts: str = "piano") -> str:
    """A small partwise score: a two-staff piano part (and with parts="split", the bars as two one-staff parts)."""
    def note(staff, name, q, chord, staves=True):
        grace = q == 0
        return (f"<note>{'<grace/>' if grace else ''}{'<chord/>' if chord else ''}"
                f"<pitch><step>{name[0]}</step><octave>{name[-1]}</octave></pitch>"
                f"{'' if grace else f'<duration>{int(q * 2)}</duration>'}<voice>{staff}</voice>"
                f"{f'<staff>{staff}</staff>' if staves else ''}</note>")

    def measures(staffs, staves):
        out = []
        for k, bar in enumerate(bars):
            body = ("<attributes><divisions>2</divisions><time><beats>4</beats><beat-type>4</beat-type></time>"
                    + ("<staves>2</staves>" if staves else "") + "</attributes>") if k == 0 else ""
            for s in staffs:
                ns = [n for n in bar if n[0] == s]
                body += "".join(note(*n, staves=staves) for n in ns)
                used = sum(n[2] for n in ns if not n[3])
                if s != staffs[-1] and used:
                    body += f"<backup><duration>{int(used * 2)}</duration></backup>"
            out.append(f'<measure number="{k + 1}">{body}</measure>')
        return "".join(out)

    if parts == "piano":
        return ('<score-partwise><part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>'
                f'<part id="P1">{measures([1, 2], True)}</part></score-partwise>')
    return ('<score-partwise><part-list><score-part id="P1"><part-name>Voice</part-name></score-part>'
            '<score-part id="P2"><part-name>Voice</part-name></score-part></part-list>'
            f'<part id="P1">{measures([1], False)}</part><part id="P2">{measures([2], False)}</part></score-partwise>')


def bars_of(tmp_path, bars, name="s", parts="piano") -> omr_bench.Reading:
    p = tmp_path / f"{name}.musicxml"
    p.write_text(score_xml(bars, parts), encoding="utf-8")
    return omr_bench.piano_bars(p)


def test_piano_bars_reads_hands_onsets_lengths_and_pitches(tmp_path):
    r = bars_of(tmp_path, [C_BAR])
    assert r.problem == "" and r.numbers == [1]
    got = {(n.hand, n.onset, n.length, n.pitch) for n in r.bars[0]}
    assert got == {("R", 0, 1, 72), ("R", 0, 1, 76), ("R", 1, 1, 74), ("R", 2, 2, 76), ("L", 0, 4, 48)}


def test_a_reading_without_a_two_staff_piano_part_has_no_bars(tmp_path):
    r = bars_of(tmp_path, [C_BAR], parts="split")
    assert r.bars == [] and "no two-staff piano part" in r.problem


def test_identical_scores_are_all_right(tmp_path):
    t = bars_of(tmp_path, [C_BAR, C_BAR], "t")
    cmp = omr_bench.compare(t, bars_of(tmp_path, [C_BAR, C_BAR], "r"))
    assert cmp.right == 2 and cmp.score == 1.0 and not cmp.extra_bars


@pytest.mark.parametrize("bar, cause", [
    ([(1, "C5", 1, False), (1, "E5", 1, True), (1, "D5", 1, False), (1, "E5", 2, False), (2, "C4", 4, False)], "octave"),
    ([(1, "C5", 1, False), (1, "E5", 1, True), (1, "D5", 1, False), (1, "F5", 2, False), (2, "C3", 4, False)], "pitch"),
    ([(1, "C5", 1, False), (1, "E5", 1, True), (1, "D5", 1, False), (1, "E5", 1, False), (2, "C3", 4, False)], "duration"),
    ([(1, "C5", 1, False), (1, "E5", 1, True), (1, "C3", 4, True), (1, "D5", 1, False), (1, "E5", 2, False)], "hand"),
    ([(1, "C5", 1, False), (1, "D5", 1, False), (1, "E5", 2, False), (2, "C3", 4, False)], "missing"),
    ([(1, "C5", 1, False), (1, "E5", 1, True), (1, "G5", 1, True), (1, "D5", 1, False), (1, "E5", 2, False),
      (2, "C3", 4, False)], "extra"),
])
def test_each_kind_of_mistake_is_named(tmp_path, bar, cause):
    cmp = omr_bench.compare(bars_of(tmp_path, [C_BAR], "t"), bars_of(tmp_path, [bar], "r"))
    assert cmp.bars[0].cause == cause and cmp.right == 0


def test_onset_mistake(tmp_path):
    t = [(1, "C5", 2, False), (1, "D5", 2, False), (2, "C3", 4, False)]
    r = [(1, "C5", 1, False), (1, "D5", 3, False), (2, "C3", 4, False)]
    res = omr_bench.compare(bars_of(tmp_path, [t], "t"), bars_of(tmp_path, [r], "r")).bars[0]
    assert res.cause in ("onset", "duration") and res.right == 1 and res.onset == 1


def test_a_lost_bar_costs_only_that_bar(tmp_path):
    a = [(1, "C5", 4, False), (2, "C3", 4, False)]
    b = [(1, "G5", 4, False), (2, "G3", 4, False)]
    c = [(1, "E5", 4, False), (2, "E3", 4, False)]
    cmp = omr_bench.compare(bars_of(tmp_path, [a, b, c], "t"), bars_of(tmp_path, [a, c], "r"))
    assert [x.cause for x in cmp.bars] == ["", "lost", ""] and cmp.right == 2


def test_an_added_bar_lowers_the_score(tmp_path):
    a = [(1, "C5", 4, False), (2, "C3", 4, False)]
    c = [(1, "E5", 4, False), (2, "E3", 4, False)]
    empty = [(1, "B5", 4, False)]
    cmp = omr_bench.compare(bars_of(tmp_path, [a, c], "t"), bars_of(tmp_path, [a, empty, c], "r"))
    assert cmp.right == 2 and cmp.extra_bars == [2] and cmp.score == pytest.approx(2 / 3)


def test_only_and_flags(tmp_path):
    a = [(1, "C5", 4, False), (2, "C3", 4, False)]
    wrong = [(1, "D5", 4, False), (2, "C3", 4, False)]
    t = bars_of(tmp_path, [a, a, a], "t")
    cmp = omr_bench.compare(t, bars_of(tmp_path, [a, wrong, wrong], "r"), only={1, 2}, flagged={2})
    assert [b.bar for b in cmp.bars] == [1, 2]
    assert [(b.exact, b.flagged) for b in cmp.bars] == [(True, False), (False, True)]


def test_a_pickup_padded_with_rests_in_front_is_right(tmp_path):
    t = [(1, "C5", 1, False), (2, "C3", 1, False)]
    r = [(1, "B4", 3, False), (1, "C5", 1, False), (2, "B2", 3, False), (2, "C3", 1, False)]
    t2 = [(1, "D5", 4, False), (2, "D3", 4, False)]
    truth = bars_of(tmp_path, [t, t2], "t")
    read = bars_of(tmp_path, [r, t2], "r")
    read.bars[0] = [n for n in read.bars[0] if n.pitch in (72, 48)]  # the rests before the notes
    assert omr_bench.compare(truth, read).right == 2


def test_grace_notes_match_on_hand_and_pitch(tmp_path):
    t = [(1, "B4", 0, False), (1, "C5", 4, False), (2, "C3", 4, False)]
    assert omr_bench.compare(bars_of(tmp_path, [t], "t"), bars_of(tmp_path, [t], "r")).right == 1


def test_summaries_rank_damages_against_the_clean_reading(tmp_path):
    a = [(1, "C5", 4, False), (2, "C3", 4, False)]
    wrong = [(1, "D5", 4, False), (2, "C3", 4, False)]
    t = bars_of(tmp_path, [a, a], "t")
    rows = []
    for dmg, read in (("clean", [a, a]), ("blur", [a, wrong]), ("curl", [wrong, wrong])):
        case = {"case": dmg, "source": "x", "engraver": "e", "layout": "piano", "damage": dmg}
        rows += omr_bench.rows(case, "final", omr_bench.compare(t, bars_of(tmp_path, read, dmg)))
    worst = omr_bench.summarise(rows, "damage", "final")
    assert [d["damage"] for d in worst] == ["curl", "blur", "clean"] and worst[0]["causes"] == {"pitch": 2}
    cost = omr_bench.lost_to(rows, "final")
    assert [(d["damage"], d["lost"], d["cost"]) for d in cost] == [("curl", 2, 1.0), ("blur", 1, 0.5)]
    omr_bench.write_csv(tmp_path / "r.csv", rows)
    assert (tmp_path / "r.csv").read_text(encoding="utf-8").startswith("case,source")


def test_excerpt_keeps_the_first_bars_and_ends_them(tmp_path):
    root = ET.fromstring(score_xml([C_BAR] * 5))
    ex = omr_bench.excerpt(root, 3)
    ms = ex.findall("part/measure")
    assert len(ms) == 3 and ms[-1].find("barline/bar-style").text == "light-heavy"
    assert len(root.findall("part/measure")) == 5  # the source is untouched


def test_add_voice_makes_a_songbook_page_with_the_same_piano(tmp_path):
    root = ET.fromstring(score_xml([C_BAR, [(1, "G5", 4, False), (2, "G2", 4, False)]]))
    root.find("part/measure/attributes").append(ET.fromstring("<clef number='1'><sign>G</sign><line>2</line></clef>"))
    song = omr_bench.add_voice(root)
    p = tmp_path / "song.musicxml"
    p.write_text(ET.tostring(song, encoding="unicode"), encoding="utf-8")
    parts = song.findall("part")
    assert [sp.findtext("part-name") for sp in song.findall("part-list/score-part")] == ["Voice", "Piano"]
    sung = [(n.findtext("pitch/step"), n.findtext("duration")) for n in parts[0].iter("note") if n.find("pitch") is not None]
    assert sung == [("C", "2"), ("D", "2"), ("E", "4"), ("G", "8")]  # the top line, one note per chord
    assert [x.findtext("text") for x in parts[0].iter("lyric")][:2] == ["Sing", "we"]
    assert [h.findtext("root/root-step") for h in parts[1].iter("harmony")] == ["C", "G"]
    original = tmp_path / "orig.musicxml"
    original.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    assert omr_bench.compare(omr_bench.piano_bars(original), omr_bench.piano_bars(p)).score == 1.0


def test_damages_keep_pages_readable_sizes():
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    from pianopipe import omr_pages

    a = np.full((660, 510), 255, np.uint8)
    for top in (100, 300):  # two staves of five lines from x=80, a brace left of them
        for k in range(5):
            a[top + 8 * k, 80:480] = 0
        a[top - 10:top + 42, 70:73] = 0
    page = Image.fromarray(a)
    x, space = omr_pages.staff_left(page)
    assert x == 80 and space == 8
    for kind in omr_pages.DAMAGES:
        pages, dpi = omr_pages.damage([page], kind, "test")
        assert len(pages) == 1 and pages[0].mode == "L" and dpi in (150, 200, 300)
        again, _ = omr_pages.damage([page], kind, "test")
        assert np.array_equal(np.asarray(pages[0]), np.asarray(again[0])), kind  # seeded: the same every run
    brace, _ = omr_pages.damage([page], "crop-brace", "t")
    assert brace[0].width == 510 - 82  # cut just right of the staff lines' start: the brace is gone
    assert omr_pages.damage([page], "lowdpi", "t")[0][0].size == (255, 330)


def _bench_tool():
    """Quest Piano Trainer's tools/omr_bench.py (not in the public pianopipe repo, where these tests skip)."""
    import sys
    from pathlib import Path

    tools = Path(__file__).resolve().parents[2] / "tools"
    if not (tools / "omr_bench.py").exists():
        pytest.skip("tools/omr_bench.py is in the Quest Piano Trainer repo only")
    sys.path.insert(0, str(tools))
    import omr_bench

    return omr_bench


def test_bar_numbers_lists():
    tool = _bench_tool()

    assert tool.bar_set("1-3, 7") == {1, 2, 3, 7} and tool.bar_set([2, 4]) == {2, 4} and tool.bar_set(None) is None


def test_another_engine_reads_each_page_and_its_pages_are_joined(tmp_path):
    import sys

    Image = pytest.importorskip("PIL.Image")
    tool = _bench_tool()

    cdir = tmp_path / "cases" / "x"
    cdir.mkdir(parents=True)
    pages = [Image.new("L", (50, 60), 255), Image.new("L", (50, 60), 255)]
    pages[0].save(cdir / "input.tif", save_all=True, append_images=pages[1:])
    (cdir / "truth.musicxml").write_text(score_xml([C_BAR, C_BAR]), encoding="utf-8")
    (tmp_path / "page.musicxml").write_text(score_xml([C_BAR]), encoding="utf-8")
    fake = tmp_path / "engine.py"  # "reads" every page as one bar of C_BAR
    fake.write_text(f"import shutil, sys\nshutil.copyfile(r'{tmp_path / 'page.musicxml'}', sys.argv[2])\n", encoding="utf-8")
    case = {"case": "x", "source": "s", "engraver": "e", "layout": "piano", "damage": "clean", "kind": "scan",
            "set": "generated", "input": str(cdir / "input.tif"), "truth": str(cdir / "truth.musicxml")}
    edir = tmp_path / "fake"
    tool.read_pages([case], edir, [sys.executable, str(fake), "{in}", "{out}"], 2, False)
    assert (edir / "x.musicxml").exists() and len(list((edir / "pages").glob("*.png"))) == 2
    rows, info = tool.chain_and_score((case, edir, "fake", False))
    assert info["final"]["right"] == 2 and info["final"]["bars"] == 2


def test_alignment_ties_go_to_the_bar_read_exactly(tmp_path):
    whole = [(1, "C5", 4, False), (2, "C3", 4, False)]
    halves = [(1, "C5", 2, False), (1, "C5", 2, False), (2, "C3", 4, False)]
    cmp = omr_bench.compare(bars_of(tmp_path, [whole, halves], "t"), bars_of(tmp_path, [whole], "r"))
    assert [b.cause for b in cmp.bars] == ["", "lost"]


def test_a_full_first_bar_shifted_is_not_a_pickup(tmp_path):
    t = [(1, "C5", 2, False), (1, "D5", 2, False), (2, "C3", 4, False)]
    r = [(1, "B4", 1, False), (1, "C5", 2, False), (1, "D5", 1, False), (2, "B2", 1, False), (2, "C3", 3, False)]
    read = bars_of(tmp_path, [r], "r")
    read.bars[0] = [n for n in read.bars[0] if n.pitch != 71 and n.pitch != 47]
    assert omr_bench.compare(bars_of(tmp_path, [t], "t"), read).right == 0


def test_octave_errors_are_found_before_notes_are_matched_across_onsets(tmp_path):
    t = [(1, "C4", 2, False), (1, "C5", 2, False), (2, "C3", 4, False)]
    r = [(1, "C5", 2, False), (1, "C6", 2, False), (2, "C3", 4, False)]
    res = omr_bench.compare(bars_of(tmp_path, [t], "t"), bars_of(tmp_path, [r], "r")).bars[0]
    assert res.cause == "octave" and res.octave == 2


def test_lost_bars_count_as_unflagged_when_flags_are_known(tmp_path):
    a = [(1, "C5", 4, False), (2, "C3", 4, False)]
    b = [(1, "G5", 4, False), (2, "G3", 4, False)]
    cmp = omr_bench.compare(bars_of(tmp_path, [a, b], "t"), bars_of(tmp_path, [a], "r"), flagged=set())
    assert [x.flagged for x in cmp.bars] == [False, False]


def test_damage_cost_counts_only_bars_added_beyond_the_clean_reading(tmp_path):
    a = [(1, "C5", 4, False), (2, "C3", 4, False)]
    extra = [(1, "B5", 4, False)]
    t = bars_of(tmp_path, [a], "t")
    rows = []
    for dmg in ("clean", "blur"):
        case = {"case": dmg, "source": "x", "engraver": "e", "layout": "piano", "damage": dmg}
        rows += omr_bench.rows(case, "final", omr_bench.compare(t, bars_of(tmp_path, [a, extra], dmg)))
    assert [(d["damage"], d["added"], d["cost"]) for d in omr_bench.lost_to(rows, "final")] == [("blur", 0, 0.0)]


def test_an_answer_may_be_two_one_staff_parts_but_a_reading_may_not(tmp_path):
    p = tmp_path / "split.musicxml"
    p.write_text(score_xml([C_BAR], parts="split"), encoding="utf-8")
    answer = omr_bench.piano_bars(p, answer=True)
    assert answer.problem == "" and {(n.hand, n.pitch) for n in answer.bars[0]} >= {("R", 72), ("L", 48)}
    assert omr_bench.compare(answer, bars_of(tmp_path, [C_BAR])).right == 1
    assert omr_bench.piano_bars(p).bars == []


def test_alignment_of_long_pieces_with_many_bars_missing():
    """Float sums along the edges once broke the traceback (an OLiMPiC page: hundreds of bars, few read)."""
    bar = lambda p: [omr_bench.BarNote("R", 0, 4, p)]  # noqa: E731
    truth = [bar(60 + k % 24) for k in range(300)]
    read = [bar(60 + k % 24) for k in range(0, 300, 37)]
    pairs = omr_bench.align(truth, read)
    assert len([p for p in pairs if p[0] is not None]) == 300 and len([p for p in pairs if p[1] is not None]) == len(read)
    assert omr_bench.align(truth, []) == [(k, None) for k in range(300)]
