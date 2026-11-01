"""The reading cards (pianopipe.cards; docs/learning-mode.md §7 "Reading flash cards", §14 L8)."""
import json

import pytest

from pianopipe import cards, engrave, midi_io

needs_engraver = pytest.mark.skipif(not engrave.available(), reason="pip install pianopipe[sheet]")


def test_bank_runs_from_landmarks_to_patterns():
    bank = cards.bank()
    assert cards.problems(bank) == []
    stages = [cards.STAGES.index(c.stage) for c in bank]
    assert stages == sorted(stages), "introduced stage by stage"
    assert [c.order for c in bank] == list(range(len(bank)))
    # The three landmarks the design names come first: middle C, treble G, bass F.
    assert [(c.pitches, c.hand) for c in bank[:4]] == [([60], "R"), ([60], "L"), ([67], "R"), ([53], "L")]
    by = {s: [c for c in bank if c.stage == s] for s in cards.STAGES}
    assert all(len(c.pitches) == 1 for c in by["landmarks"])
    # Steps and skips: the landmark, then a note a 2nd or 3rd away.
    for c in by["steps"]:
        a, b = c.pitches
        assert a in {lm.pitch for lm in cards.LANDMARKS} and 1 <= abs(a - b) <= 4
    # Intervals struck together (two notes), triads (three), patterns four quarters in one bar with the metre shown.
    assert {len(c.pitches) for c in by["intervals"]} == {2, 3}
    assert all(len(c.chords) == 1 for c in by["intervals"])
    assert all(len(c.chords) == 4 and c.meter for c in by["patterns"])
    assert not any(c.meter for c in bank if c.stage != "patterns")


def test_manifest_entry():
    m = cards.bank()[2].manifest()
    assert m["answer"] == "G4" and m["stage"] == 0 and m["title"] == "Treble G"


def test_problems_finds_a_black_key_and_a_repeat():
    bank = cards.bank()[:2]
    bank.append(cards.Card("x", "landmarks", "x", "R", [(0, 4, (61,))], 4))
    bank.append(cards.Card("lm-c4-treble", "landmarks", "y", "R", [(0, 4, (60,))], 4))
    found = " ".join(cards.problems(bank))
    assert "C#4 is not a white key" in found and "repeated id" in found and "the same notes" in found


@needs_engraver
def test_a_card_is_centred_on_its_page(tmp_path):
    card = next(c for c in cards.bank() if c.id == "st-g4-down2")
    data = cards.engrave_card(card, tmp_path)
    assert (data["width"], data["height"]) == (cards.CARD.width, cards.CARD.height)
    (m,) = data["measures"]
    assert len(data["pages"]) == 1 and data["bars"] == 1
    # The notation sits in the middle: the bar's staff about as far from either side, the system round the centre.
    assert abs(m["left"] - (1 - m["right"])) < 0.12
    sys = data["systems"][0]
    assert 0.3 < (sys["top"] + sys["bottom"]) / 2 < 0.7
    # A head for each note, at the pitches played, inside the bar.
    heads = [(m["notes"][i + 1], m["notes"][i + 2]) for i in range(0, len(m["notes"]), 4)]
    assert [p for p, _ in heads] == [67, 64]
    assert all(m["left"] < x < m["right"] for _, x in heads)
    on_disk = json.loads((tmp_path / f"{card.id}.sheet.json").read_text())
    assert on_disk == data
    import hashlib
    assert hashlib.sha256((tmp_path / data["pages"][0]["file"]).read_bytes()).hexdigest() == data["pages"][0]["sha256"]
    notes = midi_io.load(tmp_path / f"{card.id}.mid").notes
    assert [n.pitch for n in sorted(notes, key=lambda n: n.start)] == [67, 64]


@needs_engraver
def test_build_writes_the_app_layout(tmp_path, monkeypatch):
    first = cards.bank()[:3]
    monkeypatch.setattr(cards, "bank", lambda: first)
    manifest = cards.build(tmp_path, passages=False, log=lambda *_: None)
    assert [c["id"] for c in manifest["cards"]] == ["lm-c4-treble", "lm-c4-bass", "lm-g4"]
    assert json.loads((tmp_path / "cards.json").read_text(encoding="utf-8")) == manifest
    for c in manifest["cards"]:
        sheet = json.loads((tmp_path / "cards" / f"{c['id']}.sheet.json").read_text())
        assert sheet["pages"][0]["file"] == f"{c['id']}-p1.png"
        assert (tmp_path / "cards" / f"{c['id']}-p1.png.bytes").exists()
        assert (tmp_path / "cards" / f"{c['id']}.mid.bytes").exists()
    assert not (tmp_path / "_build").exists()
