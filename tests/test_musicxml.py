"""MusicXML import: hands from staves and voices, ties, repeats and jumps, grace notes, tempo, part shapes, SATB."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from pianopipe import musicxml
from pianopipe.musicxml import MusicXMLError
from pianopipe.process import load_notes, process_file
from pianopipe.quality import GatePolicy, OFFICIAL

from helpers import make_midi

TPB = musicxml.TPB


def note(step: str, octave: int, dur: int = 4, voice: int = 1, staff: int = 1, *extra: str, alter: int = 0) -> str:
    """One <note>; durations are in divisions (4 per quarter note)."""
    a = f"<alter>{alter}</alter>" if alter else ""
    return (f"<note><pitch><step>{step}</step>{a}<octave>{octave}</octave></pitch><duration>{dur}</duration>"
            f"{''.join(e for e in extra if e.startswith('<tie'))}<voice>{voice}</voice><staff>{staff}</staff>"
            f"{''.join(e for e in extra if not e.startswith('<tie'))}</note>")


def rest(dur: int = 4, voice: int = 1, staff: int = 1) -> str:
    return f"<note><rest/><duration>{dur}</duration><voice>{voice}</voice><staff>{staff}</staff></note>"


def backup(dur: int = 16) -> str:
    return f"<backup><duration>{dur}</duration></backup>"


ATTRS = ("<attributes><divisions>4</divisions><key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type>"
         "</time><staves>{staves}</staves></attributes>")


def score(parts: list[tuple], staves: int = 2, header: str = "") -> str:
    """parts: [(part name, [measure body, ...]) or (name, measures, staves)]; the first measure of each part gets
    divisions, key, time and staves."""
    plist = "".join(f'<score-part id="P{i}"><part-name>{p[0]}</part-name></score-part>' for i, p in enumerate(parts))
    body = ""
    for i, p in enumerate(parts):
        measures, n = p[1], (p[2] if len(p) > 2 else staves)
        ms = "".join(
            f'<measure number="{j + 1}">{ATTRS.format(staves=n) if j == 0 else ""}{m}</measure>' for j, m in enumerate(measures))
        body += f'<part id="P{i}">{ms}</part>'
    return f'<?xml version="1.0"?><score-partwise version="4.0">{header}<part-list>{plist}</part-list>{body}</score-partwise>'


def write(tmp_path: Path, xml: str, name: str = "s.musicxml") -> Path:
    p = tmp_path / name
    p.write_text(xml, encoding="utf-8")
    return p


def pitches(sc, hand: str) -> list[int]:
    return [n.pitch for n in sc.notes if n.hand == hand]


def bar(rh: str, lh: str) -> str:
    return rh + backup() + lh


C4, E4, G4, C3, G2 = 60, 64, 67, 48, 43


def test_two_staves_give_hands_with_chords_and_voices(tmp_path):
    rh = note("C", 4) + note("E", 4) + note("G", 4) + note("C", 5)
    lh = note("C", 3, 16, 5, 2)
    m1 = bar(rh, lh)
    sc = musicxml.load(write(tmp_path, score([("Piano", [m1])])))
    assert sc.hand_source == "staves"
    assert pitches(sc, "R") == [60, 64, 67, 72]
    assert pitches(sc, "L") == [48]
    lh_note = next(n for n in sc.notes if n.hand == "L")
    assert (lh_note.start, lh_note.end) == (0, 4 * TPB)


def test_chord_notes_share_the_onset(tmp_path):
    chord = note("C", 4, 8) + note("E", 4, 8).replace("<pitch>", "<chord/><pitch>") + note("G", 4, 8)
    sc = musicxml.load(write(tmp_path, score([("Piano", [bar(chord, note("C", 3, 16, 5, 2))])])))
    rh = sorted((n.start, n.pitch) for n in sc.notes if n.hand == "R")
    assert rh == [(0, 60), (0, 64), (2 * TPB, 67)]


def test_cross_staff_note_stays_with_its_voice(tmp_path):
    # Voice 1 (right hand) dips onto the bass staff for its second note, as in the Moonlight Sonata's triplets.
    rh = note("C", 4) + note("G", 3, 4, 1, 2) + note("C", 4) + note("E", 4)
    lh = note("C", 2, 16, 5, 2)
    sc = musicxml.load(write(tmp_path, score([("Piano", [bar(rh, lh)])])))
    assert pitches(sc, "R") == [60, 55, 60, 64]
    assert pitches(sc, "L") == [36]


def test_ties_merge_and_may_skip_a_note(tmp_path):
    rh = note("C", 4, 8) + note("C", 4, 8, 1, 1, '<tie type="start"/>')
    rh2 = note("C", 4, 8, 1, 1, '<tie type="stop"/>') + note("D", 4, 8)
    # Left hand: D3 tied over an A3 to a D3 later in the bar, meaning "hold the D".
    lh = note("D", 3, 2, 5, 2, '<tie type="start"/>') + note("A", 3, 2, 5, 2) + note("D", 3, 12, 5, 2, '<tie type="stop"/>')
    sc = musicxml.load(write(tmp_path, score([("Piano", [bar(rh, lh), bar(rh2, rest(16, 5, 2))])])))
    rh_notes = [(n.start, n.end, n.pitch) for n in sc.notes if n.hand == "R"]
    assert rh_notes == [(0, 2 * TPB, 60), (2 * TPB, 6 * TPB, 60), (6 * TPB, 8 * TPB, 62)]
    lh_notes = sorted((n.start, n.end, n.pitch) for n in sc.notes if n.hand == "L")
    assert lh_notes == [(0, 4 * TPB, 50), (TPB // 2, TPB, 57)]


def test_unisons_become_one_note(tmp_path):
    # Two voices of the right hand strike E4 together; a dotted half and a quarter.
    rh = note("E", 4, 12) + note("C", 5, 4) + backup() + note("E", 4, 4, 2) + note("G", 4, 12, 2)
    sc = musicxml.load(write(tmp_path, score([("Piano", [bar(rh, note("C", 3, 16, 5, 2))])])))
    e4 = [n for n in sc.notes if n.pitch == 64]
    assert len(e4) == 1 and e4[0].end == 3 * TPB


def _measures_played(sc) -> list[int]:
    """The right hand plays one whole note per bar whose pitch is 60 + bar number."""
    return [n.pitch - 60 for n in sorted(sc.notes, key=lambda n: n.start) if n.hand == "R"]


def _bars(count: int, extras: dict[int, str]) -> list[str]:
    steps = ["C", "D", "E", "F", "G", "A", "B"]
    out = []
    for i in range(1, count + 1):
        semis = [0, 2, 4, 5, 7, 9, 11]
        # pitch 60 + i: pick a spelling
        midi = 60 + i
        octave, pc = divmod(midi, 12)
        if pc in semis:
            body = note(steps[semis.index(pc)], octave - 1, 16)
        else:
            body = note(steps[semis.index(pc - 1)], octave - 1, 16, alter=1)
        out.append(extras.get(i, "") + body + backup() + note("C", 3, 16, 5, 2) + extras.get(-i, ""))
    return out


FWD = '<barline location="left"><repeat direction="forward"/></barline>'
BWD = '<barline location="right"><repeat direction="backward"/></barline>'


def test_repeat_and_voltas_unroll(tmp_path):
    ending1 = '<barline location="left"><ending number="1" type="start"/></barline>'
    ending1_end = '<barline location="right"><ending number="1" type="stop"/><repeat direction="backward"/></barline>'
    ending2 = '<barline location="left"><ending number="2" type="start"/></barline>'
    ending2_end = '<barline location="right"><ending number="2" type="discontinue"/></barline>'
    bars = _bars(5, {2: FWD, 3: ending1, -3: ending1_end, 4: ending2, -4: ending2_end})
    sc = musicxml.load(write(tmp_path, score([("Piano", bars)])))
    assert _measures_played(sc) == [1, 2, 3, 2, 4, 5]


def test_repeat_times(tmp_path):
    bars = _bars(2, {-1: '<barline location="right"><repeat direction="backward" times="3"/></barline>'})
    assert _measures_played(musicxml.load(write(tmp_path, score([("Piano", bars)])))) == [1, 1, 1, 2]


def test_da_capo_al_fine_skips_repeats(tmp_path):
    fine = '<direction><direction-type><words>Fine</words></direction-type><sound fine="yes"/></direction>'
    dc = '<direction><direction-type><words>D.C. al Fine</words></direction-type><sound dacapo="yes"/></direction>'
    bars = _bars(4, {-1: BWD, 2: fine, -4: dc})
    assert _measures_played(musicxml.load(write(tmp_path, score([("Piano", bars)])))) == [1, 1, 2, 3, 4, 1, 2]


def test_dal_segno_al_coda(tmp_path):
    segno = '<direction><direction-type><segno/></direction-type><sound segno="s"/></direction>'
    tocoda = '<direction><direction-type><words>To Coda</words></direction-type><sound tocoda="c"/></direction>'
    ds = '<direction><direction-type><words>D.S. al Coda</words></direction-type><sound dalsegno="s"/></direction>'
    coda = '<direction><direction-type><coda/></direction-type><sound coda="c"/></direction>'
    bars = _bars(6, {2: segno, -3: tocoda, -4: ds, 5: coda})
    assert _measures_played(musicxml.load(write(tmp_path, score([("Piano", bars)])))) == [1, 2, 3, 4, 2, 3, 5, 6]


def test_grace_notes(tmp_path):
    acc = note("D", 4, 0).replace("<duration>0</duration>", "").replace("<pitch>", '<grace slash="yes"/><pitch>')
    app = note("F", 4, 0).replace("<duration>0</duration>", "").replace("<pitch>", "<grace/><pitch>")
    rh = note("C", 4, 4) + acc + note("C", 4, 4) + app + note("E", 4, 8)
    sc = musicxml.load(write(tmp_path, score([("Piano", [bar(rh, note("C", 3, 16, 5, 2))])])))
    rh_notes = sorted((n.start, n.end, n.pitch) for n in sc.notes if n.hand == "R")
    g = TPB // 8
    # The acciaccatura takes a 32nd before the beat; the appoggiatura takes half its note.
    assert rh_notes == [(0, TPB, 60), (TPB - g, TPB, 62), (TPB, 2 * TPB, 60), (2 * TPB, 3 * TPB, 65), (3 * TPB, 4 * TPB, 64)]


def test_tempo_sound_metronome_words_and_bogus_values(tmp_path):
    bogus = '<direction><direction-type><words>q =</words></direction-type><sound tempo="450"/></direction>'
    real = ('<direction><direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>45</per-minute></metronome>'
            '</direction-type><sound tempo="45"/></direction>')
    dotted = ('<direction><direction-type><metronome><beat-unit>quarter</beat-unit><beat-unit-dot/><per-minute>60</per-minute>'
              '</metronome></direction-type></direction>')
    words = '<direction><direction-type><words>Allegro</words></direction-type></direction>'
    bars = _bars(3, {1: bogus + real, 2: dotted, 3: words})
    sc = musicxml.load(write(tmp_path, score([("Piano", bars)])))
    tempos = [(t, round(60_000_000 / m.tempo)) for t, m in sc.song.meta if m.type == "set_tempo"]
    assert tempos == [(0, 45), (4 * TPB, 90), (8 * TPB, 132)]
    assert any("450" in w for w in sc.warnings)


def test_time_and_key_signatures(tmp_path):
    sc = musicxml.load(write(tmp_path, score([("Piano", _bars(2, {}))]).replace("<fifths>0</fifths>", "<fifths>-3</fifths><mode>minor</mode>")))
    kinds = {(m.type, getattr(m, "key", None), getattr(m, "numerator", None)) for _, m in sc.song.meta}
    assert ("key_signature", "Cm", None) in kinds and ("time_signature", None, 4) in kinds


def test_fingering_and_transpose(tmp_path):
    fing = "<notations><technical><fingering>3</fingering></technical></notations>"
    rh = note("E", 4, 16, 1, 1, fing)
    sc = musicxml.load(write(tmp_path, score([("Piano", [bar(rh, note("C", 3, 16, 5, 2))])])))
    assert next(n for n in sc.notes if n.pitch == 64).finger == 3
    clar = score([("Clarinet in B-flat", [note("D", 5, 16, 1, 1)])], staves=1).replace(
        "</attributes>", "<transpose><diatonic>-1</diatonic><chromatic>-2</chromatic></transpose></attributes>")
    assert [n.pitch for n in musicxml.load(write(tmp_path, clar, "c.musicxml")).notes] == [72]


def test_mxl_container(tmp_path):
    xml = score([("Piano", [bar(note("C", 4, 16), note("C", 3, 16, 5, 2))])])
    p = tmp_path / "s.mxl"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("mimetype", "application/vnd.recordare.musicxml")
        z.writestr("META-INF/container.xml",
                   '<container><rootfiles><rootfile full-path="score/s.xml"/></rootfiles></container>')
        z.writestr("score/s.xml", xml)
    assert sorted(n.pitch for n in musicxml.load(p).notes) == [48, 60]


def test_part_shapes(tmp_path):
    # Melody part + piano: the piano is used and the melody left out (with a warning).
    melody = ("Voice", [note("C", 5, 16)], 1)
    piano = ("Piano", [bar(note("E", 4, 16), note("C", 3, 16, 5, 2))])
    sc = musicxml.load(write(tmp_path, score([melody, piano])))
    assert sorted(n.pitch for n in sc.notes) == [48, 64] and sc.warnings
    # Two one-staff keyboard parts: right and left by pitch.
    upper, lower = ("Piano", [note("C", 5, 16)]), ("Piano", [note("C", 3, 16)])
    sc = musicxml.load(write(tmp_path, score([lower, upper], staves=1), "two.musicxml"))
    assert pitches(sc, "R") == [72] and pitches(sc, "L") == [48] and sc.hand_source == "staves"
    # A piano duet is refused.
    with pytest.raises(MusicXMLError, match="duet"):
        musicxml.load(write(tmp_path, score([piano, piano]), "duet.musicxml"))


def test_satb_parts_and_reach(tmp_path):
    # The tenor G3 is 19 semitones above the bass C2, and the right hand (C4-E4) can take it: it moves up.
    parts = [("Soprano", [note("E", 4, 16)]), ("Alto", [note("C", 4, 16)]), ("Tenor", [note("G", 3, 16)]),
             ("Bass", [note("C", 2, 16)])]
    p = write(tmp_path, score(parts, staves=1))
    sc = musicxml.load(p)
    assert sc.hand_source == "satb"
    assert pitches(sc, "R") == [60, 64] and sorted(pitches(sc, "L")) == [36, 55]
    _, notes, hand_source, _ = load_notes(p)
    assert hand_source == "satb"
    assert sorted(n.pitch for n in notes if n.hand == "R") == [55, 60, 64]
    assert [n.pitch for n in notes if n.hand == "L"] == [36]


def test_satb_midi_tracks(tmp_path):
    path = make_midi(tmp_path / "hymn.mid", [
        {"name": "Soprano", "channel": 0, "notes": [(0, 960, 64)]},
        {"name": "Alto", "channel": 1, "notes": [(0, 960, 60)]},
        {"name": "Tenor", "channel": 2, "notes": [(0, 960, 52)]},
        {"name": "Bass", "channel": 3, "notes": [(0, 960, 48)]},
    ])
    _, notes, hand_source, _ = load_notes(path)
    assert hand_source == "satb"
    assert sorted(n.pitch for n in notes if n.hand == "R") == [60, 64]
    assert sorted(n.pitch for n in notes if n.hand == "L") == [48, 52]


def test_satb_flag_on_two_staff_hymn(tmp_path):
    # A hymnal's two staves (upper = S+A, lower = T+B) with a wide left hand: the manifest's satb flag fixes it.
    path = make_midi(tmp_path / "hymn.mid", [
        {"name": "upper", "channel": 0, "notes": [(0, 960, 67), (0, 960, 64)]},
        {"name": "lower", "channel": 1, "notes": [(0, 960, 60), (0, 960, 41)]},
    ])
    _, notes, hand_source, _ = load_notes(path)
    assert hand_source == "track_names"
    _, notes, hand_source, _ = load_notes(path, satb=True)
    assert hand_source == "satb"
    assert sorted(n.pitch for n in notes if n.hand == "R") == [60, 64, 67]


def test_process_file_end_to_end(tmp_path):
    bars = []
    melody = ["C", "D", "E", "F", "G", "A", "B", "C"]
    for i in range(20):
        rh = "".join(note(melody[(i + k) % 8], 5 if melody[(i + k) % 8] == "C" and k == 3 else 4, 4) for k in range(4))
        lh = note("C", 3, 8, 5, 2) + note("G", 2, 8, 5, 2)
        bars.append(bar(rh, lh))
    tempo = '<direction><direction-type><words>Moderato</words></direction-type><sound tempo="96"/></direction>'
    bars[0] = tempo + bars[0]
    p = write(tmp_path, score([("Piano", bars)]))
    entry = process_file(p, tmp_path / "out", {"id": "xml-test", "title": "XML test"}, policy=GatePolicy(min_duration_s=10))
    assert entry.hand_source == "staves"
    assert entry.levels[-1].fingering and entry.levels[-1].tempo_bpm == 96
    assert OFFICIAL.allowed_hand_sources.count("staves") == 1


def test_bad_files(tmp_path):
    with pytest.raises(MusicXMLError):
        musicxml.load(write(tmp_path, '<?xml version="1.0"?><score-timewise/>'))
    with pytest.raises(MusicXMLError):
        musicxml.load(write(tmp_path, score([("Piano", [rest(16)])]), "rests.musicxml"))


def test_lilypond_voice_tracks_named_by_staff(tmp_path):
    # LilyPond writes a track per voice named after its staff ("up", "upII", "down", "downII").
    path = make_midi(tmp_path / "ly.mid", [
        {"name": "up", "channel": 0, "notes": [(0, 960, 72)]},
        {"name": "upII", "channel": 1, "notes": [(0, 960, 64)]},
        {"name": "down", "channel": 2, "notes": [(0, 960, 48)]},
        {"name": "downII:", "channel": 3, "notes": [(0, 960, 41)]},
    ])
    _, notes, hand_source, _ = load_notes(path)
    assert hand_source == "track_names"
    assert sorted(n.pitch for n in notes if n.hand == "R") == [64, 72]
