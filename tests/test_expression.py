"""The score's expression in app-ready files (docs/formats.md, "Dynamics and articulation"; learning-mode.md §14 L32):
the sustain pedal from a MIDI file's CC 64 or a score's pedal marks, MusicXML staccato played short, hairpins as
velocity ramps."""

from __future__ import annotations

import mido

from pianopipe import midi_io, musicxml
from pianopipe.model import Note
from pianopipe.process import process_file

from helpers import TPB, make_midi, two_hand_piece
from test_musicxml import BWD, FWD, backup, bar, note, score, write


def pedal_of(path) -> list[tuple[int, int, int]]:
    """Every CC 64 in a file as (absolute tick, track, value), in file order."""
    out = []
    for i, track in enumerate(mido.MidiFile(str(path)).tracks):
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "control_change" and msg.control == 64:
                out.append((tick, i, msg.value))
    return out


def with_pedal(path, spans, channel=0, track=1):
    """Adds CC 64 down/up at the given (down, up) ticks to one track of a file made by make_midi."""
    mid = mido.MidiFile(str(path))
    t = mid.tracks[track]
    events, tick = [], 0
    for msg in t:
        tick += msg.time
        events.append((tick, 1, msg))
    for down, up in spans:
        events.append((down, 2, mido.Message("control_change", channel=channel, control=64, value=100)))
        events.append((up, 0, mido.Message("control_change", channel=channel, control=64, value=0)))
    end = [e for e in events if e[2].type == "end_of_track"]
    events = sorted((e for e in events if e[2].type != "end_of_track"), key=lambda e: (e[0], e[1]))
    new, last = mido.MidiTrack(), 0
    for tick, _, msg in events + end:
        new.append(msg.copy(time=max(0, tick - last)))
        last = max(last, tick)
    mid.tracks[track] = new
    mid.save(str(path))
    return path


def test_cc64_is_read_and_written_on_the_rh_track(tmp_path):
    rh, lh = two_hand_piece(2)
    src = make_midi(tmp_path / "pedalled.mid", [{"name": "Piano", "notes": rh + lh}])
    # Down over bar 1, changed at bar 2 (up and down at one tick), up before the end.
    with_pedal(src, [(0, 4 * TPB), (4 * TPB, 7 * TPB)])
    song = midi_io.load(src)
    assert song.pedal == [(0, 4 * TPB), (4 * TPB, 7 * TPB)]
    entry = process_file(src, tmp_path / "out")
    for lv in entry.levels:
        cc = pedal_of(tmp_path / "out" / entry.id / f"level{lv.level}.mid")
        if lv.level < 3:
            assert cc == []  # the melody levels leave the pedal unknown
            continue
        assert [t for t, _, _ in cc] == [0, 4 * TPB, 4 * TPB, 7 * TPB]
        assert [v for _, _, v in cc] == [127, 0, 127, 0]  # the change: up, then down, at one tick
        assert {tr for _, tr, _ in cc} == {1}  # the RH track


def test_overlapping_pedals_merge_and_files_without_cc64_stay_unpedalled(tmp_path):
    assert midi_io.merge_pedal([(0, 100), (50, 200), (200, 300), (400, 400)]) == [(0, 200), (200, 300)]
    assert midi_io.clip_pedal([(0, 200), (300, 500)], 100, 400) == [(100, 200), (300, 400)]
    rh, lh = two_hand_piece(1)
    src = make_midi(tmp_path / "plain.mid", [{"name": "Piano", "notes": rh + lh}])
    assert midi_io.load(src).pedal == []
    entry = process_file(src, tmp_path / "out")
    assert pedal_of(tmp_path / "out" / entry.id / "level4.mid") == []


def test_write_puts_a_left_hand_pieces_pedal_on_the_lh_track(tmp_path):
    out = tmp_path / "lh.mid"
    midi_io.write_app_midi(out, TPB, [], [Note(0, TPB, 48, hand="L")], [(0, TPB)])
    assert [(t, tr, v) for t, tr, v in pedal_of(out)] == [(0, 2, 127), (TPB, 2, 0)]


PED = '<direction placement="below"><direction-type><pedal type="{}" line="yes"/></direction-type>{}</direction>'


def ped(kind: str, offset: int = 0) -> str:
    return PED.format(kind, f"<offset>{offset}</offset>" if offset else "")


def test_musicxml_pedal_marks_through_a_repeat(tmp_path):
    lh = note("C", 3, 16, 5, 2)
    rh = note("C", 4, 16)
    # Bar 1: down at its start, changed half way, lifted at its end; bar 2 (repeated): down, lifted at its end.
    m1 = (ped("start") + rh + backup() + lh + backup(8) + ped("change") + "<forward><duration>8</duration></forward>"
          + ped("stop"))
    m2 = FWD + ped("start") + rh + backup() + lh + ped("stop") + BWD
    sc = musicxml.load(write(tmp_path, score([("Piano", [m1, m2])])))
    q = TPB
    assert sc.song.pedal == [(0, 2 * q), (2 * q, 4 * q), (4 * q, 8 * q), (8 * q, 12 * q)]


def test_musicxml_staccato_sounds_half_its_value(tmp_path):
    stacc = "<notations><articulations><staccato/></articulations></notations>"
    sharp = "<notations><articulations><staccatissimo/></articulations></notations>"
    rh = note("C", 4, 4, 1, 1, stacc) + note("D", 4, 4, 1, 1, sharp) + note("E", 4) + note("F", 4, 4, 1, 1, '<tie type="start"/>', stacc)
    m2 = note("F", 4, 16, 1, 1, '<tie type="stop"/>')
    sc = musicxml.load(write(tmp_path, score([("Piano", [bar(rh, note("C", 3, 16, 5, 2)), bar(m2, note("C", 3, 16, 5, 2))])])))
    right = [(n.pitch, n.start, n.end) for n in sc.notes if n.hand == "R"]
    q = TPB
    assert right == [(60, 0, q // 2), (62, q, q + q // 4), (64, 2 * q, 3 * q), (65, 3 * q, 8 * q)]  # a tie isn't cut


def dyn(mark: str, velocity: int) -> str:
    return (f'<direction><direction-type><dynamics><{mark}/></dynamics></direction-type>'
            f'<sound dynamics="{velocity / 0.9:.2f}"/></direction>')


def wedge(kind: str) -> str:
    return f'<direction><direction-type><wedge type="{kind}"/></direction-type></direction>'


def velocities(sc) -> list[int]:
    return [n.velocity for n in sc.notes if n.hand == "R"]


def test_a_crescendo_ramps_to_the_next_marking(tmp_path):
    lh = note("C", 3, 16, 5, 2)
    quarters = note("C", 4) + note("D", 4) + note("E", 4) + note("F", 4)
    m1 = dyn("p", 50) + wedge("crescendo") + quarters + wedge("stop") + backup() + lh
    m2 = dyn("f", 90) + note("G", 4, 16) + backup() + lh
    sc = musicxml.load(write(tmp_path, score([("Piano", [m1, m2])])))
    assert velocities(sc) == [50, 60, 70, 80, 90]


def test_a_diminuendo_with_no_marking_after_it_ends_softer_and_stays_there(tmp_path):
    lh = note("C", 3, 16, 5, 2)
    quarters = note("C", 4) + note("D", 4) + note("E", 4) + note("F", 4)
    m1 = dyn("mf", 72) + wedge("diminuendo") + quarters + wedge("stop") + backup() + lh
    m2 = note("G", 4) + note("A", 4) + note("B", 4) + note("C", 5) + backup() + lh
    m3 = dyn("f", 90) + note("G", 4, 16) + backup() + lh
    sc = musicxml.load(write(tmp_path, score([("Piano", [m1, m2, m3])])))
    v = velocities(sc)
    assert v[:4] == [72, 69, 66, 63] and v[4:8] == [60, 60, 60, 60] and v[8] == 90
    left = [n.velocity for n in sc.notes if n.hand == "L"]
    assert left == [72, 60, 90]  # the left hand follows the hairpin too


def test_a_second_hairpin_starts_from_the_level_the_first_left(tmp_path):
    # p, a crescendo with no marking after it (to 62, held), then another crescendo to f: from 62, never past 90.
    lh = note("C", 3, 16, 5, 2)
    quarters = note("C", 4) + note("D", 4) + note("E", 4) + note("F", 4)
    m1 = dyn("p", 50) + wedge("crescendo") + quarters + wedge("stop") + backup() + lh
    m2 = wedge("crescendo") + quarters + wedge("stop") + backup() + lh
    m3 = dyn("f", 90) + note("G", 4, 16) + backup() + lh
    sc = musicxml.load(write(tmp_path, score([("Piano", [m1, m2, m3])])))
    v = velocities(sc)
    assert v[:4] == [50, 53, 56, 59] and v[4:] == [62, 69, 76, 83, 90]
