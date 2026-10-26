"""Builders for small synthetic MIDI files used by the tests."""

from __future__ import annotations

from pathlib import Path

import mido

TPB = 480


def make_midi(path: Path, tracks: list[dict], tpb: int = TPB, tempo: int = 600000) -> Path:
    """tracks: [{"name": str, "channel": int, "program": int, "notes": [(start, dur, pitch), ...]}]"""
    mid = mido.MidiFile(type=1, ticks_per_beat=tpb)
    conductor = mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=tempo, time=0)])
    mid.tracks.append(conductor)
    for spec in tracks:
        ch = spec.get("channel", 0)
        events = []
        for start, dur, pitch in spec["notes"]:
            events.append((start, 1, mido.Message("note_on", channel=ch, note=pitch, velocity=80)))
            events.append((start + dur, 0, mido.Message("note_off", channel=ch, note=pitch, velocity=0)))
        events.sort(key=lambda e: (e[0], e[1]))
        track = mido.MidiTrack()
        if spec.get("name"):
            track.append(mido.MetaMessage("track_name", name=spec["name"], time=0))
        track.append(mido.Message("program_change", channel=ch, program=spec.get("program", 0), time=0))
        last = 0
        for tick, _, msg in events:
            track.append(msg.copy(time=tick - last))
            last = tick
        mid.tracks.append(track)
    mid.save(str(path))
    return path


def two_hand_piece(bars: int = 4) -> tuple[list, list]:
    """A simple melody over an Alberti-style bass, as (rh_notes, lh_notes)."""
    melody = [72, 74, 76, 77, 79, 77, 76, 74]
    rh, lh = [], []
    for bar in range(bars):
        for beat in range(4):
            t = (bar * 4 + beat) * TPB
            rh.append((t, TPB, melody[(bar * 4 + beat) % len(melody)]))
            rh.append((t, TPB, melody[(bar * 4 + beat) % len(melody)] - 4))
            for i, p in enumerate((48, 55, 52, 55)):
                lh.append((t + i * TPB // 4, TPB // 4, p))
    return rh, lh
