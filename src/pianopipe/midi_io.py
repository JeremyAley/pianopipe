"""Reading arbitrary MIDI files and writing app-ready MIDI.

App-ready MIDI (see docs/formats.md) is a type 1 file with three tracks:
track 0 holds tempo/time-signature meta events, track 1 is named "RH"
(channel 0) and track 2 is named "LH" (channel 1). A note with a finger gets
a text event "f1".."f5" right before its note-on.
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import mido

from .model import Note, Song, TrackInfo

# Meta events that describe timing and are copied into app-ready files.
_TIMING_META = {"set_tempo", "time_signature", "key_signature"}


def load(path: str | Path) -> Song:
    mid = mido.MidiFile(str(path))
    song = Song(ticks_per_beat=mid.ticks_per_beat)
    for index, track in enumerate(mid.tracks):
        info = TrackInfo(index=index, name="")
        tick = 0
        programs: dict[int, int] = defaultdict(int)
        # (channel, pitch) -> queue of (start tick, velocity, program, finger); FIFO pairing.
        open_notes: dict[tuple[int, int], deque] = defaultdict(deque)
        finger = None  # from a finger text event, for the next note-on
        for msg in track:
            tick += msg.time
            if msg.is_meta:
                if msg.type == "track_name" and not info.name:
                    info.name = msg.name.strip()
                elif msg.type in _TIMING_META:
                    song.meta.append((tick, msg.copy(time=0)))
                elif msg.type == "text":
                    finger = read_finger(msg.text)
                continue
            if msg.type == "program_change":
                programs[msg.channel] = msg.program
            elif msg.type == "note_on" and msg.velocity > 0:
                open_notes[(msg.channel, msg.note)].append((tick, msg.velocity, programs[msg.channel], finger))
                finger = None
            elif msg.type in ("note_off", "note_on"):
                queue = open_notes.get((msg.channel, msg.note))
                if queue:
                    start, vel, prog, fing = queue.popleft()
                    if tick > start:
                        info.notes.append(Note(start, tick, msg.note, vel, None, index, msg.channel, prog, fing))
        # Close notes that never got a note-off at the end of the track.
        for (channel, pitch), queue in open_notes.items():
            for start, vel, prog, fing in queue:
                if tick > start:
                    info.notes.append(Note(start, tick, pitch, vel, None, index, channel, prog, fing))
        info.notes.sort(key=lambda n: (n.start, n.pitch))
        song.tracks.append(info)
    song.meta.sort(key=lambda e: e[0])
    return song


def finger_text(finger: int) -> str:
    """The text event that gives the next note-on's finger (docs/formats.md)."""
    return f"f{finger}"


def read_finger(text: str) -> int | None:
    """The finger in a text event written by finger_text, or None for any other text."""
    if len(text) == 2 and text[0] == "f" and text[1] in "12345":
        return int(text[1])
    return None


def _events_to_track(name: str, channel: int, notes: list[Note]) -> mido.MidiTrack:
    # (tick, order, pitch, message): note-offs sort before note-ons at the same tick so repeated notes retrigger,
    # and a finger's text event sorts right before its own note-on.
    events: list[tuple[int, int, int, mido.Message]] = []
    for n in notes:
        if n.finger is not None:
            events.append((n.start, 1, n.pitch, mido.MetaMessage("text", text=finger_text(n.finger))))
        events.append((n.start, 2, n.pitch, mido.Message("note_on", channel=channel, note=n.pitch, velocity=max(1, n.velocity))))
        events.append((n.end, 0, n.pitch, mido.Message("note_off", channel=channel, note=n.pitch, velocity=0)))
    events.sort(key=lambda e: (e[0], 0 if e[1] == 0 else 1, e[2], e[1]))
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name=name, time=0))
    track.append(mido.Message("program_change", channel=channel, program=0, time=0))
    last = 0
    for tick, _, _, msg in events:
        track.append(msg.copy(time=tick - last))
        last = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    return track


def write_app_midi(path: str | Path, ticks_per_beat: int, meta: list, notes: list[Note]) -> None:
    """Write notes (each with hand set) as an app-ready MIDI file."""
    unassigned = [n for n in notes if n.hand is None]
    if unassigned:
        raise ValueError(f"{len(unassigned)} notes have no hand assigned")
    mid = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("track_name", name="Conductor", time=0))
    if not any(m.type == "set_tempo" for _, m in meta):
        conductor.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    last = 0
    for tick, msg in meta:
        conductor.append(msg.copy(time=tick - last))
        last = tick
    conductor.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(conductor)
    mid.tracks.append(_events_to_track("RH", 0, [n for n in notes if n.hand == "R"]))
    mid.tracks.append(_events_to_track("LH", 1, [n for n in notes if n.hand == "L"]))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(path))
