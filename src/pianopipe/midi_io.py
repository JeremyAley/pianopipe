"""Reading arbitrary MIDI files and writing app-ready MIDI.

App-ready MIDI (see docs/formats.md) is a type 1 file with three tracks:
track 0 holds tempo/time-signature meta events, track 1 is named "RH"
(channel 0) and track 2 is named "LH" (channel 1). A note with a finger gets
a text event "f1".."f5" right before its note-on, and a finger printed in the
score has a "printed" text event right before that. The sustain pedal (Song.pedal) is CC 64 on
the RH track: 127 down, 0 up, a change written as the up then the down at one tick.
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import mido

from .model import Note, Song, TrackInfo

PEDAL_CC = 64  # sustain; a value of 64 or more is down

# Meta events that describe timing and are copied into app-ready files.
_TIMING_META = {"set_tempo", "time_signature", "key_signature"}


def load(path: str | Path) -> Song:
    mid = mido.MidiFile(str(path))
    song = Song(ticks_per_beat=mid.ticks_per_beat)
    for index, track in enumerate(mid.tracks):
        info = TrackInfo(index=index, name="")
        tick = 0
        programs: dict[int, int] = defaultdict(int)
        # (channel, pitch) -> queue of (start tick, velocity, program, finger, printed); FIFO pairing.
        open_notes: dict[tuple[int, int], deque] = defaultdict(deque)
        finger = None  # from a finger text event, for the next note-on
        pedal_down: dict[int, int] = {}  # channel -> tick its sustain pedal went down
        printed = mark = False  # the finger was marked printed / the last text event was the mark
        for msg in track:
            tick += msg.time
            if msg.is_meta:
                if msg.type == "track_name" and not info.name:
                    info.name = msg.name.replace("\x00", "").strip()  # MuseScore pads names with NUL
                elif msg.type in _TIMING_META:
                    song.meta.append((tick, msg.copy(time=0)))
                elif msg.type == "text":
                    finger = read_finger(msg.text)
                    printed = mark and finger is not None
                    mark = msg.text.strip() == PRINTED_TEXT
                continue
            if msg.type == "program_change":
                programs[msg.channel] = msg.program
            elif msg.type == "control_change" and msg.control == PEDAL_CC:
                if msg.value >= 64:
                    pedal_down.setdefault(msg.channel, tick)
                elif msg.channel in pedal_down:
                    song.pedal.append((pedal_down.pop(msg.channel), tick))
            elif msg.type == "note_on" and msg.velocity > 0:
                open_notes[(msg.channel, msg.note)].append((tick, msg.velocity, programs[msg.channel], finger, printed))
                finger, printed, mark = None, False, False
            elif msg.type in ("note_off", "note_on"):
                queue = open_notes.get((msg.channel, msg.note))
                if queue:
                    start, vel, prog, fing, pr = queue.popleft()
                    if tick > start:
                        info.notes.append(Note(start, tick, msg.note, vel, None, index, msg.channel, prog, fing, pr))
        # Close notes that never got a note-off at the end of the track.
        for (channel, pitch), queue in open_notes.items():
            for start, vel, prog, fing, pr in queue:
                if tick > start:
                    info.notes.append(Note(start, tick, pitch, vel, None, index, channel, prog, fing, pr))
        info.notes.sort(key=lambda n: (n.start, n.pitch))
        song.tracks.append(info)
        song.pedal += [(down, tick) for down in pedal_down.values()]  # still down at the track's end
    song.meta.sort(key=lambda e: e[0])
    song.pedal = merge_pedal(song.pedal)
    return song


def merge_pedal(spans) -> list[tuple[int, int]]:
    """Pedal spans (down, up) in order, overlapping ones (another track's or channel's pedal) joined into one. Spans
    that only touch stay apart: that is a change (up and down again at one tick). Empty spans are dropped."""
    out: list[list[int]] = []
    for down, up in sorted((int(d), int(u)) for d, u in spans if u > d):
        if out and down < out[-1][1]:
            out[-1][1] = max(out[-1][1], up)
        else:
            out.append([down, up])
    return [(d, u) for d, u in out]


def clip_pedal(spans, start: int, end: int) -> list[tuple[int, int]]:
    """The pedal spans inside start..end (a level's notes), cut at its ends."""
    return [(max(d, start), min(u, end)) for d, u in spans if u > start and d < end and min(u, end) > max(d, start)]


# The text event right before a finger's that marks it printed in the score (docs/formats.md).
PRINTED_TEXT = "printed"


def finger_text(finger: int) -> str:
    """The text event that gives the next note-on's finger (docs/formats.md)."""
    return f"f{finger}"


def read_finger(text: str) -> int | None:
    """The finger in a text event written by finger_text, or None for any other text."""
    if len(text) == 2 and text[0] == "f" and text[1] in "12345":
        return int(text[1])
    return None


def _events_to_track(name: str, channel: int, notes: list[Note], pedal=()) -> mido.MidiTrack:
    # (tick, order, pitch, message): note-offs sort before note-ons at the same tick so repeated notes retrigger,
    # and a finger's text event sorts right before its own note-on, with the printed mark right before that.
    # The pedal comes up before anything else at its tick and goes down after the note-ons, so a change at one tick
    # reads as up, then down.
    events: list[tuple[int, int, int, mido.Message]] = []
    for down, up in pedal:
        events.append((down, 4, -1, mido.Message("control_change", channel=channel, control=PEDAL_CC, value=127)))
        events.append((up, -1, -1, mido.Message("control_change", channel=channel, control=PEDAL_CC, value=0)))
    for n in notes:
        if n.finger is not None:
            if n.printed:
                events.append((n.start, 1, n.pitch, mido.MetaMessage("text", text=PRINTED_TEXT)))
            events.append((n.start, 2, n.pitch, mido.MetaMessage("text", text=finger_text(n.finger))))
        events.append((n.start, 3, n.pitch, mido.Message("note_on", channel=channel, note=n.pitch, velocity=max(1, n.velocity))))
        events.append((n.end, 0, n.pitch, mido.Message("note_off", channel=channel, note=n.pitch, velocity=0)))
    events.sort(key=lambda e: (e[0], e[1] if e[1] in (-1, 4) else 0 if e[1] == 0 else 1, e[2], e[1]))
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name=name, time=0))
    track.append(mido.Message("program_change", channel=channel, program=0, time=0))
    last = 0
    for tick, _, _, msg in events:
        track.append(msg.copy(time=tick - last))
        last = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    return track


def write_app_midi(path: str | Path, ticks_per_beat: int, meta: list, notes: list[Note], pedal=()) -> None:
    """Write notes (each with hand set) as an app-ready MIDI file, with the sustain pedal's (down, up) spans as CC 64
    on the RH track (the LH track when the right hand has no notes); no spans, no CC 64 (the pedal is unknown)."""
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
    rh, lh = [n for n in notes if n.hand == "R"], [n for n in notes if n.hand == "L"]
    pedal = merge_pedal(pedal)
    mid.tracks.append(_events_to_track("RH", 0, rh, pedal if rh else ()))
    mid.tracks.append(_events_to_track("LH", 1, lh, () if rh else pedal))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(path))
