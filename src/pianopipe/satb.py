"""Four-part choir settings (hymnals, chorales) on two hands: the keyboard reduction organists use.

Soprano and alto go to the right hand, tenor and bass to the left. Where tenor and bass are more than an octave
apart, the tenor (the left hand's top note) moves to the right hand when the right hand can take it: it stays below
the right hand's notes, the right hand then spans an octave or less, and holds five notes at most. Otherwise the
note stays and the quality gate's playability check reports the stretch.
"""

from __future__ import annotations

from dataclasses import replace

from .hands import group_chords
from .model import Note

OCTAVE = 12


def reach(notes: list[Note], ticks_per_beat: int = 480) -> list[Note]:
    """Move left-hand top notes to the right hand where the left hand would stretch past an octave."""
    out: list[Note] = []
    for chord in group_chords(notes, window=max(1, ticks_per_beat // 32)):
        left = sorted((n for n in chord if n.hand == "L"), key=lambda n: n.pitch)
        right = sorted((n for n in chord if n.hand == "R"), key=lambda n: n.pitch)
        if len(left) >= 2 and left[-1].pitch - left[0].pitch > OCTAVE:
            tenor = left[-1]
            fits = (not right or (tenor.pitch <= right[0].pitch and right[-1].pitch - tenor.pitch <= OCTAVE)) and len(right) < 5
            if fits:
                moved = replace(tenor, hand="R", channel=0, track=0)
                if right and tenor.pitch == right[0].pitch:
                    moved = None  # alto and tenor in unison: the right hand already plays it
                left = left[:-1]
                right = ([moved] if moved else []) + right
        out.extend(left + right)
    return sorted(out, key=lambda n: (n.start, n.pitch))
