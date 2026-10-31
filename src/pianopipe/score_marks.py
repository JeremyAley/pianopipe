"""Marks an OMR'd score lacks, written into its MusicXML: jumps (segno, To Coda, D.S. / D.C., coda, Fine) as playback
marks, a tempo, and its title and composer.

Audiveris's OCR reads "D.S. al Coda" and "To Coda" as words but never the segno and coda signs, so the importer has
to guess where they are (musicxml.play_order warns when it does). A publisher's vector PDF has them as glyphs
(pdf_songs.jumps); set_jumps replaces the file's jump marks with them, each a <direction> with the sign or the words
and a <sound> carrying the playback attribute. Bars are numbered as the importer counts them: the first part's
measures in order, from 1. tools/sheet_to_library.py uses this (docs/catalog-expansion.md §3.2).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from .musicxml import _JUMP_WORDS

JUMP_KINDS = ("segno", "tocoda", "dalsegno", "dacapo", "coda", "fine")
_SOUND = {"segno": ("segno", "segno"), "coda": ("coda", "coda"), "tocoda": ("tocoda", "coda"),
          "dalsegno": ("dalsegno", "segno"), "dacapo": ("dacapo", "yes"), "fine": ("fine", "yes")}
_AT_END = ("tocoda", "dalsegno", "dacapo", "fine")  # printed at the end of their bar


def _words_jump(text: str) -> bool:
    text = re.sub(r"^[\W_]+", "", (text or "").strip())
    return any(rx.match(text) for _, rx in _JUMP_WORDS)


def clear_jumps(root: ET.Element) -> int:
    """Remove every jump mark from every part: <sound> jump attributes, segno and coda signs, and words that read as
    a jump. Returns how many it removed."""
    removed = 0
    for m in root.iter("measure"):
        for snd in m.iter("sound"):
            for attr in JUMP_KINDS:
                removed += snd.attrib.pop(attr, None) is not None
        for d in m.findall("direction"):
            for dt in d.findall("direction-type"):
                for el in list(dt):
                    if el.tag in ("segno", "coda") or (el.tag == "words" and _words_jump(el.text)):
                        dt.remove(el)
                        removed += 1
                if not len(dt):
                    d.remove(dt)
            snd = d.find("sound")
            if snd is not None and not snd.attrib and not len(snd):
                d.remove(snd)
            if d.find("direction-type") is None and d.find("sound") is None:
                m.remove(d)
    return removed


def _insert(m: ET.Element, el: ET.Element, at_end: bool) -> None:
    """Before the measure's first note (after its attributes), or at its end before a closing barline."""
    kids = list(m)
    if at_end:
        idx = next((i for i, k in enumerate(kids) if k.tag == "barline" and k.get("location") == "right"), len(kids))
    else:
        idx = next((i for i, k in enumerate(kids) if k.tag in ("note", "backup", "forward")), len(kids))
    m.insert(idx, el)


def set_jumps(root: ET.Element, marks: dict[str, int]) -> list[str]:
    """Replace the file's jump marks with these ({kind: bar}, bars from 1). Returns what was set, e.g. "segno 5"."""
    part = root.find("part")
    if part is None:
        return []
    measures = part.findall("measure")
    clear_jumps(root)
    done = []
    al = " al Coda" if "tocoda" in marks else " al Fine" if "fine" in marks else ""
    words = {"tocoda": "To Coda", "dalsegno": "D.S." + al, "dacapo": "D.C." + al, "fine": "Fine"}
    for kind in JUMP_KINDS:
        bar = marks.get(kind)
        if bar is None or not 1 <= bar <= len(measures):
            continue
        d = ET.Element("direction", {"placement": "above"})
        dt = ET.SubElement(d, "direction-type")
        if kind in ("segno", "coda"):
            ET.SubElement(dt, kind)
        else:
            ET.SubElement(dt, "words").text = words[kind]
        attr, value = _SOUND[kind]
        ET.SubElement(d, "sound", {attr: value})
        _insert(measures[bar - 1], d, kind in _AT_END)
        done.append(f"{kind} {bar}")
    return done


def set_tempo(root: ET.Element, bpm: float, words: str = "") -> None:
    """A tempo at the start of the first bar: the printed words, and <sound tempo> for playback."""
    part = root.find("part")
    if part is None or part.find("measure") is None:
        return
    d = ET.Element("direction", {"placement": "above"})
    if words:
        ET.SubElement(ET.SubElement(d, "direction-type"), "words").text = words
    else:
        mt = ET.SubElement(ET.SubElement(d, "direction-type"), "metronome")
        ET.SubElement(mt, "beat-unit").text = "quarter"
        ET.SubElement(mt, "per-minute").text = f"{bpm:g}"
    ET.SubElement(d, "sound", {"tempo": f"{bpm:g}"})
    _insert(part.find("measure"), d, False)


def set_title(root: ET.Element, title: str, composer: str = "") -> None:
    """The work's title (what the library shows) and its composer (the show or artist, as the library names it)."""
    work = root.find("work")
    if work is None:
        work = ET.Element("work")
        root.insert(0, work)
    wt = work.find("work-title")
    if wt is None:
        wt = ET.SubElement(work, "work-title")
    wt.text = title
    mt = root.find("movement-title")
    if mt is not None:
        mt.text = title
    if not composer:
        return
    ident = root.find("identification")
    if ident is None:
        ident = ET.Element("identification")
        kids = list(root)
        after = max((i for i, k in enumerate(kids) if k.tag in ("work", "movement-number", "movement-title")), default=-1)
        root.insert(after + 1, ident)
    creator = next((c for c in ident.findall("creator") if c.get("type") == "composer"), None)
    if creator is None:
        creator = ET.Element("creator", {"type": "composer"})
        ident.insert(0, creator)
    creator.text = composer
