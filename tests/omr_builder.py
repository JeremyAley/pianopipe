"""Builders for small synthetic Audiveris projects (.omr) and their MusicXML exports, for the OMR repair tests.

A project is a zip with one sheet#1/sheet#1.xml holding systems, parts, staves, measures (with Audiveris's voices),
inters (heads, stems, beams, flags, dots, rests, tuplets, accidentals, clefs, keys) with bounding boxes, and the
relations between them, plus optionally a drawn BINARY.png. Only what pianopipe.omr reads is written.
"""

from __future__ import annotations

import io
import struct
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path

IL = 20  # staff space in pixels
HEAD_W, HEAD_H = 24, 20
SYSTEM_HEIGHT = 600
MEASURE_W = 600
LEFT = 100
STEPS = "CDEFGAB"


def treble(name: str) -> int:
    """Audiveris pitch (staff steps from the middle line, positive downwards) of a note on a treble staff: B4 = 0."""
    return 34 - (7 * int(name[-1]) + STEPS.index(name[0]))


def bass(name: str) -> int:
    """... on a bass staff: D3 = 0."""
    return 22 - (7 * int(name[-1]) + STEPS.index(name[0]))


@dataclass
class N:
    """One chord (or rest) as printed: where it is and its symbols."""
    staff: int  # 1 = treble (right hand), 2 = bass
    x: float  # left edge of its heads, relative to the measure's left barline
    steps: tuple[int, ...] = ()
    shape: str = "NOTEHEAD_BLACK"
    stem: str | None = "up"
    beam: str | None = None  # chords with the same name share their beams
    beams: int = 0  # beams on the group (kept in the .omr only if beam_inter)
    beam_inter: bool = True  # False: the beams are drawn on the page but Audiveris didn't keep them
    flags: int = 0
    dots: int = 0
    tuplet: str | None = None  # chords with the same name share a TUPLET_THREE
    rest: str = ""
    voice: int | None = 1  # Audiveris's voice; None = left out of every voice
    alters: dict[int, str] = field(default_factory=dict)  # step -> SHARP | FLAT | NATURAL
    ties: dict[int, str] = field(default_factory=dict)  # step -> LEFT (a tie starts) | RIGHT (one ends)


@dataclass
class System:
    measures: list[list[N]]  # the piano part's chords per measure ([] when the piano is absent)
    piano: bool = True
    vocal: bool = False
    fifths: int = 0
    free_heads: list[tuple[int, int, float, int]] = field(default_factory=list)  # (measure, staff, x, step) unread marks


class _Sheet:
    def __init__(self):
        self.next = 1000
        self.inters: list[str] = []
        self.relations: list[str] = []
        self.ink: list[tuple[float, float, float, float]] = []  # boxes to paint black on the image
        self.glyphs: list[str] = []

    def id(self) -> int:
        self.next += 1
        return self.next

    def inter(self, tag: str, x: float, y: float, w: float, h: float, staff: int | None = None, ink: bool = True,
              **attrs) -> int:
        i = self.id()
        a = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
        st = f' staff="{staff}"' if staff is not None else ""
        self.inters.append(f'<{tag} {a}{st} id="{i}"><bounds x="{x}" y="{y}" w="{w}" h="{h}"/></{tag}>')
        if ink:
            self.ink.append((x, y, w, h))
        return i

    def rel(self, kind: str, source: int, target: int, **attrs) -> None:
        a = " ".join(f'{k}="{v}"' for k, v in attrs.items())
        self.relations.append(f'<relation source="{source}" target="{target}"><{kind} {a}/></relation>')


def omr_bytes(systems: list[System], image: bool = False) -> bytes:
    sh = _Sheet()
    sys_xml = []
    staff_no = 0
    for k, system in enumerate(systems):
        top = k * SYSTEM_HEIGHT
        mids = {1: top + 250, 2: top + 450}
        n_measures = max(len(system.measures), 1)
        stacks = "".join(f'<stack id="{j + 1}" left="{LEFT + j * MEASURE_W}" right="{LEFT + (j + 1) * MEASURE_W}"/>'
                         for j in range(n_measures))
        parts = []
        if system.vocal:
            staff_no += 1
            empty = "".join(f'<measure id="{j + 1}"/>' for j in range(n_measures))
            parts.append(f'<part id="1"><staff id="{staff_no}" left="{LEFT}" right="{LEFT + n_measures * MEASURE_W}">'
                         f'<lines><line><point x="{LEFT}" y="{top + 60}"/><point x="{LEFT}" y="{top + 140}"/></line>'
                         f'</lines></staff>{empty}</part>')
        free: list[int] = []
        if system.piano:
            ids = {1: staff_no + 1, 2: staff_no + 2}
            staff_no += 2
            staves = ""
            for s in (1, 2):
                clef = sh.inter("clef", LEFT + 10, mids[s] - 40, 40, 80, ids[s], ink=False,
                                kind="TREBLE" if s == 1 else "BASS", shape="G_CLEF" if s == 1 else "F_CLEF")
                key = sh.inter("key", LEFT + 50, mids[s] - 40, 30, 80, ids[s], ink=False, fifths=system.fifths)
                staves += (f'<staff id="{ids[s]}" left="{LEFT}" right="{LEFT + n_measures * MEASURE_W}"><lines>'
                           f'<line><point x="{LEFT}" y="{mids[s] - 2 * IL}"/><point x="{LEFT + 50}" y="{mids[s] + 2 * IL}"/>'
                           f'</line></lines><header><clef>{clef}</clef><key>{key}</key></header></staff>')
            measures = ""
            for j, chords in enumerate(system.measures):
                left = LEFT + j * MEASURE_W
                measures += _measure(sh, j + 1, left, chords, ids, mids)
            for j, s, x, step in system.free_heads:
                gx, gy = LEFT + j * MEASURE_W + x, mids[s] + step * IL / 2 - HEAD_H / 2
                g = sh.id()
                sh.glyphs.append(f'<glyph left="{gx}" top="{gy}" id="{g}"><run-table width="{HEAD_W}" height="{HEAD_H}"/></glyph>')
                sh.ink.append((gx, gy, HEAD_W, HEAD_H))
                free.append(g)
            parts.append(f'<part id="2">{staves}{measures}</part>')
        sys_xml.append(f'<system id="{k + 1}">{stacks}{"".join(parts)}<free-glyphs>{" ".join(map(str, free))}'
                       f'</free-glyphs><sig><inters>{{inters}}</inters><relations>{{relations}}</relations></sig></system>')
    # every inter goes in the first system's sig: the reader doesn't care which system holds it
    body = sys_xml[0].replace("{inters}", "".join(sh.inters)).replace("{relations}", "".join(sh.relations))
    rest = [s.replace("{inters}", "").replace("{relations}", "") for s in sys_xml[1:]]
    xml = (f'<?xml version="1.0"?><sheet><scale><interline main="{IL}"/><beam main-thickness="10"/>'
           f'<black-head mean-width="{HEAD_W}" mean-height="{HEAD_H}"/></scale>'
           f'<glyph-index>{"".join(sh.glyphs)}</glyph-index><page>{body}{"".join(rest)}</page></sheet>')
    buf = io.BytesIO()
    out = zipfile.ZipFile(buf, "w")
    out.writestr("book.xml", "<book/>")
    out.writestr("sheet#1/sheet#1.xml", xml)
    if image:
        width = LEFT + max(len(s.measures) for s in systems) * MEASURE_W + LEFT
        out.writestr("sheet#1/BINARY.png", _png(width, len(systems) * SYSTEM_HEIGHT, sh.ink))
    out.close()
    return buf.getvalue()


def _measure(sh: _Sheet, number: int, left: float, chords: list[N], ids: dict[int, int], mids: dict[int, float]) -> str:
    heads, rests = [], []
    voices: dict[int, list[int]] = {}
    groups: dict[str, list[tuple[N, int, float, float]]] = {}  # beam name -> (chord, stem id, stem x, stem tip y)
    tuplets: dict[str, int] = {}
    for n in chords:
        x = left + n.x
        mid = mids[n.staff]
        if n.rest:
            r = sh.inter("rest", x, mid - IL, 18, 2 * IL, ids[n.staff], shape=n.rest)
            c = sh.inter("rest-chord", x, mid - IL, 18, 2 * IL, ids[n.staff], ink=False)
            sh.rel("containment", c, r)
            rests.append(c)
        else:
            c = sh.inter("head-chord", x, mid, HEAD_W, HEAD_H, ids[n.staff], ink=False)
            hs = []
            for step in n.steps:
                hy = mid + step * IL / 2 - HEAD_H / 2
                h = sh.inter("head", x, hy, HEAD_W, HEAD_H, ids[n.staff], shape=n.shape, pitch=step)
                sh.rel("containment", c, h)
                hs.append((h, hy, step))
                for _ in range(n.dots):
                    d = sh.inter("augmentation-dot", x + HEAD_W + 6, hy + 7, 5, 5, ids[n.staff])
                    sh.rel("augmentation", d, h)
                if step in n.alters:
                    a = sh.inter("alter", x - 18, hy - 10, 14, 40, ids[n.staff], shape=n.alters[step], pitch=step)
                    sh.rel("alter-head", a, h)
                if step in n.ties:
                    t = sh.inter("slur", x + HEAD_W, hy + HEAD_H, 40, 8, ids[n.staff], ink=False, tie="true", shape="SLUR_BELOW")
                    sh.rel("slur-head", t, h, side=n.ties[step])
            if n.stem:
                ys = [hy for _, hy, _ in hs]
                if n.stem == "up":
                    sx, sy0, sy1 = x + HEAD_W - 3, min(ys) - 3.5 * IL, max(ys) + HEAD_H / 2
                    tip = sy0
                else:
                    sx, sy0, sy1 = x, min(ys) + HEAD_H / 2, max(ys) + HEAD_H + 3.5 * IL
                    tip = sy1
                s = sh.inter("stem", sx, sy0, 3, sy1 - sy0, ids[n.staff], shape="STEM")
                sh.rel("chord-stem", c, s)
                for h, _, _ in hs:
                    sh.rel("head-stem", h, s)
                if n.beam:
                    groups.setdefault(n.beam, []).append((n, s, sx, tip))
                for _ in range(n.flags):
                    f = sh.inter("flag", sx + 3, tip, 12, 30, ids[n.staff], shape=f"FLAG_{n.flags}")
                    sh.rel("flag-stem", f, s)
                    break
            if n.tuplet:
                if n.tuplet not in tuplets:
                    tuplets[n.tuplet] = sh.inter("tuplet", x, mid - 5 * IL, 12, 20, ids[n.staff], shape="TUPLET_THREE")
                sh.rel("chord-tuplet", c, tuplets[n.tuplet])
            heads.append(c)
        if n.voice is not None:
            voices.setdefault(n.voice, []).append(c)
    for members in groups.values():
        first, last = members[0], members[-1]
        up = first[0].stem == "up"
        for b in range(first[0].beams):
            y = first[3] + (b * 1.6 * IL / 2 if up else -b * 1.6 * IL / 2 - 10)
            box = (first[2], y, last[2] - first[2] + 3, 10)
            if first[0].beam_inter:
                beam = sh.inter("beam", *box, shape="BEAM")
                for _, s, _, _ in members:
                    sh.rel("beam-stem", beam, s)
            else:
                sh.ink.append(box)
    vx = "".join(f'<voice id="{v}"><slots>' + "".join(f'<entry><key>{k + 1}</key><value chord="{c}" status="BEGIN"/></entry>'
                                                       for k, c in enumerate(cs)) + "</slots></voice>"
                 for v, cs in sorted(voices.items()))
    return (f'<measure id="{number}"><head-chords>{" ".join(map(str, heads))}</head-chords>'
            f'<rest-chords>{" ".join(map(str, rests))}</rest-chords>{vx}</measure>')


def _png(w: int, h: int, ink: list[tuple[float, float, float, float]]) -> bytes:
    pixels = bytearray(b"\xff" * (w * h))
    for x, y, bw, bh in ink:
        for yy in range(max(0, int(y)), min(h, int(y + bh))):
            row = yy * w
            pixels[row + max(0, int(x)):row + min(w, int(x + bw))] = b"\x00" * (min(w, int(x + bw)) - max(0, int(x)))
    raw = b"".join(b"\x00" + bytes(pixels[r * w:(r + 1) * w]) for r in range(h))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


# ---------------------------------------------------------------- the export

def xnote(pitch: str, dur: int, voice: int = 1, staff: int = 1, x: float | None = None, chord: bool = False,
          tie: str = "") -> str:
    """A MusicXML <note>: pitch like "C#5" or "rest", duration in divisions (4 per quarter), x relative to the barline."""
    dx = f' default-x="{x * 10 / IL:g}"' if x is not None else ""
    if pitch == "rest":
        p = "<rest/>"
    else:
        alter = {"#": "<alter>1</alter>", "b": "<alter>-1</alter>"}.get(pitch[1:-1], "")
        p = f"<pitch><step>{pitch[0]}</step>{alter}<octave>{pitch[-1]}</octave></pitch>"
    t = f'<tie type="{tie}"/>' if tie else ""
    return (f'<note{dx}>{"<chord/>" if chord else ""}{p}<duration>{dur}</duration>{t}<voice>{voice}</voice>'
            f'<staff>{staff}</staff></note>')


def backup(dur: int) -> str:
    return f"<backup><duration>{dur}</duration></backup>"


def export_xml(measures: list[str], time: tuple[int, int] = (4, 4), vocal: list[str] | None = None,
               times: dict[int, tuple[int, int]] | None = None) -> str:
    """A score-partwise export: the piano part (P2, two staves) and optionally a one-staff vocal part (P1).
    `times` adds time signatures to later measures (0-based index -> (beats, beat type))."""
    def part(pid: str, ms: list[str], staves: int) -> str:
        out = []
        for k, body in enumerate(ms):
            t = (times or {}).get(k)
            attrs = (f"<attributes><divisions>4</divisions><key><fifths>0</fifths></key><time><beats>{time[0]}</beats>"
                     f"<beat-type>{time[1]}</beat-type></time><staves>{staves}</staves></attributes>") if k == 0 else (
                f"<attributes><time><beats>{t[0]}</beats><beat-type>{t[1]}</beat-type></time></attributes>" if t else "")
            out.append(f'<measure number="{k + 1}">{attrs}{body}</measure>')
        return f'<part id="{pid}">{"".join(out)}</part>'

    plist = '<score-part id="P2"><part-name>Piano</part-name></score-part>'
    parts = part("P2", measures, 2)
    if vocal is not None:
        plist = '<score-part id="P1"><part-name>Voice</part-name></score-part>' + plist
        parts = part("P1", vocal, 1) + parts
    return f'<?xml version="1.0"?><score-partwise><part-list>{plist}</part-list>{parts}</score-partwise>'


def write_project(tmp: Path, systems: list[System], measures: list[str], image: bool = False, **kw) -> tuple[Path, Path]:
    omr_path = tmp / "song.omr"
    omr_path.write_bytes(omr_bytes(systems, image))
    xml_path = tmp / "song.musicxml"
    xml_path.write_text(export_xml(measures, **kw), encoding="utf-8")
    return xml_path, omr_path
