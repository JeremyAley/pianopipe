"""Reading a bought sheet-music PDF for import: where each song starts, its title and show or artist, whether its
notation is vector (a publisher's export) or a scan, and the tempo and jumps printed in it.

- **Songs in one PDF.** A Sheet Music Direct order prints several songs into one PDF. A song's first page has its
  title as the largest text near the top (or drawn as outlines: big filled shapes over the first staff), a
  copyright notice at the bottom, and on some prints a footer "<Title> - <pages> - 1". Its credits name the show
  ("from Walt Disney's ALADDIN") or the writers ("By LAUFEY, SPENCER STEWART").
- **Vector or scan.** A page whose symbols are music-font glyphs is vector notation, which pdf_check can read
  exactly; a page that is an image is a scan, which only Audiveris reads.
- **Tempo and jumps** come from pdf_score's reading: the words over the first bar ("Moderately fast",
  "Spirited (q = ca. 132)"), and the segno and coda signs and the D.S. / D.C. / To Coda / Fine words over each bar.

tools/sheet_to_library.py runs the whole import (docs/catalog-expansion.md §3.2). Needs pymupdf (the [omr] extra).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from . import pdf_score
from .musicxml import _JUMP_WORDS, _tempo_word

SMALL_WORDS = {"a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with"}
FOOTER = re.compile(r"^(.+?)\s+-\s+(\d+)\s+-\s+(\d+)$")  # "<Title> - <pages> - <page>"
CREDIT = re.compile(r"^(?:words\s+and\s+music|music\s+and\s+lyrics|music|words|lyrics)?\s*by\b\s*(.*)$", re.I)
SHOW_OWNER = re.compile(r"^.*(?:['’]s|s['’])\s+(?=\S)")  # "Walt Disney's ", "Walt Disney Pictures' "
SHOW_PREFIX = re.compile(r"^(?:the\s+)?(?:motion\s+picture|film|movie|musical|broadway\s+musical|tv\s+series|"
                         r"television\s+series|video\s+game)\s+", re.I)
METRONOME = re.compile(r"=\s*(?:c(?:a|irca)?\.?\s*)?(\d{2,3})\b")


@dataclass
class Song:
    title: str
    artist: str  # the show ("Aladdin"), else the first credited writer
    first: int  # pages, 1-based, inclusive
    last: int
    kind: str  # "vector" (notation as glyphs: pdf-check reads it) or "scan"

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class _Text:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float


def nice_case(text: str) -> str:
    """ALL CAPS titles in title case: "ROCKIN’ ROBIN" -> "Rockin' Robin", "I SEE THE LIGHT" -> "I See the Light"."""
    text = re.sub(r"\s+", " ", text.replace("’", "'").replace("‘", "'")).strip()
    if any(c.islower() for c in text):
        return text
    words = text.split(" ")
    out = []
    for i, w in enumerate(words):
        low = w.lower()
        if 0 < i < len(words) - 1 and low in SMALL_WORDS:
            out.append(low)
        elif re.fullmatch(r"[IVX]+", w) and len(w) > 1:
            out.append(w)  # roman numerals
        else:
            out.append(re.sub(r"[a-z]", lambda m: m.group().upper(), low, count=1))
    return " ".join(out)


def title_from_name(name: str) -> str:
    """A file's name as a title to start from: "go-the-distance_scan" -> "Go the distance scan"."""
    t = re.sub(r"[-_\s]+", " ", name).strip()
    return nice_case(t[:1].upper() + t[1:]) if t else "Untitled"


def _texts(page) -> list[_Text]:
    """The page's text lines (not music-font symbols), joined from their spans."""
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip() and pdf_score._family(s["font"]) is None
                     and not any(0xE000 <= ord(c) <= 0xF8FF for c in s["text"])]
            if spans:
                text = "".join(s["text"] for s in spans).strip()
                out.append(_Text(text, min(s["bbox"][0] for s in spans), min(s["bbox"][1] for s in spans),
                                 max(s["bbox"][2] for s in spans), max(s["bbox"][3] for s in spans),
                                 max(s["size"] for s in spans)))
    return out


def page_kind(page) -> str:
    """"vector" when the page's notation is music-font glyphs, "scan" when it's mostly an image, else "text"."""
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                fam = pdf_score._family(span["font"])
                if fam and any((pdf_score.symbol(fam, c["c"]) or ("",))[0] == "head" for c in span["chars"]):
                    return "vector"
    area = page.rect.width * page.rect.height
    for info in page.get_image_info():
        x0, y0, x1, y1 = info["bbox"]
        if (x1 - x0) * (y1 - y0) > 0.4 * area:
            return "scan"
    return "text"


def _first_staff_top(page) -> float:
    hlines = pdf_score._read_page(page).hlines
    staves = pdf_score._staves(hlines)
    return staves[0].top if staves else page.rect.height * 0.3


def _outline_title(page, top: float) -> bool:
    """A title drawn as outlines: filled shapes a title's height, spanning a good width, above the first staff."""
    boxes = [d["rect"] for d in page.get_drawings()
             if d.get("fill") is not None and d["rect"].y1 < top - 10 and 10 <= d["rect"].height <= 60
             and d["rect"].width < 80]
    if len(boxes) < 4:
        return False
    return max(b.x1 for b in boxes) - min(b.x0 for b in boxes) > 0.25 * page.rect.width


def _show(text: str) -> str:
    """'from Walt Disney's POCAHONTAS' -> 'Pocahontas', 'from "HERCULES"' -> 'Hercules'."""
    s = re.sub(r"^from\s+", "", text.strip(), flags=re.I).strip("\"“”'‘’ ")
    s = SHOW_OWNER.sub("", s)
    s = SHOW_PREFIX.sub("", s).strip("\"“”'‘’ ")
    return nice_case(s)


def _writer(texts: list[_Text], k: int) -> str:
    """The first name credited on line k ("By LAUFEY, SPENCER STEWART" -> "Laufey"); a line ending in "by" credits
    the next line."""
    m = CREDIT.match(texts[k].text)
    names = m.group(1).strip() if m else ""
    if not names and k + 1 < len(texts):
        names = texts[k + 1].text
    first = re.split(r",|\s+and\s+|\s+&\s+|/", names)[0].strip()
    return nice_case(first) if first else ""


def _page_info(page) -> dict:
    """What marks this page as a song's first: its title, footer, credits and copyright."""
    h = page.rect.height
    texts = sorted(_texts(page), key=lambda t: (t.y0, t.x0))
    info: dict = {"kind": page_kind(page), "title": "", "artist": "", "footer": None, "copyright": False,
                  "outline_title": False}
    sizes = sorted(t.size for t in texts)
    body = sizes[(len(sizes) - 1) // 2] if sizes else 10.0
    top = [t for t in texts if t.y0 < 0.25 * h]
    big = max(top, key=lambda t: t.size, default=None)
    if big is not None and big.size >= 16 and big.size >= 1.6 * body:
        same = [t for t in top if abs(t.size - big.size) < 0.5 and abs(t.y0 - big.y0) < big.size]
        info["title"] = nice_case(" ".join(t.text for t in sorted(same, key=lambda t: t.x0)))
        top = [t for t in top if t not in same]  # "FROM THE START" is a title, not a show
    for t in texts:
        if t.y0 > 0.85 * h:
            m = FOOTER.match(t.text)
            if m:
                info["footer"] = (m.group(1).strip(), int(m.group(2)), int(m.group(3)))
            if "©" in t.text or re.search(r"\bcopyright\b", t.text, re.I):
                info["copyright"] = True
    for t in top:
        if re.match(r"^from\s+\S", t.text, re.I) and not info["artist"]:
            info["artist"] = _show(t.text)
    if not info["artist"]:
        for k, t in enumerate(top):
            if CREDIT.match(t.text):
                info["artist"] = _writer(top, k)
                if info["artist"]:
                    break
    if not info["title"] and info["kind"] == "vector":
        info["outline_title"] = _outline_title(page, _first_staff_top(page))
    return info


def propose(path: str | Path) -> list[Song]:
    """The songs in a PDF: one per first page found, with the title and artist printed on it."""
    import pymupdf  # the [omr] extra

    with pymupdf.open(str(path)) as doc:
        infos = [_page_info(page) for page in doc]
    starts = []
    for i, info in enumerate(infos):
        footer = info["footer"]
        first = (i == 0 or info["title"] or info["outline_title"] or (footer and footer[2] == 1)
                 or (info["copyright"] and not (footer and footer[2] > 1)))
        if i == 0 or (first and info["kind"] != "text"):
            starts.append(i)
    songs = []
    for n, s in enumerate(starts):
        end = (starts[n + 1] if n + 1 < len(starts) else len(infos)) - 1
        info = infos[s]
        footer = info["footer"]
        title = (footer[0] if footer else "") or info["title"]
        if not title:  # a scan, or a title drawn as outlines: the file's name for the user to correct
            title = title_from_name(Path(path).stem) + (f" {n + 1}" if len(starts) > 1 else "")
        kinds = [infos[p]["kind"] for p in range(s, end + 1)]
        kind = "vector" if kinds.count("vector") >= kinds.count("scan") and "vector" in kinds else "scan"
        songs.append(Song(title, info["artist"], s + 1, end + 1, kind))
    return songs


def split(path: str | Path, first: int, last: int, out: str | Path) -> Path:
    """Pages first..last (1-based) of a PDF as a PDF of their own."""
    import pymupdf

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(str(path)) as doc, pymupdf.open() as part:
        part.insert_pdf(doc, from_page=first - 1, to_page=last - 1)
        part.save(str(out))
    return out


def tempo(score: pdf_score.PdfScore, title: str = "") -> tuple[str, float] | None:
    """The tempo printed over the first bar, as (its words, quarter notes per minute): a metronome number
    ("= ca. 132") wins over words ("Moderately fast"). The title and credits don't count."""
    if not score.bars:
        return None
    found = None
    for text in score.bars[0].words:
        if text.strip().lower() == title.strip().lower() or re.match(r"^from\s|.*\bby\b", text, re.I):
            continue
        m = METRONOME.search(text)
        if m and 30 <= int(m.group(1)) <= 300:
            return text, float(m.group(1))
        bpm = _tempo_word(text)
        if bpm and found is None:
            found = (text, bpm)
    return found


def jumps(score: pdf_score.PdfScore) -> tuple[dict[str, int], list[str]]:
    """The jumps printed in the PDF: {kind: PDF bar (1-based)} for segno, tocoda, dalsegno, dacapo, coda and fine,
    and what's missing for them to play (empty when the set is complete). A coda sign beside "To Coda" is the To
    Coda; one on its own, or under the heading "Coda", after the To Coda is the coda."""
    found: dict[str, int] = {}
    al_fine = al_coda = False
    codas = []
    for b in score.bars:
        for text in b.words:
            for kind, pattern in _JUMP_WORDS:
                if pattern.match(re.sub(r"^[\W_]+", "", text.strip())):
                    kind = {"fine_word": "fine"}.get(kind, kind)
                    if kind == "coda":
                        codas.append(b.number)
                    else:
                        found.setdefault(kind, b.number)
                    if kind in ("dalsegno", "dacapo"):
                        al_fine |= bool(re.search(r"\bfine\b", text, re.I))
                        al_coda |= bool(re.search(r"\bcoda\b", text, re.I))
        for sign in b.signs:
            if sign == "segno":
                found.setdefault("segno", b.number)
            else:
                codas.append(b.number)
    to_coda = found.get("tocoda")
    after = [n for n in sorted(set(codas)) if to_coda is None or n > to_coda]
    if to_coda is None and len(after) >= 2:  # two coda signs and no words: the first is the To Coda
        found["tocoda"], after = after[0], after[1:]
    if after and "tocoda" in found:
        found["coda"] = after[0]
    if "fine" in found and not al_fine:
        del found["fine"]  # "Fine" on its own may be a lyric; only a jump "al Fine" ends there
    missing = []
    jump = found.get("dalsegno") or found.get("dacapo")
    if "dalsegno" in found and "segno" not in found:
        missing.append("D.S. but no segno")
    if "tocoda" in found and "coda" not in found:
        missing.append("To Coda but no coda sign")
    if "coda" in found and "tocoda" not in found:
        missing.append("a coda but no To Coda")
    if "tocoda" in found and jump is None:
        missing.append("To Coda but no D.S. or D.C.")
    if al_coda and "tocoda" not in found:
        missing.append("al Coda but no To Coda")
    if al_fine and "fine" not in found:
        missing.append("al Fine but no Fine")
    return found, missing
