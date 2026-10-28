"""The catalog.json format shared by the official catalog and personal sources.

See docs/formats.md for the full description. URLs are relative to the
catalog.json file unless they are absolute. Fields marked optional may be
missing; readers ignore fields they don't know.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

CATALOG_VERSION = 1
GRADE_SOURCES = ("reference", "estimated")
CHARACTERS = ("agile", "lyrical", "other")
# The song's first tag, from its full level's grade: the exam boards' own three tiers (ABRSM and Trinity put grades 1-3
# at RQF level 1, 4-5 at level 2 and 6-8 at level 3; docs/learning-mode.md 1.1). A grade below the limit gets the tag.
GRADE_TAGS = ((3.5, "beginner"), (5.5, "intermediate"), (float("inf"), "advanced"))
# Without a grade (no grade model), the full level's difficulty decides, with these limits (up to and including).
DIFFICULTY_TAGS = ((4.0, "beginner"), (6.5, "intermediate"), (float("inf"), "advanced"))


def grade_tag(grade: float | None, difficulty: float | None = None) -> str:
    """beginner, intermediate or advanced for a song whose full level has this grade (or, without one, difficulty)."""
    if grade is not None:
        return next(name for limit, name in GRADE_TAGS if grade < limit)
    return next(name for limit, name in DIFFICULTY_TAGS if (difficulty or 0.0) <= limit)


def easiest_first_key(song: "SongEntry") -> tuple:
    """Sort key for an easiest-first catalog: the full level's grade, then its difficulty, then title and id. Songs
    without a grade go after the graded ones."""
    full = song.levels[-1]
    return (full.grade is None, full.grade or 0.0, full.difficulty or 0.0, song.title.lower(), song.id)


MAX_SHEET_PAGES = 64


@dataclass
class SheetRef:
    """A level's sheet music: its sheet.json (bar map and page list, docs/formats.md) and how many pages it has."""
    url: str
    sha256: str
    pages: int


@dataclass
class Level:
    level: int
    label: str
    url: str
    sha256: str
    notes: int
    difficulty: float | None = None  # optional: 1 (easiest) to 10, see difficulty.difficulty_score
    fingering: bool = False  # optional: every note-on has a finger text event (written only when true)
    grade: float | None = None  # optional: 0 (Initial) to 10 on our grade scale, one decimal, see grading.py
    grade_source: str = ""  # optional: "reference" (documented grade) or "estimated" (from the notes)
    tempo_bpm: int | None = None  # optional: target tempo in quarter notes per minute
    skills: list[str] = field(default_factory=list)  # optional: tags the level practises, see features.skills
    sheet: SheetRef | None = None  # optional: engraved notation of this level (engrave.py)


@dataclass
class SongEntry:
    id: str
    title: str  # short display title (the app shows about 38 characters)
    full_title: str = ""  # optional: the complete title when `title` is shortened
    composer: str = ""
    license: str = ""
    attribution: str = ""
    source_url: str = ""
    hand_source: str = ""  # how hands were decided: track_names, two_tracks, two_channels, staves, satb, leadsheet, auto
    duration_s: float = 0.0
    warnings: list[str] = field(default_factory=list)
    levels: list[Level] = field(default_factory=list)
    era: str = ""  # optional: Baroque, Classical, Romantic, Impressionist, Modern
    tags: list[str] = field(default_factory=list)  # optional: grade (beginner/intermediate/advanced), form, source
    character: str = ""  # optional: agile, lyrical or other


def _known(cls, data: dict) -> dict:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in names}


@dataclass
class Catalog:
    name: str
    songs: list[SongEntry] = field(default_factory=list)
    version: int = CATALOG_VERSION

    def to_json(self) -> str:
        data = asdict(self)
        for song in data["songs"]:
            for key in ("full_title", "character"):
                if not song[key]:
                    del song[key]
            for lv in song["levels"]:
                for key in ("difficulty", "grade", "tempo_bpm", "sheet"):
                    if lv[key] is None:
                        del lv[key]
                for key in ("fingering", "grade_source", "skills"):
                    if not lv[key]:
                        del lv[key]
        return json.dumps(data, indent=2, ensure_ascii=False) + "\n"

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Catalog":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        validate(data)
        def level(lv: dict) -> Level:
            known = _known(Level, lv)
            if known.get("sheet") is not None:
                known["sheet"] = SheetRef(**_known(SheetRef, known["sheet"]))
            return Level(**known)

        songs = [SongEntry(**{**_known(SongEntry, s), "levels": [level(lv) for lv in s["levels"]]}) for s in data["songs"]]
        return cls(name=data["name"], songs=songs, version=data["version"])


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def validate(data: dict) -> None:
    """Raise ValueError if a parsed catalog.json does not match the format."""
    if data.get("version") != CATALOG_VERSION:
        raise ValueError(f"unsupported catalog version {data.get('version')!r}")
    if not isinstance(data.get("name"), str):
        raise ValueError("catalog needs a name")
    if not isinstance(data.get("songs", []), list):
        raise ValueError("songs must be a list")
    ids = set()
    for s in data.get("songs", []):
        for key in ("id", "title", "levels"):
            if key not in s:
                raise ValueError(f"song missing {key!r}: {s.get('id')}")
        if not isinstance(s["id"], str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", s["id"]):
            raise ValueError(f"bad song id {s['id']!r} (lowercase letters, digits and dashes)")
        if s["id"] in ids:
            raise ValueError(f"duplicate song id {s['id']}")
        ids.add(s["id"])
        if not s["levels"]:
            raise ValueError(f"song {s['id']} has no levels")
        if not isinstance(s.get("duration_s", 0), (int, float)) or s.get("duration_s", 0) < 0:
            raise ValueError(f"song {s['id']} has a bad duration_s")
        if not isinstance(s.get("full_title", ""), str):
            raise ValueError(f"song {s['id']} full_title must be a string")
        for key in ("warnings", "tags"):
            if not all(isinstance(w, str) for w in s.get(key, [])):
                raise ValueError(f"song {s['id']} {key} must be strings")
        if s.get("character", "agile") not in CHARACTERS:
            raise ValueError(f"song {s['id']} character {s['character']!r} is not one of {', '.join(CHARACTERS)}")
        levels = set()
        for lv in s["levels"]:
            for key in ("level", "url", "sha256"):
                if key not in lv:
                    raise ValueError(f"song {s['id']} level missing {key!r}")
            if lv["level"] in levels:
                raise ValueError(f"song {s['id']} has level {lv['level']} twice")
            levels.add(lv["level"])
            if not _SHA256.match(str(lv["sha256"])):
                raise ValueError(f"song {s['id']} level {lv['level']} sha256 is not 64 lowercase hex digits")
            if not url_allowed(lv["url"]):
                raise ValueError(f"song {s['id']} level {lv['level']} url must be relative to catalog.json or https")
            diff = lv.get("difficulty")
            if diff is not None and not (isinstance(diff, (int, float)) and 1 <= diff <= 10):
                raise ValueError(f"song {s['id']} level {lv['level']} difficulty {diff!r} is outside 1-10")
            if not isinstance(lv.get("fingering", False), bool):
                raise ValueError(f"song {s['id']} level {lv['level']} fingering must be true or false")
            where = f"song {s['id']} level {lv['level']}"
            grade = lv.get("grade")
            if grade is not None and not (_number(grade) and 0 <= grade <= 10):
                raise ValueError(f"{where} grade {grade!r} is outside 0-10")
            if lv.get("grade_source", "estimated") not in GRADE_SOURCES:
                raise ValueError(f"{where} grade_source {lv['grade_source']!r} is not reference or estimated")
            if "grade_source" in lv and grade is None:
                raise ValueError(f"{where} has a grade_source but no grade")
            bpm = lv.get("tempo_bpm")
            if bpm is not None and not (_number(bpm) and bpm > 0):
                raise ValueError(f"{where} tempo_bpm {bpm!r} must be a positive number")
            skills = lv.get("skills", [])
            if not isinstance(skills, list) or not all(isinstance(k, str) for k in skills):
                raise ValueError(f"{where} skills must be a list of strings")
            sheet = lv.get("sheet")
            if sheet is not None:
                if not isinstance(sheet, dict) or not url_allowed(sheet.get("url")):
                    raise ValueError(f"{where} sheet needs a url relative to catalog.json or https")
                if not _SHA256.match(str(sheet.get("sha256"))):
                    raise ValueError(f"{where} sheet sha256 is not 64 lowercase hex digits")
                pages = sheet.get("pages")
                if not (isinstance(pages, int) and not isinstance(pages, bool) and 1 <= pages <= MAX_SHEET_PAGES):
                    raise ValueError(f"{where} sheet pages {pages!r} is outside 1-{MAX_SHEET_PAGES}")


def url_allowed(url) -> bool:
    """A level url the app downloads: relative to catalog.json ("songs/x/level1.mid") or https. No http, file or
    other schemes, and no host-relative "//host/..." (the app's CatalogFile.UrlAllowed, keep them together)."""
    if not isinstance(url, str) or not url or url.startswith(("//", "\\")):
        return False
    if ":" in url:  # a scheme (or a drive letter)
        return url.lower().startswith("https://")
    return True


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def verify(path: str | Path) -> list[str]:
    """Check a catalog.json on disk: valid format, and every relative level file exists with its sha256.

    Returns a list of problems (empty when everything matches).
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        validate(data)
    except ValueError as exc:
        return [str(exc)]
    problems = []
    for s in data["songs"]:
        for lv in s["levels"]:
            if "://" in lv["url"]:
                continue
            file = path.parent / lv["url"]
            if not file.exists():
                problems.append(f"{s['id']} level {lv['level']}: {lv['url']} is missing")
            elif sha256_file(file) != lv["sha256"]:
                problems.append(f"{s['id']} level {lv['level']}: sha256 does not match {lv['url']}")
            sheet = lv.get("sheet")
            if sheet and "://" not in sheet["url"]:
                problems += [f"{s['id']} level {lv['level']}: {p}" for p in verify_sheet(path.parent / sheet["url"], sheet)]
    return problems


PAGE_NAME = re.compile(r"[A-Za-z0-9._-]+\.png")


def verify_sheet(path: Path, ref: dict) -> list[str]:
    """Check a sheet.json against its catalog reference, and its pages against their sha256 (docs/formats.md)."""
    if not path.exists():
        return [f"{ref['url']} is missing"]
    if sha256_file(path) != ref["sha256"]:
        return [f"sha256 does not match {ref['url']}"]
    data = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    version = data.get("version")
    if version not in (1, 2):
        problems.append(f"{ref['url']}: unsupported version {version!r}")
    pages = data.get("pages", [])
    if len(pages) != ref["pages"]:
        problems.append(f"{ref['url']}: {len(pages)} pages, the catalog says {ref['pages']}")
    for page in pages:
        name = str(page.get("file", ""))
        if not PAGE_NAME.fullmatch(name):
            problems.append(f"{ref['url']}: page file {name!r} must be a .png name next to the sheet.json")
            continue
        file = path.parent / name
        if not file.exists():
            problems.append(f"{ref['url']}: {name} is missing")
        elif sha256_file(file) != page.get("sha256"):
            problems.append(f"{ref['url']}: sha256 does not match {name}")
    systems = len(data.get("systems", []))
    for i, m in enumerate(data.get("measures", [])):
        if m.get("bar") != i + 1 or not 0 <= m.get("system", -1) < systems:
            problems.append(f"{ref['url']}: measure {i + 1} is out of order or names a missing system")
            break
        if len(m.get("onsets", [])) % 2 or len(m.get("notes", [])) % 4:
            problems.append(f"{ref['url']}: measure {i + 1}: onsets come in pairs, notes in fours")
            break
    if version == 2:
        # Repeats written once: each of the app's bars 1..bars is played by exactly one measure.
        played = sorted(b for m in data.get("measures", []) for b in m.get("plays", []))
        if played != list(range(1, int(data.get("bars", 0)) + 1)):
            problems.append(f"{ref['url']}: the measures' plays don't cover bars 1-{data.get('bars')} once each")
    return problems
