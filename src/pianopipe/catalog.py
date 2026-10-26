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
                for key in ("difficulty", "grade", "tempo_bpm"):
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
        songs = [
            SongEntry(**{**_known(SongEntry, s), "levels": [Level(**_known(Level, lv)) for lv in s["levels"]]})
            for s in data["songs"]
        ]
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
    return problems
