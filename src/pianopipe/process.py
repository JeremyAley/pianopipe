"""End-to-end processing: arbitrary MIDI in, app-ready level files and a catalog entry out."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from . import features, grading, midi_io
from .catalog import CHARACTERS, Catalog, Level, SongEntry, sha256_file
from .difficulty import LEVELS, difficulty_score, make_levels
from .fingering import assign_fingering
from .hands import split_hands
from .model import Song
from .quality import GatePolicy, QualityError, check
from .tracks import pick_piano

MAX_WARNINGS = 20
# Grade tag from the full arrangement's difficulty.
GRADES = ((4.0, "beginner"), (6.5, "intermediate"), (10.0, "advanced"))


def slugify(text: str) -> str:
    """'Für Elise' -> 'fur-elise': accents folded, anything else between words becomes one dash."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-") or "song"


def seconds_at(song: Song, tick: int) -> float:
    """Convert an absolute tick to seconds using the song's tempo map."""
    tempo, last_tick, seconds = 500000, 0, 0.0
    for t, msg in song.meta:
        if msg.type != "set_tempo" or t >= tick:
            continue
        seconds += (t - last_tick) * tempo / 1e6 / song.ticks_per_beat
        tempo, last_tick = msg.tempo, t
    return seconds + (tick - last_tick) * tempo / 1e6 / song.ticks_per_beat


def grade(difficulty: float) -> str:
    return next(name for limit, name in GRADES if difficulty <= limit)


def load_notes(src: str | Path):
    """Read a MIDI file's piano part with hands: (song, notes, hand_source, seconds(tick))."""
    song = midi_io.load(src)
    part = pick_piano(song)
    notes = part.notes
    hand_source = part.hand_source
    if hand_source == "none":
        notes = split_hands(notes, song.ticks_per_beat)
        hand_source = "auto"

    def seconds(tick: int) -> float:
        return seconds_at(song, tick)

    return song, notes, hand_source, seconds


_DEFAULT = object()


def process_file(
    src: str | Path,
    out_dir: str | Path,
    meta: dict | None = None,
    base_url: str = "",
    policy: GatePolicy | None = None,
    fingering: bool = True,
    grade_model=_DEFAULT,
    anchors=_DEFAULT,
) -> SongEntry:
    """Process one MIDI file into out_dir/<id>/level<N>.mid and return its catalog entry.

    `meta` may hold id, title, full_title, composer, license, attribution, source_url, era, tags and character (which
    overrides the computed one).
    `base_url` is prepended to level file paths in the entry (relative to catalog.json).
    With a `policy`, the quality gate runs before anything is written and a failing song raises QualityError;
    without one, only its warnings are kept. With `fingering`, every level's notes get fingers (fingering.py).
    Every level gets a `grade` estimated by `grade_model` (default: data/grade_model.json; None for no grades),
    except the full level of a piece in `anchors` (default: data/grade_anchors.csv), which gets its reference grade;
    a level is never graded above the next one.
    """
    src = Path(src)
    meta = dict(meta or {})
    song_id = meta.pop("id", None) or slugify(src.stem)
    song, notes, hand_source, seconds = load_notes(src)
    if not notes:
        raise QualityError(["no playable notes"])
    if grade_model is _DEFAULT:
        grade_model = grading.default_model()
    if anchors is _DEFAULT:
        anchors = grading.default_anchors()
    tpb = song.ticks_per_beat

    duration_s = round(seconds(max(n.end for n in notes)), 2)
    levels = make_levels(notes, song.ticks_per_beat)
    gate = check(notes, levels, hand_source, duration_s, song.ticks_per_beat, seconds, policy or GatePolicy())
    if policy is not None and not gate.ok:
        raise QualityError(gate.reasons)

    difficulties = {lvl: difficulty_score(level_notes, seconds, song.ticks_per_beat) for lvl, level_notes in levels.items()}
    tags = list(meta.pop("tags", []))
    tags.insert(0, grade(difficulties[max(levels)]))
    warnings = gate.warnings
    if len(warnings) > MAX_WARNINGS:
        warnings = warnings[: MAX_WARNINGS - 1] + [f"and {len(warnings) - MAX_WARNINGS + 1} more"]
    char = meta.pop("character", None) or features.character(levels[max(levels)], seconds, tpb)
    if char not in CHARACTERS:
        raise ValueError(f"character {char!r} is not one of {', '.join(CHARACTERS)}")
    tempo = features.tempo_bpm(song.meta, max(n.end for n in notes), tpb)
    entry = SongEntry(
        id=song_id,
        title=meta.pop("title", None) or src.stem.replace("_", " "),
        hand_source=hand_source,
        duration_s=duration_s,
        warnings=warnings,
        tags=tags,
        character=char,
        **{k: v for k, v in meta.items() if k in ("full_title", "composer", "license", "attribution", "source_url", "era")},
    )
    song_dir = Path(out_dir) / song_id
    for level, level_notes in levels.items():
        if fingering:
            level_notes = assign_fingering(level_notes, song.ticks_per_beat)
        path = song_dir / f"level{level}.mid"
        midi_io.write_app_midi(path, song.ticks_per_beat, song.meta, level_notes)
        lv = Level(
            level, LEVELS[level], f"{base_url}{song_id}/level{level}.mid", sha256_file(path), len(level_notes),
            difficulties[level], fingering, tempo_bpm=tempo, skills=features.skills(level_notes, seconds, tpb, song.meta),
        )
        if anchors and song_id in anchors and level == max(levels):
            lv.grade, lv.grade_source = anchors[song_id].grade, "reference"
        elif grade_model is not None:
            feats = features.level_features(level_notes, seconds, tpb, song.meta)
            lv.grade, lv.grade_source = grade_model.predict(feats), "estimated"
        entry.levels.append(lv)
    # A reduced level is never graded above the level it was reduced towards.
    for lower, higher in zip(entry.levels[-2::-1], entry.levels[:0:-1]):
        if lower.grade is not None and higher.grade is not None:
            lower.grade = min(lower.grade, higher.grade)
    return entry


@dataclass
class BuildResult:
    catalog: Catalog
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (file, reason)

    def report(self) -> dict:
        return {
            "songs": len(self.catalog.songs),
            "levels": sum(len(s.levels) for s in self.catalog.songs),
            "built": [
                {
                    "id": s.id, "title": s.title, "hand_source": s.hand_source, "duration_s": s.duration_s,
                    "difficulty": [lv.difficulty for lv in s.levels], "grade": [lv.grade for lv in s.levels],
                    "grade_source": s.levels[-1].grade_source, "character": s.character, "tags": s.tags,
                    "warnings": len(s.warnings),
                }
                for s in self.catalog.songs
            ],
            "skipped": [{"file": f, "reason": r} for f, r in self.skipped],
        }


def build_catalog(
    src_dir: str | Path,
    out_dir: str | Path,
    name: str,
    sources: dict | list | None = None,
    policy: GatePolicy | None = None,
    easiest_first: bool = False,
) -> BuildResult:
    """Process MIDI files into out_dir/songs/<id>/ with out_dir/catalog.json and out_dir/build-report.json.

    `sources` is either a dict mapping file names in src_dir to their metadata (every .mid/.midi in src_dir is
    built), or a manifest list of entries whose "file" is a path relative to src_dir (only those are built, in
    order). Files that fail the quality gate or can't be read are skipped and listed in the report, so one bad file
    doesn't stop a catalog build. With `easiest_first`, songs are sorted by the full level's difficulty (then title),
    so a browser's first page holds the beginner pieces.
    """
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(sources, list):
        jobs = [(src_dir / e["file"], e) for e in sources]
    else:
        files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in (".mid", ".midi"))
        jobs = [(p, (sources or {}).get(p.name)) for p in files]
    policy = policy or GatePolicy()
    result = BuildResult(Catalog(name=name))
    for path, meta in jobs:
        rel = path.relative_to(src_dir).as_posix()
        meta = {k: v for k, v in (meta or {}).items() if k not in ("file", "download_url")}
        meta["id"] = meta.get("id") or slugify(path.stem)
        if meta["id"] in {s.id for s in result.catalog.songs}:
            result.skipped.append((rel, f"duplicate id {meta['id']}"))
        elif not path.exists():
            result.skipped.append((rel, "file missing (run the fetch step)"))
        else:
            try:
                result.catalog.songs.append(process_file(path, out_dir / "songs", meta, "songs/", policy))
            except QualityError as exc:
                result.skipped.append((rel, str(exc)))
            except Exception as exc:  # noqa: BLE001 - keep building the rest of the catalog
                result.skipped.append((rel, f"error: {exc}"))
    if easiest_first:
        result.catalog.songs.sort(key=lambda s: (s.levels[-1].difficulty or 0.0, s.title.lower(), s.id))
    result.catalog.save(out_dir / "catalog.json")
    (out_dir / "build-report.json").write_text(
        json.dumps(result.report(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result
