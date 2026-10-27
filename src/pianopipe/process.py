"""End-to-end processing: arbitrary MIDI in, app-ready level files and a catalog entry out."""

from __future__ import annotations

import json
import re
import unicodedata
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path

from . import engrave, features, grading, leadsheet, midi_io, musicxml
from .catalog import CHARACTERS, Catalog, Level, SheetRef, SongEntry, easiest_first_key, grade_tag, sha256_file
from .difficulty import LEVELS, difficulty_score, make_levels
from .fingering import assign_fingering
from .hands import split_hands
from .model import Song
from .quality import GatePolicy, QualityError, check
from .satb import reach
from .tracks import pick_piano

MAX_WARNINGS = 20


def slugify(text: str) -> str:
    """'Für Elise' -> 'fur-elise': accents folded, anything else between words becomes one dash."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-") or "song"


def tempo_map(song: Song):
    """seconds(tick): absolute ticks to seconds with the song's tempo map, by binary search over the cumulative time at
    each tempo change (a scan per call was half the build time of files with thousands of tempo events)."""
    ticks, secs, tempos = [0], [0.0], [500000]
    for t, msg in sorted(((t, m) for t, m in song.meta if m.type == "set_tempo"), key=lambda e: e[0]):
        # The same arithmetic, in the same order, as summing the spans one by one, so the result doesn't change.
        secs.append(secs[-1] + (t - ticks[-1]) * tempos[-1] / 1e6 / song.ticks_per_beat)
        ticks.append(t)
        tempos.append(msg.tempo)

    def seconds(tick: int) -> float:
        i = max(0, bisect_right(ticks, tick) - 1)
        return secs[i] + (tick - ticks[i]) * tempos[i] / 1e6 / song.ticks_per_beat

    return seconds


def seconds_at(song: Song, tick: int) -> float:
    """Convert an absolute tick to seconds using the song's tempo map (for one lookup; tempo_map for many)."""
    return tempo_map(song)(tick)


def load_notes(src: str | Path, satb: bool = False):
    """Read a MIDI or MusicXML file's piano part with hands: (song, notes, hand_source, seconds(tick)).

    MusicXML hands come from the staves (musicxml.py); MIDI hands from track names, tracks or channels (tracks.py);
    an ABC lead sheet (a house arrangement) has the melody in the right hand and a generated left hand (leadsheet.py).
    With `satb` (a four-part hymn or chorale on two staves), or when the file's parts are a choir, the SATB rule's
    reach step moves the tenor to the right hand where the left hand would stretch past an octave."""
    suffix = Path(src).suffix.lower()
    if suffix in musicxml.SUFFIXES:
        score = musicxml.load(src)
        song, notes, hand_source = score.song, score.notes, score.hand_source
    elif suffix in leadsheet.SUFFIXES:
        song, _ = leadsheet.load(src)
        notes, hand_source = song.notes, "leadsheet"
    else:
        song = midi_io.load(src)
        part = pick_piano(song)
        notes, hand_source = part.notes, part.hand_source
    if hand_source == "none":
        notes = split_hands(notes, song.ticks_per_beat)
        hand_source = "auto"
    elif satb or hand_source == "satb":
        notes = reach(notes, song.ticks_per_beat)
        hand_source = "satb"

    return song, notes, hand_source, tempo_map(song)


_DEFAULT = object()


def process_file(
    src: str | Path,
    out_dir: str | Path,
    meta: dict | None = None,
    base_url: str = "",
    policy: GatePolicy | None = None,
    fingering: bool = True,
    grade_model=_DEFAULT,
    anchors: dict | None = None,
    sheets: bool = False,
) -> SongEntry:
    """Process one MIDI file into out_dir/<id>/level<N>.mid and return its catalog entry.

    `meta` may hold id, title, full_title, composer, license, attribution, source_url, era, tags, character (which
    overrides the computed one) and satb (true for a four-part hymn setting: see load_notes).
    `base_url` is prepended to level file paths in the entry (relative to catalog.json).
    With a `policy`, the quality gate runs before anything is written and a failing song raises QualityError;
    without one, only its warnings are kept. With `fingering`, every level's notes get fingers (fingering.py).
    Every level gets a `grade` estimated by `grade_model` (default: the packaged grade_model.json; None for no grades),
    except the full level of a piece in `anchors` (id -> grading.Anchor, from grading.load_anchors; the official
    build passes pipeline/data/grade_anchors.csv), which gets its reference grade. A reduced level's grade is capped
    at a share of the full level's (grading.REDUCED_SHARE) and never above the next level's.
    With `sheets`, each level is also engraved from its level file (engrave.py, needs pianopipe[sheet]):
    <id>/level<N>.sheet.json and its page images, referenced by the level's `sheet`.
    """
    if sheets and not engrave.available():
        raise RuntimeError("sheet music needs the optional packages: pip install pianopipe[sheet]")
    src = Path(src)
    meta = dict(meta or {})
    song_id = meta.pop("id", None) or slugify(src.stem)
    song, notes, hand_source, seconds = load_notes(src, satb=bool(meta.pop("satb", False)))
    if not notes:
        raise QualityError(["no playable notes"])
    if grade_model is _DEFAULT:
        grade_model = grading.default_model()
    tpb = song.ticks_per_beat

    duration_s = round(seconds(max(n.end for n in notes)), 2)
    levels = make_levels(notes, song.ticks_per_beat, song.meta)
    gate = check(notes, levels, hand_source, duration_s, song.ticks_per_beat, seconds, policy or GatePolicy())
    if policy is not None and not gate.ok:
        raise QualityError(gate.reasons)

    difficulties = {lvl: difficulty_score(level_notes, seconds, song.ticks_per_beat) for lvl, level_notes in levels.items()}
    tags = list(meta.pop("tags", []))
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
        if sheets:
            sheet = engrave.engrave(path, song_dir, f"level{level}", level, entry.title)
            lv.sheet = SheetRef(f"{base_url}{song_id}/level{level}.sheet.json", sheet.sha256, sheet.pages)
        if anchors and song_id in anchors and level == max(levels):
            lv.grade, lv.grade_source = anchors[song_id].grade, "reference"
        elif grade_model is not None:
            feats = features.level_features(level_notes, seconds, tpb, song.meta)
            lv.grade, lv.grade_source = grade_model.predict(feats), "estimated"
        entry.levels.append(lv)
    # A reduced level is never graded above a share of the full level (grading.cap_reduced), nor above the level it
    # was reduced towards.
    full = entry.levels[-1]
    if full.grade is not None:
        for lv in entry.levels[:-1]:
            if lv.grade is not None:
                lv.grade = grading.cap_reduced(lv.level, lv.grade, full.grade)
    for lower, higher in zip(entry.levels[-2::-1], entry.levels[:0:-1]):
        if lower.grade is not None and higher.grade is not None:
            lower.grade = min(lower.grade, higher.grade)
    entry.tags.insert(0, grade_tag(full.grade, full.difficulty))
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
    anchors: dict | None = None,
    sheets: bool = False,
) -> BuildResult:
    """Process MIDI and MusicXML files into out_dir/songs/<id>/ with out_dir/catalog.json and out_dir/build-report.json.

    `sources` is either a dict mapping file names in src_dir to their metadata (every .mid/.midi in src_dir is
    built), or a manifest list of entries whose "file" is a path relative to src_dir (only those are built, in
    order). Files that fail the quality gate or can't be read are skipped and listed in the report, so one bad file
    doesn't stop a catalog build. With `easiest_first`, songs are sorted by the full level's grade (then difficulty
    and title, catalog.easiest_first_key), so a browser's first page holds the beginner pieces. `anchors` gives
    reference grades (see process_file); `sheets` engraves every level (see process_file).
    """
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(sources, list):
        jobs = [(src_dir / e["file"], e) for e in sources]
    else:
        files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in (".mid", ".midi") + musicxml.SUFFIXES + leadsheet.SUFFIXES)
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
                result.catalog.songs.append(process_file(path, out_dir / "songs", meta, "songs/", policy, anchors=anchors, sheets=sheets))
            except QualityError as exc:
                result.skipped.append((rel, str(exc)))
            except Exception as exc:  # noqa: BLE001 - keep building the rest of the catalog
                result.skipped.append((rel, f"error: {exc}"))
    if easiest_first:
        result.catalog.songs.sort(key=easiest_first_key)
    result.catalog.save(out_dir / "catalog.json")
    (out_dir / "build-report.json").write_text(
        json.dumps(result.report(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result
