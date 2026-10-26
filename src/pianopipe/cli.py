"""Command line entry point: `pianopipe <command> ...`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import grading, midi_io
from .catalog import verify
from .fingering_eval import evaluate as evaluate_fingering
from .fingering_eval import report as fingering_report
from .hands import hand_accuracy, split_hands
from .process import build_catalog, process_file
from .quality import OFFICIAL
from .tracks import pick_piano


def _inspect(args) -> int:
    song = midi_io.load(args.file)
    print(f"ticks_per_beat={song.ticks_per_beat}")
    for t in song.tracks:
        chans = sorted({n.channel + 1 for n in t.notes})
        progs = sorted({n.program + 1 for n in t.notes})
        print(f"track {t.index:2d} {t.name!r:30} notes={len(t.notes):5d} channels={chans} programs={progs}")
    part = pick_piano(song)
    print(f"piano notes={len(part.notes)} hands from: {part.hand_source}")
    return 0


def _build(args) -> int:
    sources = json.loads(Path(args.sources).read_text(encoding="utf-8")) if args.sources else None
    result = build_catalog(
        args.src, args.out, args.name, sources, OFFICIAL if args.official else None, args.easiest_first
    )
    levels = sum(len(s.levels) for s in result.catalog.songs)
    print(f"wrote {len(result.catalog.songs)} songs ({levels} level files) to {Path(args.out) / 'catalog.json'}")
    for file, reason in result.skipped:
        print(f"  skipped {file}: {reason}")
    problems = verify(Path(args.out) / "catalog.json")
    for p in problems:
        print(f"  INVALID {p}")
    return 1 if problems else 0


def _validate(args) -> int:
    problems = verify(args.catalog)
    for p in problems:
        print(p)
    if not problems:
        data = json.loads(Path(args.catalog).read_text(encoding="utf-8"))
        print(f"ok: {len(data['songs'])} songs, {sum(len(s['levels']) for s in data['songs'])} level files match their sha256")
    return 1 if problems else 0


def _process(args) -> int:
    entry = process_file(args.file, args.out)
    print(json.dumps(entry.__dict__, default=lambda o: o.__dict__, indent=2))
    return 0


def _eval_hands(args) -> int:
    """Feasibility test 4: strip hands from labeled files, re-split, and score."""
    scores = []
    for path in sorted(Path(args.dir).glob("*.mid*")):
        part = pick_piano(midi_io.load(path))
        if part.hand_source == "none":
            print(f"{path.name:40} no hand labels, skipped")
            continue
        song = midi_io.load(path)
        predicted = split_hands([n.with_hand(None) for n in part.notes], song.ticks_per_beat)
        acc = hand_accuracy(predicted, part.notes)
        scores.append(acc)
        print(f"{path.name:40} {acc:6.1%}  ({len(part.notes)} notes, labels from {part.hand_source})")
    if scores:
        print(f"mean hand accuracy over {len(scores)} files: {sum(scores) / len(scores):.1%}")
    return 0


def _eval_fingering(args) -> int:
    """Score automatic fingering against the hand-written textbook cases in fingering_eval.py."""
    print(fingering_report(evaluate_fingering()))
    return 0


def _grade_fit(args) -> int:
    """Fit the grade model on the anchors and write the model and the report."""
    manifest = json.loads(Path(args.sources).read_text(encoding="utf-8"))
    anchors = grading.load_anchors(args.anchors)
    ids, titles, grades, rows, missing = grading.anchor_rows(args.raw, manifest, anchors)
    if len(ids) < 5:
        print(f"only {len(ids)} anchors have files under {args.raw}; need 5 or more")
        return 1
    res = grading.fit(ids, titles, grades, rows)
    res.model.save(args.model)
    rcm_only = sum(anchors[i].abrsm is None and anchors[i].trinity is None for i in ids)
    Path(args.report).write_text(grading.report(res, missing, rcm_only), encoding="utf-8")
    print(grading.summary(res.model))
    for name, step in (("step 1", res.step1), ("step 2", res.step2)):
        for m in step["misplaced"]:
            print(f"  {name} misplaced: {m['title']}: reference {m['reference']:g}, predicted {m['predicted']}")
    if missing:
        print(f"  no file for: {', '.join(missing)}")
    print(f"wrote {args.model} and {args.report}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pianopipe", description="Quest Piano Trainer content pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect", help="show tracks, channels and programs of a MIDI file")
    p.add_argument("file")
    p.set_defaults(func=_inspect)

    p = sub.add_parser("process", help="turn one MIDI file into app-ready level files")
    p.add_argument("file")
    p.add_argument("--out", default="out")
    p.set_defaults(func=_process)

    p = sub.add_parser("build", help="build a catalog.json from a folder of MIDI files")
    p.add_argument("src")
    p.add_argument("out")
    p.add_argument("--name", default="My songs")
    p.add_argument(
        "--sources",
        help="JSON file: a dict mapping file names to title/composer/license/attribution/source_url, "
        "or a manifest list of entries with a \"file\" path relative to src",
    )
    p.add_argument("--official", action="store_true", help="official catalog gate: hands must come from the file's structure")
    p.add_argument("--easiest-first", action="store_true", help="order songs by difficulty instead of source order")
    p.set_defaults(func=_build)

    p = sub.add_parser("validate", help="check a catalog.json and that its level files match their sha256")
    p.add_argument("catalog")
    p.set_defaults(func=_validate)

    p = sub.add_parser("eval-hands", help="score automatic hand separation against hand-labeled files")
    p.add_argument("dir")
    p.set_defaults(func=_eval_hands)

    p = sub.add_parser("eval-fingering", help="score automatic fingering against hand-written textbook fingerings")
    p.set_defaults(func=_eval_fingering)

    p = sub.add_parser("grade-fit", help="calibrate grades on the anchor pieces; writes data/grade_model.json and grade_fit.md")
    p.add_argument("raw", help="folder the manifest's file paths are relative to (data/raw)")
    p.add_argument("--sources", default=str(grading.DATA.parents[1] / "data" / "catalog" / "sources.json"))
    p.add_argument("--anchors", default=str(grading.ANCHORS_CSV))
    p.add_argument("--model", default=str(grading.MODEL_JSON))
    p.add_argument("--report", default=str(grading.REPORT_MD))
    p.set_defaults(func=_grade_fit)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
