"""Command line entry point: `pianopipe <command> ...`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, engrave, grading, midi_io, omr_repair
from .catalog import Catalog, verify
from . import fingering_eval
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
    anchors = grading.load_anchors(args.anchors) if args.anchors else None
    result = build_catalog(
        args.src, args.out, args.name, sources, OFFICIAL if args.official else None, args.easiest_first, anchors,
        sheets=args.sheets,
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


def _engrave(args) -> int:
    stem = args.stem or Path(args.file).stem
    level = args.level or 4
    sheet = engrave.engrave(args.file, args.out, stem, level, fingering=False if args.no_fingering else None,
                            keep_musicxml=args.musicxml)
    print(f"wrote {sheet.json_path} ({sheet.pages} pages, {sheet.data['bars']} bars), sha256 {sheet.sha256}")
    return 0


def _sightread(args) -> int:
    from . import sightread

    if args.bank:
        grades = sorted({int(g) for g in args.grades.split(",")}) if args.grades else list(sightread.GRADES)
        cat = sightread.build_bank(args.out, args.count, grades, sheets=not args.no_sheets, jobs=args.jobs)
        print(f"wrote {len(cat.songs)} pieces to {Path(args.out) / 'catalog.json'}")
        print(sightread.summary(cat))
        problems = verify(Path(args.out) / "catalog.json")
        for p in problems:
            print(f"  INVALID {p}")
        return 1 if problems else 0
    if args.grade is None:
        print("sightread: give --grade (and --seed), or --bank", file=sys.stderr)
        return 2
    entry = sightread.build_piece(args.grade, args.seed, args.seed, args.out, sheets=not args.no_sheets)
    # As the catalog writes it (no empty optional fields).
    print(json.dumps(json.loads(Catalog("", [sightread._entry(entry)]).to_json())["songs"][0], indent=2, ensure_ascii=False))
    return 0


def _cards(args) -> int:
    from . import cards

    manifest = cards.build(args.out, passages=not args.no_passages)
    print(f"wrote {Path(args.out) / 'cards.json'}: {cards.summary(manifest)}")
    return 0


def _process(args) -> int:
    melody = args.melody if args.melody in (None, True) else [m for m in args.melody.split(",") if m.strip()]
    entry = process_file(args.file, args.out, {"melody": melody} if melody else None, sheets=args.sheets)
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
    """Score automatic fingering against the hand-written textbook cases in fingering_eval.py, and with a catalog's
    files, against its printed fingers (--printed) and standard practice (--lint)."""
    print(fingering_eval.report(fingering_eval.evaluate()))
    if args.printed or args.lint:
        if not args.raw or not args.sources:
            print("--printed and --lint need --raw and --sources")
            return 1
        fingers = Path(args.fingering) if args.fingering else None
        pieces = fingering_eval.printed_pieces(Path(args.raw), Path(args.sources), fingers, only_printed=not args.lint)
    if args.printed:
        printed = [p for p in pieces if any(n.finger in fingering_eval.FINGERS for n in p.notes)]
        print(f"\ngenerated against printed fingers ({len(printed)} pieces):")
        print(fingering_eval.printed_report(fingering_eval.evaluate_printed(printed), fingering_eval.wrong_hand_chords(printed)))
    if args.lint:
        print(f"\nstandard-practice checks ({len(pieces)} pieces{', every level' if args.levels else ''}):")
        print(fingering_eval.lint_report(fingering_eval.lint(pieces, args.levels)))
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


def _omr_repair(args) -> int:
    exports = omr_repair.export_paths(args.export, args.omr)
    report = omr_repair.repair(exports if len(exports) > 1 else exports[0], args.omr, args.out)
    print(report.summary())
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_json(), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


def _pdf_check(args) -> int:
    from . import pdf_check  # needs pymupdf: the [omr] extra

    report = pdf_check.check(args.pdf, args.musicxml, args.out, args.repair_report)
    print(report.summary())
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_json(), indent=1) + "\n", encoding="utf-8")
    if args.out:
        print(f"wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pianopipe", description="Quest Piano Trainer content pipeline")
    parser.add_argument("--version", action="version", version=f"pianopipe {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect", help="show tracks, channels and programs of a MIDI file")
    p.add_argument("file")
    p.set_defaults(func=_inspect)

    p = sub.add_parser("process", help="turn one MIDI file into app-ready level files")
    p.add_argument("file")
    p.add_argument("--out", default="out")
    p.add_argument("--sheets", action="store_true", help="also engrave sheet music for every level (pianopipe[sheet])")
    p.add_argument("--melody", nargs="?", const=True, metavar="PARTS",
                   help="MusicXML: merge the voice or melody parts (or these parts, by name or id, comma-separated) "
                   "into the right hand as its top voice")
    p.set_defaults(func=_process)

    p = sub.add_parser("engrave", help="sheet music pages and bar map for one app-ready level file (pianopipe[sheet])")
    p.add_argument("file")
    p.add_argument("--out", default="out")
    p.add_argument("--level", type=int, help="the level (1-4) the file is: levels 1-2 are one line per hand, 1-3 get "
                   "finger numbers (default 4)")
    p.add_argument("--stem", help="file name stem (default: the input's), giving <stem>.sheet.json and <stem>-p<N>.png")
    p.add_argument("--no-fingering", action="store_true")
    p.add_argument("--musicxml", action="store_true", help="also keep the notation as <stem>.musicxml")
    p.set_defaults(func=_engrave)

    p = sub.add_parser("sightread", help="generate sight-reading pieces at a grade (level file and sheet music), "
                                         "or the reading collection (--bank)")
    p.add_argument("--grade", type=int, help="0 (Initial) to 5")
    p.add_argument("--seed", type=int, default=1, help="the same grade and seed always give the same piece")
    p.add_argument("--out", default="out", help="writes <out>/songs/sr-g<grade>-<seed>/level1.mid, or with --bank "
                   "<out>/catalog.json and <out>/songs/")
    p.add_argument("--bank", action="store_true", help="the reading collection: --count pieces per grade")
    p.add_argument("--count", type=int, default=200, help="pieces per grade in the bank (default 200)")
    p.add_argument("--grades", help="with --bank, comma-separated grades (default all: 0,1,2,3,4,5)")
    p.add_argument("--jobs", type=int, help="with --bank, worker processes (default: CPUs - 1)")
    p.add_argument("--no-sheets", action="store_true", help="level files only (no sheet music)")
    p.set_defaults(func=_sightread)

    p = sub.add_parser("cards", help="the reading cards and read-ahead passages, in the app's layout "
                                     "(unity/QuestPianoTrainer/Assets/Resources/Reading)")
    p.add_argument("--out", required=True, help="writes <out>/cards.json, <out>/cards/, <out>/passages.json, <out>/passages/")
    p.add_argument("--no-passages", action="store_true", help="the cards only")
    p.set_defaults(func=_cards)

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
    p.add_argument("--anchors", help="grade anchors CSV (pipeline/data/grade_anchors.csv): their full levels get the reference grade")
    p.add_argument("--easiest-first", action="store_true", help="order songs by the full level's grade (easiest first) instead of source order")
    p.add_argument("--sheets", action="store_true", help="also engrave sheet music for every level (pianopipe[sheet])")
    p.set_defaults(func=_build)

    p = sub.add_parser("validate", help="check a catalog.json and that its level files match their sha256")
    p.add_argument("catalog")
    p.set_defaults(func=_validate)

    p = sub.add_parser("eval-hands", help="score automatic hand separation against hand-labeled files")
    p.add_argument("dir")
    p.set_defaults(func=_eval_hands)

    p = sub.add_parser("eval-fingering", help="score automatic fingering against textbook and printed fingerings")
    p.add_argument("--printed", action="store_true", help="also score it against the catalog's printed fingers")
    p.add_argument("--lint", action="store_true", help="also check every catalog piece against standard practice")
    p.add_argument("--levels", action="store_true", help="with --lint, check every level, not only the full piece")
    p.add_argument("--raw", help="folder the manifest's file paths are relative to (data/raw)")
    p.add_argument("--sources", help="the catalog manifest (data/catalog/sources.json)")
    p.add_argument("--fingering", help="Mutopia's printed fingers (data/catalog/fingering)")
    p.set_defaults(func=_eval_fingering)

    p = sub.add_parser("grade-fit", help="calibrate grades on the anchor pieces; writes the grade model and a fit report")
    p.add_argument("raw", help="folder the manifest's file paths are relative to (data/raw)")
    p.add_argument("--sources", required=True, help="the catalog manifest (data/catalog/sources.json)")
    p.add_argument("--anchors", required=True, help="the anchors CSV (pipeline/data/grade_anchors.csv)")
    p.add_argument("--report", required=True, help="where the fit report goes (pipeline/data/grade_fit.md)")
    p.add_argument("--model", default=str(grading.MODEL_JSON), help="where the model goes (default: the package's grade_model.json)")
    p.set_defaults(func=_grade_fit)

    p = sub.add_parser("omr-repair", help="re-time the piano part of an Audiveris export from its .omr project's geometry")
    p.add_argument("export", nargs="+", help="Audiveris's MusicXML export (.mxl); several movements (<song>.mvt1.mxl, "
                   ".mvt2.mxl) are joined in order; a glob, or a folder for the exports named after the project")
    p.add_argument("omr", help="the Audiveris project (.omr) it came from")
    p.add_argument("-o", "--out", required=True, help="where the repaired MusicXML goes (.musicxml)")
    p.add_argument("--json", help="also write the per-bar report as JSON (needs_review, reasons)")
    p.set_defaults(func=_omr_repair)

    p = sub.add_parser("pdf-check", help="check a MusicXML file bar by bar against the vector PDF it was read from, "
                                         "and rewrite the bars that differ from it")
    p.add_argument("pdf", help="the publisher's PDF (Finale, Sibelius or MuseScore export with real notation)")
    p.add_argument("musicxml", help="the MusicXML read from it (omr-repair's output)")
    p.add_argument("-o", "--out", help="write a copy with differing bars rewritten from the PDF (otherwise: audit only)")
    p.add_argument("--json", help="also write the per-bar report as JSON (needs_review, found, filled, reasons)")
    p.add_argument("--repair-report", help="omr-repair's JSON report: its needs_review bars that match the PDF are cleared")
    p.set_defaults(func=_pdf_check)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
