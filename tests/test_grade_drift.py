"""Drift check for the grade model: the anchors' full-level features are stored in tests/data/anchor_features.json.

The stored model must still grade them as it did when the fixture was written, and, where the anchors' raw files are
at hand (PIANOPIPE_RAW, or data/raw in the Quest Piano Trainer repo), today's feature code must still compute the
same features from them. A change that moves either beyond the tolerance needs a refit (`pianopipe grade-fit`) and a
new fixture:

    python tests/test_grade_drift.py <raw dir> <sources.json> <grade_anchors.csv>
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pianopipe import grading  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "data" / "anchor_features.json"
TOLERANCE = 0.2  # grades a prediction may move
FEATURE_TOLERANCE = 1e-6  # relative change of a recomputed feature


def _raw() -> Path | None:
    env = os.environ.get("PIANOPIPE_RAW")
    candidates = [Path(env)] if env else []
    candidates.append(Path(__file__).resolve().parents[2] / "data" / "raw")  # pipeline/ inside Quest Piano Trainer
    return next((p for p in candidates if p.is_dir()), None)


def test_stored_model_still_grades_the_anchors_as_before():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    model = grading.default_model()
    assert len(fixture["anchors"]) >= 26
    for a in fixture["anchors"]:
        assert abs(model.predict(a["features"]) - a["predicted"]) <= TOLERANCE, a["id"]
        assert abs(a["predicted"] - a["grade"]) < 2.0, a["id"]  # in-sample; the LOO report has the honest errors
    errors = [abs(model.predict(a["features"]) - a["grade"]) for a in fixture["anchors"]]
    assert sum(errors) / len(errors) <= fixture["mae"] + 0.05


def test_features_from_the_raw_files_have_not_drifted():
    raw = _raw()
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if raw is None or not all((raw / a["file"]).exists() for a in fixture["anchors"]):
        pytest.skip("the anchors' raw files aren't here (set PIANOPIPE_RAW to the catalog's data/raw)")
    anchors = {a["id"]: grading.Anchor(a["id"], a["title"], a["grade"], abrsm=a["grade"]) for a in fixture["anchors"]}
    ids, _, _, rows, missing = grading.anchor_rows(raw, fixture["anchors"], anchors)
    assert not missing
    model = grading.default_model()
    for i, row, a in zip(ids, rows, fixture["anchors"]):
        assert i == a["id"]
        for name, stored in a["features"].items():
            assert row[name] == pytest.approx(stored, rel=FEATURE_TOLERANCE, abs=1e-9), f"{i}: {name}"
        assert abs(model.predict(row) - a["predicted"]) <= TOLERANCE, i


def write_fixture(raw: Path, sources: Path, anchors_csv: Path) -> None:
    manifest = json.loads(sources.read_text(encoding="utf-8"))
    anchors = grading.load_anchors(anchors_csv)
    ids, titles, grades, rows, missing = grading.anchor_rows(raw, manifest, anchors)
    if missing:
        raise SystemExit(f"no file for {', '.join(missing)}")
    model = grading.default_model()
    files = {e["id"]: e["file"] for e in manifest}
    out = [
        {"id": i, "title": t, "file": files[i], "grade": g, "predicted": model.predict(r), "features": r}
        for i, t, g, r in zip(ids, titles, grades, rows)
    ]
    mae = round(sum(abs(a["predicted"] - a["grade"]) for a in out) / len(out), 3)
    FIXTURE.parent.mkdir(exist_ok=True)
    FIXTURE.write_text(json.dumps({"mae": mae, "anchors": out}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {FIXTURE} ({len(out)} anchors, in-sample MAE {mae})")


if __name__ == "__main__":
    write_fixture(*(Path(a) for a in sys.argv[1:4]))
