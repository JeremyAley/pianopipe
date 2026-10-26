"""Grades on our 0-10 scale (0 Initial, 1-8 grades, 9-10 diploma), calibrated on pieces with documented exam grades.

data/grade_anchors.csv holds the anchors. `pianopipe grade-fit` measures how well the full level's `difficulty`
alone predicts their grades (step 1: isotonic regression), then fits a small ridge model on cheap features
(step 2, features.py), and writes the package's grade_model.json and data/grade_fit.md. Catalog builds only apply the
stored model, so they stay deterministic. Pure Python (no numpy): 26 anchors and a handful of features.
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .features import FEATURES

DATA = Path(__file__).resolve().parents[2] / "data"
ANCHORS_CSV = DATA / "grade_anchors.csv"
MODEL_JSON = Path(__file__).with_name("grade_model.json")  # inside the package, so an installed pianopipe has it
REPORT_MD = DATA / "grade_fit.md"
RCM_TO_GRADE = {1: 1, 2: 1.5, 3: 2, 4: 3, 5: 4, 6: 4.5, 7: 5, 8: 6, 9: 7, 10: 8}
RIDGE_GRID = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
MAX_FEATURES = 4
MIN_GAIN = 0.03  # LOO MAE (grades) a feature must save to be added
MISPLACED = 1.5  # |predicted - reference| that counts as misplaced
# Length stops counting past 1,500 notes, and every feature is clipped to the anchors' range (GradeModel.ranges).
# Without them, pieces longer or more chromatic than every anchor (Schubert's impromptus, the Ballade) were
# extrapolated to 10, melody level included. Together they cost 0.07 grades of LOO MAE (0.59 -> 0.66; nested 0.65 -> 0.73); see
# data/grade_fit.md, Limitations, for the alternatives tried.
CAPS = {"log_notes": math.log(1500)}


def capped(feats: dict[str, float], caps: dict[str, float] = CAPS) -> dict[str, float]:
    return {k: min(v, caps[k]) if k in caps else v for k, v in feats.items()}


# ---- anchors ----

@dataclass
class Anchor:
    id: str
    title: str
    grade: float
    abrsm: float | None = None
    trinity: float | None = None
    rcm: int | None = None
    henle: int | None = None
    sources: str = ""
    notes: str = ""


def anchor_grade(abrsm: float | None, trinity: float | None, rcm: int | None) -> float | None:
    """Mean of the ABRSM and Trinity grades if either exists, else the RCM level through our concordance."""
    boards = [g for g in (abrsm, trinity) if g is not None]
    if boards:
        return sum(boards) / len(boards)
    return RCM_TO_GRADE.get(rcm) if rcm is not None else None


def load_anchors(path: str | Path = ANCHORS_CSV) -> dict[str, Anchor]:
    text = Path(path).read_text(encoding="utf-8")
    rows = csv.DictReader(io.StringIO("".join(l for l in text.splitlines(True) if not l.startswith("#"))))
    num = lambda v, t=float: t(v) if v.strip() else None  # noqa: E731
    out = {}
    for r in rows:
        a = Anchor(
            r["id"], r["title"], float(r["grade"]), num(r["abrsm"]), num(r["trinity"]), num(r["rcm"], int),
            num(r["henle"], int), r["sources"], r["notes"],
        )
        if a.id in out:
            raise ValueError(f"anchor {a.id} listed twice")
        if anchor_grade(a.abrsm, a.trinity, a.rcm) != a.grade:
            raise ValueError(f"anchor {a.id}: grade {a.grade} doesn't follow the rule")
        out[a.id] = a
    return out


# ---- statistics ----

def ranks(xs: list[float]) -> list[float]:
    """1-based ranks, ties get their average rank."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(a: list[float], b: list[float]) -> float:
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else 0.0


def spearman(a: list[float], b: list[float]) -> float:
    return pearson(ranks(a), ranks(b))


def isotonic_fit(xs: list[float], ys: list[float]) -> list[tuple[float, float]]:
    """Non-decreasing fit by pool adjacent violators; returns knots (block mean x, block mean y)."""
    blocks: list[list[float]] = []  # [sum x, sum y, weight]
    for x, y in sorted(zip(xs, ys)):
        if blocks and blocks[-1][0] / blocks[-1][2] == x:  # tied x: one point
            blocks[-1][0] += x
            blocks[-1][1] += y
            blocks[-1][2] += 1
        else:
            blocks.append([x, y, 1])
        while len(blocks) > 1 and blocks[-2][1] / blocks[-2][2] >= blocks[-1][1] / blocks[-1][2]:
            sx, sy, w = blocks.pop()
            blocks[-1][0] += sx
            blocks[-1][1] += sy
            blocks[-1][2] += w
    return [(sx / w, sy / w) for sx, sy, w in blocks]


def isotonic_predict(knots: list[tuple[float, float]], x: float) -> float:
    """Linear between knots, flat beyond the ends."""
    if x <= knots[0][0]:
        return knots[0][1]
    for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return knots[-1][1]


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting (small symmetric positive definite systems)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        for r in range(c + 1, n):
            f = m[r][c] / m[c][c]
            for k in range(c, n + 1):
                m[r][k] -= f * m[c][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (m[r][n] - sum(m[r][k] * x[k] for k in range(r + 1, n))) / m[r][r]
    return x


@dataclass
class GradeModel:
    """grade = intercept + sum(coef * (feature - mean) / std), clamped to 0-10 and rounded to one decimal."""

    features: list[str]
    mean: list[float]
    std: list[float]
    coef: list[float]
    intercept: float
    ridge: float
    fit: dict = field(default_factory=dict)  # summary of the fit report, for the build's printout
    caps: dict = field(default_factory=dict)  # feature -> largest value that counts (see CAPS)
    ranges: dict = field(default_factory=dict)  # feature -> [min, max] seen in the anchors; values are clipped to it

    def raw(self, feats: dict[str, float]) -> float:
        feats = capped(feats, self.caps)
        feats = {k: min(max(v, self.ranges[k][0]), self.ranges[k][1]) if k in self.ranges else v for k, v in feats.items()}
        return self.intercept + sum(
            c * (feats[f] - m) / s for f, m, s, c in zip(self.features, self.mean, self.std, self.coef)
        )

    def predict(self, feats: dict[str, float]) -> float:
        return round(min(10.0, max(0.0, self.raw(feats))), 1)

    def save(self, path: str | Path = MODEL_JSON) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path = MODEL_JSON) -> "GradeModel":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def ridge_fit(rows: list[dict[str, float]], ys: list[float], names: list[str], lam: float) -> GradeModel:
    """Ridge regression on standardised features; the intercept (mean grade) isn't penalised."""
    n = len(rows)
    mean = [sum(r[f] for r in rows) / n for f in names]
    std = [math.sqrt(sum((r[f] - m) ** 2 for r in rows) / n) or 1.0 for f, m in zip(names, mean)]
    x = [[(r[f] - m) / s for f, m, s in zip(names, mean, std)] for r in rows]
    my = sum(ys) / n
    a = [[sum(row[i] * row[j] for row in x) + (lam if i == j else 0.0) for j in range(len(names))] for i in range(len(names))]
    b = [sum(row[i] * (y - my) for row, y in zip(x, ys)) for i in range(len(names))]
    coef = _solve(a, b) if names else []
    # A piece outside the anchors' range on some feature is graded as if at the edge of that range: the anchors say
    # nothing about what lies beyond, and extrapolating gave long or very chromatic pieces 10s.
    ranges = {f: [min(r[f] for r in rows), max(r[f] for r in rows)] for f in names}
    return GradeModel(list(names), mean, std, coef, my, lam, ranges=ranges)


def _loo_ridge(rows, ys, names, lam) -> list[float]:
    preds = []
    for i in range(len(rows)):
        m = ridge_fit(rows[:i] + rows[i + 1:], ys[:i] + ys[i + 1:], names, lam)
        preds.append(m.raw(rows[i]))
    return preds


def mae(pred: list[float], ys: list[float]) -> float:
    return sum(abs(p - y) for p, y in zip(pred, ys)) / len(ys)


def _clamp(p: float) -> float:
    return min(10.0, max(0.0, p))


def best_ridge(rows, ys, names) -> tuple[float, float]:
    """(LOO MAE, ridge strength) of the strength with the lowest LOO MAE."""
    return min((mae([_clamp(p) for p in _loo_ridge(rows, ys, names, lam)], ys), lam) for lam in RIDGE_GRID)


def _all_positive(rows, ys, names, lam) -> bool:
    return all(c > 0 for c in ridge_fit(rows, ys, names, lam).coef)


def select_features(rows, ys, candidates=FEATURES) -> tuple[list[str], float, float]:
    """Forward selection by LOO MAE: add the feature that helps most while it saves MIN_GAIN, up to MAX_FEATURES.

    Every feature measures something that makes a piece harder, so a set where one gets a negative coefficient
    (a collinearity artefact that would make, say, a faster version rate lower) isn't considered.
    Returns (features, ridge strength, LOO MAE).
    """
    chosen: list[str] = []
    score, lam = best_ridge(rows, ys, chosen)
    while len(chosen) < MAX_FEATURES:
        trials = [(best_ridge(rows, ys, chosen + [f]), f) for f in candidates if f not in chosen]
        trials = [((s, l), f) for (s, l), f in trials if _all_positive(rows, ys, chosen + [f], l)]
        if not trials:
            break
        (s, l), f = min(trials)
        if score - s < MIN_GAIN:
            break
        chosen, score, lam = chosen + [f], s, l
    return chosen, lam, score


# ---- the fit ----

@dataclass
class FitResult:
    ids: list[str]
    titles: list[str]
    grades: list[float]
    rows: list[dict[str, float]]
    step1: dict
    step2: dict
    model: GradeModel


def _misplaced(ids, titles, grades, preds) -> list[dict]:
    out = [
        {"id": i, "title": t, "reference": g, "predicted": round(p, 1)}
        for i, t, g, p in zip(ids, titles, grades, preds)
        if abs(p - g) >= MISPLACED
    ]
    return sorted(out, key=lambda m: -abs(m["predicted"] - m["reference"]))


def fit(ids: list[str], titles: list[str], grades: list[float], rows: list[dict[str, float]]) -> FitResult:
    """Step 1 (isotonic difficulty -> grade) and step 2 (ridge on selected features), both scored leave-one-out.
    Features are capped (CAPS) before step 2, and the model applies the same caps."""
    n = len(ids)
    rows = [capped(r) for r in rows]
    diff = [r["difficulty"] for r in rows]
    iso_loo = [
        isotonic_predict(isotonic_fit(diff[:i] + diff[i + 1:], grades[:i] + grades[i + 1:]), diff[i]) for i in range(n)
    ]
    step1 = {
        "spearman_difficulty": round(spearman(diff, grades), 3),
        "loo_spearman": round(spearman(iso_loo, grades), 3),
        "loo_mae": round(mae(iso_loo, grades), 2),
        "knots": [(round(x, 2), round(y, 2)) for x, y in isotonic_fit(diff, grades)],
        "loo_pred": [round(p, 2) for p in iso_loo],
        "misplaced": _misplaced(ids, titles, grades, iso_loo),
    }

    names, lam, _ = select_features(rows, grades)
    loo = [_clamp(p) for p in _loo_ridge(rows, grades, names, lam)]
    # Nested LOO: the feature selection and ridge strength are chosen again without the held-out piece, so this
    # number doesn't benefit from having picked the features on the same 26 pieces.
    nested = []
    for i in range(n):
        r, g = rows[:i] + rows[i + 1:], grades[:i] + grades[i + 1:]
        f, l, _ = select_features(r, g)
        nested.append(_clamp(ridge_fit(r, g, f, l).raw(rows[i])))
    model = ridge_fit(rows, grades, names, lam)
    model.caps = {k: v for k, v in CAPS.items() if k in names}
    single ={f: round(spearman([r[f] for r in rows], grades), 3) for f in FEATURES}
    step2 = {
        "features": names,
        "ridge": lam,
        "loo_spearman": round(spearman(loo, grades), 3),
        "loo_mae": round(mae(loo, grades), 2),
        "nested_loo_spearman": round(spearman(nested, grades), 3),
        "nested_loo_mae": round(mae(nested, grades), 2),
        "feature_spearman": single,
        "loo_pred": [round(p, 2) for p in loo],
        "misplaced": _misplaced(ids, titles, grades, loo),
    }
    model.fit = {
        "anchors": n,
        "step1": {k: step1[k] for k in ("spearman_difficulty", "loo_spearman", "loo_mae")},
        "step2": {k: step2[k] for k in ("loo_spearman", "loo_mae", "nested_loo_spearman", "nested_loo_mae")},
    }
    return FitResult(ids, titles, grades, rows, step1, step2, model)


def anchor_rows(raw: str | Path, manifest: list[dict], anchors: dict[str, Anchor]):
    """Features of every anchor's full level, read from the raw MIDI files the manifest points at.

    Returns (ids, titles, grades, rows, missing ids)."""
    from .features import level_features
    from .process import load_notes  # here to avoid an import cycle

    by_id = {e.get("id"): e for e in manifest}
    ids, titles, grades, rows, missing = [], [], [], [], []
    for a in anchors.values():
        entry = by_id.get(a.id)
        path = Path(raw) / entry["file"] if entry else None
        if path is None or not path.exists():
            missing.append(a.id)
            continue
        song, notes, _, seconds = load_notes(path)
        ids.append(a.id)
        titles.append(a.title)
        grades.append(a.grade)
        rows.append(level_features(notes, seconds, song.ticks_per_beat, song.meta))
    return ids, titles, grades, rows, missing


def report(res: FitResult, missing: list[str] = (), rcm_only: int | None = None) -> str:
    """The human-readable fit report (data/grade_fit.md). `rcm_only`: anchors graded through the RCM concordance."""
    s1, s2, m = res.step1, res.step2, res.model
    lines = [
        "# Grade fit",
        "",
        "Generated by `pianopipe grade-fit` (pipeline/src/pianopipe/grading.py); don't edit by hand.",
        f"Anchors: {len(res.ids)} catalog pieces with documented grades (pipeline/data/grade_anchors.csv), each "
        "scored on its full level. Our scale: 0 Initial, 1-8 grades, 9-10 diploma.",
        "All errors are leave-one-out (LOO): each piece is predicted by a fit that never saw it. "
        f"Misplaced means |predicted - reference| >= {MISPLACED}.",
        "",
        "## Step 1: difficulty alone (isotonic regression)",
        "",
        f"- Spearman(difficulty, grade): {s1['spearman_difficulty']}",
        f"- LOO Spearman: {s1['loo_spearman']}, LOO MAE: {s1['loo_mae']} grades",
        f"- Knots (difficulty -> grade): {', '.join(f'{x} -> {y}' for x, y in s1['knots'])}",
        f"- Misplaced: {len(s1['misplaced'])} of {len(res.ids)}",
        "",
    ]
    lines += _table(s1["misplaced"])
    lines += [
        "",
        "Below the 0.7 bar in docs/learning-mode.md 1.5, so step 2 adds features.",
        "",
        "## Step 2: cheap features (ridge regression)",
        "",
        "Candidate features (features.level_features): " + ", ".join(f"`{f}`" for f in FEATURES) + ". "
        "No pedal: CC64 isn't loaded by midi_io.",
        "",
        "Each feature's own Spearman with the grade:",
        "",
        "| Feature | Spearman |",
        "| --- | --- |",
    ]
    lines += [f"| `{f}` | {v} |" for f, v in sorted(s2["feature_spearman"].items(), key=lambda kv: -abs(kv[1]))]
    lines += [
        "",
        f"Forward selection by LOO MAE (a feature must save {MIN_GAIN} grades, at most {MAX_FEATURES}), ridge strength "
        f"from {list(RIDGE_GRID)} by LOO, on standardised features. A feature set where any coefficient comes out "
        "negative is skipped: every feature measures something that makes a piece harder.",
        "",
        f"- Chosen: {', '.join(f'`{f}`' for f in s2['features'])}; ridge strength {s2['ridge']}",
        f"- LOO Spearman: {s2['loo_spearman']}, LOO MAE: {s2['loo_mae']} grades (features chosen on all {len(res.ids)}, "
        "so slightly optimistic)",
        f"- Nested LOO (selection redone without the held-out piece): Spearman {s2['nested_loo_spearman']}, "
        f"MAE {s2['nested_loo_mae']} grades. This is the honest number.",
        f"- Misplaced (LOO): {len(s2['misplaced'])} of {len(res.ids)}",
        "",
        "Model (pianopipe/grade_model.json): grade = intercept + sum(coef * (feature - mean) / std), clamped to 0-10.",
        "",
        "| Feature | Mean | Std | Coef (grades per std) |",
        "| --- | --- | --- | --- |",
    ]
    lines += [f"| `{f}` | {mu:.3f} | {sd:.3f} | {c:+.2f} |" for f, mu, sd, c in zip(m.features, m.mean, m.std, m.coef)]
    lines += [f"| intercept | | | {m.intercept:.2f} |", ""]
    lines += _table(s2["misplaced"])
    lines += [
        "",
        "## Per piece",
        "",
        "| Piece | Reference | difficulty | Step 1 (LOO) | Step 2 (LOO) |",
        "| --- | --- | --- | --- | --- |",
    ]
    order = sorted(range(len(res.ids)), key=lambda i: (res.grades[i], res.ids[i]))
    lines += [
        f"| {res.titles[i]} | {res.grades[i]:g} | {res.rows[i]['difficulty']} | {s1['loo_pred'][i]:.1f} | {s2['loo_pred'][i]:.1f} |"
        for i in order
    ]
    if missing:
        lines += ["", f"Anchors without a file (not used): {', '.join(missing)}"]
    lines += [
        "",
        "## Limitations",
        "",
        f"- {len(res.ids)} anchors is a small set; a feature that looks useful here may not generalise, which is why the "
        "model is kept to a few features and the nested LOO number is reported.",
        "- The RCM-to-grade concordance is our own synthesis (grade_anchors.csv header)"
        + (f", and {rcm_only} of the anchors rest on it alone." if rcm_only is not None else "."),
        "- MIDI shows notes, not how they must sound: voicing, tone, pedalling and rubato, which make slow pieces hard, "
        "are invisible here (pedal could be added once CC64 is loaded).",
        "- Length (`log_notes`) carries much of the fit: longer pieces sit higher on exam lists. It counts notes as the "
        "file has them, so a file with written-out repeats rates longer than one without.",
        "- Length stops counting at 1,500 notes and every feature is clipped to the anchors' range, so nothing is "
        "extrapolated: before, pieces longer or more chromatic than every anchor got 10 (Schubert's impromptus, even "
        "at melody level). Leave-one-out MAE / nested MAE of the options tried: uncapped 0.59 / 0.65; cap 1,500 alone "
        "0.64 / 0.68; cap 2,500 alone 0.59 / 0.60 (but Schubert stays at 9.3-10); clipping alone 0.63 / 0.68; "
        "clipping + cap 1,500 (chosen) 0.66 / 0.73. Density per minute instead of length (nested 1.18) and a "
        "per-window key measure (LOO 0.76) lost too much. The cost of the chosen option: long virtuoso pieces read "
        "low (La Campanella 6.7, Moonlight 3rd movement 7.2), because nothing here sees speed and no anchor is above "
        "grade 8. More anchors (PSyllabus, below) are the real fix.",
        "- The grades are for the full arrangement. Reduced levels get the same model's estimate from their own notes, "
        "which the anchors don't test.",
        "- Next step: PSyllabus (CC BY 4.0, 7,901 pieces with levels from ABRSM, RCM, Trinity and others, "
        "https://doi.org/10.5281/zenodo.14794592) is the obvious bigger training set; keep these anchors as the check.",
        "",
    ]
    return "\n".join(lines)


def _table(misplaced: list[dict]) -> list[str]:
    if not misplaced:
        return ["None misplaced."]
    out = ["| Piece | Reference | Predicted |", "| --- | --- | --- |"]
    out += [f"| {m['title']} | {m['reference']:g} | {m['predicted']:.1f} |" for m in misplaced]
    return out


def summary(model: GradeModel) -> str:
    """One paragraph for build logs."""
    f = model.fit
    if not f:
        return "grade model: no fit summary"
    s1, s2 = f["step1"], f["step2"]
    return (
        f"grade fit on {f['anchors']} anchors: difficulty alone Spearman {s1['spearman_difficulty']} "
        f"(isotonic LOO MAE {s1['loo_mae']}); model {'+'.join(model.features)} LOO Spearman {s2['loo_spearman']}, "
        f"MAE {s2['loo_mae']} (nested LOO {s2['nested_loo_spearman']}, MAE {s2['nested_loo_mae']})"
    )


_DEFAULTS: dict = {}


def default_model() -> GradeModel | None:
    """The model shipped in the package (grade_model.json beside this file), or None if it's missing."""
    if "model" not in _DEFAULTS:
        _DEFAULTS["model"] = GradeModel.load() if MODEL_JSON.exists() else None
    return _DEFAULTS["model"]


def default_anchors() -> dict[str, Anchor]:
    if "anchors" not in _DEFAULTS:
        _DEFAULTS["anchors"] = load_anchors() if ANCHORS_CSV.exists() else {}
    return _DEFAULTS["anchors"]
