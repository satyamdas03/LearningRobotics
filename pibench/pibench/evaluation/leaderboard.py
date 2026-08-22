"""Static leaderboard generation for PIBench results.

The leaderboard scans ``output/results_*.json`` files produced by the runner,
computes per-predictor aggregates, and writes both a JSON summary and a
self-contained HTML page.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from pibench.core.runner import RunResult
from pibench.evaluation.metrics import accuracy_per_concept, calibration_metrics


class SuiteResultSummary(BaseModel):
    suite: str
    n_total: int
    n_correct: float
    accuracy: float


class ConceptResultSummary(BaseModel):
    concept: str
    n_total: int
    n_correct: float
    accuracy: float


class CalibrationSummary(BaseModel):
    ece: float
    brier: float
    nll: float | None
    n_with_confidence: int
    n_total: int


class LeaderboardEntry(BaseModel):
    predictor: str
    overall_accuracy: float
    n_total: int
    n_correct: float
    suite_metrics: list[SuiteResultSummary]
    concept_metrics: list[ConceptResultSummary]
    calibration: CalibrationSummary
    results_path: str | None = None


class Leaderboard(BaseModel):
    entries: list[LeaderboardEntry] = Field(default_factory=list)


def _load_run_result(path: Path) -> RunResult:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return RunResult.model_validate(data)


def build_leaderboard(
    result_paths: list[Path] | None = None,
    results_glob: str = "output/results_*.json",
) -> Leaderboard:
    """Build a leaderboard from result JSON files.

    If ``result_paths`` is omitted, the current working directory is searched
    with ``results_glob``.
    """
    if result_paths is None:
        result_paths = sorted(Path.cwd().glob(results_glob))

    entries: list[LeaderboardEntry] = []
    for path in result_paths:
        result = _load_run_result(path)
        all_results = [r for s in result.suites for r in s.results]

        suite_summaries: list[SuiteResultSummary] = []
        for sr in result.suites:
            total = len(sr.results)
            correct = sum(r.score for r in sr.results)
            suite_summaries.append(
                SuiteResultSummary(
                    suite=sr.suite,
                    n_total=total,
                    n_correct=correct,
                    accuracy=correct / total if total else 0.0,
                )
            )

        concept_acc = accuracy_per_concept(all_results)
        concept_summaries = [
            ConceptResultSummary(
                concept=concept,
                n_total=data["n_total"],
                n_correct=data["n_correct"],
                accuracy=data["accuracy"],
            )
            for concept, data in concept_acc.items()
        ]

        cal = calibration_metrics(all_results)
        calibration_summary = CalibrationSummary(
            ece=cal.ece,
            brier=cal.brier,
            nll=cal.nll,
            n_with_confidence=cal.n_with_confidence,
            n_total=cal.n_total,
        )

        entries.append(
            LeaderboardEntry(
                predictor=result.predictor,
                overall_accuracy=result.overall_accuracy,
                n_total=len(all_results),
                n_correct=sum(r.score for r in all_results),
                suite_metrics=suite_summaries,
                concept_metrics=concept_summaries,
                calibration=calibration_summary,
                results_path=str(path),
            )
        )

    # Sort by overall accuracy descending, then by name.
    entries.sort(key=lambda e: (-e.overall_accuracy, e.predictor))
    return Leaderboard(entries=entries)


def _html_table(
    title: str,
    headers: list[str],
    rows: list[list[str]],
) -> str:
    header_html = "".join(f"<th>{h}</th>" for h in headers)
    rows_html = ""
    for row in rows:
        rows_html += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>\n"
    return f"""
    <h2>{title}</h2>
    <table>
      <thead><tr>{header_html}</tr></thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
    """


def render_leaderboard_html(leaderboard: Leaderboard) -> str:
    """Render a self-contained HTML leaderboard."""
    entries = leaderboard.entries

    # Overall table.
    overall_rows: list[list[str]] = []
    for e in entries:
        cal = e.calibration
        cal_str = f"ECE={cal.ece:.2%}, Brier={cal.brier:.3f}"
        if cal.nll is not None:
            cal_str += f", NLL={cal.nll:.3f}"
        overall_rows.append(
            [
                e.predictor,
                f"{e.overall_accuracy:.1%}",
                f"{int(e.n_correct)}/{e.n_total}",
                f"{cal.n_with_confidence}/{cal.n_total}",
                cal_str,
            ]
        )

    # Collect all suite names and concept names across entries for pivot tables.
    suite_names = sorted({sm.suite for e in entries for sm in e.suite_metrics})
    concept_names = sorted({cm.concept for e in entries for cm in e.concept_metrics})

    suite_rows: list[list[str]] = []
    for e in entries:
        acc_by_suite = {sm.suite: sm for sm in e.suite_metrics}
        row = [e.predictor]
        for name in suite_names:
            sm = acc_by_suite.get(name)
            row.append(f"{sm.accuracy:.1%}" if sm else "—")
        suite_rows.append(row)

    concept_rows: list[list[str]] = []
    for e in entries:
        acc_by_concept = {cm.concept: cm for cm in e.concept_metrics}
        row = [e.predictor]
        for name in concept_names:
            cm = acc_by_concept.get(name)
            row.append(f"{cm.accuracy:.1%}" if cm else "—")
        concept_rows.append(row)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PIBench Leaderboard</title>
  <style>
    :root {{
      --bg: #ffffff;
      --fg: #1a1a1a;
      --muted: #555555;
      --border: #dddddd;
      --accent: #2563eb;
      --accent-soft: #dbeafe;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #0f172a;
        --fg: #e2e8f0;
        --muted: #94a3b8;
        --border: #334155;
        --accent: #60a5fa;
        --accent-soft: #1e293b;
      }}
    }}
    body {{
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--fg);
      margin: 0;
      padding: 2rem;
      line-height: 1.5;
    }}
    h1, h2 {{ color: var(--accent); }}
    .container {{
      max-width: 1100px;
      margin: 0 auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 1rem 0 2rem;
      font-size: 0.95rem;
    }}
    th, td {{
      border: 1px solid var(--border);
      padding: 0.5rem 0.75rem;
      text-align: left;
    }}
    th {{
      background: var(--accent-soft);
      color: var(--accent);
      font-weight: 600;
    }}
    tr:nth-child(even) {{ background: var(--accent-soft); }}
    .meta {{
      color: var(--muted);
      font-size: 0.9rem;
      margin-bottom: 1.5rem;
    }}
    .scroll {{
      overflow-x: auto;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>PIBench Leaderboard</h1>
    <p class="meta">Generated automatically from <code>output/results_*.json</code>.</p>

    {_html_table(
        "Overall",
        ["Predictor", "Accuracy", "Correct / Total", "With Confidence", "Calibration"],
        overall_rows,
    )}

    <div class="scroll">
      {_html_table(
          "Per-Suite Accuracy",
          ["Predictor"] + suite_names,
          suite_rows,
      )}
    </div>

    <div class="scroll">
      {_html_table(
          "Per-Concept Accuracy",
          ["Predictor"] + concept_names,
          concept_rows,
      )}
    </div>
  </div>
</body>
</html>
"""
    return html


def write_leaderboard(
    leaderboard: Leaderboard,
    json_path: Path | str,
    html_path: Path | str | None = None,
) -> tuple[Path, Path | None]:
    """Write the leaderboard to JSON and optional HTML."""
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(leaderboard.model_dump(), f, indent=2)

    html_written: Path | None = None
    if html_path is not None:
        html_path = Path(html_path)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(render_leaderboard_html(leaderboard), encoding="utf-8")
        html_written = html_path

    return json_path, html_written
