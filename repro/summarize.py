#!/usr/bin/env python3
"""Aggregate results/summary.csv into a per-model, per-condition table.

Writes results/summary.md (mean and standard deviation of customer welfare and
needs-met rate across runs) so you can compare against the paper's findings.

For the experiment-specific metrics the paper reports (payments redirected to
manipulative sellers, position and first-proposal bias), use the upstream
analysis scripts, which read the same databases:
  experiments/malicious/analyze_results.py, plot_results.py
  experiments/position/generate_position_data.py, generate_proposal_data.py, plot_*.py
  experiments/consideration_set_size/analyze_consideration_set_size.py, create_plot.py
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SUMMARY_CSV = REPO_ROOT / "results" / "summary.csv"
OUT_MD = REPO_ROOT / "results" / "summary.md"


def fmt(values: list[float], digits: int = 1) -> str:
    if not values:
        return "–"
    mean = statistics.fmean(values)
    if len(values) == 1:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {statistics.stdev(values):.{digits}f}"


def main() -> int:
    if not SUMMARY_CSV.exists():
        print(f"No results yet: {SUMMARY_CSV.relative_to(REPO_ROOT)} not found.")
        return 1

    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    with SUMMARY_CSV.open() as f:
        for row in csv.DictReader(f):
            groups[(row["experiment"], row["condition"], row["model"])].append(row)

    lines = [
        "# Reproduction results",
        "",
        "Mean ± standard deviation across runs. Customer welfare is the total utility",
        "customers receive, as computed by upstream analytics: when a purchase meets the customer's needs, twice the customer's stated value of the requested items, minus everything they paid.",
        "",
    ]
    current_exp = None
    for (exp, cond, model), rows in sorted(groups.items()):
        if exp != current_exp:
            current_exp = exp
            lines += [
                f"## {exp}",
                "",
                "| Condition | Model | Runs | Customer welfare | Needs met (%) | Failed LLM calls |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        welfare = [float(r["customer_welfare"]) for r in rows]
        needs = [
            100 * int(r["customers_with_needs_met"]) / max(1, int(r["customers"])) for r in rows
        ]
        failed = sum(int(r["failed_llm_calls"]) for r in rows)
        lines.append(
            f"| {cond} | {model} | {len(rows)} | {fmt(welfare)} | {fmt(needs)} | {failed} |"
        )
        next_keys = [k for k in sorted(groups) if k > (exp, cond, model)]
        if not next_keys or next_keys[0][0] != exp:
            lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWritten to {OUT_MD.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
