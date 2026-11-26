"""Quality gates: compare a run against a committed baseline and report regressions.

CI fails when any gated metric moves in the wrong direction by more than its
tolerance. Refresh the baseline deliberately with `--update-baseline` once a
change is accepted.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Rule:
    metric: str
    higher_is_better: bool = True
    tolerance: float = 0.01


def check_against_baseline(current: dict[str, float], baseline: dict[str, float], rules: list[Rule]) -> list[str]:
    failures = []
    for rule in rules:
        if rule.metric not in baseline or rule.metric not in current:
            continue
        cur, base = current[rule.metric], baseline[rule.metric]
        if cur is None or base is None:
            continue
        delta = cur - base
        regressed = delta < -rule.tolerance if rule.higher_is_better else delta > rule.tolerance
        if regressed:
            arrow = "dropped" if rule.higher_is_better else "rose"
            failures.append(
                f"{rule.metric} {arrow} from {base:.4f} to {cur:.4f} "
                f"(change {delta:+.4f}, tolerance {rule.tolerance})"
            )
    return failures


def load_baseline(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def write_baseline(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_step_summary(markdown: str) -> None:
    """Append to the GitHub Actions job summary when running in CI."""
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if target:
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(markdown + "\n")
