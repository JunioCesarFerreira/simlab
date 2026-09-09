"""Freeze and compare the NSGA metrics regression baseline.

    python -m tests.regression.baseline --write      # (re-)freeze
    python -m tests.regression.baseline --check      # compare, exit 1 on drift

The baseline is a *record of current behaviour*, not a target to preserve.
Phases 1-4 of the fix plan change these numbers on purpose; the point is that
each phase's diff is inspectable and attributable to one change.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

from .kernel import BASELINE_CONFIGS, SEEDS, run_grid

BASELINE_PATH = Path(__file__).with_name("baseline.json")

# Bumped whenever a phase of the fix plan deliberately changes the numbers, so
# a stale baseline reports as "wrong stage" instead of as a silent regression.
BASELINE_STAGE = "phase-2"

# Comparison tolerance. Runs are bit-for-bit reproducible on one machine; this
# absorbs only last-ulp differences from another BLAS/libm build.
TOLERANCE = 1e-9

_TRACKED_PACKAGES = ("numpy", "pymoo", "deap", "moocore")


def _versions() -> dict:
    return {p: importlib.metadata.version(p) for p in _TRACKED_PACKAGES}


def build(progress=None) -> dict:
    return {
        "stage": BASELINE_STAGE,
        "seeds": list(SEEDS),
        "versions": _versions(),
        "runs": run_grid(progress=progress),
    }


def _walk(node, path=""):
    """Yield (dotted path, value) for every leaf of a nested JSON structure."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(value, f"{path}[{index}]")
    else:
        yield path, node


def compare(actual: dict, expected: dict, tolerance: float = TOLERANCE) -> list[str]:
    """Return a list of human-readable differences; empty means identical."""
    if actual.get("stage") != expected.get("stage"):
        return [f"stage: {expected.get('stage')!r} -> {actual.get('stage')!r}"]

    actual_leaves = dict(_walk(actual["runs"]))
    expected_leaves = dict(_walk(expected["runs"]))

    differences: list[str] = []
    for key in expected_leaves.keys() | actual_leaves.keys():
        if key not in expected_leaves:
            differences.append(f"{key}: absent in baseline -> {actual_leaves[key]!r}")
            continue
        if key not in actual_leaves:
            differences.append(f"{key}: {expected_leaves[key]!r} -> absent in run")
            continue
        before, now = expected_leaves[key], actual_leaves[key]
        if isinstance(before, (int, float)) and isinstance(now, (int, float)):
            if abs(float(now) - float(before)) > tolerance:
                differences.append(f"{key}: {before!r} -> {now!r}")
        elif before != now:
            differences.append(f"{key}: {before!r} -> {now!r}")
    return sorted(differences)


def load() -> dict:
    return json.loads(BASELINE_PATH.read_text())


def summary(result: dict) -> str:
    """Per-config headline table: what a phase of the fix plan is judged on.

    ``max drop`` is the largest single-step decrease in HV over every seed —
    the quantity the audit used to show that the offspring series looks like it
    regresses while the surviving population does not.
    """
    import statistics

    grouped: dict[str, list[dict]] = {}
    for run in result["runs"]:
        grouped.setdefault(run["label"], []).append(run["history"])

    header = (
        f"{'config':22} {'HV off':>10} {'HV surv':>10} "
        f"{'drop off':>10} {'drop surv':>10} {'GD surv':>10}"
    )
    lines = [header, "-" * len(header)]
    for label, histories in grouped.items():
        def worst_drop(field: str) -> float:
            return max(
                max(a[field]["hv"] - b[field]["hv"] for a, b in zip(h, h[1:]))
                for h in histories
            )

        def final_mean(field: str, metric: str) -> float:
            return statistics.fmean(h[-1][field][metric] for h in histories)

        lines.append(
            f"{label:22} {final_mean('offspring', 'hv'):10.6f} "
            f"{final_mean('survivors', 'hv'):10.6f} {worst_drop('offspring'):10.6f} "
            f"{worst_drop('survivors'):10.6f} {final_mean('survivors', 'gd'):10.6f}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="(re-)freeze the baseline")
    group.add_argument("--check", action="store_true", help="compare against the baseline")
    group.add_argument("--summary", action="store_true", help="print the frozen baseline's table")
    parser.add_argument("--max-differences", type=int, default=25)
    args = parser.parse_args()

    if args.summary:
        print(summary(load()))
        return 0

    result = build(progress=lambda label: print(label, "done", flush=True))

    if args.write:
        BASELINE_PATH.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {BASELINE_PATH} ({BASELINE_STAGE}, {len(result['runs'])} runs)")
        print(summary(result))
        return 0

    expected = load()
    if expected["versions"] != result["versions"]:
        print(f"warning: library versions differ: {expected['versions']} != {result['versions']}")
    differences = compare(result, expected)
    if not differences:
        print(f"baseline matches ({BASELINE_STAGE}, {len(result['runs'])} runs)")
        return 0
    print(summary(result))
    print(f"{len(differences)} difference(s) against the {expected['stage']} baseline:")
    for line in differences[: args.max_differences]:
        print(" ", line)
    if len(differences) > args.max_differences:
        print(f"  ... and {len(differences) - args.max_differences} more")
    return 1


if __name__ == "__main__":
    sys.exit(main())
