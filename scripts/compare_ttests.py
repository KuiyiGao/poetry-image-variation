#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path

from summarize import GROUPS, ROOT, load_datasets


def compare_groups(a, b, stats) -> dict:
    if len(set(a)) == 1 and len(set(b)) == 1:
        return {
            method: {"t": None, "p": None, "status": "zero_within_group_variance"}
            for method in ("student_equal_variance", "welch")
        }
    results = {}
    for method, equal_var in (("student_equal_variance", True), ("welch", False)):
        test = stats.ttest_ind(a, b, equal_var=equal_var, alternative="two-sided")
        t, p = float(test.statistic), float(test.pvalue)
        finite = math.isfinite(t) and math.isfinite(p)
        results[method] = {
            "t": t if finite else None,
            "p": p if finite else None,
            "status": "computed" if finite else "nonfinite_result",
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Student and Welch tests on saved poem-level scores.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    try:
        from scipy import stats
    except ImportError:
        parser.error("SciPy is required. Install requirements.txt, or run summarize.py for an offline audit.")
    results = {}
    try:
        datasets = load_datasets(args.data_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for model, rows in datasets.items():
        a, b = ([row["Variance"] for row in rows if row["Group"] == group] for group in GROUPS)
        results[model] = compare_groups(a, b, stats)
    print(json.dumps({
        "unit_of_analysis": "Poem; counts are read from each model table",
        "group_counts": {
            model: {group: sum(row["Group"] == group for row in rows) for group in GROUPS}
            for model, rows in datasets.items()
        },
        "warning": "Exploratory only. Duplicate cross-model scores and selection/length confounds remain unresolved.",
        "results": results,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
