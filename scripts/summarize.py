#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median, variance

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("Imagism", "Romanticism")


def load_scores(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Group", "Title", "Word_Count", "Variance", "Samples_Count"}
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)):
            raise ValueError(f"{path}: duplicate column names")
        missing = required - set(headers)
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        rows = []
        seen = set()
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}:{line}: row has a different field count from the header")
            if not row["Title"].strip():
                raise ValueError(f"{path}:{line}: Title must not be blank")
            key = row["Group"], row["Title"]
            if key in seen:
                raise ValueError(f"{path}:{line}: duplicate poem {key}")
            seen.add(key)
            if row["Group"] not in GROUPS:
                raise ValueError(f"{path}:{line}: unexpected group {row['Group']!r}")
            for column in ("Word_Count", "Samples_Count"):
                try:
                    row[column] = int(row[column])
                except ValueError as error:
                    raise ValueError(f"{path}:{line}: {column} must be a positive integer") from error
                if row[column] <= 0:
                    raise ValueError(f"{path}:{line}: {column} must be positive")
            try:
                row["Variance"] = float(row["Variance"])
            except ValueError as error:
                raise ValueError(f"{path}:{line}: Variance must be a number") from error
            if not math.isfinite(row["Variance"]) or row["Variance"] < 0:
                raise ValueError(f"{path}:{line}: score must be finite and nonnegative")
            rows.append(row)
    if any(sum(row["Group"] == group for row in rows) < 2 for group in GROUPS):
        raise ValueError(f"{path}: each group needs at least two poem-level scores")
    return rows


def summarize(rows: list[dict]) -> dict:
    groups = {}
    for group in GROUPS:
        selected = [row for row in rows if row["Group"] == group]
        values = [row["Variance"] for row in selected]
        groups[group] = {
            "poems": len(selected),
            "reported_images": sum(row["Samples_Count"] for row in selected),
            "mean_score": mean(values),
            "median_score": median(values),
            "minimum_score": min(values),
            "maximum_score": max(values),
            "mean_word_count": mean(row["Word_Count"] for row in selected),
        }
    a, b = ([row["Variance"] for row in rows if row["Group"] == group] for group in GROUPS)
    n_a, n_b = len(a), len(b)
    var_a, var_b = variance(a), variance(b)
    pooled = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    standard_error = math.sqrt(pooled * (1 / n_a + 1 / n_b))
    welch_error_squared = var_a / n_a + var_b / n_b
    welch_denominator = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    return {
        "groups": groups,
        "mean_difference_romanticism_minus_imagism": mean(b) - mean(a),
        "equal_variance_t_statistic_imagism_minus_romanticism": (
            (mean(a) - mean(b)) / standard_error if standard_error else None
        ),
        "equal_variance_degrees_of_freedom": n_a + n_b - 2,
        "welch_degrees_of_freedom": (
            welch_error_squared**2 / welch_denominator if welch_denominator else None
        ),
        "p_values": "Not computed by this dependency-free audit. Run compare_ttests.py with SciPy.",
    }


def load_datasets(data_dir: Path) -> dict:
    datasets = {
        model: load_scores(data_dir / f"{model}.csv")
        for model in ("flux", "stable_diffusion")
    }
    keys = [{(row["Group"], row["Title"]) for row in rows} for rows in datasets.values()]
    if keys[0] != keys[1]:
        raise ValueError(f"{data_dir}: model tables must contain the same poem/group keys")
    return datasets


def audit(data_dir: Path) -> dict:
    datasets = load_datasets(data_dir)
    matches = []
    other = {(row["Group"], row["Title"]): row for row in datasets["stable_diffusion"]}
    for row in datasets["flux"]:
        key = row["Group"], row["Title"]
        if key in other and row["Variance"] == other[key]["Variance"]:
            matches.append({"group": key[0], "title": key[1], "score": row["Variance"]})
    return {
        "unit_of_analysis": "One retained aggregate score per poem and model; group counts are reported below.",
        "source_sha256": {
            model: hashlib.sha256((data_dir / f"{model}.csv").read_bytes()).hexdigest()
            for model in datasets
        },
        "models": {model: summarize(rows) for model, rows in datasets.items()},
        "exact_cross_model_score_matches": matches,
        "interpretation": "Descriptive audit only. Matching scores require source verification; they are not corrected here.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit saved poem-level scores from local CSV files.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, help="Optional JSON destination; otherwise prints to stdout")
    args = parser.parse_args()
    try:
        payload = json.dumps(audit(args.data_dir), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not args.output:
        print(payload, end="")


if __name__ == "__main__":
    main()
