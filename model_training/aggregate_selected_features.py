#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

def parse_feature_list(value: str) -> list[str]:
    if value is None:
        return []
    cleaned = value.strip()
    if not cleaned:
        return []
    return [item.strip() for item in cleaned.split(",") if item.strip()]

def aggregate_features(csv_path: Path) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    past_by_target: dict[str, set[str]] = defaultdict(set)
    input_by_target: dict[str, set[str]] = defaultdict(set)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            target = (row.get("target") or "").strip()
            if not target:
                continue

            selected_past = parse_feature_list(row.get("selected_past", ""))
            selected_input = parse_feature_list(row.get("selected_input", ""))

            for feature in selected_past:
                past_by_target[target].add(feature)
            for feature in selected_input:
                input_by_target[target].add(feature)

    past_sorted = {target: sorted(features) for target, features in past_by_target.items()}
    input_sorted = {target: sorted(features) for target, features in input_by_target.items()}
    return past_sorted, input_sorted

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate selected_past and selected_input columns by target."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="/home/kpiotr6/Documents/stuida/praca_magisterska/proj/outputs/linear_regression_forward_selection_summary.csv",
        help="Path to the forward selection summary CSV.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    past_agg, input_agg = aggregate_features(csv_path)
    targets = sorted(set(past_agg) | set(input_agg))
    aggregated = {
        target: {
            "past": past_agg.get(target, []),
            "input": input_agg.get(target, []),
        }
        for target in targets
    }
    print(json.dumps(aggregated, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
