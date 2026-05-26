#!/usr/bin/env python3
"""Find the best model (lowest MAPE) per target from `evaluation_values/` CSVs.

By default, if `evaluation_metrics_target_*.csv` files exist, they are used.
Otherwise, if `evaluation_metrics_all_targets.csv` exists, it is used (it
contains `MAPE_<target>` columns for all targets). As a last resort, the script
loads any `evaluation_metrics*.csv` files in the given folder.

Outputs a CSV with the best row per target (or per target+chunk lengths), and
includes the corresponding MAE/RMSE/MAPE values for that selection.

Tip: use `--only-all-targets` to force reading only `evaluation_metrics_all_targets.csv`,
or `--also-all-targets` to additionally write a separate "all_targets-only" output.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to read CSV: {path}: {exc}") from exc


_MAE_RMSE_MAPE_TARGET_RE = re.compile(r"^(MAE|RMSE|MAPE)_(.+)$")


def _extract_target_metric_columns(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Return mapping: target -> {metric_name -> column_name} for MAE/RMSE/MAPE."""
    out: dict[str, dict[str, str]] = {}
    for col in df.columns:
        match = _MAE_RMSE_MAPE_TARGET_RE.match(str(col))
        if not match:
            continue
        metric_name, target = match.group(1), match.group(2)
        out.setdefault(target, {})[metric_name] = col
    return out


def _to_long_from_wide_targets(
    df: pd.DataFrame,
    *,
    source_file: str,
    id_cols: list[str],
) -> pd.DataFrame:
    target_to_metrics = _extract_target_metric_columns(df)
    if not target_to_metrics:
        return pd.DataFrame(columns=[*id_cols, "Target", "MAE", "RMSE", "MAPE", "SourceFile"])

    parts: list[pd.DataFrame] = []
    for target, metrics_map in sorted(target_to_metrics.items()):
        if not {"MAE", "RMSE", "MAPE"}.issubset(set(metrics_map.keys())):
            # Skip partially-populated targets.
            continue
        sub = df[id_cols].copy()
        sub["Target"] = target
        sub["MAE"] = pd.to_numeric(df[metrics_map["MAE"]], errors="coerce")
        sub["RMSE"] = pd.to_numeric(df[metrics_map["RMSE"]], errors="coerce")
        sub["MAPE"] = pd.to_numeric(df[metrics_map["MAPE"]], errors="coerce")
        sub["SourceFile"] = source_file
        parts.append(sub)

    if not parts:
        return pd.DataFrame(columns=[*id_cols, "Target", "MAE", "RMSE", "MAPE", "SourceFile"])
    return pd.concat(parts, ignore_index=True)


def _infer_target_from_filename(path: Path) -> str | None:
    # e.g. evaluation_metrics_target_return.csv -> return
    m = re.match(r"evaluation_metrics_target_(.+)\.csv$", path.name)
    if m:
        return m.group(1)
    return None


def _to_long_from_target_file(
    df: pd.DataFrame,
    *,
    source_file: str,
    default_target: str | None,
    id_cols: list[str],
) -> pd.DataFrame:
    # Prefer metric-specific target cols (MAE_<target>, RMSE_<target>, MAPE_<target>) if present.
    if _extract_target_metric_columns(df):
        return _to_long_from_wide_targets(df, source_file=source_file, id_cols=id_cols)

    required = {"MAE", "RMSE", "MAPE"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame(columns=[*id_cols, "Target", "MAE", "RMSE", "MAPE", "SourceFile"])

    target = default_target
    if not target and "RunTag" in df.columns:
        # e.g. target_return
        run_tag = str(df.iloc[0]["RunTag"]) if len(df) else ""
        if run_tag.startswith("target_"):
            target = run_tag[len("target_") :]

    if not target:
        target = "unknown"

    out = df[id_cols].copy()
    out["Target"] = target
    out["MAE"] = pd.to_numeric(df["MAE"], errors="coerce")
    out["RMSE"] = pd.to_numeric(df["RMSE"], errors="coerce")
    out["MAPE"] = pd.to_numeric(df["MAPE"], errors="coerce")
    out["SourceFile"] = source_file
    return out


def load_evaluation_long(
    evaluation_dir: Path,
    *,
    use_all_files: bool = False,
    only_all_targets: bool = False,
) -> pd.DataFrame:
    evaluation_dir = evaluation_dir.resolve()
    if not evaluation_dir.exists() or not evaluation_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {evaluation_dir}")

    all_targets_path = evaluation_dir / "evaluation_metrics_all_targets.csv"
    per_target_paths = sorted(evaluation_dir.glob("evaluation_metrics_target_*.csv"))

    csv_paths: list[Path]
    if only_all_targets:
        if not all_targets_path.exists():
            raise FileNotFoundError(f"File not found: {all_targets_path}")
        csv_paths = [all_targets_path]
    elif use_all_files:
        csv_paths = sorted(evaluation_dir.glob("evaluation_metrics*.csv"))
    else:
        # Auto mode:
        # - Prefer per-target files if they exist (most direct source per target)
        # - Otherwise fall back to all_targets if present
        # - Otherwise load whatever evaluation_metrics*.csv exists
        if per_target_paths:
            csv_paths = per_target_paths
        elif all_targets_path.exists():
            csv_paths = [all_targets_path]
        else:
            csv_paths = sorted(evaluation_dir.glob("evaluation_metrics*.csv"))

    if not csv_paths:
        raise FileNotFoundError(f"No evaluation CSVs found in: {evaluation_dir}")

    # Common columns seen in your CSVs.
    id_cols = [c for c in ["RunTag", "InputChunkLength", "OutputChunkLength", "Model"] if c]

    long_parts: list[pd.DataFrame] = []

    for path in csv_paths:
        df = _read_csv(path)
        missing = [c for c in id_cols if c not in df.columns]
        if missing:
            # Skip files that are not in expected format.
            continue

        if path.name == "evaluation_metrics_all_targets.csv":
            long_df = _to_long_from_wide_targets(
                df, source_file=path.name, id_cols=id_cols
            )
        else:
            default_target = _infer_target_from_filename(path)
            long_df = _to_long_from_target_file(
                df,
                source_file=path.name,
                default_target=default_target,
                id_cols=id_cols,
            )

        if not long_df.empty:
            long_parts.append(long_df)

    if not long_parts:
        raise RuntimeError(
            "No usable evaluation rows found (missing required columns like Model/InputChunkLength/OutputChunkLength)."
        )

    long = pd.concat(long_parts, ignore_index=True)

    # Normalize types
    long["InputChunkLength"] = pd.to_numeric(long["InputChunkLength"], errors="coerce")
    long["OutputChunkLength"] = pd.to_numeric(long["OutputChunkLength"], errors="coerce")

    # Drop unusable metric values
    long = long.dropna(subset=["MAPE", "MAE", "RMSE"])

    # Drop exact duplicates that can happen when loading both all_targets + per-target files.
    long = long.drop_duplicates(
        subset=["Target", "Model", "InputChunkLength", "OutputChunkLength", "MAPE", "MAE", "RMSE"]
    )

    return long


def select_best(
    long_df: pd.DataFrame,
    *,
    group_by_chunks: bool = False,
    input_chunk: int | None = None,
    output_chunk: int | None = None,
) -> pd.DataFrame:
    df = long_df.copy()

    if input_chunk is not None:
        df = df[df["InputChunkLength"] == input_chunk]
    if output_chunk is not None:
        df = df[df["OutputChunkLength"] == output_chunk]

    if df.empty:
        raise RuntimeError("No rows left after applying filters (input/output chunk lengths).")

    group_cols = ["Target"]
    if group_by_chunks:
        group_cols += ["InputChunkLength", "OutputChunkLength"]

    # Deterministic tie-breaker: MAPE, then Model name.
    df = df.sort_values([*group_cols, "MAPE", "Model"], ascending=[True] * len(group_cols) + [True, True])

    best = df.groupby(group_cols, as_index=False).first()

    # Keep just what we need (and make it obvious).
    out_cols = [*group_cols, "Model", "MAE", "RMSE", "MAPE"]
    if not group_by_chunks:
        out_cols += ["InputChunkLength", "OutputChunkLength"]
    if "SourceFile" in best.columns:
        out_cols += ["SourceFile"]

    best = best[out_cols].rename(
        columns={
            "Model": "BestModel",
            "MAE": "BestMAE",
            "RMSE": "BestRMSE",
            "MAPE": "BestMAPE",
        }
    )

    return best


def _default_output_path(evaluation_dir: Path, group_by_chunks: bool) -> Path:
    name = "best_model_per_target_by_chunks.csv" if group_by_chunks else "best_model_per_target.csv"
    return evaluation_dir / name


def _default_output_path_all_targets(evaluation_dir: Path, group_by_chunks: bool) -> Path:
    name = (
        "best_model_per_target_all_targets_by_chunks.csv"
        if group_by_chunks
        else "best_model_per_target_all_targets.csv"
    )
    return evaluation_dir / name


def _derive_out_path(base_out_path: Path, *, all_targets: bool, by_chunks: bool) -> Path:
    """Derive an output path with stable suffix ordering.

    Ordering is always: `_all_targets` then `_by_chunks`.
    Any existing occurrences at the end of the stem are stripped first.
    """

    stem = base_out_path.stem
    for marker in ("_all_targets_by_chunks", "_all_targets", "_by_chunks"):
        if stem.endswith(marker):
            stem = stem[: -len(marker)]

    suffix = ""
    if all_targets:
        suffix += "_all_targets"
    if by_chunks:
        suffix += "_by_chunks"

    new_name = f"{stem}{suffix}{base_out_path.suffix or '.csv'}"
    return base_out_path.with_name(new_name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find best model per target based on MAPE from evaluation CSVs."
    )
    parser.add_argument(
        "--evaluation-dir",
        default="evaluation_values",
        help="Folder containing evaluation CSV files (default: evaluation_values).",
    )
    parser.add_argument(
        "--use-all-files",
        action="store_true",
        help="Load all evaluation_metrics*.csv (may include duplicates across all_targets/per-target files).",
    )
    parser.add_argument(
        "--only-all-targets",
        action="store_true",
        help="Use only evaluation_metrics_all_targets.csv (if it exists).",
    )
    parser.add_argument(
        "--also-all-targets",
        action="store_true",
        help=(
            "Additionally compute and write best models based only on evaluation_metrics_all_targets.csv "
            "(useful when per-target files exist but you want the all-targets view too)."
        ),
    )
    parser.add_argument(
        "--also-by-chunks",
        action="store_true",
        help=(
            "Additionally write the by-chunks output(s) (i.e., best per target + (InputChunkLength, OutputChunkLength)) "
            "without needing a second run with --group-by-chunks."
        ),
    )
    parser.add_argument(
        "--group-by-chunks",
        action="store_true",
        help="Select best model per (target, InputChunkLength, OutputChunkLength).",
    )
    parser.add_argument(
        "--input-chunk",
        type=int,
        default=None,
        help="If set, only consider this InputChunkLength.",
    )
    parser.add_argument(
        "--output-chunk",
        type=int,
        default=None,
        help="If set, only consider this OutputChunkLength.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path. Default writes into evaluation dir.",
    )

    args = parser.parse_args()

    evaluation_dir = Path(args.evaluation_dir)
    long_df = load_evaluation_long(
        evaluation_dir,
        use_all_files=bool(args.use_all_files),
        only_all_targets=bool(args.only_all_targets),
    )

    best = select_best(
        long_df,
        group_by_chunks=bool(args.group_by_chunks),
        input_chunk=args.input_chunk,
        output_chunk=args.output_chunk,
    )

    out_path = Path(args.out) if args.out else _default_output_path(evaluation_dir, bool(args.group_by_chunks))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best.to_csv(out_path, index=False)

    # Print a compact view to stdout
    with pd.option_context("display.max_rows", 200, "display.max_columns", 50, "display.width", 140):
        print(best.to_string(index=False))

    print(f"\nWrote: {out_path}")

    if bool(args.also_by_chunks) and not bool(args.group_by_chunks):
        best_by_chunks = select_best(
            long_df,
            group_by_chunks=True,
            input_chunk=args.input_chunk,
            output_chunk=args.output_chunk,
        )

        if args.out:
            out_path_by_chunks = _derive_out_path(out_path, all_targets=False, by_chunks=True)
        else:
            out_path_by_chunks = _default_output_path(evaluation_dir, True)

        out_path_by_chunks.parent.mkdir(parents=True, exist_ok=True)
        best_by_chunks.to_csv(out_path_by_chunks, index=False)

        with pd.option_context(
            "display.max_rows", 200, "display.max_columns", 50, "display.width", 140
        ):
            print("\nBy-chunks best models:")
            print(best_by_chunks.to_string(index=False))

        print(f"\nWrote: {out_path_by_chunks}")

    if bool(args.also_all_targets) and not bool(args.only_all_targets):
        long_df_all_targets = load_evaluation_long(
            evaluation_dir,
            use_all_files=False,
            only_all_targets=True,
        )
        best_all_targets = select_best(
            long_df_all_targets,
            group_by_chunks=bool(args.group_by_chunks),
            input_chunk=args.input_chunk,
            output_chunk=args.output_chunk,
        )

        if args.out:
            out_path_all_targets = _derive_out_path(
                out_path, all_targets=True, by_chunks=bool(args.group_by_chunks)
            )
        else:
            out_path_all_targets = _default_output_path_all_targets(
                evaluation_dir, bool(args.group_by_chunks)
            )

        out_path_all_targets.parent.mkdir(parents=True, exist_ok=True)
        best_all_targets.to_csv(out_path_all_targets, index=False)

        with pd.option_context(
            "display.max_rows", 200, "display.max_columns", 50, "display.width", 140
        ):
            print("\nAll-targets-only best models:")
            print(best_all_targets.to_string(index=False))

        print(f"\nWrote: {out_path_all_targets}")

        if bool(args.also_by_chunks) and not bool(args.group_by_chunks):
            best_all_targets_by_chunks = select_best(
                long_df_all_targets,
                group_by_chunks=True,
                input_chunk=args.input_chunk,
                output_chunk=args.output_chunk,
            )

            if args.out:
                out_path_all_targets_by_chunks = _derive_out_path(
                    out_path, all_targets=True, by_chunks=True
                )
            else:
                out_path_all_targets_by_chunks = _default_output_path_all_targets(
                    evaluation_dir, True
                )

            out_path_all_targets_by_chunks.parent.mkdir(parents=True, exist_ok=True)
            best_all_targets_by_chunks.to_csv(out_path_all_targets_by_chunks, index=False)

            with pd.option_context(
                "display.max_rows", 200, "display.max_columns", 50, "display.width", 140
            ):
                print("\nAll-targets-only by-chunks best models:")
                print(best_all_targets_by_chunks.to_string(index=False))

            print(f"\nWrote: {out_path_all_targets_by_chunks}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
