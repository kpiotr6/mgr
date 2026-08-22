"""Shared loading and ranking for the approach-comparison charts.

Both ``plot_rank_shares.py`` (Form G) and ``plot_approach_model_interaction.py``
(Form H) rank the three modelling approaches against each other, so the ranking
lives here rather than being copied into each script -- if the two drifted apart
the charts would quietly disagree.

Every run is scaled by the NaiveLastValue baseline at the same target, lookback
window and horizon:

    MASE = MAE_model / MAE_naive

Ranks are then taken across the approaches available for a target, lowest MASE
first. ``gran1_blain`` has no ``all_targets`` run, so it is ranked of two while
the other targets are ranked of three.
"""

from pathlib import Path

import pandas as pd

BASELINE_MODEL = "NaiveLastValue"
APPROACH_ORDER = ["simple", "per_target", "all_targets"]
TARGET_ORDER = ["return", "first_chamber_filling", "second_chamber_filling", "gran1_blain"]
KEYS = ["Target", "Model", "InputChunkLength", "OutputChunkLength"]


def _approach_and_targets(frame: pd.DataFrame, run_tag: str) -> tuple[str, list[str]]:
    if run_tag == "all_targets":
        targets = [c[len("MAE_"):] for c in frame.columns if c.startswith("MAE_")]
        return "all_targets", targets
    if run_tag.startswith("simple_"):
        return "simple", [run_tag[len("simple_"):]]
    if run_tag.startswith("target_"):
        return "per_target", [run_tag[len("target_"):]]
    raise ValueError(f"Unrecognised RunTag: {run_tag}")


def load_metrics(evaluation_dir: Path) -> pd.DataFrame:
    """One tidy row per target x approach x model x lookback x horizon."""
    frames = []
    for path in sorted(evaluation_dir.glob("evaluation_metrics_*.csv")):
        frame = pd.read_csv(path)
        approach, targets = _approach_and_targets(frame, frame["RunTag"].iloc[0])
        for target in targets:
            column = f"MAE_{target}"
            if column not in frame.columns:
                raise ValueError(f"{path.name} has no {column} column")
            tidy = frame[["InputChunkLength", "OutputChunkLength", "Model", column]].copy()
            tidy.columns = ["InputChunkLength", "OutputChunkLength", "Model", "MAE"]
            tidy["Target"] = target
            tidy["Approach"] = approach
            frames.append(tidy)

    if not frames:
        raise FileNotFoundError(f"No evaluation_metrics_*.csv under {evaluation_dir}")

    metrics = pd.concat(frames, ignore_index=True)

    # The baseline is per configuration, not per model -- Model must stay out of the key.
    baseline_keys = ["Target", "InputChunkLength", "OutputChunkLength", "Approach"]
    baseline = metrics[metrics["Model"] == BASELINE_MODEL].set_index(baseline_keys)["MAE"]
    lookup = metrics.set_index(baseline_keys).index.map(baseline)
    metrics["MAE_baseline"] = lookup
    if metrics["MAE_baseline"].isna().any():
        raise ValueError(f"Missing {BASELINE_MODEL} baseline for some configurations")

    metrics["MASE"] = metrics["MAE"] / metrics["MAE_baseline"]
    metrics = metrics[metrics["Model"] != BASELINE_MODEL].reset_index(drop=True)
    metrics["Model"] = metrics["Model"].str.replace("NeuralForecast_", "", regex=False)
    return metrics


def approaches_for(metrics: pd.DataFrame, target: str) -> list[str]:
    present = set(metrics.loc[metrics["Target"] == target, "Approach"])
    return [a for a in APPROACH_ORDER if a in present]


def rank_table(metrics: pd.DataFrame, target: str) -> pd.DataFrame:
    """Rank the approaches within every matched configuration of one target.

    Returns a long frame with one row per configuration x approach, carrying the
    1-based ``Rank`` and how many approaches it was ranked against.
    """
    approaches = approaches_for(metrics, target)
    wide = (
        metrics[metrics["Target"] == target]
        .pivot_table(index=KEYS, columns="Approach", values="MASE")
        .reindex(columns=approaches)
        .dropna()
    )
    if wide.empty:
        return pd.DataFrame(columns=KEYS + ["Approach", "Rank", "Depth"])

    ranks = wide.rank(axis=1, method="min", ascending=True)  # lower MASE is better
    long = ranks.stack().rename("Rank").reset_index()
    long = long.rename(columns={long.columns[-2]: "Approach"})
    long["Depth"] = len(approaches)
    return long


def rank_summary(long: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean rank, first-place share and per-place shares, grouped by `by`."""
    if long.empty:
        return pd.DataFrame()
    grouped = long.groupby(by, observed=True)
    summary = grouped.agg(MeanRank=("Rank", "mean"), N=("Rank", "size")).reset_index()
    for place in (1, 2, 3):
        shares = grouped["Rank"].apply(lambda s, p=place: (s == p).mean()).rename(f"Place{place}")
        summary = summary.merge(shares.reset_index(), on=by, how="left")
    return summary
