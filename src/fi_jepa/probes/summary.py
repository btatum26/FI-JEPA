from __future__ import annotations

from collections import defaultdict
import math
from statistics import mean, median
from typing import Iterable


RESULT_LABELS = {"SUPPORTED", "PROMISING", "INCONCLUSIVE", "FAILED", "INVALID"}


# ============================================================================
# RESULT LABELS
# ============================================================================


def classify_summary_result(
    task_type: str,
    *,
    window_count: int,
    windows_improved: int,
    primary_metric: float,
    worst_metric: float,
    stable_direction: bool = True,
    invalid: bool = False,
    bootstrap_supports_improvement: bool | None = None,
) -> str:
    """Assign one compact evaluation label from baseline-relative window results.

    Regression metrics are RMSE ratios where lower is better. Classification
    primary/worst metrics are ROC-AUC where higher is better. Bootstrap support
    upgrades strong stable results but its absence does not make a run invalid.
    """
    if invalid or window_count == 0 or not math.isfinite(primary_metric):
        return "INVALID"
    required_windows = min(2, window_count)
    if task_type == "regression":
        if (
            windows_improved >= required_windows
            and primary_metric < 1.0
            and worst_metric <= 1.1
            and stable_direction
            and bootstrap_supports_improvement is not False
        ):
            return "SUPPORTED"
        if windows_improved >= required_windows or primary_metric < 1.0:
            return "PROMISING" if stable_direction else "INCONCLUSIVE"
        if windows_improved == 0 and primary_metric >= 1.0:
            return "FAILED"
        return "INCONCLUSIVE"
    if task_type == "classification":
        if (
            windows_improved >= required_windows
            and primary_metric >= 0.55
            and worst_metric >= 0.5
            and stable_direction
            and bootstrap_supports_improvement is not False
        ):
            return "SUPPORTED"
        if windows_improved >= required_windows or primary_metric > 0.55:
            return "PROMISING" if stable_direction else "INCONCLUSIVE"
        if windows_improved == 0 and primary_metric <= 0.5:
            return "FAILED"
        return "INCONCLUSIVE"
    raise ValueError(f"Unknown summary task type: {task_type!r}")


# ============================================================================
# REPORT AGGREGATION
# ============================================================================


def _finite(values: Iterable[object]) -> list[float]:
    """Return finite floating-point values from a report field sequence."""
    output: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            output.append(number)
    return output


def _bootstrap_support(report: dict[str, object], identity: tuple[str, str, str]) -> bool | None:
    """Read pooled regression bootstrap support without consulting oracle metrics."""
    raw_target, model_name, feature_family = identity
    for row in report.get("stability_summary", []):
        if (
            row.get("task_type") == "regression"
            and row.get("raw_target") == raw_target
            and row.get("model_name") == model_name
            and row.get("feature_family") == feature_family
        ):
            interval = row.get("bootstrap_interval_95")
            if isinstance(interval, list) and len(interval) == 2:
                return float(interval[1]) < 0.0
    return None


def _incremental_rows(report: dict[str, object], expected_windows: int) -> list[dict[str, object]]:
    """Aggregate original-unit hand-plus-representation comparisons by target and model."""
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in report.get("incremental_hand_plus_z_comparisons", []):
        if row.get("score_space") == "original":
            grouped[(str(row["raw_target"]), str(row["model_name"]))].append(row)

    output: list[dict[str, object]] = []
    result_rows = report.get("results", [])
    for (target, model_name), rows in grouped.items():
        ratios = _finite(row.get("rmse_ratio_vs_hand") for row in rows)
        correlations = _finite(
            row.get("pearson_correlation")
            for row in result_rows
            if row.get("task_type") == "regression"
            and row.get("raw_target") == target
            and row.get("model_name") == model_name
            and row.get("feature_family") == "hand_market_features_plus_residual_z"
            and row.get("score_space") == "original"
        )
        signs = {1 if value > 0.0 else -1 if value < 0.0 else 0 for value in correlations}
        stable_direction = not ({-1, 1} <= signs)
        invalid_count = sum(
            int(row.get("invalid_prediction_count", 0))
            for row in result_rows
            if row.get("task_type") == "regression"
            and row.get("raw_target") == target
            and row.get("model_name") == model_name
            and row.get("feature_family") == "hand_market_features_plus_residual_z"
        )
        window_count = len({str(row["validation_window_name"]) for row in rows})
        median_ratio = median(ratios) if ratios else float("nan")
        worst_ratio = max(ratios) if ratios else float("nan")
        improved = sum(value < 1.0 for value in ratios)
        bootstrap_support = _bootstrap_support(
            report, (target, model_name, "hand_market_features_plus_residual_z")
        )
        output.append(
            {
                "target": target,
                "model_name": model_name,
                "window_count": window_count,
                "windows_improved": improved,
                "median_rmse_ratio": median_ratio,
                "worst_rmse_ratio": worst_ratio,
                "stable_direction": stable_direction,
                "label": classify_summary_result(
                    "regression",
                    window_count=window_count,
                    windows_improved=improved,
                    primary_metric=median_ratio,
                    worst_metric=worst_ratio,
                    stable_direction=stable_direction,
                    invalid=invalid_count > 0 or window_count < expected_windows,
                    bootstrap_supports_improvement=bootstrap_support,
                ),
            }
        )
    return sorted(
        output,
        key=lambda row: (
            float(row["median_rmse_ratio"]),
            -int(row["windows_improved"]),
            float(row["worst_rmse_ratio"]),
        ),
    )


def _classification_rows(report: dict[str, object], expected_windows: int) -> list[dict[str, object]]:
    """Aggregate classification heads and compare Brier scores with the class prior."""
    results = report.get("results", [])
    prior: dict[tuple[str, str], float] = {}
    for row in results:
        if row.get("task_type") == "classification" and row.get("predictor_name") == "class_prior":
            prior[(str(row["validation_window_name"]), str(row["classification_label"]))] = float(
                row["brier_score"]
            )
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in results:
        if row.get("task_type") == "classification" and row.get("predictor_kind") == "model":
            grouped[
                (
                    str(row["classification_label"]),
                    str(row["model_name"]),
                    str(row["feature_family"]),
                )
            ].append(row)

    output: list[dict[str, object]] = []
    for (label_name, model_name, feature_family), rows in grouped.items():
        aucs = _finite(row.get("roc_auc") for row in rows)
        brier_ratios = []
        for row in rows:
            baseline = prior.get((str(row["validation_window_name"]), label_name))
            if baseline is not None and baseline > 0.0:
                brier_ratios.append(float(row["brier_score"]) / baseline)
        window_count = len({str(row["validation_window_name"]) for row in rows})
        windows_improved = sum(value < 1.0 for value in brier_ratios)
        mean_auc = mean(aucs) if aucs else float("nan")
        worst_auc = min(aucs) if aucs else float("nan")
        mean_brier_ratio = mean(brier_ratios) if brier_ratios else float("nan")
        invalid = (
            any(int(row.get("invalid_prediction_count", 0)) > 0 for row in rows)
            or window_count < expected_windows
        )
        output.append(
            {
                "classification_label": label_name,
                "model_name": model_name,
                "feature_family": feature_family,
                "window_count": window_count,
                "mean_auc": mean_auc,
                "worst_auc": worst_auc,
                "mean_brier_ratio": mean_brier_ratio,
                "windows_improved": windows_improved,
                "label": classify_summary_result(
                    "classification",
                    window_count=window_count,
                    windows_improved=windows_improved,
                    primary_metric=mean_auc,
                    worst_metric=worst_auc,
                    invalid=invalid,
                ),
            }
        )
    return sorted(output, key=lambda row: (-float(row["mean_auc"]), float(row["mean_brier_ratio"])))


def _representation_rows(report: dict[str, object]) -> list[dict[str, object]]:
    """Flatten selected-PCA per-window diagnostics for the Markdown table."""
    diagnostics = report.get("representation_diagnostics", {})
    representations = diagnostics.get("representations", {}) if isinstance(diagnostics, dict) else {}
    selected = representations.get("selected_pca_representation", {})
    rows: list[dict[str, object]] = []
    for window in selected.get("windows", []):
        metrics = window.get("metrics", {})

        def value(name: str, field: str = "validation_value") -> object:
            metric = metrics.get(name, {})
            return metric.get(field) if isinstance(metric, dict) else None

        rows.append(
            {
                "window": window.get("validation_window_name"),
                "effective_rank": value("effective_rank"),
                "rank_percentile": value("effective_rank", "validation_percentile"),
                "pairwise_cosine": value("mean_pairwise_cosine"),
                "mean_norm": value("mean_vector_norm"),
            }
        )
    return rows


def _warnings(report: dict[str, object], expected_windows: int) -> list[str]:
    """Return concrete warnings without allowing oracle diagnostics into conclusions."""
    warnings: list[str] = []
    grids = report.get("resolved_probe_config", {})
    grid_names = {
        "ridge": "ridge_alphas",
        "huber": "huber_alphas",
        "elastic_net": "elastic_net_alphas",
        "logistic": "logistic_alphas",
    }
    boundary_hits: set[str] = set()
    for row in report.get("parameter_selection_by_fold", []):
        model_name = str(row.get("model_name"))
        grid = _finite(grids.get(grid_names.get(model_name, ""), []))
        selected = row.get("selected_alpha")
        if grid and selected is not None and float(selected) in {min(grid), max(grid)}:
            boundary_hits.add(model_name)
    if boundary_hits:
        warnings.append(f"Selected alpha reached a search-grid boundary for: {', '.join(sorted(boundary_hits))}.")

    invalid_count = sum(int(row.get("invalid_prediction_count", 0)) for row in report.get("results", []))
    if invalid_count:
        warnings.append(f"Invalid predictions were recorded: {invalid_count} rows.")

    grouped_correlations: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in report.get("results", []):
        if row.get("task_type") == "regression" and row.get("predictor_kind") == "model":
            value = row.get("pearson_correlation")
            if value is not None and math.isfinite(float(value)):
                grouped_correlations[
                    (str(row.get("raw_target")), str(row.get("model_name")), str(row.get("feature_family")))
                ].append(float(value))
    reversals = [key for key, values in grouped_correlations.items() if min(values) < 0.0 < max(values)]
    if reversals:
        warnings.append(f"Correlation sign reversals occurred in {len(reversals)} target/model combinations.")

    def contains_oracle_metric(value: object) -> bool:
        if isinstance(value, dict):
            return any(
                ("oracle" in str(key).lower() and not str(key).lower().endswith("included"))
                or contains_oracle_metric(child)
                for key, child in value.items()
            )
        if isinstance(value, list):
            return any(contains_oracle_metric(child) for child in value)
        return False

    if contains_oracle_metric(report):
        warnings.append("Oracle-only metrics are present and are excluded from all labels and rankings.")

    incomplete = [
        row
        for row in [*report.get("final_regression_summary", []), *report.get("final_classification_summary", [])]
        if int(row.get("window_count", 0)) < expected_windows
    ]
    if incomplete:
        warnings.append(f"Missing validation windows affect {len(incomplete)} summarized combinations.")
    if not _representation_rows(report):
        warnings.append("Representation diagnostics were not available in this probe artifact.")
    return warnings


# ============================================================================
# MARKDOWN RENDERING
# ============================================================================


def _number(value: object) -> str:
    """Format one report number compactly for Markdown."""
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.4f}" if math.isfinite(number) else "-"


def build_summary_markdown(report: dict[str, object]) -> str:
    """Render the complete small human-facing summary from one probe report."""
    observed_windows = sorted(
        {
            str(row["validation_window_name"])
            for row in report.get("results", [])
            if row.get("validation_window_name")
        }
    )
    configured_windows = [
        str(row["validation_window_name"])
        for row in report.get("validation_windows", [])
        if row.get("validation_window_name")
    ]
    windows = configured_windows or observed_windows
    expected_windows = len(windows)
    regression = _incremental_rows(report, expected_windows)
    classification = _classification_rows(report, expected_windows)
    representation = _representation_rows(report)
    warnings = _warnings(report, expected_windows)
    lines = [
        "# FI-JEPA Evaluation Summary",
        "",
        "## Run Identity",
        "",
        f"- Checkpoint: `{report.get('checkpoint_id', '-')}`",
        f"- Checkpoint step: {report.get('checkpoint_step', '-')}",
        f"- Representation variant: `{report.get('representation_variant', '-')}`",
        f"- Probe configuration: `{report.get('resolved_probe_config', {})}`",
        f"- Validation windows: {', '.join(windows) if windows else '-'}",
        "",
        "## Representation Diagnostics",
        "",
    ]
    if representation:
        lines.extend(
            [
                "| Window | Effective rank | Matched-train percentile | Pairwise cosine | Mean norm |",
                "|---|---:|---:|---:|---:|",
                *[
                    f"| {row['window']} | {_number(row['effective_rank'])} | {_number(row['rank_percentile'])} | "
                    f"{_number(row['pairwise_cosine'])} | {_number(row['mean_norm'])} |"
                    for row in representation
                ],
            ]
        )
    else:
        lines.append("Representation diagnostics are unavailable for this probe artifact.")
    lines.extend(["", "## Incremental Regression Results", ""])
    if regression:
        lines.extend(
            [
                "| Result | Target | Model | Median RMSE ratio vs hand | Windows improved | Worst ratio |",
                "|---|---|---|---:|---:|---:|",
                *[
                    f"| {row['label']} | {row['target']} | {row['model_name']} | "
                    f"{_number(row['median_rmse_ratio'])} | {row['windows_improved']}/{row['window_count']} | "
                    f"{_number(row['worst_rmse_ratio'])} |"
                    for row in regression
                ],
            ]
        )
    else:
        lines.append("No original-unit incremental regression comparisons were available.")
    lines.extend(["", "## Classification Results", ""])
    if classification:
        lines.extend(
            [
                "| Result | Label | Model / features | Mean AUC | Worst AUC | Mean Brier ratio | Windows improved |",
                "|---|---|---|---:|---:|---:|---:|",
                *[
                    f"| {row['label']} | {row['classification_label']} | {row['model_name']} / "
                    f"{row['feature_family']} | {_number(row['mean_auc'])} | {_number(row['worst_auc'])} | "
                    f"{_number(row['mean_brier_ratio'])} | {row['windows_improved']}/{row['window_count']} |"
                    for row in classification
                ],
            ]
        )
    else:
        lines.append("No classification model results were available.")
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {warning}" for warning in warnings] or ["- None."])
    lines.append("")
    return "\n".join(lines)
