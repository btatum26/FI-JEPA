from __future__ import annotations

from fi_jepa.probes.summary import (
    RESULT_LABELS,
    build_summary_markdown,
    classify_summary_result,
)


# ============================================================================
# RESULT LABELS
# ============================================================================


def test_summary_labels_use_only_the_five_allowed_outcomes() -> None:
    labels = {
        classify_summary_result(
            "regression",
            window_count=3,
            windows_improved=3,
            primary_metric=0.8,
            worst_metric=1.02,
        ),
        classify_summary_result(
            "regression",
            window_count=3,
            windows_improved=2,
            primary_metric=0.95,
            worst_metric=1.3,
            stable_direction=False,
        ),
        classify_summary_result(
            "regression",
            window_count=3,
            windows_improved=0,
            primary_metric=1.2,
            worst_metric=1.4,
        ),
        classify_summary_result(
            "classification",
            window_count=3,
            windows_improved=2,
            primary_metric=0.57,
            worst_metric=0.51,
        ),
        classify_summary_result(
            "classification",
            window_count=3,
            windows_improved=1,
            primary_metric=0.6,
            worst_metric=0.45,
        ),
        classify_summary_result(
            "classification",
            window_count=0,
            windows_improved=0,
            primary_metric=0.5,
            worst_metric=0.5,
        ),
    }

    assert labels == RESULT_LABELS


# ============================================================================
# SUMMARY RENDERING
# ============================================================================


def _report() -> dict[str, object]:
    windows = ("first", "second", "third")
    results: list[dict[str, object]] = []
    for index, window in enumerate(windows):
        results.extend(
            [
                {
                    "validation_window_name": window,
                    "task_type": "regression",
                    "raw_target": "future_return_21d",
                    "model_name": "ridge",
                    "feature_family": "hand_market_features_plus_residual_z",
                    "score_space": "original",
                    "predictor_kind": "model",
                    "pearson_correlation": (0.2, 0.3, 0.1)[index],
                    "invalid_prediction_count": 0,
                },
                {
                    "validation_window_name": window,
                    "task_type": "classification",
                    "classification_label": "positive_return",
                    "model_name": "class_prior",
                    "feature_family": "constant",
                    "predictor_name": "class_prior",
                    "predictor_kind": "baseline",
                    "roc_auc": 0.5,
                    "brier_score": 0.2,
                    "invalid_prediction_count": 0,
                },
                {
                    "validation_window_name": window,
                    "task_type": "classification",
                    "classification_label": "positive_return",
                    "model_name": "logistic",
                    "feature_family": "z_only",
                    "predictor_name": "logistic__z_only",
                    "predictor_kind": "model",
                    "roc_auc": (0.7, 0.65, 0.6)[index],
                    "brier_score": (0.1, 0.12, 0.15)[index],
                    "invalid_prediction_count": 0,
                },
            ]
        )
    # Separate rows exercise invalid-prediction and sign-reversal warnings.
    results.extend(
        [
            {
                "validation_window_name": "first",
                "task_type": "regression",
                "raw_target": "future_realized_vol_21d",
                "model_name": "huber",
                "feature_family": "z_only",
                "predictor_kind": "model",
                "pearson_correlation": -0.2,
                "invalid_prediction_count": 1,
            },
            {
                "validation_window_name": "second",
                "task_type": "regression",
                "raw_target": "future_realized_vol_21d",
                "model_name": "huber",
                "feature_family": "z_only",
                "predictor_kind": "model",
                "pearson_correlation": 0.2,
                "invalid_prediction_count": 0,
            },
        ]
    )
    rank_metrics = {
        "effective_rank": {"validation_value": 8.5, "validation_percentile": 40.0},
        "mean_pairwise_cosine": {"validation_value": 0.12},
        "mean_vector_norm": {"validation_value": 3.4},
    }
    return {
        "checkpoint_id": "step_11000",
        "checkpoint_step": 11000,
        "representation_variant": "pooled_pca_16",
        "resolved_probe_config": {
            "ridge_alphas": [0.1, 1.0],
            "huber_alphas": [0.01, 0.1],
            "elastic_net_alphas": [0.01, 0.1],
            "logistic_alphas": [0.1, 1.0],
        },
        "parameter_selection_by_fold": [
            {"model_name": "ridge", "selected_alpha": 0.1},
        ],
        "results": results,
        "incremental_hand_plus_z_comparisons": [
            {
                "validation_window_name": window,
                "raw_target": "future_return_21d",
                "model_name": "ridge",
                "score_space": "original",
                "rmse_ratio_vs_hand": ratio,
            }
            for window, ratio in zip(windows, (0.8, 0.9, 1.05), strict=True)
        ],
        "stability_summary": [
            {
                "task_type": "regression",
                "raw_target": "future_return_21d",
                "model_name": "ridge",
                "feature_family": "hand_market_features_plus_residual_z",
                "bootstrap_interval_95": [-0.2, -0.01],
            }
        ],
        "final_regression_summary": [{"window_count": 2}],
        "final_classification_summary": [{"window_count": 3}],
        "representation_diagnostics": {
            "representations": {
                "selected_pca_representation": {
                    "windows": [
                        {"validation_window_name": window, "metrics": rank_metrics}
                        for window in windows
                    ]
                }
            }
        },
        "oracle_validation_recalibration": {"slope": 999.0},
    }


def test_summary_contains_required_sections_rankings_metrics_and_warnings() -> None:
    summary = build_summary_markdown(_report())

    assert "## Run Identity" in summary
    assert "## Representation Diagnostics" in summary
    assert "## Incremental Regression Results" in summary
    assert "## Classification Results" in summary
    assert "## Warnings" in summary
    assert "SUPPORTED | future_return_21d | ridge | 0.9000 | 2/3 | 1.0500" in summary
    assert "SUPPORTED | positive_return | logistic / z_only | 0.6500 | 0.6000" in summary
    assert "Selected alpha reached a search-grid boundary" in summary
    assert "Invalid predictions were recorded" in summary
    assert "Correlation sign reversals occurred" in summary
    assert "Oracle-only metrics are present" in summary
    assert "Missing validation windows affect" in summary
    assert "999" not in summary
