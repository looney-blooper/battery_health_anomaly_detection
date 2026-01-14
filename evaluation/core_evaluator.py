"""
Core Single-Battery Offline Evaluation Logic

PURE LOGIC ONLY.
- No file I/O
- No model loading
- No MLflow
- No ZenML

This logic is reused by:
- ZenML evaluation pipeline
- Reference scripts
"""

import numpy as np


def evaluate_single_battery(
    window_errors: np.ndarray,
    weeks_per_month: int = 4,
    green_sigma: float = 1.0,
    red_sigma: float = 3.0,
):
    """
    Parameters
    ----------
    window_errors : np.ndarray
        Reconstruction error per window (ordered by time)

    Returns
    -------
    logs : list[dict]
        Weekly evaluation timeline
    """

    windows_per_week = len(window_errors) // max(1, len(window_errors) // 12)
    weeks = np.array_split(
        window_errors,
        len(window_errors) // windows_per_week
    )

    baseline_mean = None
    baseline_std = None
    baseline_p95 = None

    previous_week_red = False
    red_in_current_month = False

    logs = []

    for week_idx, week_errors in enumerate(weeks):
        month_idx = week_idx // weeks_per_month

        weekly_mean = np.mean(week_errors)
        weekly_p95 = np.percentile(week_errors, 95)

        decision = "🟢"

        if baseline_mean is not None:
            is_red = weekly_mean > baseline_mean + red_sigma * baseline_std
            is_yellow = weekly_mean > baseline_mean + green_sigma * baseline_std

            if is_red and previous_week_red:
                decision = "🔴"
                red_in_current_month = True
            elif is_red or is_yellow:
                decision = "🟡"

            previous_week_red = is_red
        else:
            decision = "🟢"  # warm-up month

        logs.append({
            "week_index": week_idx,
            "month_index": month_idx,
            "weekly_mean_error": weekly_mean,
            "weekly_p95_error": weekly_p95,
            "baseline_mean": baseline_mean,
            "baseline_p95": baseline_p95,
            "decision": decision,
        })

        # Month boundary
        if (week_idx + 1) % weeks_per_month == 0:
            month_errors = np.concatenate(
                weeks[week_idx - weeks_per_month + 1 : week_idx + 1]
            )

            if not red_in_current_month:
                baseline_mean = np.mean(month_errors)
                baseline_std = np.std(month_errors)
                baseline_p95 = np.percentile(month_errors, 95)

            previous_week_red = False
            red_in_current_month = False

    return logs
