from zenml import step
import numpy as np
import pandas as pd
import torch
import mlflow
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path

from models.factory import get_model
from data_preprocessing.load_and_clean import load_and_clean
from data_preprocessing.extract_window import extract_lstm_windows
from evaluation.core_evaluator import evaluate_single_battery


@step
def offline_evaluate(
    battery_csv: str,
    model_type: str,
    window_size: int,
    n_features: int,
    model_weights_path: str,
    weeks_per_month: int = 4,
    green_sigma: float = 1.0,
    red_sigma: float = 3.0,
):
    """
    Runs single-battery offline evaluation using the shared core evaluator.
    Logs metrics, CSV timeline, and plots to MLflow.
    """

    # -------------------------
    # Load + preprocess
    # -------------------------
    df = load_and_clean(battery_csv)
    windows = extract_lstm_windows(df)
    X = torch.tensor(windows, dtype=torch.float32)

    # -------------------------
    # Load model (candidate)
    # -------------------------
    model = get_model(
        model_type,
        window_size=window_size,
        n_features=n_features,
    )
    model.load_state_dict(torch.load(model_weights_path, map_location="cpu", weights_only=True))
    model.eval()

    # -------------------------
    # Reconstruction errors
    # -------------------------
    with torch.no_grad():
        recon = model(X)
        errors = torch.mean((recon - X) ** 2, dim=(1, 2)).numpy()

    # -------------------------
    # Core evaluation (ONE source of truth)
    # -------------------------
    logs = evaluate_single_battery(
        errors,
        weeks_per_month=weeks_per_month,
        green_sigma=green_sigma,
        red_sigma=red_sigma,
    )

    timeline = pd.DataFrame(logs)

    # -------------------------
    # System-level metrics
    # -------------------------
    false_reds = (timeline["decision"] == "🔴").sum()
    first_yellow = timeline.index[timeline["decision"] == "🟡"].min()
    first_red = timeline.index[timeline["decision"] == "🔴"].min()

    mlflow.log_metric("false_red_count", int(false_reds))
    mlflow.log_metric(
        "weeks_to_first_yellow",
        int(first_yellow) if pd.notna(first_yellow) else -1
    )
    mlflow.log_metric(
        "weeks_to_first_red",
        int(first_red) if pd.notna(first_red) else -1
    )
    mlflow.log_metric(
        "mean_weekly_error",
        float(timeline["weekly_mean_error"].mean())
    )
    mlflow.log_metric(
        "max_weekly_error",
        float(timeline["weekly_mean_error"].max())
    )

    # -------------------------
    # Log CSV artifact
    # -------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "weekly_timeline.csv"
        timeline.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, artifact_path="evaluation")

    # -------------------------
    # Plot timeline (MLflow)
    # -------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        plot_path = Path(tmpdir) / "weekly_error_timeline.png"

        plt.figure(figsize=(10, 4))
        plt.plot(
            timeline["week_index"],
            timeline["weekly_mean_error"],
            label="Weekly Mean",
        )
        plt.plot(
            timeline["week_index"],
            timeline["weekly_p95_error"],
            label="Weekly P95",
        )
        plt.xlabel("Week")
        plt.ylabel("Reconstruction Error")
        plt.title("Weekly Battery Error Timeline")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        mlflow.log_artifact(plot_path, artifact_path="evaluation_plots")

    return timeline
