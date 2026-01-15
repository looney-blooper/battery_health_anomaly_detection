from zenml import pipeline
from steps.offline_evaluate import offline_evaluate


@pipeline
def evaluation_pipeline(
    battery_csv: str,
    model_type: str,
    window_size: int,
    n_features: int,
    model_weights_path: str,
):
    offline_evaluate(
        battery_csv=battery_csv,
        model_type=model_type,
        window_size=window_size,
        n_features=n_features,
        model_weights_path=model_weights_path,
    )
