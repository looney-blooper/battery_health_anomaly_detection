from steps. ingest_data import ingest_data
from steps. preprocess import preprocess_data
from steps. train_model import train_model
from steps. evaluate_model import evaluate_model
from steps. register_model import register_model

from zenml import pipeline

@pipeline
def training_pipeline(
    csv_path: str,
    model_type: str,
    window_size: int,
    n_features: int,
    epochs: int = 50,
    lr: float = 1e-3,
):
    windows = preprocess_data(csv_path=csv_path)
    model = train_model(
        windows=windows,
        model_type=model_type,
        window_size=window_size,
        n_features=n_features,
        epochs=epochs,
        lr=lr,
    )
    metric = evaluate_model(model, windows)

    register_model(
        model,
        model_name="battery_health_autoencoder"
    )

    