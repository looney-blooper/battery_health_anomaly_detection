"""
To run this pipelise, use this command, 
format :
python -m pipelines.evaluation_pipeline \
--battery_csv <csv_file_path> \
--model_type <model_type> \
--window_size <window_size> \
--n_features <n_features> \

optional:
--lr <learning_rate> \
--epochs <num_epochs> \

for example:
python -m pipelines.evaluation_pipeline \
--battery_csv data/raw/regular_alt_batteries/battery10.csv \
--model_type LSTMDeepAutoEncoder \
--window_size 60 \
--n_features 4 \
"""


import argparse
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
    print(f"Training model type: {model_type}")
    model_path = train_model(
        windows=windows,
        model_type=model_type,
        window_size=window_size,
        n_features=n_features,
        epochs=epochs,
        lr=lr,
    )
    metric = evaluate_model(model_path, windows, model_type, window_size, n_features)


    register_model(
        model_path,
        model_name=f"battery_health_{model_type}"
    )

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--model_type", required=True, choices=["DenseAutoEncoder", "LSTM2LayerAutoencoder", "LSTMDeepAutoEncoder"])
    parser.add_argument("--window_size", type=int, required=True)
    parser.add_argument("--n_features", type=int, required=True)

    args = parser.parse_args()

    training_pipeline(
        csv_path=args.csv_path,
        model_type=args.model_type,
        window_size=args.window_size,
        n_features=args.n_features,
    )