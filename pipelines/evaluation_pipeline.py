import argparse

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--battery_csv", required=True)
    parser.add_argument(
        "--model_type",
        required=True,
        choices=["DenseAutoEncoder", "LSTM2LayerAutoencoder", "LSTMDeepAutoEncoder"],
    )
    parser.add_argument("--window_size", type=int, required=True)
    parser.add_argument("--n_features", type=int, required=True)
    parser.add_argument("--model_weights_path", required=True)

    args = parser.parse_args()

    evaluation_pipeline(
        battery_csv=args.battery_csv,
        model_type=args.model_type,
        window_size=args.window_size,
        n_features=args.n_features,
        model_weights_path=args.model_weights_path,
    )

    # To run the pipeline, use the command line:
    