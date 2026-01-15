"""
To run this pipelise, use this command, 
format :
python -m pipelines.evaluation_pipeline \
--battery_csv <csv_file_path> \
--model_type <model_type> \
--window_size <window_size> \
--n_features <n_features> \
--model_weights_path <path_where_model_weights_stored>

for example:
python -m pipelines.evaluation_pipeline \
--battery_csv data/raw/regular_alt_batteries/battery10.csv \
--model_type LSTMDeepAutoEncoder \
--window_size 60 \
--n_features 4 \
--model_weights_path trained_models/LSTMDeepAutoEncoder_model.pt
"""


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
