import mlflow
import torch
from pathlib import Path

MODEL_NAME = "battery_health_autoencoder"
MODEL_STAGE = "Production"

LOCAL_MODEL_PATH = Path("edge_models/base_model.pt")

def sync_model():

    print("Syncing model from MLflow registry...")

    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

    model = mlflow.pytorch.load_model(model_uri)

    LOCAL_MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), LOCAL_MODEL_PATH)

    print(f"Model synced and saved to {LOCAL_MODEL_PATH}")


if __name__ == "__main__":
    sync_model()
