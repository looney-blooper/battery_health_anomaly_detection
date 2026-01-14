from zenml import step
import mlflow
import torch
from pathlib import Path

@step
def register_model(
    model,
    model_name: str,
):
    """
    Registers trained model to MLflow Model Registry.
    """
    with mlflow.start_run(nested=True):
        # Save model temporarily
        tmp_path = Path("tmp_model")
        tmp_path.mkdir(exist_ok=True)

        model_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), model_path)

        mlflow.log_artifact(model_path, artifact_path="model")

        result = mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
            name=model_name
        )

    return result.name
