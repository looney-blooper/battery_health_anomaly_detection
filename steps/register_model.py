from zenml import step
import mlflow


@step
def register_model(
    model_path: str,
    model_name: str,
):
    """
    Registers trained model to MLflow Model Registry.
    """

    mlflow.log_artifact(model_path, artifact_path="model")

    result = mlflow.register_model(
        model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
        name=model_name,
    )

    return result.name
