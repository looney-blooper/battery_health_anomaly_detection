import mlflow

MODEL_NAME = "battery_health_autoencoder"
VERSION = "3"   # example
STAGE = "Staging"  # or "Production"

client = mlflow.tracking.MlflowClient()

client.transition_model_version_stage(
    name=MODEL_NAME,
    version=VERSION,
    stage=STAGE,
    archive_existing_versions=True
)

print(f"Model {MODEL_NAME} v{VERSION} promoted to {STAGE}")
