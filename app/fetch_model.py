"""Build-time: download the latest registered digits-rf model from DagsHub into ./model.

Auth via MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD (DagsHub token).
"""
import mlflow

mlflow.set_tracking_uri("https://dagshub.com/ChenAuCarre/mlflow-with-dagshub.mlflow")
client = mlflow.MlflowClient()
latest = max(client.search_model_versions("name='digits-rf'"), key=lambda v: int(v.version))
print(f"Fetching digits-rf version {latest.version}")
mlflow.artifacts.download_artifacts(artifact_uri=latest.source, dst_path="model")
