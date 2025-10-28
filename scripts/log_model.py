# scripts/log_model.py
import mlflow
import mlflow.keras
import os

# MLflow côté réseau docker
mlflow.set_tracking_uri("http://mlflow:5050")

EXPERIMENT = "florisense-s3"
mlflow.set_experiment(EXPERIMENT)

# Chemin du modèle Keras déjà entraîné (adapte au tien)
MODEL_PATH = "/app/20251016-122103_mobile--v2.keras"  # <- mets le bon chemin dans l'image

with mlflow.start_run(run_name="upload_keras_model"):
    # Log le modèle Keras
    model_info = mlflow.keras.log_model(
        keras_model=MODEL_PATH,
        artifact_path="model",
        registered_model_name="florisense_model",
        input_example=None,
    )
    print("Registered model:", model_info.model_uri)