# scripts/register_from_minio.py
import os, boto3, mlflow, tempfile
import tensorflow as tf

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5050")
ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123")

BUCKET = "mlflow-artifacts"
OBJECT_KEY = "20251016-122103_mobilenetv2.keras"   # <-- le nom que tu vois dans MinIO
REGISTERED_NAME = "florisense_model"
EXPERIMENT_NAME = "florisens"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

s3 = boto3.client("s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id=AWS_KEY,
    aws_secret_access_key=AWS_SECRET,
    region_name=os.getenv("AWS_REGION","us-east-1"),
)

with tempfile.TemporaryDirectory() as td:
    local_path = os.path.join(td, "model.keras")
    s3.download_file(BUCKET, OBJECT_KEY, local_path)
    model = tf.keras.models.load_model(local_path)

    with mlflow.start_run(run_name="import-from-minio"):
        # Log + register en une seule passe
        mlflow.tensorflow.log_model(
            model,
            artifact_path="model",
            registered_model_name=REGISTERED_NAME
        )
        # Optionnel : quelques tags/params
        mlflow.set_tag("source", "minio")
        mlflow.log_param("file", OBJECT_KEY)

print("✅ Modèle enregistré dans le registry sous le nom:", REGISTERED_NAME)