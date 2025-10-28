# Partir de l'image MLflow officielle
FROM ghcr.io/mlflow/mlflow:v2.16.0

# Passer en root pour pouvoir installer des packages
USER root

# Installer boto3, la dépendance manquante pour Minio/S3
RUN pip install boto3

# Revenir à l'utilisateur non-root par défaut de l'image
USER 1001