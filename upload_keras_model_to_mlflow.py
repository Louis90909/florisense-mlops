# upload_keras_model_to_mlflow.py
import mlflow
import mlflow.keras  # <--- CHANGEMENT 1 : Importer la saveur Keras
import tensorflow as tf
import os

# --- Configuration ---
MLFLOW_TRACKING_URI = "http://localhost:5050"
LOCAL_MODEL_PATH = "20251016-122103_mobilenetv2.keras" # Le fichier .keras que tu m'as donné
EXPERIMENT_NAME = "florisense-s3"
REGISTERED_MODEL_NAME = "florisense_model" # LE NOM EXACT que ton API attend !
# ---------------------

# 1. Se connecter au serveur MLflow qui tourne dans Docker
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
exp_id = mlflow.set_experiment(EXPERIMENT_NAME).experiment_id

if not os.path.exists(LOCAL_MODEL_PATH):
    print(f"Erreur: Fichier modèle non trouvé à l'emplacement: {LOCAL_MODEL_PATH}")
    exit()

print(f"Chargement du modèle local depuis : {LOCAL_MODEL_PATH}")
model = tf.keras.models.load_model(LOCAL_MODEL_PATH)
print("Modèle local chargé.")

# 2. Démarrer un run ET logger le modèle
with mlflow.start_run(experiment_id=exp_id, run_name="Upload Production Model MobileNetV2") as run:
    print(f"Run démarré (ID: {run.info.run_id}) dans l'expériment {EXPERIMENT_NAME}")
    
    # Logue tes paramètres et métriques d'origine
    mlflow.log_param("model_architecture", "MobileNetV2")
    mlflow.log_metric("val_accuracy", 0.92)
    mlflow.log_metric("val_loss", 0.18)
    mlflow.set_tag("status", "production-candidate")

# 3. L'ÉTAPE CRUCIALE : Logger le modèle
    print("Enregistrement du modèle dans MLflow (avec la saveur Keras)...")
    
    # --- CHANGEMENT 2 : Utiliser mlflow.keras.log_model ---
    mlflow.keras.log_model(
        model,
        artifact_path="model", # Nom du dossier dans les artefacts du run
        
        # Ceci crée la "Version 1" du modèle "florisense_model"
        registered_model_name=REGISTERED_MODEL_NAME
        
        # On n'a plus besoin de 'save_format'. 
        # mlflow.keras est assez malin pour Keras 3
    )
    # --- FIN DE LA CORRECTION ---
    
    print(f"Modèle enregistré et versionné avec succès sous le nom: {REGISTERED_MODEL_NAME}")

print("Upload terminé.")