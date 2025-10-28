# main.py
import os
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import numpy as np
# --- Ne pas importer mlflow ou mlflow.keras ici pour le moment ---
# import mlflow
# import mlflow.keras 
import tensorflow as tf # <-- Assure-toi que tensorflow est importé
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import traceback

# --- Configuration (on n'utilise plus MLflow URI ici) ---
# MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5050")
# MODEL_URI = os.getenv("MODEL_URI", "models:/florisense_model/1")
LOCAL_MODEL_FILENAME = "20251016-122103_mobilenetv2.keras" # <-- Nom du fichier local
ml_models = {}

# --- Fonction Lifespan MODIFIÉE ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("----- DÉMARRAGE DE L'APPLICATION API -----")
    # print(f"Tentative de connexion à MLflow: {MLFLOW_TRACKING_URI}") # Pas besoin ici
    # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI) # Pas besoin ici
    
    print(f"Tentative de chargement direct du modèle Keras: {LOCAL_MODEL_FILENAME}")
    try:
        # --- CHANGEMENT PRINCIPAL: Charger directement depuis le fichier ---
        ml_models["florisense"] = tf.keras.models.load_model(LOCAL_MODEL_FILENAME)
        print("✅ Modèle Keras chargé directement avec succès.")
        # --- FIN DU CHANGEMENT ---
        
    except Exception as e:
        print("❌ ERREUR CRITIQUE PENDANT LE CHARGEMENT DIRECT DU MODÈLE:")
        print(f"   Type d'erreur: {type(e).__name__}")
        print(f"   Message: {e}")
        print("   Traceback complet:")
        traceback.print_exc() 
        
    print("----- Fin de la phase de démarrage (lifespan) -----")
    yield 
    
    print("----- Arrêt de l'application API -----")
    ml_models.clear()

# --- Reste du code (inchangé) ---
app = FastAPI(lifespan=lifespan)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    model_loaded = "florisense" in ml_models
    print(f"[/health] Vérification: Modèle chargé = {model_loaded}")
    # On met le nom de fichier local dans la réponse pour info
    return {"ok": True, "model_file": LOCAL_MODEL_FILENAME, "model_loaded": model_loaded} 

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print("[/predict] Requête reçue.")
    model = ml_models.get("florisense")
    
    if not model:
        print("❌ [/predict] Erreur: Modèle non chargé.")
        return {"error": "Le modèle n'est pas chargé. Vérifiez les logs de l'API."}, 503

    try: 
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB").resize((224, 224))
        x = np.array(img, dtype=np.float32) / 255.0
        
        print("[/predict] Image prétraitée, lancement de la prédiction...")
        preds = model.predict(x[None, ...]) 
        print(f"[/predict] Prédiction brute obtenue: {preds}")
        

        labels = ["pissenlit", "herbe"]
        confidence = float(np.max(preds[0]))
        predicted_class_index = int(np.argmax(preds[0]))
        prediction_label = labels[predicted_class_index]
        
        print(f"[/predict] Résultat: {prediction_label} (Confiance: {confidence:.2f})")
        return {
            "prediction": prediction_label,
            "confidence": confidence
        }
    except Exception as e:
        print(f"❌ [/predict] Erreur pendant la prédiction: {e}")
        traceback.print_exc()
        return {"error": f"Erreur pendant la prédiction: {e}"}, 500