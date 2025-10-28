import os, io, json
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import mlflow
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf

MODEL_URI = os.getenv("MODEL_URI", "models:/florisense_model/1")
CLASSES = [c.strip() for c in os.getenv("CLASSES","").split(",") if c.strip()]
TOP_K = int(os.getenv("TOP_K","5"))

app = FastAPI()
model = None
input_size = (224, 224)

def load():
    global model
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","http://localhost:5050"))
    model = mlflow.keras.load_model(MODEL_URI)

def prepare(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(input_size)
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, 0)
    return preprocess_input(x)

@app.on_event("startup")
def _startup():
    load()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image")
    x = prepare(img)
    y = model.predict(x)
    y = np.squeeze(y).astype(float)
    if y.ndim == 0:
        y = np.array([float(y)])
    if CLASSES and len(CLASSES) == y.shape[-1]:
        idx = np.argsort(y)[::-1][:TOP_K]
        preds = [{"label": CLASSES[i], "prob": float(y[i])} for i in idx]
        return JSONResponse({"model_uri": MODEL_URI, "top_k": preds})
    else:
        idx = np.argsort(y)[::-1][:TOP_K]
        preds = [{"index": int(i), "prob": float(y[i])} for i in idx]
        return JSONResponse({"model_uri": MODEL_URI, "top_k": preds})

@app.get("/health")
def health():
    ok = model is not None
    return {"ok": ok, "model_uri": MODEL_URI}