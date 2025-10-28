# Utiliser une image Python 3.11 de base
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Installer TOUTES les dépendances depuis requirements.txt
# Keras 3.10.0 va installer la bonne version de TensorFlow (>=2.16)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .
# Copier le modèle local
COPY 20251016-122103_mobilenetv2.keras .

EXPOSE 8000
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]