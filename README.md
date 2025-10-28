# **FloriSense – MLOps Project: Image Classification**

## **Objectif**

**FloriSense** est une application MLOps complète permettant de classifier des images de plantes — en distinguant les **pissenlits** (mauvaises herbes) de **l’herbe saine** — à l’aide d’un modèle **Deep Learning (MobileNetV2)** déployé via **FastAPI** et suivi avec **MLflow**.

L’objectif est d’illustrer **tout le cycle de vie d’un modèle ML** :

➡️ du développement au déploiement, en passant par le suivi, la reproductibilité et l’observabilité.

---

## **Fonctionnalités principales**

| **Composant** | **Description** | **Technologie** |
| --- | --- | --- |
| **API de Prédiction** | Endpoint /predict recevant une image et retournant une prédiction JSON (classe + score de confiance). | FastAPI |
| **Interface Web** | Application statique pour tester les prédictions en direct. | Nginx + HTML/JS |
| **Suivi des Modèles** | Suivi des expériences, runs et artefacts. | MLflow |
| **Stockage d’Artefacts** | Stockage S3-compatible des modèles et artefacts MLflow. | Minio |
| **Orchestration** | Lancement complet via un simple docker-compose up. | Docker Compose |
| **Tracking et versioning** | Expériences versionnées automatiquement dans MLflow et artefacts stockés sur Minio. | MLflow + S3 |

## **Architecture et choix techniques**

L’infrastructure est composée de **microservices indépendants** gérés par docker-compose :

### **1. API (FastAPI)**

- Sert le modèle Keras (MobileNetV2).
- Documentée automatiquement via /docs.
- Chargement du modèle au démarrage avec lifespan FastAPI pour éviter les race conditions.
- **Choix technique :** FastAPI pour sa rapidité, sa gestion asynchrone et son intégration aisée avec TensorFlow/Keras.

### **2. Web (Nginx)**

- Sert le fichier statique florisense_webapp.html et l’interface utilisateur.

### **3. MLflow Server**

- Serveur de tracking pour les expériences et modèles.
- **Choix technique :** Dockerfile MLflow personnalisé incluant boto3 pour la compatibilité S3 (Minio).

### **4. Minio Server**

- Stockage S3 local des artefacts MLflow (mlflow-artifacts).
- Accès UI : http://localhost:9001.

### **5. Minio Client (mc)**

- Script de démarrage automatique créant le bucket mlflow-artifacts.

---

## **Défi technique résolu : compatibilité**

## **Keras 3**

Le principal challenge du projet a été le **chargement du modèle .keras (format Keras 3)** dans un environnement Dockerisé.

### **Problèmes rencontrés**

- Erreurs ModuleNotFoundError ou TypeError :
    
    ex. keras.src.models.functional cannot be imported
    
    → dues à une incompatibilité entre les versions de Keras lors du chargement.
    

### **Solutions adoptées**

✅ Base python:3.11-slim pour un contrôle total.

✅ Versions fixées dans requirements.txt :

## **Installation et lancement**

### **Prérequis**

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)
- [Git](https://git-scm.com/)

## **Utilisation**

### **Interface Web**

- Accès : [http://localhost:8080](http://localhost:8080/)
- Chargez une image → cliquez sur **Obtenir le Verdict**.

### **API Swagger**

- Accès : http://localhost:8000/docs
- Testez directement /predict via l’interface interactive.

### **MLflow UI**

- Accès : [http://localhost:5050](http://localhost:5050/)
- Suivez vos expériences, runs et modèles.

### **Minio UI**

- Accès : [http://localhost:9001](http://localhost:9001/)
- Identifiants :
    - **Login:** minioadmin
    - **Password:** minioadmin123
- Bucket : mlflow-artifacts

---

## **Structure du projet**

florisense-mlops/

├── 20251016-122103_mobilenetv2.keras      # Modèle Keras 3

├── Dockerfile                             # Image FastAPI + TensorFlow

├── docker-compose.yml                     # Orchestration des services

├── florisense_webapp.html                 # Interface utilisateur

├── [main.py](http://main.py/)                                # API FastAPI

├── requirements.txt                       # Dépendances Python

├── upload_keras_model_to_mlflow.py        # Enregistrement du modèle sur MLflow

├── log_model.py                           # Script de logging MLflow

├── .gitignore                             # Fichiers ignorés

└── state/                                 # Dossier de stockage MLflow local

**Écosystème MLOps intégré**

| **Étape** | **Outil** | **Rôle** |
| --- | --- | --- |
| **Versioning & CI/CD** | GitHub + Docker Compose | Reproductibilité |
| **Tracking** | MLflow | Suivi des modèles et métriques |
| **Stockage d’artefacts** | Minio (S3 local) | Stockage des modèles |
| **Serving** | FastAPI | API RESTful |
| **Infrastructure** | Docker | Isolation et portabilité |
| **Monitoring (à venir)** | Grafana / Prometheus (optionnel) | Suivi des métriques API & modèles |

## **Perspectives d’évolution (Roadmap)**

- Déploiement sur **Kubernetes / Minikube** avec Helm Charts.
- Ajout d’un pipeline de **retraining automatique (Airflow / Prefect)**.
- Monitoring complet (Prometheus + Grafana).
- Tests automatisés et CI/CD GitHub Actions.
- Intégration d’un **Feature Store (Feast)** pour la reproductibilité.
