FROM python:3.11-slim

# Installer mlflow et dépendances
RUN pip install mlflow sqlalchemy

# Créer dossier de travail
WORKDIR /app

# Créer dossier artifacts pour stocker modèles (volume persistant)
RUN mkdir artifacts

# Expose le port MLflow
EXPOSE 5000

# Commande pour lancer MLflow server
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "./artifacts", \
     "--host", "0.0.0.0", "--port", "5000"]