import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report, roc_curve

from config import *
from evaluate_model import find_best_threshold, compute_metrics
from mlflow_tracking import configure_mlflow, log_model_with_example, save_and_log_plot

import mlflow

def main():
    # Configuration MLflow
    configure_mlflow(MLFLOW_TRACKING_URI, EXPERIMENT_NAME)

    # Chargement des données
    data_train = pd.read_csv(TRAIN_PATH, compression="zip", index_col=0)
    data_test = pd.read_csv(TEST_PATH, compression="zip", index_col=0)
    TARGET = pd.read_csv(TARGET_PATH, index_col=0).squeeze()

    X = data_train.drop(TARGET_COL, axis=1)
    y = data_train[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y, random_state=42)

    # Réduction de mémoire
    for col in X_train.columns:
        if X_train[col].dtype == "float64":
            X_train[col] = X_train[col].astype(np.float32)
        elif X_train[col].dtype == "int64":
            X_train[col] = X_train[col].astype(np.int32)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, (model, param_grid) in MODELS_CONFIG.items():
        with mlflow.start_run(run_name=model_name):
            pipe = Pipeline([
                ("sampling", SMOTE()),
                ("model", model)
            ])
            print(f"\nModèle : {model_name}")
            start_time = time.time()

            grid = GridSearchCV(pipe, param_grid, scoring='roc_auc', cv=cv, n_jobs=1)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_

            y_train_proba = best_model.predict_proba(X_train)[:, 1]
            y_test_proba = best_model.predict_proba(X_test)[:, 1]
            threshold = find_best_threshold(y_test, y_test_proba, COUT_FN, COUT_FP)

            # Métriques
            metrics_test = compute_metrics(y_test, y_test_proba, threshold)
            metrics_train = compute_metrics(y_train, y_train_proba, threshold)
            training_time = time.time() - start_time

            # Log MLflow
            mlflow.log_param("Best Threshold", threshold)
            mlflow.log_metric("Training_Time_s", training_time)
            for k, v in metrics_train.items():
                mlflow.log_metric(f"{k} Train", v)
            for k, v in metrics_test.items():
                mlflow.log_metric(f"{k} Test", v)

            # Modèle
            log_model_with_example(best_model, model_name, X_train)

            # Matrice de confusion
            y_test_pred = (y_test_proba >= threshold).astype(int)
            conf_matrix = confusion_matrix(y_test, y_test_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title(f"{model_name} - Matrice de Confusion")
            save_and_log_plot(plt.gcf(), model_name, "confusion_matrix")

            # Distribution des scores
            plt.figure(figsize=(10, 6))
            sns.histplot(y_test_proba, bins=50, kde=True, color='skyblue')
            plt.title(f"{model_name} - Distribution des Scores")
            save_and_log_plot(plt.gcf(), model_name, "score_distribution")

            # Courbe ROC
            fpr, tpr, _ = roc_curve(y_test, y_test_proba)
            plt.figure(figsize=(10, 6))
            plt.plot(fpr, tpr, label="ROC curve")
            plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
            plt.title(f"{model_name} - ROC Curve")
            save_and_log_plot(plt.gcf(), model_name, "roc_curve")

            # Classification Report
            print(classification_report(y_test, y_test_pred, digits=3))

if __name__ == "__main__":
    main()