import os
import pandas as pd
import time
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping  # <--- IMPORT IMPORTANT

def main():
    base_path = "data"

    # Charger les données
    app_train_df = pd.read_csv(os.path.join(base_path, "application_train.csv"))
    app_test_df = pd.read_csv(os.path.join(base_path, "application_test.csv"))

    # Filtrer les colonnes utiles
    feats = [f for f in app_train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    app_train_df = app_train_df[feats]
    app_test_df = app_test_df[feats]

    # Auto-détection des types
    numerical = app_train_df.select_dtypes(exclude='object').columns.tolist()
    categorical = app_train_df.select_dtypes(include='object').columns.tolist()

    # Créer le mapping des colonnes avec ColumnMapping
    column_mapping = ColumnMapping()
    column_mapping.numerical_features = numerical
    column_mapping.categorical_features = categorical

    # Créer et exécuter le rapport
    start = time.time()
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=app_train_df,
        current_data=app_test_df,
        column_mapping=column_mapping  # <-- objet ColumnMapping ici
    )
    report.save_html("data_drift_analysis.html")
    print(f"Rapport généré en {time.time() - start:.2f} secondes.")

if __name__ == "__main__":
    main()