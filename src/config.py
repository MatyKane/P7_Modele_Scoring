import os

# Fichiers de données
TRAIN_PATH = "data_train_final.zip"
TEST_PATH = "data_test_final.zip"
TARGET_PATH = "TARGET.csv"

# Colonne cible
TARGET_COL = "TARGET"

# MLflow
MLFLOW_TRACKING_URI = f"file:///{os.path.abspath('./mlruns').replace(os.sep, '/')}"
EXPERIMENT_NAME = "Classification_Modeles"

# Coût métier
COUT_FN = 10
COUT_FP = 1

# Modèles à tester
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

MODELS_CONFIG = {
    'LogisticRegression': (LogisticRegression(max_iter=1000), {
        'model__C': [0.01, 1, 10],
        'model__solver': ['lbfgs']
    }),
    'RandomForest': (RandomForestClassifier(), {
        'model__n_estimators': [50, 100],
        'model__max_depth': [None, 5, 10],
        'model__min_samples_split': [2, 5],
        'model__min_samples_leaf': [1, 2],
        'model__max_features': ['sqrt']
    }),
    'XGBoost': (XGBClassifier(eval_metric='logloss'), {
        'model__n_estimators': [50, 100],
        'model__max_depth': [3, 6],
        'model__learning_rate': [0.1, 0.2],
        'model__subsample': [1.0],
        'model__colsample_bytree': [0.7]
    }),
    'LightGBM': (LGBMClassifier(), {
        'model__n_estimators': [50, 100],
        'model__num_leaves': [31, 50],
        'model__learning_rate': [0.1, 0.2],
        'model__subsample': [1.0],
        'model__colsample_bytree': [0.7]
    })
}