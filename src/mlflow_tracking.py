import mlflow
import os
import tempfile
import matplotlib.pyplot as plt

def configure_mlflow(uri, experiment_name):
    os.makedirs(uri.replace("file:///", ""), exist_ok=True)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)

def log_model_with_example(model, model_name, X_train):
    input_example = X_train.iloc[:1]
    mlflow.sklearn.log_model(model, model_name, input_example=input_example)

def save_and_log_plot(fig, model_name, plot_name):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        fig.savefig(tmpfile.name, bbox_inches="tight")
        mlflow.log_artifact(tmpfile.name, artifact_path=f"plots/{model_name}/{plot_name}")
        plt.close(fig)
        os.remove(tmpfile.name)


def log_feature_importance(best_model, X_train, model_name):
    try:
        if hasattr(best_model.named_steps['model'], 'feature_importances_'):
            importances = best_model.named_steps['model'].feature_importances_
        elif hasattr(best_model.named_steps['model'], 'coef_'):
            importances = np.abs(best_model.named_steps['model'].coef_).flatten()
        else:
            print(f"Pas de feature importance disponible pour {model_name}")
            return

        feature_names = X_train.columns
        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=feat_imp.values[:20], y=feat_imp.index[:20], palette='viridis')
        plt.title(f"{model_name} - Top 20 Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        save_and_log_plot(plt.gcf(), model_name, "feature_importance")

    except Exception as e:
        print(f"Erreur lors de l'affichage des importances : {e}")

        
def log_shap_summary(best_model, X_train, model_name):
    try:
        import shap
        explainer = shap.Explainer(best_model.named_steps['model'], X_train)
        shap_values = explainer(X_train)

        plt.figure()
        shap.summary_plot(shap_values, X_train, show=False)
        save_and_log_plot(plt.gcf(), model_name, "shap_summary")

        plt.figure()
        shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
        save_and_log_plot(plt.gcf(), model_name, "shap_bar")
    except Exception as e:
        print(f"Erreur lors du calcul SHAP pour {model_name} : {e}")