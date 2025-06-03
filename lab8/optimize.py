import pandas as pd
import mlflow
from mlflow.models.signature import infer_signature
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly
import seaborn as sns
import pickle
import os
import xgboost
import sklearn
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
)
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings("ignore")

optuna.logging.set_verbosity(optuna.logging.WARNING)

# reproducibilidad
rs = 42

# cargar datos
df = pd.read_csv('./water_potability.csv')
train_val, test = train_test_split(df, test_size=0.1, random_state=rs)
train, val = train_test_split(train_val, test_size=2/9, random_state=rs)
X_train, y_train = train.drop(columns=['Potability']), train['Potability']
X_val, y_val = val.drop(columns=['Potability']), val['Potability']
X_test, y_test = test.drop(columns=['Potability']), test['Potability']

# nombre del experimento
experiment_name = f"Optuna_XGBoost_{rs}"
mlflow.set_experiment(experiment_name)

def objective(trial):
    # hiperparámetros
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)  
    max_depth = trial.suggest_int("max_depth", 2, 6)  
    n_estimators = trial.suggest_int("n_estimators", 100, 1000)  
    max_leaves = trial.suggest_int("max_leaves", 10, 50)  
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)  
    min_child_weight = trial.suggest_int("min_child_weight", 3, 10)  
    reg_alpha = trial.suggest_float("reg_alpha", 0.0, 10.0)
    reg_lambda = trial.suggest_float("reg_lambda", 1.0, 10.0)

    run_name = f"XGBoost lr={learning_rate:.3f}, md={max_depth}, ne={n_estimators}"
    
    with mlflow.start_run(run_name=run_name, nested=True):
        params = {
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'n_estimators': n_estimators,
            'max_leaves': max_leaves,
            'colsample_bytree': colsample_bytree,
            'min_child_weight': min_child_weight,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }

        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('classifier', XGBClassifier(**params, random_state=rs))
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='macro')

        mlflow.log_params(params)
        mlflow.log_metric("valid_f1", f1)

        input_example = X_val.iloc[:1]
        signature = infer_signature(X_val, y_pred[:1])

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            input_example=input_example,
            signature=signature
        )
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        return f1

def get_best_model(experiment_id):
    runs = mlflow.search_runs([experiment_id])
    best_model_id = runs.sort_values("metrics.valid_f1", ascending=False)["run_id"].iloc[0]
    best_model = mlflow.sklearn.load_model("runs:/" + best_model_id + "/model")
    return best_model

def optimize_model():
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=rs))
    with mlflow.start_run(run_name="Optuna XGBoost Optimization") as run:
        study.optimize(objective, timeout=600)

        # log de mejores hiperparámetros
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metric("best_valid_f1", study.best_value)

        # guardar visualizaciones de historial de optimización
        os.makedirs("plots", exist_ok=True)
        fig1 = plot_optimization_history(study)
        fig2 = plot_parallel_coordinate(study)
        fig1.write_image("plots/opt_history.png")
        fig2.write_image("plots/parallel_coordinate.png")
        mlflow.log_artifact("plots/opt_history.png", artifact_path="plots")
        mlflow.log_artifact("plots/parallel_coordinate.png", artifact_path="plots")

        # cargar y guardar mejor modelo
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        best_model = get_best_model(experiment_id)
        os.makedirs("models", exist_ok=True)
        with open("models/best_model.pkl", "wb") as f:
            pickle.dump(best_model, f)
        mlflow.log_artifact("models/best_model.pkl", artifact_path="models")

        # guardar importancia de variables
        xgb_model = best_model.named_steps['classifier']
        importances = xgb_model.feature_importances_
        features = X_train.columns
        importance_df = pd.DataFrame({
            "feature": features,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

        plt.figure(figsize=(8, 6))
        sns.barplot(data=importance_df, x="importance", y="feature")
        plt.title("Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig("plots/feature_importance.png")

        # log de versiones
        mlflow.log_param("xgboost_version", xgboost.__version__)
        mlflow.log_param("sklearn_version", sklearn.__version__)
        mlflow.log_param("optuna_version", optuna.__version__)

        print("Best trial:", study.best_trial.params)
        print("Best F1 score:", study.best_value)

if __name__ == "__main__":
    optimize_model()
