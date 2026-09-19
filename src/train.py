"""
Módulo de Treinamento e Otimização Bayesiana com LightGBM, Optuna e MLflow.
"""

from typing import Dict, Any, Tuple
import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Silenciar logs verbosos do Optuna durante os trials
optuna.logging.set_verbosity(optuna.logging.WARNING)


class CreditModelTrainer:
    """
    Treinador do modelo LightGBM para predição de inadimplência de crédito.
    """

    def __init__(self, experiment_name: str = "credit_risk_lightgbm"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(self.experiment_name)

    def optimize_hyperparameters(
        self, X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 15
    ) -> Dict[str, Any]:
        """
        Executa busca bayesiana de hiperparâmetros via Optuna com Validação Cruzada.
        """

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "binary",
                "metric": "auc",
                "boosting_type": "gbdt",
                "verbosity": -1,
                "random_state": 42,
                "n_estimators": trial.suggest_int("n_estimators", 50, 250),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
            }

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []

            for train_idx, val_idx in skf.split(X_train, y_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model = lgb.LGBMClassifier(**params)
                model.fit(X_tr, y_tr)

                preds = model.predict_proba(X_val)[:, 1]
                cv_scores.append(roc_auc_score(y_val, preds))

            return float(np.mean(cv_scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        print(f"-> Melhor ROC-AUC na Validação Cruzada: {study.best_value:.4f}")
        return study.best_params

    def train_and_log(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        best_params: Dict[str, Any],
    ) -> Tuple[lgb.LGBMClassifier, str]:
        """
        Treina o modelo final com os melhores parâmetros e registra os artefatos no MLflow.
        """
        with mlflow.start_run() as run:
            # 1. Instanciar e Treinar Modelo Final
            model = lgb.LGBMClassifier(**best_params, objective="binary", verbosity=-1, random_state=42)
            model.fit(X_train, y_train)

            # 2. Predições e Avaliação Inicial
            train_preds = model.predict_proba(X_train)[:, 1]
            test_preds = model.predict_proba(X_test)[:, 1]

            train_auc = roc_auc_score(y_train, train_preds)
            test_auc = roc_auc_score(y_test, test_preds)

            # 3. Log de Parâmetros e Métricas no MLflow (Usando flavor nativo do LightGBM)
            mlflow.log_params(best_params)
            mlflow.log_metric("train_roc_auc", train_auc)
            mlflow.log_metric("test_roc_auc", test_auc)
            
            # Registro do modelo utilizando o módulo nativo do LightGBM
            mlflow.lightgbm.log_model(model, artifact_path="lightgbm_credit_model")

            print("=== REGISTRO NO MLFLOW COMPLETO ===")
            print(f"Run ID       : {run.info.run_id}")
            print(f"Train ROC-AUC: {train_auc:.4f}")
            print(f"Test ROC-AUC : {test_auc:.4f}")

            return model, run.info.run_id


if __name__ == "__main__":
    from src.data_pipeline import CreditDataPipeline

    # 1. Carregar dados
    pipeline = CreditDataPipeline(default_rate=0.08, random_seed=42)
    X_train, X_test, y_train, y_test = pipeline.get_train_test_data(sample_size=5000)

    # 2. Otimizar e Treinar
    trainer = CreditModelTrainer()
    print("Iniciando Otimização Bayesiana de Hiperparâmetros (Optuna)...")
    best_params = trainer.optimize_hyperparameters(X_train, y_train, n_trials=10)

    print("\nTreinando Modelo Final e Registrando no MLflow...")
    final_model, run_id = trainer.train_and_log(X_train, y_train, X_test, y_test, best_params)