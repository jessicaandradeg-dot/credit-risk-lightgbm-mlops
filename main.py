"""
Orquestrador Principal da Pipeline de Risco de Crédito e MLOps.
Executa o ciclo completo: Ingestão/Engenharia -> Treinamento/Optuna -> Registro MLflow -> Avaliação Financeira.
"""

import sys
from src.data_pipeline import CreditDataPipeline
from src.train import CreditModelTrainer
from src.evaluate import FinancialCostEvaluator


def run_pipeline():
    print("==========================================================")
    print("   ETAPA 1: SIMULAÇÃO DE DADOS & ENGENHARIA DE FEATURES   ")
    print("==========================================================")
    pipeline = CreditDataPipeline(default_rate=0.08, random_seed=42)
    X_train, X_test, y_train, y_test = pipeline.get_train_test_data(sample_size=10000)
    print(f"-> Base Treino: {X_train.shape[0]} registros | Base Teste: {X_test.shape[0]} registros")
    print(f"-> Taxa de Default Observada: {y_train.mean():.2%}")

    print("\n==========================================================")
    print("   ETAPA 2: OTIMIZAÇÃO BAYESIANA & TREINAMENTO LIGHTGBM   ")
    print("==========================================================")
    trainer = CreditModelTrainer(experiment_name="credit_risk_lightgbm_production")
    print("-> Rodando 15 trials de otimização via Optuna...")
    best_params = trainer.optimize_hyperparameters(X_train, y_train, n_trials=15)

    print("\n-> Treinando modelo final e registrando no MLflow...")
    model, run_id = trainer.train_and_log(X_train, y_train, X_test, y_test, best_params)

    print("\n==========================================================")
    print("   ETAPA 3: AVALIAÇÃO DE IMPACTO POR MATRIZ DE CUSTO      ")
    print("==========================================================")
    test_probs = model.predict_proba(X_test)[:, 1]
    evaluator = FinancialCostEvaluator(cost_false_negative=5000.0, cost_false_positive=500.0)

    optimal_thresh, min_loss, results_df = evaluator.find_optimal_threshold(y_test.values, test_probs)
    standard_loss = evaluator.compute_financial_loss(y_test.values, test_probs, threshold=0.50)
    savings = standard_loss - min_loss

    print(f"-> Perda (Threshold Padrão 0.50) : R$ {standard_loss:,.2f}")
    print(f"-> Perda (Threshold Ótimo {optimal_thresh:.2f})  : R$ {min_loss:,.2f}")
    print(f"-> ECONOMIA TOTAL ESTIMADA        : R$ {savings:,.2f} ({savings/standard_loss:.1%})")

    evaluator.plot_cost_curve(results_df, optimal_thresh, min_loss)
    print("-> Gráfico 'data/cost_threshold_curve.png' atualizado.")
    print("==========================================================")
    print("               PIPELINE CONCLUÍDA COM SUCESSO!            ")
    print("==========================================================")


if __name__ == "__main__":
    run_pipeline()