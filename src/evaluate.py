"""
Módulo de Avaliação por Matriz de Custo Financeiro e Otimização de Threshold.
Calcula o ponto de corte ideal para maximizar o resultado financeiro da operação de crédito.
"""

from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class FinancialCostEvaluator:
    """
    Avaliador Financeiro de Risco de Crédito baseado em Matriz de Custo Assimétrica.
    """

    def __init__(
        self,
        cost_false_negative: float = 5000.0,  # Prejuízo médio por inadimplência (FN)
        cost_false_positive: float = 500.0,   # Custo de oportunidade por recusa indevida (FP)
    ):
        self.cost_fn = cost_false_negative
        self.cost_fp = cost_false_positive

    def compute_financial_loss(
        self, y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float
    ) -> float:
        """
        Calcula a perda financeira total em Reais (R$) para um determinado threshold.
        """
        y_pred = (y_pred_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        total_loss = (fn * self.cost_fn) + (fp * self.cost_fp)
        return float(total_loss)

    def find_optimal_threshold(
        self, y_true: np.ndarray, y_pred_prob: np.ndarray
    ) -> Tuple[float, float, pd.DataFrame]:
        """
        Varre uma grade de thresholds e encontra aquele que minimiza a perda financeira.
        """
        thresholds = np.linspace(0.01, 0.99, 99)
        loss_results = []

        for thresh in thresholds:
            loss = self.compute_financial_loss(y_true, y_pred_prob, thresh)
            loss_results.append({"threshold": thresh, "total_loss_brl": loss})

        results_df = pd.DataFrame(loss_results)
        best_row = results_df.loc[results_df["total_loss_brl"].idxmin()]

        optimal_threshold = float(best_row["threshold"])
        min_loss = float(best_row["total_loss_brl"])

        return optimal_threshold, min_loss, results_df

    def plot_cost_curve(
        self,
        results_df: pd.DataFrame,
        optimal_threshold: float,
        min_loss: float,
        output_path: str = "data/cost_threshold_curve.png",
    ) -> None:
        """Gera e salva o gráfico da curva de custo financeiro vs threshold."""
        plt.figure(figsize=(9, 5))
        sns.set_theme(style="whitegrid")

        plt.plot(
            results_df["threshold"],
            results_df["total_loss_brl"],
            color="#1D3557",
            linewidth=2.5,
            label="Perda Financeira (R$)",
        )
        plt.axvline(
            optimal_threshold,
            color="#E63946",
            linestyle="--",
            linewidth=1.8,
            label=f"Threshold Ótimo ({optimal_threshold:.2f})",
        )
        plt.axvline(
            0.50,
            color="#457B9D",
            linestyle=":",
            linewidth=1.5,
            label="Threshold Padrão (0.50)",
        )

        plt.title(
            "Otimização do Threshold de Corte por Matriz de Custo Financeiro",
            fontsize=12,
            fontweight="bold",
        )
        plt.xlabel("Probability Threshold (Corte de Default)")
        plt.ylabel("Perda Financeira Total Estimada (R$)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()


if __name__ == "__main__":
    from src.data_pipeline import CreditDataPipeline
    from src.train import CreditModelTrainer

    # 1. Carregar dados e modelo
    pipeline = CreditDataPipeline(default_rate=0.08, random_seed=42)
    X_train, X_test, y_train, y_test = pipeline.get_train_test_data(sample_size=5000)

    trainer = CreditModelTrainer()
    best_params = trainer.optimize_hyperparameters(X_train, y_train, n_trials=5)
    model, _ = trainer.train_and_log(X_train, y_train, X_test, y_test, best_params)

    # 2. Avaliar financeiramente no conjunto de teste
    test_probs = model.predict_proba(X_test)[:, 1]
    evaluator = FinancialCostEvaluator(
        cost_false_negative=5000.0, cost_false_positive=500.0
    )

    optimal_thresh, min_loss, results_df = evaluator.find_optimal_threshold(
        y_test.values, test_probs
    )
    standard_loss = evaluator.compute_financial_loss(
        y_test.values, test_probs, threshold=0.50
    )

    savings = standard_loss - min_loss

    print("\n=== AVALIAÇÃO DE IMPACTO FINANCEIRO DA OPERAÇÃO ===")
    print(f"Perda com Threshold Padrão (0.50): R$ {standard_loss:,.2f}")
    print(f"Perda com Threshold Ótimo ({optimal_thresh:.2f}) : R$ {min_loss:,.2f}")
    print(f"Economia Gerada pela Otimização : R$ {savings:,.2f} ({savings/standard_loss if standard_loss > 0 else 0:.1%})")

    evaluator.plot_cost_curve(results_df, optimal_thresh, min_loss)
    print("-> Gráfico 'data/cost_threshold_curve.png' salvo com sucesso!")