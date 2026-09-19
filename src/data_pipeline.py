"""
Módulo de Simulação de Dados de Crédito e Engenharia de Features.
Gera microdados de proponentes de crédito bancário com desbalanceamento de classe.
"""

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class CreditDataPipeline:
    """
    Gerador e Pré-processador de Dados de Risco de Crédito.
    """

    def __init__(self, default_rate: float = 0.08, random_seed: int = 42):
        """
        :param default_rate: Proporção esperada de inadimplentes (padrão: 8%).
        :param random_seed: Semente para reprodutibilidade.
        """
        self.default_rate = default_rate
        self.random_seed = random_seed

    def generate_raw_data(self, sample_size: int = 10000) -> pd.DataFrame:
        """Gera dataset bruto simulando proposta de crédito de varejo."""
        np.random.seed(self.random_seed)

        # 1. Variáveis Cadastrais e Financeiras
        age = np.random.normal(38, 11, sample_size).clip(18, 75)
        monthly_income = np.random.exponential(4500, sample_size) + 1200
        bureau_score = np.random.normal(600, 150, sample_size).clip(200, 950)
        number_of_credit_inquiries = np.random.poisson(2, sample_size)
        delinquency_history_months = np.random.negative_binomial(1, 0.3, sample_size)
        loan_amount_requested = monthly_income * np.random.uniform(1.5, 8.0, sample_size)

        # 2. Mecanismo de Inadimplência (Logit do Default)
        logit_default = (
            -2.5
            - 0.006 * (bureau_score - 600)
            + 0.35 * delinquency_history_months
            + 0.15 * number_of_credit_inquiries
            + 0.0001 * (loan_amount_requested / monthly_income * 1000)
            - 0.0001 * (monthly_income - 3000)
        )
        prob_default = 1.0 / (1.0 + np.exp(-logit_default))

        # Ajuste para manter a taxa de default desejada
        threshold = np.quantile(prob_default, 1.0 - self.default_rate)
        default_label = (prob_default >= threshold).astype(int)

        df = pd.DataFrame(
            {
                "age": age,
                "monthly_income": monthly_income,
                "bureau_score": bureau_score,
                "number_of_credit_inquiries": number_of_credit_inquiries,
                "delinquency_history_months": delinquency_history_months,
                "loan_amount_requested": loan_amount_requested,
                "default": default_label,
            }
        )

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria variáveis derivadas e métricas de risco para o modelo."""
        df_feat = df.copy()

        # 1. Comprometimento do Empréstimo em Relação à Renda
        df_feat["loan_to_income_ratio"] = df_feat["loan_amount_requested"] / df_feat["monthly_income"]

        # 2. Indicador Combinado de Risco de Histórico
        df_feat["risk_index"] = (
            df_feat["delinquency_history_months"] * 1.5 + df_feat["number_of_credit_inquiries"]
        )

        # 3. Faixa de Pontuação de Bureau
        df_feat["bureau_score_ratio"] = df_feat["bureau_score"] / 950.0

        return df_feat

    def get_train_test_data(
        self, test_size: float = 0.2, sample_size: int = 10000
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Executa a geração, engenharia de features e faz o split estratificado.
        """
        raw_df = self.generate_raw_data(sample_size=sample_size)
        feat_df = self.engineer_features(raw_df)

        X = feat_df.drop(columns=["default"])
        y = feat_df["default"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_seed, stratify=y
        )

        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    pipeline = CreditDataPipeline(default_rate=0.08, random_seed=42)
    X_train, X_test, y_train, y_test = pipeline.get_train_test_data(sample_size=5000)

    print("=== PIPELINE DE DADOS DE CRÉDITO ===")
    print(f"Dimensões do Treino : {X_train.shape}")
    print(f"Dimensões do Teste  : {X_test.shape}")
    print(f"Taxa de Default (Treino): {y_train.mean():.2%}")
    print(f"Taxa de Default (Teste) : {y_test.mean():.2%}")
    print("\n--- Primeiras Linhas das Features ---")
    print(X_train.head())