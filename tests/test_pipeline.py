"""
Módulo de Testes Unitários e Validação Funcional (Pytest).
"""

import pytest
import numpy as np
import pandas as pd
from src.data_pipeline import CreditDataPipeline
from src.evaluate import FinancialCostEvaluator


@pytest.fixture
def sample_data():
    pipeline = CreditDataPipeline(default_rate=0.08, random_seed=42)
    return pipeline.get_train_test_data(sample_size=1000)


def test_data_pipeline_dimensions(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    assert len(X_train) == 800
    assert len(X_test) == 200
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_stratification_target(sample_data):
    _, _, y_train, y_test = sample_data
    # Garante que a proporção do target de default em treino e teste difere por menos de 2%
    assert abs(y_train.mean() - y_test.mean()) < 0.02


def test_engineered_features_existence(sample_data):
    X_train, _, _, _ = sample_data
    expected_cols = ["loan_to_income_ratio", "risk_index", "bureau_score_ratio"]
    for col in expected_cols:
        assert col in X_train.columns


def test_financial_evaluator_cost_calculation():
    evaluator = FinancialCostEvaluator(cost_false_negative=5000.0, cost_false_positive=500.0)
    
    y_true = np.array([1, 0, 0, 1])
    y_probs = np.array([0.1, 0.8, 0.2, 0.9])  # FN no índice 0, FP no índice 1
    
    # Com threshold 0.5:
    # Predições: [0, 1, 0, 1]
    # Índices: idx 0 (verdadeiro 1, predito 0) = FN -> R$ 5,000
    # Índices: idx 1 (verdadeiro 0, predito 1) = FP -> R$ 500
    # Perda Esperada: 5000 + 500 = R$ 5,500
    loss = evaluator.compute_financial_loss(y_true, y_probs, threshold=0.50)
    assert loss == 5500.0