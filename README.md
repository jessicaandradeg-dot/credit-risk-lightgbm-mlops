# 💳 Credit Risk Modeling with Cost-Sensitive LightGBM & MLOps Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.5+-00979D.svg)](https://mlflow.org/)
[![Optuna](https://img.shields.io/badge/Optuna-3.0+-blueviolet.svg)](https://optuna.org/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Solution for retail credit risk modeling with asymmetric financial loss optimization, hyperparameter search via Optuna, experiment tracking with MLflow, and strict unit testing.

---

## 🎯 Contexto de Negócio & Motivação

Na concessão de crédito de varejo bancário, modelos de Machine Learning enfrentam dois desafios críticos:
1. **Desbalanceamento de Classe Severo:** A taxa de inadimplência (*default*) costuma variar entre 5% e 10% da carteira.
2. **Custo Assimétrico do Erro:** 
   * **Falso Negativo (FN - Aprovar um inadimplente):** Gera perda do saldo devedor principal ($\text{R}\$ 5.000,00$ em média).
   * **Falso Positivo (FP - Recusar um bom cliente):** Gera apenas o custo de oportunidade do spread não realizado ($\text{R}\$ 500,00$ em média).

Modelos otimizados apenas por acurácia ou ROC-AUC com ponto de corte fixo em $0,50$ são subótimos do ponto de vista financeiro. Este projeto resolve esse impasse ao alinhar a otimização estatística do **LightGBM** com uma **Matriz de Custo Financeiro Assimétrica**.

---

## 🏗️ Arquitetura da Solução

```text
credit-risk-lightgbm-mlops/
├── data/                         # Artefatos e gráficos de diagnóstico
│   └── cost_threshold_curve.png
├── src/                          # Módulos core da aplicação
│   ├── __init__.py
│   ├── data_pipeline.py          # Gerador sintético de microdados e Feature Engineering
│   ├── train.py                  # Otimização com Optuna e Rastreamento via MLflow
│   └── evaluate.py               # Otimizador do Threshold da Matriz de Custo Financeiro
├── tests/                        # Suíte de testes unitários funcionais
│   └── test_pipeline.py
├── main.py                       # Orquestrador da pipeline fim-a-fim
├── pytest.ini                    # Configurações do Pytest e PYTHONPATH
├── requirements.txt              # Dependências do projeto
└── README.md                     # Documentação técnica do projeto