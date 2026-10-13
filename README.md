# 💳 Credit Risk Modeling with Cost-Sensitive LightGBM & MLOps Pipeline

Projeto de modelagem de risco de crédito com foco em decisão financeira mais eficiente, redução de perdas e pipeline reprodutível em produção.

## Problema de negócio

Em crédito de varejo, o problema não é apenas prever inadimplência, mas decidir com base no impacto financeiro real de cada erro.

- aprovar um cliente ruim pode gerar perdas significativas
- recusar um cliente bom gera perda de oportunidade e receita

Modelos tradicionais otimizados por acurácia ou ROC-AUC nem sempre maximizam valor financeiro. Em cenários bancários, a otimização correta depende do custo econômico associado a falso positivo e falso negativo.

Este projeto usa LightGBM com matriz de custo assimétrica para orientar decisões de crédito com foco em perda líquida e eficiência de carteira.

## O que foi construído

- gerador sintético de dados de crédito
- feature engineering para risco
- tuning de hiperparâmetros com Optuna
- treinamento de LightGBM com custo assimétrico
- busca do threshold ótimo com base em perda financeira
- rastreio de experimentos com MLflow
- testes unitários para pipeline

## Metodologia

A abordagem segue uma lógica bastante alinhada à prática de risco bancário:

1. gerar base de microdados com comportamento realista de crédito
2. definir custo financeiro para falso positivo e falso negativo
3. treinar modelo que optimize perda neta e não só acurácia
4. ajustar cutoff com base em custo marginal
5. comparar resultados por métricas financeiras e operacionais

O ponto central é converter a avaliação de modelo em linguagem de negócio: valor real do erro, não apenas classificação binária.

## Resultados e KPI

### Métricas principais

- AUC e recall
- precisão por classe
- perda financeira estimada
- custo total por política de corte
- volume de aprovação e taxa de default esperada

### Exemplo ilustrativo

| Política | Aprovados | Default esperado | Perda líquida | Observação |
|---|---:|---:|---:|---|
| Threshold 0.50 | alto | alto | maior custo | pouco alinhado ao custo financeiro |
| Threshold ótimo | moderado | controlado | menor perda | melhor equilíbrio financeiro |
| Política conservadora | baixo | baixo | menor risco | menor volume e menor receita |

O principal insight é que o melhor modelo não é o que maximiza acurácia pura, e sim o que minimiza custo financeiro real.

## Benchmark comparativo

O projeto permite comparar:

- cutoff fixo em 0.50
- cutoff otimizado por custo financeiro
- política por sensibilidade de risco
- estratégia de validação financeira por carteira

Esse tipo de benchmark mostra maturidade em modelagem aplicada e comunica bem com recrutadores, porque reforça que a solução foi pensada para decisão de negócio e não apenas para performance estatística.

## Stack

- Python 3.10+
- LightGBM
- Optuna
- MLflow
- Pandas
- NumPy
- Scikit-learn
- Pytest
- Jupyter / notebook de análise

## Estrutura do projeto

```text
credit-risk-lightgbm-mlops/
├── data/
├── src/
│   ├── data_pipeline.py
│   ├── train.py
│   ├── evaluate.py
│   └── __init__.py
├── tests/
├── main.py
├── README.md
├── requirements.txt
├── pytest.ini
└── .gitignore
```


## Link para artigo / benchmark / tabela

- [Curva de threshold de custo](#)
- [Relatório de performance financeira](#)
- [Notebook de tuning e validação](#)
- [Comparativo de políticas de corte](#)


