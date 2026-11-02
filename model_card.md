# Model Card: Churn Prediction API (Random Forest)

## Visão Geral do Modelo
- **Desenvolvedor:** Selma Miki Masuzawa
- **Modelo:** Random Forest Classifier (Scikit-Learn)
- **Versão:** 3.0
- **Objetivo:** Prever a propensão de cancelamento (Churn) de clientes de uma operadora de telecomunicações com base em dados demográficos, serviços contratados e histórico de faturação.

## Performance e Métricas
- **Algoritmo:** Ensemble de árvores de decisão (Random Forest Pipeline com ColumnTransformer para pré-processamento de colunas numéricas e categóricas).
- **Métricas de Avaliação:** Acurácia, Precisão, Recall e F1-Score validadas via testes de integração automatizados (`pytest`).
- **Comportamento:** Retorna a predição binária (`0` para mantido, `1` para churn) acompanhada de uma probabilidade calculada (`predict_proba`) e categorização de risco automatizada.

## Dados Utilizados
- Base de dados de telecomunicações contendo variáveis como `Tenure Months`, `Monthly Charges`, `Total Charges`, tipo de contrato, métodos de pagamento e serviços adicionais (Internet, Segurança Online, Streaming, etc.). Retirados no Kaggle: [text](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset?resource=download)

## Limitações e Considerações
1. **Viés de Dados:** O modelo foi treinado com base num dataset histórico específico. Mudanças drásticas no comportamento de mercado ou nos planos da operadora podem impactar a precisão das previsões.
2. **Variáveis Categóricas:** Valores fora do domínio treinado (ex: novos serviços de internet não previstos no treino original) podem gerar exceções ou exigir reajustes no pipeline.
3. **Uso de Apoio à Decisão:** As recomendações geradas pela API têm caráter puramente consultivo e preventivo para equipas de CRM, não devendo substituir a análise humana individualizada.