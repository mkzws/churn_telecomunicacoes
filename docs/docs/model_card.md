# 📇 Model Card - Baseline de Predição de Churn (Etapa 1)

Este documento fornece informações detalhadas sobre o modelo baseline de Machine Learning desenvolvido para a predição de cancelamento de clientes, mapeando suas características, desempenho e restrições operacionais.

---

## 1. Identificação do Modelo
*   **Nome do Modelo:** Baseline de Predição de Churn
*   **Versão:** 1.0 (MVP Inicial).
*   **Data de Criação:** Outubro de 2026.
*   **Desenvolvedores:** Grupo de Engenharia de Machine Learning (Pos Tech FIAP).

---

## 2. Descrição do Modelo
*   **Problema Resolvido:** O modelo aborda a perda acelerada de receita recorrente na operadora, identificando clientes residenciais e corporativos com alta propensão a cancelar os serviços (*churn*) antes do encerramento contratual definitivo.
*   **Tipo de Modelo:** Classificação Binária Supervisionada (0: Não cancela / Ativo; 1: Cancela / Churn).
*   **Algoritmo Utilizado:** Regressão Logística (*Scikit-Learn*) com sementes fixadas para fins de reprodutibilidade científica.

---

## 3. Dados de Treinamento e Pré-processamento
*   **Fonte dos Dados:** Banco de dados histórico consolidado de clientes de telecomunicações (*Telco Customer Churn Dataset*).
*   **Tamanho do Conjunto de Dados:** 7.043 registros totais.
    *   *Treino (80%):* 5.634 registros.
    *   *Teste (20%):* 1.409 registros.
*   **Pré-processamento Realizado:**
    1.  **Limpeza e Tipagem:** Conversão manual da coluna `Total Charges` de formato textual para ponto flutuante (*float64*) e preenchimento de nulos ocultos com `0.0` para clientes sem tempo de casa (`Tenure Months = 0`).
    2.  **Mitigação de Vazamento de Dados (*Data Leakage*):** Exclusão mandatória de colunas geradas pós-evento ou puramente estatísticas do cálculo preditivo (ex: `Churn Score`, `Churn Value` e `Churn Reason`), blindando a integridade das inferências.
    3.  **Codificação:** Aplicação de *One-Hot Encoding* (`pd.get_dummies`) com remoção da primeira categoria dummy (`drop_first=True`) para converter variáveis nominais de texto (como tipo de contrato e tecnologia de internet) em vetores numéricos binários.

---

## 4. Desempenho
Os resultados consolidados obtidos pelo algoritmo de Regressão Logística diretamente no conjunto de teste (dados inéditos) foram:

*   **Acurácia de Teste:** 79,70%
*   **Precisão de Teste:** 67,17%
*   **Recall (Sensibilidade) de Teste:** 55,75%
*   **F1-Score de Teste:** 60,93%

*Nota sobre subgrupos:* Avaliações preliminares na análise exploratória (EDA) demonstram que o desempenho do modelo e a taxa de erro tendem a variar conforme o tempo de contrato do cliente. Há um volume substancialmente maior de cancelamentos nos primeiros 5 meses de jornada, exigindo cautela na calibração do limiar (*threshold*) para clientes novos.

---

## 5. Limitações
*   **Premissa de Linearidade:** A Regressão Logística assume limites de decisão lineares. Ela falha em capturar interações complexas e não-lineares ocultas entre as variáveis (como o impacto conjunto de tarifas altas aplicadas estritamente à tecnologia de Fibra Óptica).
*   **Falsos Negativos:** O indicador de *Recall* em 55,75% aponta que o modelo deixa passar aproximadamente 44,25% dos clientes que efetivamente cancelarão o serviço sem disparar nenhum alerta preventivo.
*   **Sensibilidade Temporal:** O modelo não considera janelas de tempo dinâmicas ou mudanças sazonais macroeconômicas bruscas que alterem o comportamento do mercado de telecomunicações no curto prazo.

---

## 6. Considerações Éticas
*   **Privacidade e LGPD:** Todos os dados de identificação direta (como `Customer ID`, nomes ou códigos cadastrais internos) foram explicitamente removidos do conjunto de treinamento. O modelo atua puramente sobre padrões comportamentais e financeiros agregados.
*   **Viés de Descontos:** Há um risco social e financeiro colateral se o modelo perpetuar a concessão de descontos e benefícios apenas para clientes de alto padrão econômico (maior CLTV), marginalizando faixas populacionais de menor receita recorrente de receberem ações de retenção proativas.

---

## 7. Instruções de Uso
*   **Implementação:** O modelo deve receber as variáveis cadastrais de faturamento, tipo de plano e serviços de suporte de forma tabular.
*   **Exemplo de Entrada (JSON Equivalente para API):**
    ```json
    {
      "Tenure Months": 12,
      "Monthly Charges": 70.5,
      "Total Charges": 846.0,
      "Contract": "Month-to-month",
      "Internet Service": "Fiber optic",
      "Tech Support": "No"
    }
    ```
*   **Exemplo de Saída:**
    *   `Prediction: 1` (Risco detectado - encaminhar para CS).
    *   `Probability_of_Churn: 0.7420` (Score probabilístico gerado pela função `predict_proba`).
