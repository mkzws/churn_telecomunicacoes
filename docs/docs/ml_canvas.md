# 🗺️ Machine Learning Canvas - Sistema de Predição de Churn

Este canvas serve como o artefato central de planejamento estratégico, técnico e operacional para o desenvolvimento do modelo preditivo de cancelamento de clientes.

---

### 1. Problema a Ser Resolvido
*   **Contexto de Negócio:** A operadora está sofrendo uma perda acelerada de clientes ativos, o que impacta diretamente a receita recorrente mensal (MRR).
*   **O Objetivo da IA:** Identificar e classificar proativamente quais clientes residenciais e corporativos possuem alto risco de cancelamento contratual iminente (*churn*).
*   **Alinhamento Estratégico:** Permitir que o time de Customer Success realize ações preventivas de fidelização focadas estritamente nos clientes corretos antes que o desligamento ocorra.

### 2. Dados
*   **Fontes de Dados:** Banco de dados histórico e unificado de faturamento, CRM e contratação de serviços (*Telco Customer Churn*).
*   **Tipo de Dados:** Estruturados, tabulares, contendo 7.043 registros no total.
*   **Qualidade e Engenharia Inicial:** Variáveis financeiras convertidas para ponto flutuante (*float64*). Espaços em branco em clientes novos (`Tenure Months = 0`) preenchidos com `0.0`.
*   **Mitigação de Leakage:** Colunas geradas pós-evento (como motivos exatos de saída e scores futuros do sistema) foram descartadas para blindar o poder preditivo real.

### 3. Modelo
*   **Tipo de Aprendizado:** Supervisionado.
*   **Tarefa de ML:** Classificação Binária (0: Cliente Ativo / 1: Churn Confirmado).
*   **Algoritmo Baseline (Etapa 1):** Regressão Logística (*Scikit-Learn*).
*   **Justificativa da Escolha:** Modelo linear simples, matematicamente transparente e altamente interpretável. Ideal para estabelecer uma linha de base robusta e validar se os dados possuem sinal preditivo útil antes de partirmos para arquiteturas mais complexas.

### 4. Avaliação
*   **Métrica Técnica Primária:** **F1-Score** (Resultado obtido no conjunto de teste: **0.6093**).
*   **Justificativa Técnica:** Devido ao desbalanceamento da base, a acurácia isolada (79.70%) traria uma falsa sensação de segurança. O F1-Score garante o equilíbrio harmônico necessário entre a *Precisão* (evitar gastos de retenção com falsos alarmes) e o *Recall* (evitar deixar clientes em risco passarem despercebidos).
*   **Métricas Técnicas Secundárias:** Acurácia (79.70%), Precisão (67.17%) e Recall (55.75%).

### 5. Implementação
*   **Estratégia de Disponibilização:** O pipeline produtivo final será isolado em módulos Python (`src/`) e servido como um microsserviço de inferência HTTP de baixa latência utilizando o framework **FastAPI**.
*   **Padrão de Consumo:** Inferência semanal em lote (*batch processing*) alimentando de forma automática o painel de atendimento do CRM.

### 6. Resultados
*   **Expectativa de Negócio:** Reduzir o Churn Rate geral da operadora em **15%** no primeiro trimestre pós-implantação.
*   **Impacto Financeiro:** Mitigar a perda de receita recorrente através da blindagem ativa do grupo de clientes de alto valor (aqueles identificados na EDA com CLTV elevado).
*   **Critério de Sucesso Técnico:** Superar o F1-Score do baseline (0.6093) nas próximas iterações utilizando modelos baseados em árvores ou redes neurais simples.

### 7. Recursos Necessários
*   **Equipe:** Cientistas de Dados (experimentação e validação), Engenheiros de Machine Learning (módulos produtivos e API) e Especialistas de CRM (validação de regras de negócio).
*   **Tecnologias e Ferramentas:** Ecossistema *Python 3*, *Scikit-Learn* (modelagem), *FastAPI* (serviço), *Pytest* (testes automatizados de qualidade) e *Git/GitHub* (versionamento).

### 8. Riscos
*   **Risco Técnico (Overfitting):** Algoritmos de árvore futuros podem se sobreajustar ao comportamento histórico, perdendo a capacidade de generalizar para novos assinantes.
*   **Risco Operacional (Capacidade do Time):** O modelo pode disparar mais alertas do que a equipe de Customer Success tem capacidade humana de ligar e negociar, gerando gargalos.
*   **Risco Ético e Viés:** O modelo pode priorizar ações de retenção apenas para assinantes com faturas caras (maior CLTV), criando um viés discriminatório contra clientes de pacotes econômicos.
