# 📱 FIAPMobile - Sistema Preditivo de Churn de Clientes

Este repositório contém o desenvolvimento do **Tech Challenge (Fase 1)** da pós-graduação em **Machine Learning Engineering** da FIAP. O objetivo deste projeto é construir uma solução de ponta a ponta, desde a análise exploratória até o serviço de inferência em produção, para combater a evasão de clientes (*churn*) da operadora FIAPMobile.

---

## 🎥 Vídeo de Apresentação (Método STAR)
Conforme exigido pelo edital, a apresentação executiva e técnica do projeto (com duração máxima de 5 minutos) está disponível através do link abaixo:
*   👉 

### 📝 Resumo Executivo (Estrutura STAR):
*   **Situation (Situação):** A operadora de telecomunicações FIAPMobile enfrenta uma perda acelerada de clientes ativos, o que impacta severamente a receita recorrente mensal (MRR).
*   **Task (Tarefa):** Nosso grupo recebeu a tarefa de mapear os dados históricos de 7.043 clientes, realizar uma análise exploratória robusta, treinar um modelo baseline reprodutível e estruturar um ecossistema pronto para produção.
*   **Action (Ação):** Concluímos a Etapa 1 mapeando o negócio via **ML Canvas**, executando uma EDA no Jupyter Notebook que mitigou vazamentos de dados (*Data Leakage*), tratando inconsistências de faturamento e estabelecendo o primeiro modelo baseline com **Regressão Logística**.
*   **Result (Resultado):** O modelo baseline legítimo estabeleceu uma **Acurácia de Teste de 79,70%** e um **F1-Score de 0.6093**, criando uma linha de base científica e documentada via **Model Card** para as próximas evoluções.

---

## 📂 Estrutura do Repositório
O projeto adota uma arquitetura padrão de mercado para projetos de Ciência de Dados e Machine Learning:

```text
├── data/                       # Armazenamento de dados do projeto
│   └── raw/                    # Dataset bruto Telco_customer_churn.csv
├── docs/                       # Documentações estratégicas e de negócio
│   ├── ml_canvas.md            # Planejamento estratégico e KPIs do projeto
│   └── model_card.md           # Características e limitações do modelo baseline
├── notebooks/                  # Ambiente de experimentação e laboratório
│   └── etapa1_eda_baseline.ipynb # Notebook contendo a EDA visual e o modelo baseline
├── src/                        # Código-fonte produtivo modular (.py)
├── tests/                      # Testes automatizados (Pytest)
├── .gitignore                  # Filtro para não versionar arquivos locais/pesados
├── README.md                   # Instruções gerais e documentação principal
└── requirements.txt            # Gerenciamento de dependências do Python
```

---

## 🛠️ Instruções de Setup e Execução

Siga os passos abaixo para clonar o repositório, configurar o ambiente virtual e rodar as análises locais:

### 1. Clonar o Repositório
Abra o seu terminal e execute o comando:
```bash
git clone https://github.com/mkzws/churn_telecomunicacoes
cd SEU_REPOSITORIO
```

### 2. Configurar o Ambiente Virtual (venv)
Crie e ative um ambiente isolado para o Python não conflitar com pacotes globais do seu computador:

*   **Windows:**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
*   **Mac/Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

### 3. Instalar as Dependências Obrigatórias
Com a venv ativa, instale os pacotes necessários listados no arquivo de requisitos:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Executar o Notebook de EDA e Baseline
Para abrir o laboratório de testes e rodar a análise exploratória ou o treinamento do modelo:
```bash
jupyter notebook
```
Navegue até a pasta `notebooks/` e abra o arquivo `eda_regressao.ipynb`. Execute as células sequencialmente (`Shift + Enter`).

---

## 📏 Resumo das Métricas do Baseline Real

*   **Acurácia de Teste:** 79,70%
*   **Precisão de Teste:** 67,17%
*   **Recall de Teste:** 55,75%
*   **F1-Score de Teste:** 60,93%

*Nota Técnica: O modelo sofreu uma severa limpeza para remover colunas geradas pós-evento (como `Churn Score`), garantindo que a performance de 79.70% reflita uma capacidade preditiva real de mercado, livre de vazamentos de dados.*
