"""src/ml_project/train.py"""

import sys
from ml_project.data import load_raw_data, prepare_data
from ml_project.model import evaluate_model, save_model, train_model


def main():
    print("🚀 Iniciando o pipeline de treinamento...")

    try:
        # 1. Carrega os dados brutos
        print(f"📂 Carregando dados de: {DATA_PATH if 'DATA_PATH' in globals() else 'data/raw/Telco_customer_churn.csv'}...")
        df = load_raw_data()
        
        # 2. Prepara os dados (limpeza e divisão treino/teste)
        print("🧹 Limpando e dividindo os dados...")
        X_train, X_test, y_train, y_test = prepare_data(df)

        # 3. Treina o modelo
        print("🤖 Treinando o modelo (RandomForest + Pipeline)...")
        pipeline = train_model(X_train, y_train)

        # 4. Avalia o modelo
        print("📊 Avaliando o desempenho...")
        metrics = evaluate_model(pipeline, X_test, y_test)
        print(f"✨ ROC-AUC Score: {metrics['roc_auc']:.4f}")

        # 5. Salva o artefato
        print("💾 Salvando o modelo treinado...")
        save_model(pipeline)
        print("✅ Treinamento concluído com sucesso! Modelo salvo em models/churn_model.pkl")

    except Exception as e:
        print(f"❌ Erro durante o treinamento: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    from config import DATA_PATH
    main()