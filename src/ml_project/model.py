import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline

from ml_project.config import MODEL_DIR, MODEL_PATH, RANDOM_STATE
from ml_project.data import build_preprocessor


def create_pipeline() -> Pipeline:
    """Cria o pipeline completo contendo pré-processamento e modelo."""
    preprocessor = build_preprocessor()
    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        class_weight="balanced"  # Lida com desbalanceamento de Churn
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def train_model(X_train, y_train) -> Pipeline:
    """Treina o pipeline contendo o modelo."""
    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(pipeline: Pipeline, X_test, y_test) -> dict:
    """Gera métricas de desempenho para o modelo de Churn."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "report": classification_report(y_test, y_pred, output_dict=True),
    }
    return metrics


def save_model(pipeline: Pipeline) -> None:
    """Salva o pipeline treinado."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)


def load_trained_model() -> Pipeline:
    """Carrega o pipeline treinado."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo não encontrado em: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)