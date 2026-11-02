from fastapi.testclient import TestClient
from src.ml_project.main import app 

client = TestClient(app)

def test_health_check():
    """Testa se o endpoint de saúde retorna status healthy e modelo carregado."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_predict_churn_success():
    """Testa uma predição bem-sucedida com dados válidos de um cliente."""
    payload = {
        "Tenure Months": 12,
        "Monthly Charges": 70.50,
        "Total Charges": 846.00,
        "Gender": "Female",
        "Senior Citizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "Fiber optic",
        "Online Security": "No",
        "Online Backup": "Yes",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "Yes",
        "Streaming Movies": "No",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Electronic check"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Valida se a resposta contém as chaves esperadas e tipos corretos
    assert "prediction" in data
    assert "probability" in data
    assert isinstance(data["prediction"], int)
    assert isinstance(data["probability"], float)

def test_predict_validation_error():
    """Testa se o Pydantic rejeita dados inválidos com código 422."""
    invalid_payload = {
        "Tenure Months": "nao_e_um_numero"
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422