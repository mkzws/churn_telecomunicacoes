"""src/ml_project/config.py"""

from pathlib import Path

# Direitórios principais
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
DATA_PATH = DATA_DIR / "Telco_customer_churn.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "churn_model.pkl"

# Variável alvo (usaremos o Churn Value que é numérico: 0 ou 1)
TARGET = "Churn Value"

NUMERICAL_FEATURES = ["Tenure Months", "Monthly Charges", "Total Charges"]

CATEGORICAL_FEATURES = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2