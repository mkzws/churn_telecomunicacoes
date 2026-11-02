"""src/ml_project/api.py"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from ml_project.model import load_trained_model

app = FastAPI(title="Telco Churn Prediction API", version="0.1.0")


class CustomerData(BaseModel):
    Gender: str = Field(..., alias="Gender")
    Senior_Citizen: str = Field("No", alias="Senior Citizen")
    Partner: str = "Yes"
    Dependents: str = "No"
    Tenure_Months: int = Field(1, alias="Tenure Months")
    Phone_Service: str = Field("Yes", alias="Phone Service")
    Multiple_Lines: str = Field("No", alias="Multiple Lines")
    Internet_Service: str = Field("Fiber optic", alias="Internet Service")
    Online_Security: str = Field("No", alias="Online Security")
    Online_Backup: str = Field("Yes", alias="Online Backup")
    Device_Protection: str = Field("No", alias="Device Protection")
    Tech_Support: str = Field("No", alias="Tech Support")
    Streaming_TV: str = Field("No", alias="Streaming TV")
    Streaming_Movies: str = Field("No", alias="Streaming Movies")
    Contract: str = "Month-to-month"
    Paperless_Billing: str = Field("Yes", alias="Paperless Billing")
    Payment_Method: str = Field("Electronic check", alias="Payment Method")
    Monthly_Charges: float = Field(70.35, alias="Monthly Charges")
    Total_Charges: float = Field(70.35, alias="Total Charges")

    class Config:
        populate_by_name = True


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict_churn(customer: CustomerData):
    try:
        model_pipeline = load_trained_model()
        
        # Converte a requisição Pydantic em DataFrame usando os aliases (nomes reais das colunas)
        input_df = pd.DataFrame([customer.model_dump(by_alias=True)])
        
        prediction = model_pipeline.predict(input_df)[0]
        churn_probability = model_pipeline.predict_proba(input_df)[0][1]

        return {
            "churn_prediction": int(prediction),
            "churn_label": "Yes" if prediction == 1 else "No",
            "churn_probability": round(float(churn_probability), 4)
        }
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Modelo não treinado. Execute o treinamento primeiro.")
    except Exception as err:
        raise HTTPException(status_code=400, detail=str(err))