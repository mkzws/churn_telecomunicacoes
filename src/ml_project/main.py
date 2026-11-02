from collections import defaultdict, deque
from datetime import datetime, timedelta
import math
import os
from pathlib import Path
import traceback
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
import jwt
import numpy as np
import pandas as pd
import joblib

# Configurações de Segurança e JWT
SECRET_KEY = os.getenv("JWT_SECRET", "sua-chave-secreta-super-segura-aqui")
ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 30

# Rate Limiter Config
REQUEST_LIMIT = 10  # Máximo de requisições por IP
WINDOW_SECONDS = 60 # Janela em segundos
request_history = defaultdict(deque)

# Base de dados de utilizadores simulada
USERS_DB = {
    "admin": {"password": "senha123", "role": "admin"},
    "user": {"password": "user123", "role": "user"}
}

# Modelos Pydantic para Autenticação
class LoginRequest(BaseModel):
    username: str = Field(..., description="Nome de usuário para login", examples=["admin"])
    password: str = Field(..., description="Senha do usuário", examples=["senha123"])

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = TOKEN_EXPIRE_MINUTES * 60

# Modelos Pydantic para Churn
class ClientDataRequest(BaseModel):
    tenure_months: int = Field(..., alias="Tenure Months", ge=0, examples=[12])
    monthly_charges: float = Field(..., alias="Monthly Charges", ge=0, examples=[70.50])
    total_charges: float = Field(..., alias="Total Charges", ge=0, examples=[846.00])
    gender: str = Field(..., alias="Gender", examples=["Female"])
    senior_citizen: int = Field(..., alias="Senior Citizen", ge=0, le=1, examples=[0])
    partner: str = Field(..., alias="Partner", examples=["Yes"])
    dependents: str = Field(..., alias="Dependents", examples=["No"])
    phone_service: str = Field(..., alias="Phone Service", examples=["Yes"])
    multiple_lines: str = Field(..., alias="Multiple Lines", examples=["No"])
    internet_service: str = Field(..., alias="Internet Service", examples=["Fiber optic"])
    online_security: str = Field(..., alias="Online Security", examples=["No"])
    online_backup: str = Field(..., alias="Online Backup", examples=["Yes"])
    device_protection: str = Field(..., alias="Device Protection", examples=["No"])
    tech_support: str = Field(..., alias="Tech Support", examples=["No"])
    streaming_tv: str = Field(..., alias="Streaming TV", examples=["Yes"])
    streaming_movies: str = Field(..., alias="Streaming Movies", examples=["No"])
    contract: str = Field(..., alias="Contract", examples=["Month-to-month"])
    paperless_billing: str = Field(..., alias="Paperless Billing", examples=["Yes"])
    payment_method: str = Field(..., alias="Payment Method", examples=["Electronic check"])

class ChurnResponse(BaseModel):
    sucesso: bool
    prediction: int
    status_cliente: str
    nivel_risco: str
    probability: Optional[float] = None
    recomendacao: str
    usuario: str

# Funções de JWT
def create_token(username: str, role: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": username,
        "role": role,
        "exp": expire
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

security = HTTPBearer()

def get_current_user(token: str = Depends(security)) -> dict:
    try:
        payload = jwt.decode(token.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return {"username": payload["sub"], "role": payload["role"]}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expirado")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token inválido")

# Inicialização da Aplicação
app = FastAPI(
    title="API de Churn de Clientes com JWT",
    description="API de Previsão de Churn protegida por Token JWT e com Rate Limiting.",
    version="3.0"
)

# Carregamento do Modelo Campeão
# CARREGAMENTO ROBUSTO (Colocar este no lugar)
possible_paths = [
    Path("churn_model.pkl"),
    Path(__file__).parent / "churn_model.pkl",
    Path("models/churn_model.pkl"),
    Path(__file__).parent.parent / "churn_model.pkl"
]

model = None
MODELO_CARREGADO = False

for path in possible_paths:
    if path.exists():
        try:
            model = joblib.load(path)
            print(f"Modelo carregado com sucesso de: {path.absolute()}")
            MODELO_CARREGADO = True
            break
        except Exception as e:
            print(f"[Erro] Falha ao carregar o modelo em {path}: {e}")

if not MODELO_CARREGADO:
    print("[Aviso Crítico] Ficheiro 'churn_model.pkl' não foi encontrado em nenhum dos caminhos esperados!")

# Middleware de Rate Limiter (igual ao padrão da aula)
@app.middleware("http")
async def rate_limiter(request, call_next):
    client_ip = request.client.host if request.client else "desconhecido"
    now = time.time() if 'time' in globals() else datetime.utcnow().timestamp()
    history = request_history[client_ip]

    while history and now - history[0] > WINDOW_SECONDS:
        history.popleft()

    if len(history) >= REQUEST_LIMIT:
        retry_after = max(1, math.ceil(WINDOW_SECONDS - (now - history[0])))
        return JSONResponse(
            status_code=429,
            content={
                "detail": f"Limite de requisições excedido. Máximo de {REQUEST_LIMIT} por {WINDOW_SECONDS} segundos.",
                "retry_after": retry_after
            },
            headers={"Retry-After": str(retry_after)}
        )

    history.append(now)
    response = await call_next(request)
    response.headers["X-RateLimit-Remaining"] = str(REQUEST_LIMIT - len(history))
    response.headers["X-RateLimit-Limit"] = str(REQUEST_LIMIT)
    return response

# Endpoints
@app.get("/")
def home():
    return {
        "nome": "API de Predição de Churn",
        "versao": "3.0 FASTAPI",
        "endpoints": {
            "POST /login": "Autenticação e obtenção do token JWT",
            "GET /me": "Verifica os dados do utilizador autenticado",
            "GET /health": "Verifica o estado da API e modelo",
            "POST /predict": "Realiza a predição de churn (Protegido)"
        },
        "modelo_carregado": MODELO_CARREGADO
    }

@app.get("/health")
def health():
    return {
        "status": "healthy" if MODELO_CARREGADO else "unhealthy",
        "modelo_carregado": MODELO_CARREGADO
    }

@app.post("/login", response_model=TokenResponse, tags=["Auth"])
def login(credenciais: LoginRequest):
    user = USERS_DB.get(credenciais.username)
    if not user or user["password"] != credenciais.password:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Credenciais inválidas")

    token = create_token(credenciais.username, user["role"])
    return TokenResponse(access_token=token)

@app.get("/me", tags=["Auth"])
def get_me(current_user: dict = Depends(get_current_user)):
    return {
        "username": current_user["username"],
        "role": current_user["role"],
        "message": "Você está autenticado com sucesso!"
    }

@app.post("/predict", response_model=ChurnResponse, tags=["Predict"])
def predict(payload: ClientDataRequest, current_user: dict = Depends(get_current_user)):
    if not MODELO_CARREGADO or model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")
    
    try:
        dict_data = payload.model_dump(by_alias=True)
        df = pd.DataFrame([dict_data])
        
        # Limpeza numérica
        numeric_cols = ["Tenure Months", "Monthly Charges", "Total Charges"]
        for col in numeric_cols:
            if col in df.columns:
                if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
                    df[col] = df[col].astype(str).str.strip().str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        # Limpeza categórica
        categorical_cols = [
            "Gender", "Senior Citizen", "Partner", "Dependents", "Phone Service", 
            "Multiple Lines", "Internet Service", "Online Security", "Online Backup", 
            "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies", 
            "Contract", "Paperless Billing", "Payment Method"
        ]
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        # 1. Predição com o modelo
        prediction = model.predict(df)
        
        probability = 0.0
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(df).max())

        # 2. 👉 COLOCAR AQUI A LÓGICA DO TEXTO AMIGÁVEL 👈
        is_churn = int(prediction[0])
        if is_churn == 1:
            status_cliente = "Cliente em risco de cancelamento (Churn)"
            nivel_risco = "Alto" if probability > 0.7 else "Moderado"
            recomendacao = "Recomenda-se oferecer desconto, upgrade de contrato ou contacto preventivo."
        else:
            status_cliente = "Cliente fidelizado / Estável"
            nivel_risco = "Baixo"
            recomendacao = "Nenhuma ação imediata necessária."

        # 3. Retornar com os novos campos amigáveis
        return ChurnResponse(
            sucesso=True,
            prediction=is_churn,
            status_cliente=status_cliente,
            nivel_risco=nivel_risco,
            probability=probability,
            recomendacao=recomendacao,
            usuario=current_user["username"]
        )
        
    except Exception as e:
        print("--- ERRO COMPLETO DE PREDIÇÃO ---")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Erro ao processar a predição: {str(e)}")