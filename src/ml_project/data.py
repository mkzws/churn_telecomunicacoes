"""src/ml_project/data.py"""

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_project.config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)


def load_raw_data(data_path: str = DATA_PATH) -> pd.DataFrame:
    """Carrega o dataset bruto delimitado por ponto e vírgula."""
    df = pd.read_csv(data_path, sep=";")
    
    # Remove espaços em branco nas pontas dos nomes de todas as colunas
    df.columns = df.columns.str.strip()
    
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa e formata os dados do Telco Churn."""
    df = df.copy()

    # Trata as colunas numéricas convertendo vírgulas em pontos, caso existam
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            # Se a coluna for do tipo object (string), substitui vírgula por ponto
            if df[col].dtype == object:
                df[col] = df[col].str.replace(",", ".", regex=False)
                df[col] = df[col].replace(" ", np.nan)
            
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(0.0)

    return df


def build_preprocessor() -> ColumnTransformer:
    """Cria o pré-processador de atributos (Pipeline de Feature Engineering)."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


def prepare_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prepara as matrizes de treino e teste."""
    df_clean = clean_data(df)
    X = df_clean[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df_clean[TARGET]

    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    