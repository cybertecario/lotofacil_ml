import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import joblib
import logging

# Configurar logging
logging.basicConfig(
    filename="lotofacil_backtest.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def prepare_features(df, w_freq=30):
    """Prepara features para treinamento."""
    logging.info("Preparando features para treinamento...")
    X, y = [], []
    for concurso in df["Concurso"][w_freq:]:
        recent = df[df["Concurso"] <= concurso].tail(w_freq)
        balls = recent[[f"Bola{i}" for i in range(1, 16)]].values
        
        # Frequência
        freq = np.sum(balls == np.arange(1, 26)[:, None, None], axis=(1, 2)) / w_freq
        
        # Atraso
        delays = np.zeros(25)
        last_seen = {ball: concurso for ball in range(1, 26)}
        for i, draw in recent.iloc[::-1].iterrows():
            for ball in draw[1:]:
                if last_seen[ball] == concurso:
                    delays[ball-1] = concurso - i
                    last_seen[ball] = i
        
        # Correlação
        balls_matrix = np.zeros((25, len(recent)))
        for i, draw in enumerate(recent.values):
            for ball in draw[1:]:
                balls_matrix[ball-1, i] = 1
        corr_matrix = np.corrcoef(balls_matrix)
        corr_sum = np.sum(corr_matrix, axis=0)
        
        # Features
        X.append(np.concatenate([freq, delays, corr_sum]))
        
        # Rótulo
        draw = df[df["Concurso"] == concurso][[f"Bola{i}" for i in range(1, 16)]].values[0]
        y.append(np.isin(np.arange(1, 26), draw).astype(int))
    
    logging.info("Features preparadas. X: %s, y: %s", str(np.shape(X)), str(np.shape(y)))
    return np.array(X), np.array(y)

def train_models():
    """Treina todos os modelos e salva em .pkl."""
    df = pd.read_csv("base_Lotofacil.csv", sep=";")
    X, y = prepare_features(df)
    
    models = {
        "rf": RandomForestClassifier(n_estimators=100, random_state=42),
        "lgbm": LGBMClassifier(n_estimators=100, random_state=42),
        "xgb": XGBClassifier(n_estimators=100, random_state=42),
        "catboost": CatBoostClassifier(n_estimators=100, verbose=False, random_state=42),
        "lr": LogisticRegression(multi_class="ovr", random_state=42),
        "nb": GaussianNB(),
        "mlp": MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42)
    }

    for name, model in models.items():
        logging.info("Treinando modelo %s...", name)
        model.fit(X, y)
        joblib.dump(model, f"{name}_model.pkl")
        logging.info("Modelo %s salvo como %s_model.pkl", name, name)

if __name__ == "__main__":
    train_models()