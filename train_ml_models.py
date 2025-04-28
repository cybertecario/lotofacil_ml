import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def train_models():
    # Carregar dados
    data = pd.read_csv('base_Lotofacil.csv', sep=';')
    
    # Preparar features (X) e target (y)
    X = data[['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6', 'Bola7', 
              'Bola8', 'Bola9', 'Bola10', 'Bola11', 'Bola12', 'Bola13', 'Bola14', 'Bola15']]
    y = X.shift(-1).dropna()  # Prever próximo sorteio
    X = X.iloc[:-1]  # Alinhar com y
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Treinar modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Salvar modelo
    joblib.dump(model, 'lotofacil_model.pkl')
    
    print("Modelo treinado e salvo como lotofacil_model.pkl")
    return model

if __name__ == "__main__":
    train_models()
