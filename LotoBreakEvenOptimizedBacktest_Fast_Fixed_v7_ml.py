import pandas as pd
import numpy as np
from scipy.stats import entropy
from concurrent.futures import ThreadPoolExecutor
import time
import pickle
import os
import logging
import hashlib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from prophet import Prophet
import joblib

# Configurar logging
logging.basicConfig(
    filename="lotofacil_backtest.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def get_file_hash(file_path):
    """Calcula o hash MD5 do arquivo."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_data(file_path):
    """Carrega e valida o CSV."""
    logging.info("Carregando CSV: %s", file_path)
    df = pd.read_csv(file_path, sep=";")
    required_cols = ["Concurso"] + [f"Bola{i}" for i in range(1, 16)]
    if not all(col in df.columns for col in required_cols):
        raise ValueError("CSV inválido. Colunas esperadas: Concurso, Bola1, ..., Bola15")
    if df.isnull().any().any():
        raise ValueError("CSV contém valores nulos")
    return df

def calculate_entropy(game):
    """Calcula a entropia de um jogo."""
    hist, _ = np.histogram(game, bins=range(1, 27))
    hist = hist / hist.sum()
    return entropy(hist)

def get_winning_patterns(df, concurso, w_hist=50):
    """Identifica números frequentes."""
    recent_draws = df[df["Concurso"] <= concurso].tail(w_hist)
    balls = recent_draws[[f"Bola{i}" for i in range(1, 16)]].values
    freq = np.sum(balls == np.arange(1, 26)[:, None, None], axis=(1, 2))
    return np.argsort(freq)[-10:] + 1

def is_valid_game(game, corr_matrix, winning_numbers):
    """Valida um jogo."""
    sum_valid = 190 <= np.sum(game) <= 200
    ranges = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25)]
    moldura_valid = all(2 <= sum(start <= ball <= end for ball in game) <= 4 for start, end in ranges)
    pairs = len([x for x in game if x % 2 == 0])
    pairs_valid = pairs in [7, 8]
    consecutives = sum(1 for i in range(len(game)-1) if game[i+1] == game[i]+1)
    consec_valid = consecutives <= 2
    entropy_valid = calculate_entropy(game) > 3.2
    game_indices = [ball-1 for ball in game]
    corr_sum = np.sum(corr_matrix[np.ix_(game_indices, game_indices)])
    cluster_valid = corr_sum < 8
    winning_valid = sum(ball in winning_numbers for ball in game) >= 5
    return sum_valid and moldura_valid and pairs_valid and consec_valid and entropy_valid and cluster_valid and winning_valid

def check_duplicate_games(games):
    """Remove jogos duplicados."""
    unique_games = []
    for game in games:
        if not unique_games or all(np.sum(np.abs(game - ugame)) > 5 for ugame in unique_games):
            unique_games.append(game)
    return np.array(unique_games)

def normal_strategy(concurso, n, stats_cache, delay_cache, cansaco_cache, corr_cache, winning_cache, w_freq, C_linha, C_par):
    """Estratégia baseada em frequência."""
    freq = stats_cache[concurso]
    corr_matrix = corr_cache[concurso]
    winning_numbers = winning_cache[concurso]
    games = []
    for _ in range(n):
        while True:
            probs = freq / freq.sum()
            game = np.random.choice(range(1, 26), size=15, replace=False, p=probs)
            game = np.sort(game)
            if is_valid_game(game, corr_matrix, winning_numbers):
                games.append(game)
                break
    return check_duplicate_games(games)[:n]

def hibrido_strategy(concurso, n, stats_cache, delay_cache, cansaco_cache, corr_cache, winning_cache, w_freq, C_linha, C_par):
    """Estratégia baseada em frequência e atraso."""
    delay = delay_cache[concurso]
    freq = stats_cache[concurso]
    corr_matrix = corr_cache[concurso]
    winning_numbers = winning_cache[concurso]
    score = freq - C_linha * delay
    games = []
    for _ in range(n):
        while True:
            selected = np.argsort(score)[-15:] + 1
            game = np.sort(selected)
            if is_valid_game(game, corr_matrix, winning_numbers):
                games.append(game)
                break
            score += np.random.normal(0, 0.1, len(score))
    return check_duplicate_games(games)[:n]

def azarao_strategy(concurso, n, stats_cache, delay_cache, cansaco_cache, corr_cache, winning_cache, w_freq, C_linha, C_par):
    """Estratégia baseada em baixa correlação."""
    cansaco = cansaco_cache[concurso]
    corr_matrix = corr_cache[concurso]
    winning_numbers = winning_cache[concurso]
    score = -np.sum(corr_matrix, axis=0) - C_par * cansaco
    games = []
    for _ in range(n):
        while True:
            selected = np.argsort(score)[-15:] + 1
            game = np.sort(selected)
            if is_valid_game(game, corr_matrix, winning_numbers):
                games.append(game)
                break
            score += np.random.normal(0, 0.1, len(score))
    return check_duplicate_games(games)[:n]

def ml_strategy(concurso, n, stats_cache, delay_cache, cansaco_cache, corr_cache, winning_cache, w_freq, C_linha, C_par):
    """Estratégia baseada em ensemble de modelos ML."""
    freq = stats_cache[concurso]
    delay = delay_cache[concurso]
    corr_matrix = corr_cache[concurso]
    winning_numbers = winning_cache[concurso]
    
    # Carregar modelos
    models = {
        "rf": joblib.load("rf_model.pkl"),
        "lgbm": joblib.load("lgbm_model.pkl"),
        "xgb": joblib.load("xgb_model.pkl"),
        "catboost": joblib.load("catboost_model.pkl"),
        "lr": joblib.load("lr_model.pkl"),
        "nb": joblib.load("nb_model.pkl"),
        "mlp": joblib.load("mlp_model.pkl")
    }
    
    # Preparar features
    X_concurso = np.concatenate([freq, delay, np.sum(corr_matrix, axis=0)]).reshape(1, -1)
    
    # Previsões do ensemble
    probs = np.zeros(25)
    weights = {"rf": 0.2, "lgbm": 0.2, "xgb": 0.2, "catboost": 0.2, "lr": 0.1, "nb": 0.05, "mlp": 0.05}
    for name, model in models.items():
        prob = model.predict_proba(X_concurso)[:, 1]  # Probabilidade de cada número ser sorteado
        probs += weights[name] * prob
    
    # Adicionar previsões do Prophet
    prophet_df = pd.DataFrame({"ds": pd.date_range(start="2020-01-01", periods=len(freq), freq="D"), "y": freq})
    prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    prophet_model.fit(prophet_df)
    future = prophet_model.make_future_dataframe(periods=1)
    forecast = prophet_model.predict(future)
    prophet_probs = np.maximum(forecast["yhat"].iloc[-1], 0) / np.sum(np.maximum(forecast["yhat"].iloc[-1], 0))
    probs += 0.05 * prophet_probs
    
    probs /= probs.sum()
    
    games = []
    for _ in range(n):
        while True:
            game = np.random.choice(range(1, 26), size=15, replace=False, p=probs)
            game = np.sort(game)
            if is_valid_game(game, corr_matrix, winning_numbers):
                games.append(game)
                break
    return check_duplicate_games(games)[:n]

def precompute_statistics(df, w_freq, w_hist=50):
    """Pré-calcula estatísticas com cache."""
    cache_file = "stats_cache.pkl"
    csv_hash = get_file_hash("base_Lotofacil.csv")
    cache_params = {"w_freq": w_freq, "w_hist": w_hist, "csv_hash": csv_hash}

    if os.path.exists(cache_file):
        logging.info("Verificando cache de estatísticas...")
        with open(cache_file, "rb") as f:
            cached_data = pickle.load(f)
        if cached_data["params"] == cache_params:
            logging.info("Cache válido encontrado.")
            return cached_data["stats_cache"], cached_data["delay_cache"], \
                   cached_data["cansaco_cache"], cached_data["corr_cache"], \
                   cached_data["winning_cache"]

    logging.info("Calculando estatísticas...")
    stats_cache = {}
    delay_cache = {}
    cansaco_cache = {}
    corr_cache = {}
    winning_cache = {}

    for concurso in df["Concurso"]:
        recent_draws = df[df["Concurso"] <= concurso].tail(w_freq)
        balls = recent_draws[[f"Bola{i}" for i in range(1, 16)]].values

        freq = np.sum(balls == np.arange(1, 26)[:, None, None], axis=(1, 2))
        stats_cache[concurso] = freq / w_freq

        cansaco = np.array([np.sum(balls == ball) for ball in range(1, 26)]) / w_freq
        cansaco_cache[concurso] = cansaco

        last_seen = {ball: concurso for ball in range(1, 26)}
        delays = {ball: 0 for ball in range(1, 26)}
        for i, draw in recent_draws.iloc[::-1].iterrows():
            for ball in draw[1:]:
                if last_seen[ball] == concurso:
                    delays[ball] = concurso - i
                    last_seen[ball] = i
        delay_cache[concurso] = np.array([delays[ball] for ball in range(1, 26)])

        balls_matrix = np.zeros((25, len(recent_draws)))
        for i, draw in enumerate(recent_draws.values):
            for ball in draw[1:]:
                balls_matrix[ball-1, i] = 1
        corr_matrix = np.corrcoef(balls_matrix)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        corr_cache[concurso] = corr_matrix

        winning_cache[concurso] = get_winning_patterns(df, concurso, w_hist)

    cache_data = {
        "params": cache_params,
        "stats_cache": stats_cache,
        "delay_cache": delay_cache,
        "cansaco_cache": cansaco_cache,
        "corr_cache": corr_cache,
        "winning_cache": winning_cache
    }
    with open(cache_file, "wb") as f:
        pickle.dump(cache_data, f)
    logging.info("Estatísticas salvas no cache: %s", cache_file)

    return stats_cache, delay_cache, cansaco_cache, corr_cache, winning_cache

def backtest_games(week_df, stats_cache, delay_cache, cansaco_cache, corr_cache, winning_cache, n_normal, n_hibrido, n_azarao, n_ml, w_freq, C_linha, C_par):
    """Executa backtesting para uma semana."""
    week_return = 0
    week_cost = 20 * 3 * len(week_df)  # 20 jogos
    hits_per_game = []
    games_cache_file = f"games_cache_week_{week_df['Concurso'].iloc[0]}.pkl"
    cache_params = {"n_normal": n_normal, "n_hibrido": n_hibrido, "n_azarao": n_azarao, "n_ml": n_ml, "w_freq": w_freq, "C_linha": C_linha, "C_par": C_par}

    if os.path.exists(games_cache_file):
        with open(games_cache_file, "rb") as f:
            cached_data = pickle.load(f)
        if cached_data["params"] == cache_params:
            logging.info("Cache de jogos encontrado para semana %d", week_df['Concurso'].iloc[0])
            games_per_contest = cached_data["games"]
        else:
            games_per_contest = None
    else:
        games_per_contest = None

    for idx, draw in week_df.iterrows():
        concurso = draw["Concurso"]
        draw_array = draw[[f"Bola{i}" for i in range(1, 16)]].values.astype(int)

        if games_per_contest is None or concurso not in games_per_contest:
            normal_games = normal_strategy(concurso, n_normal, stats_cache, delay_cache, cansaco_cache, corr_cache, winning_cache, w_freq, C_linha, C_par)
            hibrido_games = hibrido_strategy(concurso, n_hibrido, stats_cache, delay_cache, cansaco_cache, corr_cache, winning_cache, w_freq, C_linha, C_par)
            azarao_games = azarao_strategy(concurso, n_azarao, stats_cache, delay_cache, cansaco_cache, corr_cache, winning_cache, w_freq, C_linha, C_par)
            ml_games = ml_strategy(concurso, n_ml, stats_cache, delay_cache, cansaco_cache, corr_cache, winning_cache, w_freq, C_linha, C_par)
            games_array = np.concatenate([normal_games, hibrido_games, azarao_games, ml_games])
            if games_per_contest is None:
                games_per_contest = {}
            games_per_contest[concurso] = games_array
        else:
            games_array = games_per_contest[concurso]

        hits = np.sum(np.isin(games_array, draw_array[:, None]), axis=2)
        hits_per_game.extend(hits.flatten())

        week_return += np.sum((hits == 11) * 6 + (hits == 12) * 12 + (hits == 13) * 30 + (hits == 14) * 2000 + (hits == 15) * 2000000)

    if games_per_contest is not None:
        cache_data = {"params": cache_params, "games": games_per_contest}
        with open(games_cache_file, "wb") as f:
            pickle.dump(cache_data, f)
        logging.info("Jogos salvos no cache: %s", games_cache_file)

    return {
        "Return": week_return,
        "Cost": week_cost,
        "Net": week_return - week_cost,
        "Hits": hits_per_game
    }

def main():
    """Função principal."""
    start_time = time.time()
    logging.info("Iniciando backtest da Lotofácil")

    df = load_data("base_Lotofacil.csv")
    logging.info("CSV carregado. Concursos: %d", len(df))

    w_freq = 30
    C_linha = 0.02
    C_par = 0.15
    n_normal = 6
    n_hibrido = 4
    n_azarao = 4
    n_ml = 6
    weeks = 4  # 1 mês (4 semanas)

    stats_cache, delay_cache, cansaco_cache, corr_cache, winning_cache = precompute_statistics(df, w_freq)
    logging.info("Estatísticas pré-calculadas")

    df = df.tail(24)  # 4 semanas × 6 concursos
    weeks_data = [df.iloc[i:i+6] for i in range(0, len(df), 6)]

    logging.info("Iniciando backtesting para %d semanas", weeks)
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(backtest_games, week_df, stats_cache, delay_cache, cansaco_cache, corr_cache, 
                           winning_cache, n_normal, n_hibrido, n_azarao, n_ml, w_freq, C_linha, C_par)
            for week_df in weeks_data
        ]
        for idx, future in enumerate(futures):
            result = future.result()
            results.append(result)
            logging.info("Semana %d processada. Retorno: R$%.2f, Custo: R$%.2f", idx+1, result["Return"], result["Cost"])

    results_df = pd.DataFrame(results)
    results_df["Hits"] = results_df["Hits"].apply(lambda x: np.array(x))
    results_df["Breakeven"] = results_df["Return"] >= results_df["Cost"]

    summary = {
        "Total_Return": results_df["Return"].sum(),
        "Total_Cost": results_df["Cost"].sum(),
        "Total_Net": results_df["Net"].sum(),
        "Total_Weeks": len(results_df),
        "Mean_Weekly_Return": results_df["Return"].mean(),
        "Mean_Weekly_Cost": results_df["Cost"].mean(),
        "Mean_Weekly_Net": results_df["Net"].mean(),
        "Breakeven_Rate": results_df["Breakeven"].mean()
    }

    results_df.to_csv("backtest_results_LotoV7.csv", index=False)
    logging.info("Resultados salvos em backtest_results_LotoV7.csv")

    print(f"Retorno Total: R$ {summary['Total_Return']:.2f}")
    print(f"Custo Total: R$ {summary['Total_Cost']:.2f}")
    print(f"Lucro Líquido Total: R$ {summary['Total_Net']:.2f}")
    print(f"Número de Semanas: {summary['Total_Weeks']}")
    print(f"Retorno Médio Semanal: R$ {summary['Mean_Weekly_Return']:.2f}")
    print(f"Custo Médio Semanal: R$ {summary['Mean_Weekly_Cost']:.2f}")
    print(f"Lucro Médio Semanal: R$ {summary['Mean_Weekly_Net']:.2f}")
    print(f"Taxa de Empate: {summary['Breakeven_Rate']:.2%}")
    print(f"Tempo de Execução: {time.time() - start_time:.2f} segundos")

    logging.info("Backtest concluído. Tempo total: %.2f segundos", time.time() - start_time)

    return summary

if __name__ == "__main__":
    main()