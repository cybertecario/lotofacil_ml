from flask import Flask, request, jsonify
import os
import pandas as pd
from LotoBreakEvenOptimizedBacktest_Fast_Fixed_v7_ml import main as run_backtest
from train_ml_models import train_models

app = Flask(__name__, static_folder="static", static_url_path="/")

@app.route("/")
def serve_html():
    return app.send_static_file("index.html")

@app.route("/train", methods=["POST"])
def train():
    try:
        if not os.path.exists("base_Lotofacil.csv"):
            return jsonify({"error": "Por favor, faça upload do base_Lotofacil.csv primeiro"}), 400
        
        train_models()  # Treina os modelos
        return jsonify({"message": "Modelos treinados com sucesso! Arquivos .pkl gerados."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/process", methods=["POST"])
def process_csv():
    try:
        file = request.files["csv"]
        file.save("base_Lotofacil.csv")
        weeks = int(request.form.get("weeks", 4))
        
        # Verificar se os modelos existem
        model_files = ["rf_model.pkl", "lgbm_model.pkl", "xgb_model.pkl", "catboost_model.pkl", 
                       "lr_model.pkl", "nb_model.pkl", "mlp_model.pkl"]
        if not all(os.path.exists(f) for f in model_files):
            return jsonify({"error": "Modelos não encontrados. Treine os modelos primeiro."}), 400
        
        result = run_backtest()
        return jsonify({
            "total_bet": result["Total_Cost"],
            "total_won": result["Total_Return"],
            "net_profit": result["Total_Net"],
            "weekly_profit": result["Mean_Weekly_Net"],
            "roi": (result["Total_Net"] / result["Total_Cost"]) * 100 if result["Total_Cost"] > 0 else 0
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))