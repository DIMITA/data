from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Activer CORS pour toutes les routes

# Charger le modèle entraîné
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "Landsapp API Flask prête 🚀"

@app.route("/api", methods=["POST"])
def predict():
    try:
        data = request.json

        # Extraire les 8 paramètres attendus
        features = [
            data["ni"],
            data["phosphore"],
            data["potassium"],
            data["magnesium"],
            data["ph"],
            data["temperature"],
            data["pluviometrie"],
            data["humidite"]
        ]

        features_array = np.array(features).reshape(1, -1)

        prediction = model.predict(features_array)

        return jsonify({"cultures": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
