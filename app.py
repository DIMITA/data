from flask import Flask, request, jsonify
import pickle
import numpy as np

# Charger le modèle entraîné (RandomForestClassifier)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

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
