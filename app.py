from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)  # Activer CORS pour toutes les routes

# Charger le modèle et les données nécessaires
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

@app.route('/api', methods=['POST'])
def predict():
    try:
        # Récupérer les données de la requête
        data = request.get_json()
        
        # Créer un DataFrame avec les données reçues
        input_data = pd.DataFrame([data])
        
        # Vérifier que toutes les colonnes nécessaires sont présentes
        missing_columns = set(feature_columns) - set(input_data.columns)
        if missing_columns:
            return jsonify({
                'error': f'Colonnes manquantes : {missing_columns}'
            }), 400
        
        # Sélectionner uniquement les colonnes nécessaires dans le bon ordre
        input_data = input_data[feature_columns]
        
        # Faire la prédiction
        prediction = model.predict(input_data)
        
        # Convertir la prédiction en nom de plante
        plant_name = label_encoder.inverse_transform(prediction)[0]
        
        return jsonify({
            'prediction': plant_name
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
