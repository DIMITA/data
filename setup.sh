#!/bin/bash

# Créer l'environnement virtuel
python3 -m venv venv

# Activer l'environnement virtuel
source venv/bin/activate

# Installer les dépendances avec des versions spécifiques compatibles
pip install flask==2.0.1 werkzeug==2.0.1 jupyter==1.0.0 nbconvert==6.0.7

# Créer le dossier uploads
mkdir -p uploads

echo "Environnement configuré avec succès !"
echo "Pour activer l'environnement virtuel, exécutez : source venv/bin/activate"
echo "Pour lancer l'API, exécutez : python app.py" 