from flask import Flask, request, jsonify
import os
import subprocess
import sys

app = Flask(__name__)

@app.route('/', methods=['POST'])
def convert_notebook():
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier n\'a été envoyé'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    
    if not file.filename.endswith('.ipynb'):
        return jsonify({'error': 'Le fichier doit être un notebook Jupyter (.ipynb)'}), 400
    
    # Sauvegarder le fichier temporairement
    temp_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(temp_path)
    
    try:
        # Utiliser le script de conversion
        subprocess.check_call([sys.executable, 'convert_notebook.py', temp_path])
        
        # Lire le contenu du fichier converti
        py_file = temp_path.replace('.ipynb', '.py')
        with open(py_file, 'r') as f:
            content = f.read()
        
        # Nettoyer les fichiers temporaires
        os.remove(temp_path)
        os.remove(py_file)
        
        return jsonify({
            'success': True,
            'content': content
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 