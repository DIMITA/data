import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import pickle

# === Config ===
notebook_path = "LandsappTest.ipynb"
pickle_path = "model.pkl"
model_variable_name = "RF"  # Le nom de la variable dans ton notebook

# === Lire et exécuter le notebook ===
with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
env = {}

try:
    ep.preprocess(notebook, {'metadata': {'path': '.'}})
    for cell in notebook.cells:
        if cell.cell_type == "code":
            exec(cell.source, env)
except Exception as e:
    print(f"[⚠️] Erreur pendant l'exécution : {e}")

# === Sauvegarder le modèle ===
if model_variable_name in env:
    with open(pickle_path, "wb") as f:
        pickle.dump(env[model_variable_name], f)
    print(f"[✅] Modèle '{model_variable_name}' sauvegardé dans '{pickle_path}'")
else:
    print(f"[❌] Variable '{model_variable_name}' non trouvée.")
