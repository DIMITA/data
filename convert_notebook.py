#!/usr/bin/env python3
import os
import sys
import subprocess
import venv
import tempfile
import shutil

def create_venv():
    """Crée un environnement virtuel temporaire."""
    temp_dir = tempfile.mkdtemp()
    venv.create(temp_dir, with_pip=True)
    return temp_dir

def check_dependencies():
    """Vérifie si les dépendances nécessaires sont installées."""
    try:
        subprocess.check_call([sys.executable, "-c", "import jupyter"], 
                            stdout=subprocess.DEVNULL, 
                            stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

def install_dependencies():
    """Installe les dépendances nécessaires dans un environnement virtuel."""
    print("Création d'un environnement virtuel temporaire...")
    venv_dir = create_venv()
    
    # Activer l'environnement virtuel
    if sys.platform == "win32":
        python_path = os.path.join(venv_dir, "Scripts", "python.exe")
        pip_path = os.path.join(venv_dir, "Scripts", "pip.exe")
    else:
        python_path = os.path.join(venv_dir, "bin", "python")
        pip_path = os.path.join(venv_dir, "bin", "pip")
    
    print("Installation des dépendances...")
    subprocess.check_call([pip_path, "install", "jupyter"])
    
    return python_path

def convert_notebook(notebook_path, python_path):
    """Convertit un notebook Jupyter en script Python."""
    try:
        subprocess.check_call([python_path, "-m", "jupyter", "nbconvert", "--to", "script", notebook_path])
        print(f"Conversion réussie ! Le script a été créé : {notebook_path.replace('.ipynb', '.py')}")
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de la conversion : {e}")
        sys.exit(1)

def main():
    # Vérifier les arguments
    if len(sys.argv) != 2:
        print("Usage: python convert_notebook.py <chemin_vers_notebook.ipynb>")
        sys.exit(1)

    notebook_path = sys.argv[1]
    
    # Vérifier si le fichier existe
    if not os.path.exists(notebook_path):
        print(f"Erreur : Le fichier {notebook_path} n'existe pas")
        sys.exit(1)

    # Vérifier si c'est un fichier .ipynb
    if not notebook_path.endswith('.ipynb'):
        print("Erreur : Le fichier doit avoir l'extension .ipynb")
        sys.exit(1)

    # Vérifier et installer les dépendances si nécessaire
    if not check_dependencies():
        print("Jupyter n'est pas installé. Installation en cours...")
        python_path = install_dependencies()
    else:
        python_path = sys.executable

    # Convertir le notebook
    convert_notebook(notebook_path, python_path)

if __name__ == "__main__":
    main() 