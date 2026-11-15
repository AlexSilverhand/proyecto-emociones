# src/download_data.py
import kagglehub
from pathlib import Path

# Descarga el dataset en la carpeta data/ del proyecto
# La ruta se resuelve de forma relativa a la raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

print(f"Descargando dataset en: {DATA_DIR}")
path = kagglehub.dataset_download(
    "ananthu017/emotion-detection-fer",
    output_dir=str(DATA_DIR)
)
print(f"Dataset descargado en: {path}")