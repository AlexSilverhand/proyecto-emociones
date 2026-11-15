# src/evaluate.py
import argparse
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize

# Asegurar que encuentra los módulos
sys.path.append(str(Path(__file__).resolve().parent))
from dataset import get_dataloaders
from models import get_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "confusion_matrices").mkdir(exist_ok=True)
(RESULTS_DIR / "roc_curves").mkdir(exist_ok=True)
(RESULTS_DIR / "metrics").mkdir(exist_ok=True)

def evaluate(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # 1. Cargar Datos (Ahora devuelve 4 variables, ignoramos train y val)
    _, _, test_loader, classes = get_dataloaders(model_name, batch_size=32)
    
    # 2. Cargar Modelo
    model = get_model(model_name, num_classes=len(classes), pretrained=False, freeze_features=False)
    
    weights_path = MODELS_DIR / f"{model_name}_best.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"No se encontraron los pesos en {weights_path}. Entrena el modelo primero.")
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    # 3. Inferencia en el Test Set
    print("Evaluando en el conjunto de prueba...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # 4. Matriz de Confusión
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.tight_layout()
    cm_path = RESULTS_DIR / "confusion_matrices" / f"{model_name}_cm.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"Matriz de confusión guardada en: {cm_path}")

    # 5. Reporte de Métricas
    report = classification_report(all_labels, all_preds, target_names=classes, digits=4)
    report_path = RESULTS_DIR / "metrics" / f"{model_name}_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print("\n--- Reporte de Clasificación ---")
    print(report)
    print(f"Reporte guardado en: {report_path}")

    # 6. Curvas ROC (One-vs-Rest)
    y_bin = label_binarize(all_labels, classes=range(len(classes)))
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(classes):
        # Solo calculamos ROC si la clase tiene ambas etiquetas (0 y 1) en el test
        if len(np.unique(y_bin[:, i])) > 1:
            fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
            
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Curvas ROC - {model_name}')
    plt.legend(loc='lower right')
    plt.tight_layout()
    roc_path = RESULTS_DIR / "roc_curves" / f"{model_name}_roc.png"
    plt.savefig(roc_path)
    plt.close()
    print(f"Curvas ROC guardadas en: {roc_path}")
    
    print("\nEvaluación completada con éxito.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["resnet18", "alexnet"])
    args = parser.parse_args()
    evaluate(args.model)