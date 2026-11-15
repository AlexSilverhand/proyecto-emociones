# src/evaluate.py
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize
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
    _, test_loader, classes = get_dataloaders(model_name, batch_size=32)

    model = get_model(model_name, num_classes=len(classes), pretrained=False)
    model.load_state_dict(torch.load(MODELS_DIR / f"{model_name}_best.pth", map_location=device))
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

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

    # Matriz de confusión
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrices" / f"{model_name}_cm.png")
    plt.close()

    # Reporte de métricas
    report = classification_report(all_labels, all_preds,
                                   target_names=classes, digits=4)
    with open(RESULTS_DIR / "metrics" / f"{model_name}_report.txt", "w") as f:
        f.write(report)
    print(report)

    # Curvas ROC (one-vs-rest)
    y_bin = label_binarize(all_labels, classes=range(len(classes)))
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Curvas ROC - {model_name}')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curves" / f"{model_name}_roc.png")
    plt.close()

    print(f"Resultados guardados en {RESULTS_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["resnet18", "alexnet"])
    args = parser.parse_args()
    evaluate(args.model)