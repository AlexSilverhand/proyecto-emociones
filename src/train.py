# src/train.py
import argparse
import sys
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

# Asegurar que encuentra los módulos
sys.path.append(str(Path(__file__).resolve().parent))
from dataset import get_dataloaders
from models import get_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "loss_curves").mkdir(parents=True, exist_ok=True)


def train(model_name, epochs=20, lr=1e-3, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # 1. Cargar Datos (Ahora devuelve 4 variables)
    train_loader, val_loader, test_loader, classes = get_dataloaders(model_name, batch_size)
    print(f"Clases: {classes}")

    # 2. Calcular pesos de clase para el desbalance (Class Weights)
    # Obtenemos las etiquetas del conjunto de entrenamiento
    # train_loader.dataset es un TransformedSubset, accedemos a sus indices
    train_indices = train_loader.dataset.indices
    base_dataset = train_loader.dataset.dataset
    train_labels = [base_dataset.targets[i] for i in train_indices]
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Pesos de clase calculados (para balancear): {class_weights.cpu().numpy()}")

    # 3. Crear Modelo
    # Si es AlexNet, congelamos las capas convolucionales inicialmente
    freeze = (model_name == "alexnet")
    model = get_model(model_name, num_classes=len(classes), pretrained=True, freeze_features=freeze).to(device)
    
    # Si congelamos, el optimizador solo actualizará las capas que requieren gradiente
    params_to_update = [p for p in model.parameters() if p.requires_grad]

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(params_to_update, lr=lr)

    best_val_loss = float('inf')
    save_path = MODELS_DIR / f"{model_name}_best.pth"
    
    # Guardar las clases para no adivinar en predict_external.py
    with open(MODELS_DIR / f"{model_name}_classes.json", "w") as f:
        json.dump(classes, f)

    # Listas para las gráficas
    history_train_loss = []
    history_val_loss = []
    history_val_acc = []

    for epoch in range(epochs):
        # --- ENTRENAMIENTO ---
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        history_train_loss.append(avg_train_loss)

        # --- VALIDACIÓN (Ahora usamos val_loader, NO test_loader) ---
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        history_val_loss.append(avg_val_loss)
        history_val_acc.append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%")

        # Guardar el mejor modelo basado en Validación Loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> Mejor modelo guardado (Val Loss={best_val_loss:.4f}, Val Acc={val_acc:.2f}%)")

    # --- GENERAR GRÁFICAS ---
    print("Generando gráficas de entrenamiento...")
    
    # Gráfica de Pérdida (Loss)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), history_train_loss, label='Train Loss', marker='o')
    plt.plot(range(1, epochs + 1), history_val_loss, label='Val Loss', marker='s')
    plt.title(f'Curva de Pérdida (Loss) - {model_name}')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "loss_curves" / f"{model_name}_loss.png")
    plt.close()

    # Gráfica de Precisión (Accuracy)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), history_val_acc, label='Val Accuracy', color='green', marker='^')
    plt.title(f'Curva de Precisión (Accuracy) - {model_name}')
    plt.xlabel('Época')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "loss_curves" / f"{model_name}_accuracy.png")
    plt.close()

    print(f"Entrenamiento finalizado. Mejor Val Loss: {best_val_loss:.4f}")
    print(f"Modelo guardado en: {save_path}")
    print(f"Gráficas guardadas en: {RESULTS_DIR / 'loss_curves'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["resnet18", "alexnet"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    train(args.model, args.epochs, args.lr, args.batch_size)