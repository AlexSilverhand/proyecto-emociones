# src/evaluate_external.py
import os
import csv
import json
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import sys

# Asegurar que encuentra los módulos
sys.path.append(str(Path(__file__).resolve().parent))
from models import get_model
from dataset import get_transforms_resnet, get_transforms_alexnet

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
EXTERNAL_DIR = PROJECT_ROOT / "external_tests"
RESULTS_DIR = PROJECT_ROOT / "results" / "external_snapshots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_model(model_name, classes, device):
    model = get_model(model_name, num_classes=len(classes), pretrained=False, freeze_features=False)
    weights_path = MODELS_DIR / f"{model_name}_best.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(model, image_path, transform, classes, device):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
        
    return classes[pred.item()], conf.item(), image

def create_snapshot(image, real_class, pred_class, conf, model_name, save_path):
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    
    # Definir color (verde si acierta, rojo si falla)
    color = 'green' if real_class.lower() in pred_class.lower() or pred_class.lower() in real_class.lower() else 'red'
    
    title = f"Modelo: {model_name.upper()}\n"
    title += f"Real: {real_class.upper()}\n"
    title += f"Predicción: {pred_class.upper()} ({conf:.2%})"
    
    plt.title(title, color=color, fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Obtener imágenes
    valid_extensions = {".jpg", ".jpeg", ".png"}
    images = [f for f in EXTERNAL_DIR.iterdir() if f.suffix.lower() in valid_extensions]
    
    if not images:
        print(f"No se encontraron imágenes en {EXTERNAL_DIR}. Agrega fotos .jpg o .png.")
        return
        
    print(f"Se encontraron {len(images)} imágenes para evaluar.")

    # Cargar clases y modelos
    with open(MODELS_DIR / "resnet18_classes.json", "r") as f:
        classes = json.load(f)
        
    resnet = load_model("resnet18", classes, device)
    alexnet = load_model("alexnet", classes, device)
    
    transform_resnet = get_transforms_resnet(train=False)
    transform_alexnet = get_transforms_alexnet(train=False)

    results = []
    
    for img_path in images:
        # Asumimos que el nombre del archivo es la emoción real (ej. "feliz.jpg")
        real_emotion = img_path.stem 
        
        # Predecir con ResNet-18
        pred_r, conf_r, img_obj = predict_image(resnet, img_path, transform_resnet, classes, device)
        snap_path_r = RESULTS_DIR / f"{img_path.stem}_resnet18.png"
        create_snapshot(img_obj, real_emotion, pred_r, conf_r, "ResNet-18", snap_path_r)
        
        # Predecir con AlexNet
        pred_a, conf_a, _ = predict_image(alexnet, img_path, transform_alexnet, classes, device)
        snap_path_a = RESULTS_DIR / f"{img_path.stem}_alexnet.png"
        create_snapshot(img_obj, real_emotion, pred_a, conf_a, "AlexNet", snap_path_a)
        
        results.append({
            "Imagen": img_path.name,
            "Emoción Real (Nombre)": real_emotion,
            "Predicción ResNet-18": pred_r,
            "Confianza ResNet-18": f"{conf_r:.2%}",
            "Predicción AlexNet": pred_a,
            "Confianza AlexNet": f"{conf_a:.2%}"
        })
        
        print(f"[{img_path.name}] Real: {real_emotion} | ResNet: {pred_r} ({conf_r:.1%}) | AlexNet: {pred_a} ({conf_a:.1%})")

    # Guardar tabla CSV
    csv_path = PROJECT_ROOT / "results" / "tabla_externa.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nTabla comparativa guardada en: {csv_path}")
    print(f"Instantáneas guardadas en: {RESULTS_DIR}")

if __name__ == "__main__":
    main()