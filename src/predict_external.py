# src/predict_external.py
import argparse
import json
import torch
from PIL import Image
from pathlib import Path
import sys

# Asegurar que encuentra los módulos
sys.path.append(str(Path(__file__).resolve().parent))
from models import get_model
from dataset import get_transforms_resnet, get_transforms_alexnet

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

def predict(model_name, image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Cargar el mapeo de clases guardado durante el entrenamiento
    classes_path = MODELS_DIR / f"{model_name}_classes.json"
    if not classes_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo {classes_path}. "
            "Asegúrate de entrenar el modelo primero."
        )
    
    with open(classes_path, "r") as f:
        classes = json.load(f)

    # 2. Cargar las transformaciones correctas (sin augmentation)
    if model_name == "resnet18":
        transform = get_transforms_resnet(train=False)
    elif model_name == "alexnet":
        transform = get_transforms_alexnet(train=False)
    else:
        raise ValueError("model_name debe ser 'resnet18' o 'alexnet'")

    # 3. Procesar la imagen
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error al abrir la imagen: {e}")
        return

    tensor = transform(image).unsqueeze(0).to(device) # Añadir dimensión de batch

    # 4. Cargar el modelo
    # Nota: freeze_features=False es irrelevante aquí porque solo haremos forward pass,
    # pero debe coincidir con la estructura para cargar los pesos correctamente.
    model = get_model(model_name, num_classes=len(classes), pretrained=False, freeze_features=False)
    
    weights_path = MODELS_DIR / f"{model_name}_best.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"No se encontraron los pesos en {weights_path}")
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    # 5. Inferencia
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        
        # Obtener las 3 predicciones más probables para ver si hay confusión
        top3_prob, top3_idx = torch.topk(probs, 3, dim=1)
        
        conf = top3_prob[0][0].item()
        pred_idx = top3_idx[0][0].item()
        pred_class = classes[pred_idx]

    print(f"\n--- Resultado de la Predicción ({model_name}) ---")
    print(f"Imagen: {image_path}")
    print(f"Predicción principal: {pred_class.upper()} (Confianza: {conf:.2%})")
    print("Top 3 probabilidades:")
    for i in range(3):
        idx = top3_idx[0][i].item()
        prob = top3_prob[0][i].item()
        print(f"  {i+1}. {classes[idx]}: {prob:.2%}")
    print("-" * 45)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["resnet18", "alexnet"])
    parser.add_argument("--image", type=str, required=True, help="Ruta a la imagen a predecir")
    args = parser.parse_args()
    predict(args.model, args.image)