# src/predict_external.py
import argparse
import torch
from PIL import Image
from pathlib import Path
from models import get_model
from dataset import get_transforms_resnet, get_transforms_alexnet

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

def predict(model_name, image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar clases (asumimos el orden de ImageFolder)
    # Puedes guardar las clases en un archivo durante el entrenamiento para más seguridad
    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    if model_name == "resnet18":
        transform = get_transforms_resnet(train=False)
    else:
        transform = get_transforms_alexnet(train=False)

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    model = get_model(model_name, num_classes=len(classes), pretrained=False)
    model.load_state_dict(torch.load(MODELS_DIR / f"{model_name}_best.pth", map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

    print(f"Imagen: {image_path}")
    print(f"Predicción: {classes[pred.item()]} (confianza: {conf.item():.2%})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["resnet18", "alexnet"])
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()
    predict(args.model, args.image)