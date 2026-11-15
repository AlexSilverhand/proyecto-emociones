# src/models.py
import torch.nn as nn
import torchvision.models as models

def get_model(model_name, num_classes=7, pretrained=True):
    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        # Reemplazar la capa fully connected final
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    elif model_name == "alexnet":
        model = models.alexnet(pretrained=pretrained)
        # Reemplazar la última capa del classifier
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    else:
        raise ValueError("model_name debe ser 'resnet18' o 'alexnet'")
    
    return model