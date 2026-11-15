# src/models.py
import torch.nn as nn
import torchvision.models as models

def get_model(model_name, num_classes=7, pretrained=True, freeze_features=False):
    """
    Carga un modelo pre-entrenado y adapta la capa final para 'num_classes'.
    
    Args:
        model_name (str): 'resnet18' o 'alexnet'.
        num_classes (int): Número de emociones (7 por defecto).
        pretrained (bool): Usar pesos de ImageNet.
        freeze_features (bool): Si es True, congela las capas convolucionales (útil para AlexNet).
    """
    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        
        # Congelar capas iniciales si se solicita (opcional para ResNet)
        if freeze_features:
            for param in model.parameters():
                param.requires_grad = False
                
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
        
        # Congelar el extractor de características (capas convolucionales)
        if freeze_features:
            for param in model.features.parameters():
                param.requires_grad = False
                
        # Reemplazar la última capa del clasificador
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