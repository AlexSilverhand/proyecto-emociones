# src/dataset.py
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Buscar automáticamente la carpeta que contiene 'train' y 'test'
def find_dataset_root():
    for path in DATA_DIR.rglob("train"):
        if path.is_dir() and (path.parent / "test").is_dir():
            return path.parent
    raise FileNotFoundError(
        f"No se encontró la carpeta 'train' dentro de {DATA_DIR}. "
        "Revisa que el dataset se haya descargado bien."
    )

DATASET_ROOT = find_dataset_root()
print(f"Dataset encontrado en: {DATASET_ROOT}")

# ---------------------------------------------------------
# Transformaciones
# ---------------------------------------------------------
def get_transforms_resnet(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validación y Test: SIN augmentación
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_transforms_alexnet(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validación y Test: SIN augmentación
        return transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# ---------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------
def get_dataloaders(model_name, batch_size=32):
    if model_name == "resnet18":
        transform_train = get_transforms_resnet(train=True)
        transform_eval = get_transforms_resnet(train=False)
    elif model_name == "alexnet":
        transform_train = get_transforms_alexnet(train=True)
        transform_eval = get_transforms_alexnet(train=False)
    else:
        raise ValueError("model_name debe ser 'resnet18' o 'alexnet'")

    # 1. Dataset de entrenamiento completo (con augmentación)
    full_train_dataset = ImageFolder(root=DATASET_ROOT / "train", transform=transform_train)
    
    # 2. Dataset de test (sin augmentación, solo para evaluación final)
    test_dataset = ImageFolder(root=DATASET_ROOT / "test", transform=transform_eval)

    # 3. Separar Train en Train (85%) y Validación (15%)
    train_size = int(0.85 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    # Usamos un generador con semilla fija para que la división sea siempre la misma
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=generator)

    # IMPORTANTE: La validación no debe tener data augmentation.
    # Como random_split comparte el dataset base, creamos un "wrapper" para cambiarle la transformación a la validación.
    class SubsetWithTransform(torch.utils.data.Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform
        def __getitem__(self, index):
            x, y = self.subset[index]
            # El subset ya devuelve el tensor transformado, así que no podemos re-transformarlo aquí fácilmente.
            # La forma correcta es que el dataset base no tenga transformación y aplicar aquí.
            return x, y
        def __len__(self):
            return len(self.subset)

    # Para evitar complicaciones con random_split y transformaciones, 
    # vamos a hacer un truco: cargamos el dataset dos veces (uno para train y otro para val) 
    # y usamos el subset.
    
    # Alternativa limpia:
    # Como es más fácil, crearemos un dataset sin transformación y le aplicaremos la transformación en el subset.
    # Pero para no sobrecomplicar el código, usaremos una clase que envuelva el subset.
    
    # Vamos a redefinir el dataset base sin transformación
    base_dataset = ImageFolder(root=DATASET_ROOT / "train", transform=None)
    
    # Hacemos la división sobre el dataset base
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(base_dataset)), [train_size, val_size], generator=generator
    )
    
    # Creamos un dataset personalizado que aplica la transformación correspondiente
    class TransformedSubset(torch.utils.data.Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
        def __getitem__(self, idx):
            # Obtener la imagen y etiqueta originales
            original_idx = self.indices[idx]
            image, label = self.dataset[original_idx]
            if self.transform:
                image = self.transform(image)
            return image, label
        def __len__(self):
            return len(self.indices)

    train_dataset = TransformedSubset(base_dataset, train_indices, transform_train)
    val_dataset = TransformedSubset(base_dataset, val_indices, transform_eval)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, full_train_dataset.classes