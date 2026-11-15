# Clasificación de Emociones en Rostros con CNN (Transfer Learning)

Proyecto de clasificación multiclase de emociones usando ResNet-18 y AlexNet
con transfer learning sobre el dataset FER.

## Estructura
- `data/` : dataset descargado con kagglehub (no versionado)
- `external_tests/` : imágenes externas para prueba final
- `models/` : modelos entrenados (no versionados)
- `src/` : código fuente
- `notebooks/` : experimentos exploratorios
- `results/` : matrices de confusión, curvas ROC, métricas

## Setup

```bash
pip install -r requirements.txt
python src/download_data.py
```

## Entrenamiento

```bash
python src/train.py --model resnet18
python src/train.py --model alexnet
```

## Evaluación

```bash
python src/evaluate.py --model resnet18
python src/evaluate.py --model alexnet
```

## Predicción con imágenes externas

```bash
python src/predict_external.py --model resnet18 --image external_tests/foto1.jpg
```
