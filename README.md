# Clasificación de emociones faciales

Proyecto académico para comparar ResNet-18 y AlexNet en la clasificación de siete expresiones faciales mediante transfer learning. El repositorio incluye la descarga y carga del dataset, el entrenamiento, la evaluación sobre un conjunto de prueba reservado y pruebas exploratorias con imágenes externas.

## Requisitos

- Python 3.11 o posterior
- Git
- Conexión a internet para descargar el dataset y los pesos preentrenados de ImageNet
- GPU NVIDIA opcional; si PyTorch detecta CUDA, el entrenamiento la utiliza automáticamente

## Preparar el entorno

Desde la carpeta raíz del repositorio, crea y activa un entorno virtual.

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**macOS o Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si se necesita una instalación de PyTorch compatible con una versión específica de CUDA, instala primero la variante indicada por el [selector oficial de PyTorch](https://pytorch.org/get-started/locally/) y después instala las dependencias restantes.

## Descargar los datos

El dataset se descarga con KaggleHub dentro de `data/`:

```bash
python src/download_data.py
```

El código busca automáticamente una carpeta que contenga las particiones `train/` y `test/`. Cada una debe tener una subcarpeta por clase. Las siete clases esperadas son `angry`, `disgusted`, `fearful`, `happy`, `neutral`, `sad` y `surprised`.

## Entrenar

Entrena cada arquitectura desde la raíz del proyecto:

```bash
python src/train.py --model resnet18
python src/train.py --model alexnet
```

Por defecto, el entrenamiento usa 20 épocas, tasa de aprendizaje `0.001` y lotes de 32 imágenes. Estos valores pueden cambiarse desde la línea de comandos:

```bash
python src/train.py --model resnet18 --epochs 20 --lr 0.001 --batch_size 32
```

El flujo separa el directorio `train/` en 85% para entrenamiento y 15% para validación, usando la semilla 42. La partición `test/` se reserva para la evaluación final. La división actual es reproducible, pero no estratificada por clase.

ResNet-18 ajusta todos sus parámetros. AlexNet empieza con sus capas convolucionales congeladas y ajusta el clasificador. Ambos modelos usan pesos por clase para reducir el efecto del desbalance y seleccionan el mejor checkpoint según la pérdida de validación.

## Evaluar

Después de entrenar un modelo, evalúalo sobre el conjunto de prueba:

```bash
python src/evaluate.py --model resnet18
python src/evaluate.py --model alexnet
```

La evaluación genera una matriz de confusión, un reporte de precisión, recall y F1 por clase, y una curva ROC multiclase.

## Probar imágenes externas

Para predecir una imagen individual:

```bash
python src/predict_external.py --model resnet18 --image external_tests/foto.jpg
```

El script muestra la predicción principal y las tres clases más probables. Requiere que el modelo ya se haya entrenado y que existan sus pesos y el archivo JSON con el orden de clases.

Para evaluar todas las imágenes de `external_tests/` con ambos modelos:

```bash
python src/evaluate_external.py
```

Se admiten imágenes `.jpg`, `.jpeg` y `.png`. El script genera `results/tabla_externa.csv` y guarda una instantánea por imagen y modelo en `results/external_snapshots/`. El CSV usa el nombre base del archivo como etiqueta; por ejemplo, `happy.jpg` produce la etiqueta `happy`. Para que la etiqueta coincida exactamente con una clase, evita sufijos como `happy_01.jpg` o normalízalos antes de calcular aciertos. Las pruebas externas disponibles, incluidas imágenes de Willem Dafoe, son una muestra pequeña y exploratoria; no sustituyen la evaluación sobre datos de prueba representativos.

## Archivos generados

- `models/`: pesos `.pth` y archivos JSON con el orden de las clases.
- `results/loss_curves/`: pérdida de entrenamiento y validación, y accuracy de validación.
- `results/confusion_matrices/`: matrices de confusión.
- `results/metrics/`: reportes de clasificación.
- `results/roc_curves/`: curvas ROC.
- `results/tabla_externa.csv`: predicciones y confianzas para las imágenes externas.
- `results/external_snapshots/`: imágenes con las predicciones anotadas y gráficas de la evaluación externa.

Los datos, los pesos de los modelos y los resultados generados están excluidos del control de versiones. Cada integrante debe descargarlos o generarlos localmente.

## Estructura del proyecto

```text
src/
  dataset.py             Carga de datos, transformaciones y particiones
  download_data.py       Descarga del dataset
  models.py              Definición de ResNet-18 y AlexNet
  train.py               Entrenamiento y selección con validación
  evaluate.py            Evaluación sobre el conjunto de prueba
  predict_external.py    Predicción de una imagen
  evaluate_external.py   Evaluación de imágenes externas
data/                    Dataset local
external_tests/          Imágenes para pruebas exploratorias
models/                  Pesos y etiquetas guardados localmente
results/                 Métricas, gráficas y resultados generados
notebooks/               Experimentos exploratorios
```
