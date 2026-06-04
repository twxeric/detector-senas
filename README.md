# Detector de Lenguaje de Señas

Detector de letras en tiempo real usando Python, MediaPipe y scikit-learn.

## Requisitos
- Python 3.11
- Webcam

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/TU_USUARIO/detector_señas.git
cd detector_señas

# Crear entorno virtual
python -m venv venv_señas
venv_señas\Scripts\activate   # Windows
source venv_señas/bin/activate  # Linux/Mac

# Instalar librerías
pip install -r requirements.txt
```

## Modelo MediaPipe

Descarga el archivo `hand_landmarker.task` desde:
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

Y colócalo en la raíz del proyecto.

## Uso

```bash
# 1. Recolectar datos
python 1_recolectar_datos.py

# 2. Entrenar modelo
python 2_entrenar_modelo.py

# 3. Detectar en tiempo real
python 3_detectar_tiempo_real.py
```

## Letras detectadas
A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y