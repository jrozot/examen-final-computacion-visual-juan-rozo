# Procesamiento Visual

Programa de consola que aplica transformaciones a una imagen con OpenCV y un
detector YOLO preentrenado. La imagen resultante se guarda en el mismo
directorio de la original con un sufijo (ej. `imagen_grey.jpg`).

## Dependencias

- Python3 - pip - venv
- opencv-python
- numpy
- ultralytics (YOLO)

## Instalación

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
**NOTA:** La instalación de librerías puede tomar varios minutos.

## Uso

```bash
python src/main.py imagen.jpg --grey
python src/main.py imagen.jpg --HSV
python src/main.py imagen.jpg --soften            # GaussianBlur (por defecto)
python src/main.py imagen.jpg --soften --median   # MedianBlur
python src/main.py imagen.jpg --edges
python src/main.py imagen.jpg --blur-gaussian
python src/main.py imagen.jpg --detector-yolo
```

**NOTA:** Se pueden combinar varias opciones en una misma ejecución.

**NOTA:** La imagen fuente debe estar en el mismo directorio de ejecución.

**NOTA:** La imagen resultado se almacena en el mismo directorio de ejecución.

## Parámetros usados

- **Grises:** `cv2.cvtColor` con `COLOR_BGR2GRAY`.
- **HSV:** `cv2.cvtColor` con `COLOR_BGR2HSV` (segunda representación de color).
- **Suavizado (`--soften`):** `GaussianBlur` kernel 5x5 por defecto; `--median`
  cambia a `medianBlur` de tamaño 5.
- **Bordes:** `Canny` con umbrales 100 y 200.
- **Blur gaussiano (`--blur-gaussian`):** `GaussianBlur` kernel 15x15.
- **YOLO:** modelo `yolo11n.pt` (el más pequeño), se descarga automáticamente
  la primera vez; se guarda la imagen anotada con las detecciones.

## Justificación

Se prioriza la simplicidad: un único archivo, OpenCV para todas las
transformaciones clásicas y Ultralytics YOLO11n por ser un modelo preentrenado
ligero que no requiere entrenamiento. Cada transformación es una llamada
directa de la librería para mantener el código mínimo.
