import argparse
import os
import cv2


def save(path, suffix, img):
    base, ext = os.path.splitext(path)
    out = f"{base}_{suffix}{ext}"
    cv2.imwrite(out, img)
    print(f"Guardado: {out}")


def main():
    p = argparse.ArgumentParser(description="Procesamiento visual con OpenCV/YOLO")
    p.add_argument("imagen", help="Ruta de la imagen de entrada")
    p.add_argument("--grey", action="store_true", help="Escala de grises")
    p.add_argument("--HSV", action="store_true", help="Espacio de color HSV")
    p.add_argument("--soften", action="store_true", help="Suavizado")
    p.add_argument("--median", action="store_true", help="Usar MedianBlur en --soften (por defecto Gaussian)")
    p.add_argument("--edges", action="store_true", help="Bordes con Canny")
    p.add_argument("--blur-gaussian", action="store_true", help="Desenfoque Gaussiano")
    p.add_argument("--detector-yolo", action="store_true", help="Detección con YOLO11n")
    args = p.parse_args()

    img = cv2.imread(args.imagen)
    if img is None:
        raise SystemExit(f"No se pudo cargar la imagen: {args.imagen}")

    if args.grey:
        save(args.imagen, "grey", cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    if args.HSV:
        save(args.imagen, "hsv", cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

    if args.soften:
        out = cv2.medianBlur(img, 5) if args.median else cv2.GaussianBlur(img, (5, 5), 0)
        save(args.imagen, "soften", out)

    if args.edges:
        save(args.imagen, "edges", cv2.Canny(img, 100, 200))

    if args.blur_gaussian:
        save(args.imagen, "blur", cv2.GaussianBlur(img, (15, 15), 0))

    if args.detector_yolo:
        from ultralytics import YOLO
        res = YOLO("yolo11n.pt")(img)[0]
        save(args.imagen, "yolo", res.plot())


if __name__ == "__main__":
    main()
