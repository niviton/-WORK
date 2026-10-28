"""Captura da GoPro (modo webcam) e carga da calibração."""
import cv2
import numpy as np
import yaml


def open_camera(index=0, width=1280, height=720):
    # CAP_DSHOW abre a GoPro Webcam mais rápido no Windows
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # reduz latência
    if not cap.isOpened():
        raise RuntimeError(f"Não abriu a câmera {index}. GoPro Webcam está ativa?")
    return cap


def save_calibration(path, K, dist, size, rms):
    data = {
        "image_width": size[0],
        "image_height": size[1],
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.ravel().tolist(),
        "rms": float(rms),
    }
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


def load_calibration(path):
    with open(path) as f:
        data = yaml.safe_load(f)
    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64)
    return K, dist, (data["image_width"], data["image_height"])
