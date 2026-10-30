"""Captura da GoPro (modo webcam) e carga da calibração."""
import time
import urllib.request

import cv2
import numpy as np
import yaml

SERVER_URL = "http://127.0.0.1:5001"  # não usar "localhost": no Windows tenta IPv6 antes (+2 s)


class RemoteCapture:
    """Imita cv2.VideoCapture lendo do servidor da câmera (vision.common.camera_server).

    Assim vários programas usam a GoPro sem abrir/fechar o dispositivo (o que a trava).
    """

    def __init__(self, url=SERVER_URL):
        self.url, self.last_id, self.size = url, None, (0, 0)
        self.last_capture_time = None    # horário (servidor) em que o frame foi capturado
        info = urllib.request.urlopen(f"{url}/info", timeout=5).read().decode().split("x")
        self.size = (int(info[0]), int(info[1]))

    def isOpened(self):
        return True

    def read(self, timeout=3.0):
        # Espera um frame NOVO (não repete o mesmo frame duas vezes)
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                r = urllib.request.urlopen(f"{self.url}/raw.jpg", timeout=2)
            except OSError:
                time.sleep(0.2)
                continue
            fid = r.headers.get("X-Frame-Id")
            if fid != self.last_id:
                self.last_id = fid
                self.last_capture_time = float(r.headers.get("X-Capture-Time", 0))
                frame = cv2.imdecode(np.frombuffer(r.read(), np.uint8), cv2.IMREAD_COLOR)
                return frame is not None, frame
            time.sleep(0.005)
        return False, None

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_WIDTH: self.size[0],
                cv2.CAP_PROP_FRAME_HEIGHT: self.size[1]}.get(prop, 0)

    def set(self, *_):
        return False

    def release(self):
        pass


def open_camera(index=0, width=1920, height=1080):
    """index: número do dispositivo (2 = GoPro) ou "srv" para usar o servidor da câmera."""
    if isinstance(index, str):
        if index == "srv" or index.startswith("http"):
            return RemoteCapture(SERVER_URL if index == "srv" else index)
        index = int(index)
    # CAP_DSHOW abre a GoPro Webcam mais rápido no Windows
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # 1080p sem perder fps
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


def scale_calibration(K, calib_size, frame_size):
    """Ajusta K quando a câmera roda em resolução diferente da calibrada (mesmo aspecto)."""
    s = frame_size[0] / calib_size[0]
    K = K.copy()
    K[:2] *= s
    return K


def load_calibration(path):
    with open(path) as f:
        data = yaml.safe_load(f)
    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64)
    return K, dist, (data["image_width"], data["image_height"])
