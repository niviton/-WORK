"""Servidor da câmera: abre a GoPro UMA vez e entrega os frames a quem pedir.

  python -m vision.common.camera_server --camera 2
  (deixe rodando; nos outros programas use --camera srv)

Endpoints (porta 5001):
  /raw.jpg  frame mais recente (JPEG q95), com cabeçalhos X-Frame-Id e X-Capture-Time
  /info     "1920x1080"

No ROS 2 este papel é do nó de câmera (usb_cam/v4l2_camera), publicando /image_raw.
"""
import argparse
import logging
import threading
import time

import cv2
from flask import Flask, Response

from vision.common.camera import open_camera


class Source:
    def __init__(self, index, width, height):
        self.index, self.width, self.height = index, width, height
        self.lock = threading.Lock()
        self.jpeg, self.frame_id, self.capture_time = None, 0, 0.0
        self.size = (width, height)
        self.status = "abrindo"

    def run(self):
        cap = open_camera(self.index, self.width, self.height)
        self.size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print(f"Câmera {self.index} aberta: {self.size[0]}x{self.size[1]}")
        black_since, n, t_fps = None, 0, time.time()
        while True:
            ok, frame = cap.read()
            t = time.time()
            if not ok or frame.max() < 5:
                # GoPro iniciando (~8 s) ou travada. Não reabre: reabrir é o que a trava.
                black_since = black_since or t
                if t - black_since > 30 and self.status != "sem sinal":
                    self.status = "sem sinal"
                    print("Sem imagem há 30 s: Stop/Start no app GoPro Webcam (o servidor continua esperando)")
                if not ok:
                    time.sleep(0.05)
                continue
            if self.status != "ok":
                print("Imagem recebida.")
            black_since, self.status = None, "ok"
            jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
            with self.lock:
                self.jpeg, self.capture_time = jpg, t
                self.frame_id += 1
            n += 1
            if t - t_fps > 10:
                print(f"{n / (t - t_fps):.1f} fps")
                n, t_fps = 0, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=2)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--port", type=int, default=5001)
    args = ap.parse_args()

    src = Source(args.camera, args.width, args.height)
    threading.Thread(target=src.run, daemon=True).start()

    app = Flask(__name__)

    @app.get("/raw.jpg")
    def raw():
        with src.lock:
            jpg, fid, t = src.jpeg, src.frame_id, src.capture_time
        if jpg is None:
            return Response(src.status, status=503)
        return Response(jpg, mimetype="image/jpeg", headers={
            "X-Frame-Id": str(fid), "X-Capture-Time": f"{t:.6f}", "Cache-Control": "no-store"})

    @app.get("/info")
    def info():
        return f"{src.size[0]}x{src.size[1]}"

    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    print(f"Servidor da câmera em http://localhost:{args.port}  (use --camera srv)")
    app.run(host="127.0.0.1", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
