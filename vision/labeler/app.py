"""Rotulador web ao vivo.

  python -m vision.labeler.app --camera 0 --calib camera_calib_aprox.yaml
  abrir http://localhost:5000

- Arraste o mouse sobre o QR/cubo, escolha o nome (ex.: cubo_3) e salve.
- O QR dentro da seleção passa a ser reconhecido com esse nome (cubes_registry.yaml).
- Cada salvamento grava imagem + anotação YOLO-seg em datasets/cubes/ para treinar depois.
- "Auto-captura" grava sozinha os cubos já conhecidos (gera dataset rápido).
"""
import argparse
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from flask import Flask, Response, jsonify, request, send_file

from vision.common.camera import load_calibration, open_camera
from vision.cubes.detector import CubeDetector

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "cubes_registry.yaml"
DATASET = ROOT / "datasets" / "cubes"


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None        # frame cru (para salvar)
        self.jpeg = None         # frame com desenho (para o navegador)
        self.cubes = []
        self.auto = False
        self.last_auto = 0.0
        self.registry = yaml.safe_load(REGISTRY.read_text()) if REGISTRY.exists() else {}
        self.registry = self.registry or {}

    def classes(self):
        return sorted(set(self.registry.values()))


def save_sample(frame, polys_labels, classes):
    """Grava imagem + rótulo no formato YOLO-seg (classe x1 y1 x2 y2 ... normalizado)."""
    (DATASET / "images").mkdir(parents=True, exist_ok=True)
    (DATASET / "labels").mkdir(parents=True, exist_ok=True)
    name = time.strftime("%Y%m%d_%H%M%S_") + f"{int(time.time() * 1000) % 1000:03d}"
    h, w = frame.shape[:2]
    lines = []
    for poly, label in polys_labels:
        pts = np.asarray(poly, float) / [w, h]
        lines.append(f"{classes.index(label)} " + " ".join(f"{v:.6f}" for v in pts.ravel()))
    cv2.imwrite(str(DATASET / "images" / f"{name}.jpg"), frame)
    (DATASET / "labels" / f"{name}.txt").write_text("\n".join(lines) + "\n")
    (DATASET / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(DATASET), "train": "images", "val": "images",
         "names": dict(enumerate(classes))}, allow_unicode=True))
    return name


def capture_loop(st, cap, detector):
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        cubes = detector.detect(frame)
        view = frame.copy()
        for c in cubes:
            name = st.registry.get(c.id)
            color = (0, 255, 0) if name else (0, 165, 255)  # verde = conhecido, laranja = novo
            cv2.polylines(view, [c.corners.astype(int)], True, color, 2)
            cv2.drawFrameAxes(view, detector.K, detector.dist, c.rvec, c.tvec, 0.03)
            txt = f"{name or '?'} [{c.id}] z={c.tvec[2] * 100:.1f}cm"
            org = tuple(c.corners[0].astype(int) - [0, 10])
            cv2.putText(view, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        with st.lock:
            st.frame, st.cubes = frame, cubes
            st.jpeg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
            known = [(c.corners, st.registry[c.id]) for c in cubes if c.id in st.registry]
            if st.auto and known and time.time() - st.last_auto > 1.0:
                st.last_auto = time.time()
                save_sample(frame, known, st.classes())


def create_app(st):
    app = Flask(__name__)

    @app.get("/")
    def index():
        return send_file(Path(__file__).with_name("index.html"))

    @app.get("/video")
    def video():
        def gen():
            while True:
                with st.lock:
                    jpg = st.jpeg
                if jpg:
                    yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                time.sleep(0.03)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")

    @app.get("/api/state")
    def state():
        with st.lock:
            h, w = st.frame.shape[:2] if st.frame is not None else (0, 0)
            return jsonify(
                width=w, height=h, auto=st.auto, registry=st.registry,
                classes=st.classes(), samples=len(list((DATASET / "images").glob("*.jpg")))
                if (DATASET / "images").exists() else 0,
                cubes=[{"qr": c.id, "name": st.registry.get(c.id),
                        "corners": c.corners.tolist(),
                        "xyz_cm": (c.tvec * 100).round(1).tolist()} for c in st.cubes])

    @app.post("/api/label")
    def label():
        """body: {box: [x1,y1,x2,y2] em pixels da imagem, label: 'cubo_3'}"""
        data = request.get_json()
        name = data["label"].strip()
        x1, y1, x2, y2 = data["box"]
        with st.lock:
            frame = st.frame.copy()
            # QR(s) cujo centro cai dentro da seleção recebem o nome
            linked = []
            for c in st.cubes:
                cx, cy = c.corners.mean(axis=0)
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    st.registry[c.id] = name
                    linked.append(c)
            if name not in st.registry.values():
                st.registry[f"__manual__{name}"] = name  # classe sem QR lido ainda
            REGISTRY.write_text(yaml.safe_dump(st.registry, allow_unicode=True))
            # Se achou o QR, usa os cantos exatos; senão, o retângulo desenhado
            polys = [(c.corners, name) for c in linked] or \
                    [([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], name)]
            sample = save_sample(frame, polys, st.classes())
        return jsonify(ok=True, sample=sample, linked=[c.id for c in linked])

    @app.post("/api/auto")
    def auto():
        st.auto = bool(request.get_json()["on"])
        return jsonify(auto=st.auto)

    @app.post("/api/forget")
    def forget():
        qr = request.get_json()["qr"]
        with st.lock:
            st.registry.pop(qr, None)
            REGISTRY.write_text(yaml.safe_dump(st.registry, allow_unicode=True))
        return jsonify(ok=True)

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--calib", default="camera_calib.yaml")
    ap.add_argument("--qr-size", type=float, default=0.04)
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    K, dist, (w, h) = load_calibration(args.calib)
    cap = open_camera(args.camera, w, h)
    st = State()
    threading.Thread(target=capture_loop, args=(st, cap, CubeDetector(K, dist, args.qr_size)),
                     daemon=True).start()
    print(f"Abra http://localhost:{args.port}")
    create_app(st).run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
