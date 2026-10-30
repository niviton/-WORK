"""Rotulador web ao vivo.

  python -m vision.labeler.app --camera 0 --calib camera_calib_aprox.yaml
  abrir http://localhost:5000

- Arraste o mouse sobre o QR/cubo, escolha o nome (ex.: cubo_3) e salve.
- O QR dentro da seleÃ§Ã£o passa a ser reconhecido com esse nome (cubes_registry.yaml).
- Cada salvamento grava imagem + anotaÃ§Ã£o YOLO-seg em datasets/cubes/ para treinar depois.
- "Auto-captura" grava sozinha os cubos jÃ¡ conhecidos (gera dataset rÃ¡pido).
"""
import argparse
import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from flask import Flask, Response, jsonify, request, send_file

from vision.common.camera import load_calibration, open_camera, scale_calibration
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

    def name_of(self, tag_id):
        # O ID do AprilTag jÃ¡ Ã© o nÃºmero do cubo; o cadastro sÃ³ renomeia se quiser
        return self.registry.get(tag_id, f"cubo_{tag_id}")

    def classes(self):
        seen = {self.name_of(c.id) for c in self.cubes}
        return sorted(set(self.registry.values()) | seen)


def save_sample(frame, polys_labels, classes):
    """Grava imagem + rÃ³tulo no formato YOLO-seg (classe x1 y1 x2 y2 ... normalizado)."""
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


def no_signal_jpeg(w, h):
    img = np.zeros((h, w, 3), np.uint8)
    for i, line in enumerate(["SEM SINAL DA CAMERA",
                              "GoPro: app GoPro Webcam > Start Webcam (reconectando...)"]):
        cv2.putText(img, line, (40, h // 2 + i * 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4 - i * 0.5, (0, 0, 255), 3 - i)
    return cv2.imencode(".jpg", img)[1].tobytes()


def capture_loop(st, open_cap, detector):
    cap = open_cap()
    black_since = None
    while True:
        ok, frame = cap.read()
        # GoPro sem transmitir entrega frames 100% pretos: avisa e tenta reabrir
        if not ok or frame.max() < 5:
            black_since = black_since or time.time()
            if time.time() - black_since > 3:
                h, w = (frame.shape[:2] if ok else (1080, 1920))
                with st.lock:
                    st.jpeg, st.cubes = no_signal_jpeg(w, h), []
                cap.release()
                time.sleep(2)
                try:
                    cap = open_cap()
                except RuntimeError:
                    pass  # cÃ¢mera ainda indisponÃ­vel; tenta de novo no prÃ³ximo ciclo
                black_since = time.time()
            time.sleep(0.01)
            continue
        black_since = None
        cubes = detector.detect(frame)
        view = frame.copy()
        for c in cubes:
            name = st.name_of(c.id)
            color = (0, 255, 0)
            cv2.polylines(view, [c.corners.astype(int)], True, color, 2)
            cv2.drawFrameAxes(view, detector.K, detector.dist, c.rvec, c.tvec, 0.03)
            txt = f"{name} [tag {c.id}] z={c.tvec[2] * 100:.1f}cm"
            org = tuple(c.corners[0].astype(int) - [0, 10])
            cv2.putText(view, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        with st.lock:
            st.frame, st.cubes = frame, cubes
            st.jpeg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
            known = [(c.corners, st.name_of(c.id)) for c in cubes]
            if st.auto and known and time.time() - st.last_auto > 1.0:
                st.last_auto = time.time()
                save_sample(frame, known, st.classes())


def create_app(st):
    app = Flask(__name__)

    @app.get("/")
    def index():
        return send_file(Path(__file__).with_name("index.html"))

    @app.get("/frame.jpg")
    def frame():
        # Um frame por requisiÃ§Ã£o: mais robusto que MJPEG (sem conexÃ£o presa no navegador)
        with st.lock:
            jpg = st.jpeg
        if not jpg:
            return Response(status=503)
        return Response(jpg, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/state")
    def state():
        with st.lock:
            h, w = st.frame.shape[:2] if st.frame is not None else (0, 0)
            return jsonify(
                width=w, height=h, auto=st.auto, registry=st.registry,
                classes=st.classes(), samples=len(list((DATASET / "images").glob("*.jpg")))
                if (DATASET / "images").exists() else 0,
                cubes=[{"qr": c.id, "name": st.name_of(c.id),
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
            # QR(s) cujo centro cai dentro da seleÃ§Ã£o recebem o nome
            linked = []
            for c in st.cubes:
                cx, cy = c.corners.mean(axis=0)
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    st.registry[c.id] = name
                    linked.append(c)
            if name not in st.registry.values():
                st.registry[f"__manual__{name}"] = name  # classe sem QR lido ainda
            REGISTRY.write_text(yaml.safe_dump(st.registry, allow_unicode=True))
            # Se achou o QR, usa os cantos exatos; senÃ£o, o retÃ¢ngulo desenhado
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
    ap.add_argument("--calib", default="camera_calib_gopro.yaml")
    ap.add_argument("--tag-size", type=float, default=0.032, help="lado do quadrado preto (m)")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    args = ap.parse_args()

    K, dist, calib_size = load_calibration(args.calib)
    cap = open_camera(args.camera, args.width, args.height)
    real = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    K = scale_calibration(K, calib_size, real)
    print(f"CÃ¢mera {args.camera}: {real[0]}x{real[1]}")
    st = State()
    open_cap = lambda: open_camera(args.camera, args.width, args.height)
    threading.Thread(target=capture_loop, args=(st, open_cap, CubeDetector(K, dist, args.tag_size)),
                     daemon=True).start()
    logging.getLogger("werkzeug").setLevel(logging.ERROR)  # sem log de cada requisiÃ§Ã£o
    print(f"Abra http://localhost:{args.port}")
    create_app(st).run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
