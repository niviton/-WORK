"""Compara a distância medida pela câmera com a medida na régua.

Da lente até o centro do tag:
  python -m vision.tools.validate --calib camera_calib_webcam.yaml --tag 3 --real 50

Entre os centros de dois tags (mais fácil de medir, não depende de achar a lente):
  python -m vision.tools.validate --calib camera_calib_webcam.yaml --tag 3 --tag2 8 --real 20

--real em cm. Cada medição é acrescentada a validation.csv.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

from vision.common.camera import load_calibration, open_camera, scale_calibration
from vision.cubes.detector import CubeDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tag2")
    ap.add_argument("--real", type=float, required=True, help="distância na régua (cm)")
    ap.add_argument("--camera", default="srv", help="srv (servidor da câmera) ou índice do dispositivo")
    ap.add_argument("--tag-size", type=float, default=0.032)
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--out", default="validation.csv")
    args = ap.parse_args()

    K, dist, calib_size = load_calibration(args.calib)
    cap = open_camera(args.camera)
    K = scale_calibration(K, calib_size, (int(cap.get(3)), int(cap.get(4))))
    det = CubeDetector(K, dist, args.tag_size, smooth=0, hold=0)

    values, t0 = [], time.time()
    while len(values) < args.samples and time.time() - t0 < 30:
        ok, frame = cap.read()
        if not ok:
            continue
        cubes = {c.id: c for c in det.detect(frame)}
        if args.tag not in cubes or (args.tag2 and args.tag2 not in cubes):
            continue
        a = cubes[args.tag].tvec
        b = cubes[args.tag2].tvec if args.tag2 else np.zeros(3)
        values.append(np.linalg.norm(a - b) * 100)
    cap.release()
    if len(values) < 10:
        raise SystemExit(f"Tag não encontrado o bastante ({len(values)} leituras). Está visível?")

    measured = float(np.median(values))
    err = measured - args.real
    print(f"medido {measured:.2f} cm | régua {args.real:.2f} cm | erro {err:+.2f} cm "
          f"({100 * err / args.real:+.1f}%) | ruído ±{np.std(values):.2f} cm | {len(values)} leituras")

    out = Path(args.out)
    new = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["calib", "tag", "tag2", "real_cm", "medido_cm", "erro_cm"])
        w.writerow([args.calib, args.tag, args.tag2 or "", args.real, round(measured, 2), round(err, 2)])


if __name__ == "__main__":
    main()
