"""Descobre onde a câmera está no robô (T_base_cam) usando um cubo como referência.

1) Coloque um cubo num ponto MEDIDO do frame do robô (x frente, y esquerda, a partir do
   centro da base_link), com o tag para cima e o "topo" do tag (lado de cima do desenho)
   apontando para a FRENTE do robô.
2) python -m vision.calibration.extrinsic --tag 3 --x 0.40 --y 0.00 --z 0.0
   (--z = altura da superfície onde o cubo está, em relação à base_link)
3) Grava robot_config.yaml, usado pelo módulo da garra.
"""
import argparse
import time

import numpy as np
import yaml

from vision.common.camera import load_calibration, open_camera, scale_calibration
from vision.cubes.detector import CubeDetector
from vision.cubes.grasp import CUBE_SIZE, to_T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="ID do tag de referência")
    ap.add_argument("--x", type=float, required=True)
    ap.add_argument("--y", type=float, required=True)
    ap.add_argument("--z", type=float, default=0.0, help="altura da superfície (m)")
    ap.add_argument("--camera", type=int, default=2)
    ap.add_argument("--calib", default="camera_calib_gopro.yaml")
    ap.add_argument("--tag-size", type=float, default=0.032)
    ap.add_argument("--out", default="robot_config.yaml")
    args = ap.parse_args()

    K, dist, calib_size = load_calibration(args.calib)
    cap = open_camera(args.camera)
    K = scale_calibration(K, calib_size, (int(cap.get(3)), int(cap.get(4))))
    det = CubeDetector(K, dist, args.tag_size, smooth=0)

    # Junta 30 medições para reduzir ruído (espera a GoPro iniciar)
    rvecs, tvecs, t0 = [], [], time.time()
    while len(tvecs) < 30 and time.time() - t0 < 40:
        ok, frame = cap.read()
        if not ok or frame.max() < 5:
            continue
        for c in det.detect(frame):
            if c.id == args.tag:
                rvecs.append(c.rvec)
                tvecs.append(c.tvec)
    cap.release()
    if len(tvecs) < 10:
        raise SystemExit(f"Tag {args.tag} visto só {len(tvecs)} vezes. Está visível para a câmera?")

    T_cam_tag = to_T(np.median(rvecs, axis=0), np.median(tvecs, axis=0))
    # Pose conhecida do tag no robô: tag x -> robô -y, tag y -> robô +x, tag z -> robô +z
    T_base_tag = np.eye(4)
    T_base_tag[:3, :3] = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], float)
    T_base_tag[:3, 3] = [args.x, args.y, args.z + CUBE_SIZE]
    T_base_cam = T_base_tag @ np.linalg.inv(T_cam_tag)

    with open(args.out, "w") as f:
        yaml.safe_dump({"T_base_cam": T_base_cam.round(6).tolist(),
                        "reference": {"tag": args.tag, "x": args.x, "y": args.y, "z": args.z}}, f)
    pos = T_base_cam[:3, 3]
    print(f"{len(tvecs)} medições. Câmera em x={pos[0]*100:.1f} y={pos[1]*100:.1f} "
          f"z={pos[2]*100:.1f} cm (frame do robô). Salvo em {args.out}")


if __name__ == "__main__":
    main()
