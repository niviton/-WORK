"""Pose da câmera a partir da altura medida e de tags deitados na mesa.

  python -m vision.calibration.extrinsic_height --calib camera_calib_webcam.yaml --height 0.57

Origem: ponto da mesa logo abaixo da câmera. x = direção para onde a câmera aponta (no plano
da mesa), y = esquerda, z = cima. A normal dos tags dá a inclinação; a altura calculada pelos
tags é só uma conferência da medida. Os cubos precisam estar com o tag para cima.
"""
import argparse
import time

import cv2
import numpy as np
import yaml

from vision.common.camera import load_calibration, open_camera, scale_calibration
from vision.cubes.detector import CubeDetector
from vision.cubes.grasp import CUBE_SIZE


def pose_from_normal(normal, height):
    n = normal / np.linalg.norm(normal)
    forward = np.array([0.0, 0.0, 1.0]) - n[2] * n   # eixo óptico projetado na mesa
    forward /= np.linalg.norm(forward)
    T = np.eye(4)
    T[:3, :3] = np.vstack([forward, np.cross(n, forward), n])
    T[2, 3] = height
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--height", type=float, required=True, help="altura da câmera sobre a mesa (m)")
    ap.add_argument("--camera", default="srv", help="srv (servidor da câmera) ou índice do dispositivo")
    ap.add_argument("--tag-size", type=float, default=0.032)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--out", default="robot_config_webcam.yaml")
    args = ap.parse_args()

    K, dist, calib_size = load_calibration(args.calib)
    cap = open_camera(args.camera)
    K = scale_calibration(K, calib_size, (int(cap.get(3)), int(cap.get(4))))
    det = CubeDetector(K, dist, args.tag_size, smooth=0, hold=0)

    normals, points, frames, t0 = [], [], 0, time.time()
    while frames < args.frames and time.time() - t0 < 30:
        ok, frame = cap.read()
        if not ok:
            continue
        frames += 1
        for c in det.detect(frame):
            normals.append(cv2.Rodrigues(c.rvec)[0][:, 2])
            points.append(c.tvec)
    cap.release()
    if len(normals) < 10:
        raise SystemExit("Poucos tags visíveis. Deixe ao menos um cubo com o tag para cima.")

    normals, points = np.array(normals), np.array(points)
    n = np.median(normals, axis=0)
    n /= np.linalg.norm(n)
    # Descarta o que não está deitado (tag de lado, leitura espelhada) e recalcula
    flat = normals @ n > np.cos(np.radians(15))
    n = normals[flat].mean(axis=0)
    n /= np.linalg.norm(n)
    estimated = CUBE_SIZE - np.median(points[flat] @ n)
    tilt = np.degrees(np.arcsin(np.clip(-n[2], -1, 1)))

    T = pose_from_normal(n, args.height)
    with open(args.out, "w") as f:
        yaml.safe_dump({"T_base_cam": T.round(6).tolist(), "height_m": args.height}, f)
    print(f"{flat.sum()}/{len(flat)} leituras usadas | câmera inclinada {tilt:.1f} graus para baixo")
    print(f"altura pelos tags {estimated * 100:.1f} cm | medida {args.height * 100:.1f} cm | "
          f"diferença {(estimated - args.height) * 100:+.1f} cm")
    print(f"Salvo em {args.out}")


if __name__ == "__main__":
    main()
