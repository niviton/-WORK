"""Demo ao vivo com a GoPro.

  python -m vision.cubes.run_cubes --tag-size 0.03

--tag-size = lado do quadrado PRETO do AprilTag no cubo, em metros.
"""
import argparse

import cv2

from vision.common.camera import load_calibration, open_camera
from vision.cubes.detector import CubeDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-size", type=float, required=True)
    ap.add_argument("--calib", default="camera_calib.yaml")
    ap.add_argument("--camera", default="srv", help="srv (servidor da câmera) ou índice do dispositivo")
    args = ap.parse_args()

    K, dist, (w, h) = load_calibration(args.calib)
    cap = open_camera(args.camera, w, h)  # mesma resolução da calibração
    detector = CubeDetector(K, dist, args.tag_size)

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        cubes = detector.detect(frame)
        cv2.imshow("cubos", detector.draw(frame, cubes))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
