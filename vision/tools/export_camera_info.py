"""Converte a nossa calibração para o formato camera_info do ROS (camera_calibration_parsers).

  python -m vision.tools.export_camera_info camera_calib_webcam.yaml \
      ros2_ws/src/cube_vision/config/camera_info.yaml
"""
import argparse

import numpy as np
import yaml

from vision.common.camera import load_calibration


def to_camera_info(K, dist, size, name="camera"):
    def mat(rows, cols, data):
        return {"rows": rows, "cols": cols, "data": [float(v) for v in np.ravel(data)]}

    P = np.hstack([K, np.zeros((3, 1))])
    return {
        "image_width": int(size[0]),
        "image_height": int(size[1]),
        "camera_name": name,
        "camera_matrix": mat(3, 3, K),
        "distortion_model": "plumb_bob",
        "distortion_coefficients": mat(1, len(dist), dist),
        "rectification_matrix": mat(3, 3, np.eye(3)),
        "projection_matrix": mat(3, 4, P),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calib")
    ap.add_argument("out")
    args = ap.parse_args()
    K, dist, size = load_calibration(args.calib)
    with open(args.out, "w") as f:
        yaml.safe_dump(to_camera_info(K, dist, size), f, sort_keys=False)
    print(f"Salvo em {args.out}")


if __name__ == "__main__":
    main()
