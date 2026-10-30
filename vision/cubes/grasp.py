"""Da detecção do tag para o alvo da garra, no frame do robô.

Convenção do robô (ROS REP-103): base_link com x = frente, y = esquerda, z = cima, em metros.
Frame do tag (OpenCV): x = direita do tag, y = para cima do tag, z = saindo da face.

T_base_cam (4x4) = pose da câmera no frame do robô; vem de robot_config.yaml,
gerado por `python -m vision.calibration.extrinsic`.
"""
from dataclasses import dataclass

import cv2
import numpy as np
import yaml

CUBE_SIZE = 0.042   # aresta do cubo (m)


def to_T(rvec, tvec):
    T = np.eye(4)
    T[:3, :3], _ = cv2.Rodrigues(np.asarray(rvec, float))
    T[:3, 3] = np.ravel(tvec)
    return T


def load_T_base_cam(path="robot_config.yaml"):
    with open(path) as f:
        return np.array(yaml.safe_load(f)["T_base_cam"], float)


@dataclass
class GraspTarget:
    id: str
    xyz: np.ndarray       # centro do cubo no frame do robô (m)
    yaw_deg: float        # giro da garra em torno de z, em [-45, 45) (cubo tem simetria de 90°)
    width: float          # abertura mínima da garra (m)


def grasp_from_cube(cube, T_base_cam, cube_size=CUBE_SIZE):
    T_cam_tag = to_T(cube.rvec, cube.tvec)
    # Centro do cubo: meia aresta "para dentro" da face onde está o tag (-z do tag)
    center_cam = T_cam_tag @ np.array([0, 0, -cube_size / 2, 1])
    center = (T_base_cam @ center_cam)[:3]

    # Giro: direção do eixo x do tag projetada no plano do chão
    x_axis = (T_base_cam[:3, :3] @ T_cam_tag[:3, :3])[:, 0]
    yaw = np.degrees(np.arctan2(x_axis[1], x_axis[0]))
    yaw = (yaw + 45) % 90 - 45
    return GraspTarget(cube.id, center, float(yaw), cube_size)


def pixel_to_table(u, v, K, dist, T_base_cam, plane_z):
    """Ponto (x, y) na mesa a partir de um pixel, sem precisar do tag.

    Serve para a detecção do YOLO (cubo longe/de lado): intersecta o raio do pixel
    com o plano horizontal z = plane_z (ex.: altura da mesa + metade do cubo).
    """
    ray = cv2.undistortPoints(np.array([[[u, v]]], float), K, dist).ravel()
    d_cam = np.array([ray[0], ray[1], 1.0])
    R, o = T_base_cam[:3, :3], T_base_cam[:3, 3]
    d = R @ d_cam
    if abs(d[2]) < 1e-9:
        return None
    s = (plane_z - o[2]) / d[2]
    return None if s <= 0 else o + s * d
