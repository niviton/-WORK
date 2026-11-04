"""Conversões sem dependência do ROS (dá para testar em qualquer máquina).

A imagem é convertida à mão em vez de usar cv_bridge: o cv_bridge do apt é compilado para
outra versão de NumPy/OpenCV, e o detector precisa do OpenCV >= 4.8 instalado via pip.
"""
import cv2
import numpy as np


def image_to_bgr(msg):
    """sensor_msgs/Image -> array BGR (h, w, 3)."""
    conversions = {
        "bgr8": (3, None),
        "rgb8": (3, cv2.COLOR_RGB2BGR),
        "mono8": (1, cv2.COLOR_GRAY2BGR),
        "yuv422_yuy2": (2, cv2.COLOR_YUV2BGR_YUY2),
    }
    if msg.encoding not in conversions:
        raise ValueError(f"Encoding não suportado: {msg.encoding}")
    channels, code = conversions[msg.encoding]
    # step pode ser maior que width * canais; descarta o preenchimento de cada linha
    raw = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)
    img = raw[:, :msg.width * channels]
    if channels > 1:
        img = img.reshape(msg.height, msg.width, channels)
    return img.copy() if code is None else cv2.cvtColor(img, code)


def rotation_to_quaternion(R):
    """Matriz 3x3 -> quaternion (x, y, z, w) com w >= 0."""
    q = np.array([
        1 + R[0, 0] - R[1, 1] - R[2, 2],
        1 - R[0, 0] + R[1, 1] - R[2, 2],
        1 - R[0, 0] - R[1, 1] + R[2, 2],
        1 + R[0, 0] + R[1, 1] + R[2, 2],
    ])
    i = int(q.argmax())
    s = np.sqrt(max(q[i], 0.0)) * 2
    if i == 0:
        out = [s / 4, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    elif i == 1:
        out = [(R[0, 1] + R[1, 0]) / s, s / 4, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    elif i == 2:
        out = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4, (R[1, 0] - R[0, 1]) / s]
    else:
        out = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, s / 4]
    out = np.array(out)
    return out if out[3] >= 0 else -out


def cube_pose(rvec, tvec, cube_size):
    """Centro do cubo e orientação, no frame óptico da câmera.

    O tag fica numa face: o centro está meia aresta atrás dele (-z do tag).
    """
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    center = np.ravel(tvec) + R @ np.array([0.0, 0.0, -cube_size / 2])
    return center, rotation_to_quaternion(R)
