"""Detecção dos cubos pelo QR Code: ID, contorno e pose 3D.

Biblioteca pura (sem janela/câmera) para ser reutilizada depois no nó ROS 2.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Cube:
    id: str                 # texto decodificado do QR
    corners: np.ndarray     # 4x2 pixels: sup-esq, sup-dir, inf-dir, inf-esq
    rvec: np.ndarray        # rotação (Rodrigues) no frame da câmera
    tvec: np.ndarray        # posição (m) do centro do QR no frame da câmera

    @property
    def distance(self):
        return float(np.linalg.norm(self.tvec))

    @property
    def yaw_deg(self):
        """Giro do cubo em torno do eixo óptico — útil para alinhar a garra."""
        R, _ = cv2.Rodrigues(self.rvec)
        return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


class CubeDetector:
    def __init__(self, K, dist, qr_size_m):
        self.K, self.dist = K, dist
        s = qr_size_m / 2
        # Mesma ordem dos cantos devolvidos pelo OpenCV; exigida pelo IPPE_SQUARE
        self.obj_pts = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)
        self.qr = cv2.QRCodeDetector()

    def detect(self, frame):
        ok, texts, points, _ = self.qr.detectAndDecodeMulti(frame)
        if not ok or points is None:
            return []
        cubes = []
        for text, pts in zip(texts, points):
            if not text:  # achou o QR mas não leu (longe/borrado)
                continue
            pts = self._refine(frame, pts.astype(np.float32))
            found, rvec, tvec = cv2.solvePnP(self.obj_pts, pts, self.K, self.dist,
                                             flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if found:
                cubes.append(Cube(text, pts, rvec.ravel(), tvec.ravel()))
        return cubes

    @staticmethod
    def _refine(frame, pts):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        return cv2.cornerSubPix(gray, pts.reshape(-1, 1, 2), (5, 5), (-1, -1), crit).reshape(4, 2)

    def draw(self, frame, cubes, axis_len=0.03):
        for c in cubes:
            cv2.polylines(frame, [c.corners.astype(int)], True, (0, 255, 0), 2)
            cv2.drawFrameAxes(frame, self.K, self.dist, c.rvec, c.tvec, axis_len)
            x, y, z = c.tvec
            org = tuple(c.corners[0].astype(int) - [0, 10])
            label = f"{c.id}  x={x*100:.1f} y={y*100:.1f} z={z*100:.1f}cm  yaw={c.yaw_deg:.0f}"
            cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        return frame
