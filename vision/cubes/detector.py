"""Detecção dos cubos pelo AprilTag (família 36h11): ID, contorno e pose 3D.

Biblioteca pura (sem janela/câmera) para ser reutilizada depois no nó ROS 2.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Cube:
    id: str                 # ID do AprilTag (ex.: "3")
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
    def __init__(self, K, dist, tag_size_m, smooth=0.6, hold=5):
        """tag_size_m: lado do quadrado PRETO do tag, em metros."""
        self.K, self.dist = K, dist
        s = tag_size_m / 2
        # Mesma ordem dos cantos devolvidos pelo OpenCV; exigida pelo IPPE_SQUARE
        self.obj_pts = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)
        params = cv2.aruco.DetectorParameters()
        # SUBPIX: cantos precisos (o modo APRILTAG falhou nos testes com o cubo real)
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.minMarkerPerimeterRate = 0.03      # abaixo disso (~14 px de lado em 1080p) não decodifica
        params.adaptiveThreshWinSizeMin = 5       # 3 escalas: bem mais rápido que varrer 9
        params.adaptiveThreshWinSizeMax = 35
        params.adaptiveThreshWinSizeStep = 15
        self.aruco = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), params)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # Estabilização temporal: id -> {"pts", "missed"}
        self.smooth = smooth      # 0 = sem filtro, 0.6 = bem suave
        self.hold = hold          # frames que o tag "sobrevive" sem ser visto
        self.tracks = {}

    def detect(self, frame):
        gray = self.clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        corners, ids, _ = self.aruco.detectMarkers(gray)
        seen = {}
        if ids is not None:
            for pts, tag_id in zip(corners, ids.ravel()):
                seen[str(int(tag_id))] = pts.reshape(4, 2).astype(np.float32)

        for tid, pts in seen.items():
            tr = self.tracks.get(tid)
            # Filtro exponencial nos cantos; se o tag pulou muito (movimento real), não suaviza
            if tr is not None and np.abs(pts - tr["pts"]).max() < 15:
                pts = self.smooth * tr["pts"] + (1 - self.smooth) * pts
            self.tracks[tid] = {"pts": pts, "missed": 0}
        for tid in list(self.tracks):
            if tid not in seen:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.hold:
                    del self.tracks[tid]

        cubes = []
        for tid, tr in self.tracks.items():
            found, rvec, tvec = cv2.solvePnP(self.obj_pts, tr["pts"], self.K, self.dist,
                                             flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if found:
                cubes.append(Cube(tid, tr["pts"], rvec.ravel(), tvec.ravel()))
        return cubes

    def draw(self, frame, cubes, axis_len=0.03):
        for c in cubes:
            cv2.polylines(frame, [c.corners.astype(int)], True, (0, 255, 0), 2)
            cv2.drawFrameAxes(frame, self.K, self.dist, c.rvec, c.tvec, axis_len)
            x, y, z = c.tvec
            org = tuple(c.corners[0].astype(int) - [0, 10])
            label = f"{c.id}  x={x*100:.1f} y={y*100:.1f} z={z*100:.1f}cm  yaw={c.yaw_deg:.0f}"
            cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        return frame
