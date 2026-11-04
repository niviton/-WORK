"""Fita zebrada (vermelho/branco) = parede virtual do RoboCup@Work.

Sem IA: segmenta vermelho, exige branco colado ao vermelho (listras alternadas) e
formato alongado (descarta cubos vermelhos e objetos soltos). Os pixels da fita são
projetados no chão (pixel_to_table) para dar pontos (x, y) em metros no frame do robô.

Biblioteca pura: no ROS 2 os pontos viram um PointCloud2 para o costmap do Nav2.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class TapeResult:
    mask: np.ndarray                         # máscara da fita (uint8 0/255)
    contours: list = field(default_factory=list)
    points: np.ndarray = None                # Nx2 (x, y) no chão, frame do robô (m)
    nearest: np.ndarray = None               # ponto (x, y) da fita mais próximo do robô
    distance: float = None                   # distância do robô até a fita (m)


class TapeDetector:
    def __init__(self, K, dist, T_base_cam=None, floor_z=0.0,
                 min_area=200, min_elongation=3.0, step=4):
        self.K, self.dist, self.T = K, dist, T_base_cam
        self.floor_z = floor_z               # altura do chão no frame do robô (m)
        self.min_area = min_area
        self.min_elongation = min_elongation
        self.step = step                     # amostragem de pixels para projeção

    @staticmethod
    def color_masks(frame):
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
        red = cv2.inRange(hsv, (0, 90, 70), (10, 255, 255)) | \
              cv2.inRange(hsv, (165, 90, 70), (180, 255, 255))
        white = cv2.inRange(hsv, (0, 0, 170), (180, 50, 255))
        return red, white

    def detect(self, frame, exclude_polys=()):
        # Analisa em meia resolução (fita é larga; 4x mais rápido) e devolve a máscara cheia
        full_hw = frame.shape[1], frame.shape[0]
        small = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        exclude_polys = [np.asarray(p) * 0.5 for p in exclude_polys]
        res = self._detect(small, exclude_polys)
        res.mask = cv2.resize(res.mask, full_hw, interpolation=cv2.INTER_NEAREST)
        res.contours = [(c * 2).astype(np.int32) for c in res.contours]
        return res

    def _detect(self, frame, exclude_polys):
        red, white = self.color_masks(frame)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        # Branco só conta se a região branca inteira encostar no vermelho (listra vizinha);
        # regiões brancas enormes (mesa, parede) são descartadas
        _, wl, ws, _ = cv2.connectedComponentsWithStats(white)
        touching = np.unique(wl[(white > 0) & (cv2.dilate(red, k) > 0)])
        max_white = 0.02 * white.size
        keep = [t for t in touching if t and ws[t, cv2.CC_STAT_AREA] < max_white]
        lut = np.zeros(len(ws), np.uint8)  # tabela por rótulo: bem mais rápido que np.isin
        lut[keep] = 255
        white_near = lut[wl]
        tape = cv2.morphologyEx(red | white_near, cv2.MORPH_CLOSE, k, iterations=2)
        for poly in exclude_polys:           # ex.: área dos cubos detectados pelo tag
            cv2.fillPoly(tape, [np.asarray(poly, np.int32)], 0)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(tape)
        mask = np.zeros_like(tape)
        contours = []
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < self.min_area:
                continue
            comp = (labels == i).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = max(cnts, key=cv2.contourArea)
            (w, h) = cv2.minAreaRect(c)[1]
            elong = max(w, h) / max(min(w, h), 1)
            fill = cv2.contourArea(c) / max(w * h, 1)
            r_frac = (red[comp > 0] > 0).mean()
            n_stripes = cv2.connectedComponents(red & comp)[0] - 1
            # Fita: comprida, cheia (não é fio curvo), duas cores e várias listras vermelhas
            if (elong >= self.min_elongation and fill > 0.5
                    and 0.2 < r_frac < 0.8 and n_stripes >= 2):
                mask |= comp
                contours.append(c)

        res = TapeResult(mask, contours)
        if self.T is not None and contours:
            ys, xs = np.nonzero(mask[::self.step, ::self.step])
            # *2: coordenadas da meia resolução -> pixels da imagem cheia (onde vale o K)
            pts = self.project(np.stack([xs, ys], 1).astype(float) * self.step * 2)
            if len(pts):
                d = np.linalg.norm(pts, axis=1)
                res.points, res.nearest, res.distance = pts, pts[d.argmin()], float(d.min())
        return res

    def project(self, uv):
        """Versão vetorizada de pixel_to_table: Nx2 pixels -> Mx2 (x, y) no chão."""
        rays = cv2.undistortPoints(uv.reshape(-1, 1, 2), self.K, self.dist).reshape(-1, 2)
        d = np.column_stack([rays, np.ones(len(rays))]) @ self.T[:3, :3].T
        o = self.T[:3, 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (self.floor_z - o[2]) / d[:, 2]
        ok = np.isfinite(s) & (s > 0)
        return (o + s[ok, None] * d[ok])[:, :2]

    def draw(self, frame, res):
        overlay = frame.copy()
        overlay[res.mask > 0] = (255, 0, 255)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        cv2.drawContours(frame, res.contours, -1, (255, 0, 255), 3)
        if res.distance is not None:
            txt = f"FITA: {res.distance * 100:.0f} cm  (x={res.nearest[0]*100:.0f} y={res.nearest[1]*100:.0f})"
            cv2.putText(frame, txt, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 7)
            cv2.putText(frame, txt, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 0, 255), 3)
        return frame
