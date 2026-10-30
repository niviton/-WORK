"""Mede o atraso (latência) real da câmera: do mundo até o frame chegar ao programa.

  python -m vision.tools.latency --camera 2

Uma janela "piscador" alterna preto/branco. Aponte a câmera para ela (de perto,
ocupando boa parte da imagem). O programa mede o tempo entre trocar a cor na tela e
o brilho mudar no frame da câmera. Inclui o atraso do monitor (~10-30 ms).
"""
import argparse
import threading
import time

import cv2
import numpy as np

from vision.common.camera import open_camera


class Grabber:
    """Lê a câmera sem parar, guardando (horário de chegada, brilho do centro)."""

    def __init__(self, cap):
        self.cap, self.samples, self.frame = cap, [], None
        self.running = True
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                h, w = f.shape[:2]
                roi = f[h // 4:3 * h // 4, w // 4:3 * w // 4]
                self.samples.append((time.perf_counter(), float(roi.mean())))
                self.frame = f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="srv", help="srv (servidor da câmera) ou índice do dispositivo")
    ap.add_argument("--trials", type=int, default=15)
    args = ap.parse_args()

    cap = open_camera(args.camera)
    g = Grabber(cap)
    white = np.full((600, 800, 3), 255, np.uint8)
    black = np.zeros_like(white)

    # Fase 1: posicionar. Mostra a câmera ao lado até o usuário apertar ESPAÇO.
    print("Aponte a câmera para a janela 'piscador' e aperte ESPACO na janela 'camera'.")
    on = False
    while True:
        on = not on if int(time.time() * 2) % 2 == on else on
        cv2.imshow("piscador", white if on else black)
        if g.frame is not None:
            prev = cv2.resize(g.frame, (640, 360))
            msg = "aguardando GoPro..." if g.frame.max() < 5 else "ESPACO = medir  Q = sair"
            cv2.putText(prev, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("camera", prev)
        k = cv2.waitKey(30) & 0xFF
        if k == ord(" ") and g.frame is not None and g.frame.max() >= 5:
            break
        if k == ord("q"):
            return

    # Fase 2: medir. Calibra níveis de preto/branco e cronometra cada troca.
    def level(img, secs=1.0):
        cv2.imshow("piscador", img)
        t_end = time.perf_counter() + secs
        while time.perf_counter() < t_end:
            cv2.waitKey(10)
        return np.mean([b for t, b in g.samples[-5:]])

    lo, hi = level(black), level(white)
    if hi - lo < 20:
        print(f"Contraste baixo ({lo:.0f} -> {hi:.0f}). Aproxime a câmera da janela.")
        return
    mid = (lo + hi) / 2
    delays = []
    for i in range(args.trials):
        target_white = i % 2 == 0
        level(white if not target_white else black, 0.7)   # estado oposto, estável
        cv2.imshow("piscador", white if target_white else black)
        cv2.waitKey(1)
        t0 = time.perf_counter()
        n0 = len(g.samples)
        while time.perf_counter() - t0 < 3:
            cv2.waitKey(1)
            hit = [t for t, b in g.samples[n0:] if (b > mid) == target_white]
            if hit:
                delays.append((hit[0] - t0) * 1000)
                print(f"  medida {i + 1}: {delays[-1]:.0f} ms")
                break
        else:
            print(f"  medida {i + 1}: sem resposta em 3 s")

    g.running = False
    cap.release()
    cv2.destroyAllWindows()
    if delays:
        d = np.array(delays)
        fps = len(g.samples) / (g.samples[-1][0] - g.samples[0][0])
        print(f"\nLATENCIA: media {d.mean():.0f} ms | mediana {np.median(d):.0f} ms | "
              f"min {d.min():.0f} | max {d.max():.0f} | camera ~{fps:.0f} fps")
        for v in (0.3, 0.5):
            print(f"  a {v} m/s o robo anda {v * np.median(d) / 10:.1f} cm durante o atraso")


if __name__ == "__main__":
    main()
