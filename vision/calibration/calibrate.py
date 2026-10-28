"""Calibração da câmera com tabuleiro ChArUco.

1) Gerar o tabuleiro para imprimir (A4, escala 100%):
     python -m vision.calibration.calibrate --make-board
2) Medir com régua o lado de um quadrado impresso (em metros) e calibrar:
     python -m vision.calibration.calibrate --square 0.030 --marker 0.022
   ESPAÇO = captura frame, C = calibrar, Q = sair.
   Capture ~25-30 fotos: perto, longe, inclinado, nos cantos da imagem.
"""
import argparse

import cv2
import numpy as np

from vision.common.camera import open_camera, save_calibration

COLS, ROWS = 7, 5
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)


def make_board(square, marker):
    return cv2.aruco.CharucoBoard((COLS, ROWS), square, marker, DICT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-board", action="store_true")
    ap.add_argument("--square", type=float, default=0.030, help="lado do quadrado (m)")
    ap.add_argument("--marker", type=float, default=0.022, help="lado do marcador (m)")
    ap.add_argument("--camera", type=int, default=2)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--out", default="camera_calib.yaml")
    args = ap.parse_args()

    board = make_board(args.square, args.marker)
    if args.make_board:
        img = board.generateImage((COLS * 300, ROWS * 300), marginSize=40)
        cv2.imwrite("charuco_board.png", img)
        print("Salvo charuco_board.png — imprima sem 'ajustar à página' e meça o quadrado.")
        return

    detector = cv2.aruco.CharucoDetector(board)
    cap = open_camera(args.camera, args.width, args.height)
    obj_all, img_all, size = [], [], None

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        size = (frame.shape[1], frame.shape[0])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
        view = frame.copy()
        # Desenho manual: drawDetectedCornersCharuco quebra no OpenCV 5
        valid = (ch_ids is not None and ch_corners is not None
                 and len(ch_corners.reshape(-1, 2)) == len(ch_ids.ravel()))
        if valid:
            for x, y in ch_corners.reshape(-1, 2).astype(int):
                cv2.circle(view, (x, y), 8, (0, 0, 255), -1)
        else:
            ch_ids = None
        n = 0 if ch_ids is None else len(ch_ids)
        cv2.putText(view, f"capturas: {len(obj_all)}  cantos: {n}  [ESPACO] capturar  [C] calibrar  [Q] sair",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imshow("calibracao", cv2.resize(view, None, fx=0.5, fy=0.5))  # 1080p não cabe na tela
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" ") and ch_ids is not None and len(ch_ids) >= 8:
            obj, img = board.matchImagePoints(ch_corners, ch_ids)
            obj_all.append(obj)
            img_all.append(img)
        elif key == ord("c") and len(obj_all) >= 10:
            rms, K, dist, _, _ = cv2.calibrateCamera(obj_all, img_all, size, None, None)
            save_calibration(args.out, K, dist, size, rms)
            print(f"RMS = {rms:.3f} px (bom: < 0.5). Salvo em {args.out}")
            break
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
