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
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", default="camera_calib.yaml")
    args = ap.parse_args()

    board = make_board(args.square, args.marker)
    if args.make_board:
        img = board.generateImage((COLS * 300, ROWS * 300), marginSize=40)
        cv2.imwrite("charuco_board.png", img)
        print("Salvo charuco_board.png — imprima sem 'ajustar à página' e meça o quadrado.")
        return

    detector = cv2.aruco.CharucoDetector(board)
    cap = open_camera(args.camera)
    obj_all, img_all, size = [], [], None

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        size = (frame.shape[1], frame.shape[0])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
        view = frame.copy()
        if ch_ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(view, ch_corners, ch_ids)
        cv2.putText(view, f"capturas: {len(obj_all)}  [ESPACO] capturar  [C] calibrar  [Q] sair",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("calibracao", view)
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
