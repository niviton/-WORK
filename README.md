# -WORK

Garra inteligente com 2 eixos (X horizontal e Z vertical) movida por cremalheiras,
controlada por Arduino e guiada por câmera para pegar um cubo laranja.

## Arquivos (`garra/`)

| Arquivo | O que faz |
|---|---|
| `garra.ino` | Firmware da placa **Anet v1.5** (ATmega1284P, drivers A4988 1/16). Rampa de aceleração, limites de curso, movimento X+Z simultâneo (Bresenham) e comandos pela Serial. |
| `controle_pc.py` | App no computador (Tkinter): setas, clique no desenho para mover, programa de pontos gravados, botão PARAR e console. |
| `visao_cubo.py` | Visão com a GoPro (modo webcam): acha o cubo laranja pela cor (HSV) e aproxima o eixo X até o cubo ficar no ponto ideal (servo visual). |
| `visao_config.json` | Calibração da visão (cor do cubo, linha alvo, px por mm). |

## Montagem

- Motor do **X** no conector **X-MOTOR**; motor do **Z** no conector **Y-MOTOR** da placa.
- Zero (0, 0) = garra no ponto **mais alto** e **mais à direita**. X positivo = esquerda, Z positivo = descer.
- A placa precisa ser ligada com a garra no zero (não há fim de curso ainda).

## Comandos da Serial (115200 baud)

```
x 120     vai para X = 120 mm          m 120 50   X e Z juntos, em linha reta
z 50      vai para Z = 50 mm           xr 10      anda +10 mm em X (zr para Z)
h         volta ao zero                !          PARAR
?         posição atual                zero       posição atual vira o zero
```

Toda resposta termina com `ok`; a linha `POS x z curso_x curso_z` traz a posição.

## Como rodar

```
pip install pyserial opencv-python
cd garra
python controle_pc.py COM15      # controle manual
python visao_cubo.py             # visão (câmera 2 = GoPro, porta COM15)
```

## Visão RoboCup@Work (`vision/`, `ros2_ws/`)

Reconhece os cubos pelo **AprilTag 36h11** do topo (o ID do tag é o número do cubo), calcula a posição 3D do centro de cada cubo e detecta a fita zebrada vermelha e branca. O plano completo está em [`PLANO.md`](PLANO.md).

| Caminho | O que faz |
|---|---|
| `vision/common/camera_server.py` | Abre a câmera uma vez só e serve as imagens às outras ferramentas (`--camera srv`) |
| `vision/calibration/` | Calibração ChArUco da câmera e pose da câmera no robô |
| `vision/cubes/` | Detecção dos tags, pose 3D e alvo da garra (centro e giro) |
| `vision/tape/` | Fita zebrada por cor e formato, com projeção no chão |
| `vision/labeler/` | Página web ao vivo em http://localhost:5000, com rotulagem para o dataset do YOLO |
| `vision/tools/` | Verificação de distância com régua, latência e exportação de `camera_info` |
| `ros2_ws/src/cube_vision/` | Nó ROS 2 que publica `/vision/cubes` e a TF `cube_<id>` (ainda não testado em ROS 2) |

```
pip install -r requirements.txt
python -m vision.common.camera_server --camera 1          # deixe rodando
python -m vision.labeler.app --calib camera_calib_webcam.yaml --robot-config robot_config_webcam.yaml
```