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
