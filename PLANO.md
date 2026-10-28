# Sistema de Visão – RoboCup@Work

Câmera: GoPro como webcam. Desenvolvimento inicial no Windows; destino final ROS 2.

## 1. O que as regras pedem (verificar no rulebook do ano)

- **Reconhecimento de objetos**: identificar e localizar objetos nas mesas de serviço para pegá-los com o braço (testes BMT/BTT/PPT etc.).
- **Barrier tape (fita zebrada vermelha/branca)**: funciona como **parede virtual** — o robô não pode cruzá-la. Precisa ser vista pela câmera e entrar no mapa de custos como obstáculo.
- Rulebook mais recente: https://atwork.robocup.org/wp-content/uploads/2023/07/Rulebook.pdf

> Atenção: o @Work oficial usa peças industriais (parafusos, porcas, perfis). Os "cubos com QR Code" provavelmente são de uma etapa regional/adaptação. Confirmar com a organização **qual marcador** é usado (QR Code, ArUco ou AprilTag) — isso muda a abordagem.

## 2. Estratégia recomendada: híbrida (clássico + IA)

O marcador em cima do cubo já é um "código" feito para máquina ler. Não é preciso a IA *adivinhar* qual cubo é — o marcador diz.

| Tarefa | Ferramenta | Por quê |
|---|---|---|
| **Qual cubo é** (ID) | Decodificar QR (`cv2.QRCodeDetector` / `pyzbar`) ou ArUco/AprilTag | 100% de certeza do ID, sem treino |
| **Onde está o cubo** (caixa/contorno) | YOLO11 **segmentação** (`yolo11n-seg`) | Contorno preciso mesmo inclinado, funciona quando o QR está longe/borrado |
| **Os 4 cantos do topo** | Cantos do QR + `cv2.approxPolyDP` no contorno da máscara | É o "quadrado perfeitinho" dos vídeos |
| **Posição 3D (x,y,z) para a garra** | `cv2.solvePnP` com cantos + tamanho real do marcador + calibração da câmera | Pose 6D do cubo sem câmera de profundidade |
| **Fita zebrada** | YOLO-seg (classe `barrier_tape`) + homografia do chão (IPM) | Máscara → distância métrica ao robô → obstáculo no costmap |

Fluxo por frame:

```
frame → undistort → YOLO-seg (cubo, fita)
        ├─ cubo: recorte → decodifica QR → ID
        │        cantos → solvePnP → pose no frame da câmera → TF → base do braço → IK
        └─ fita: máscara → projeção no chão (homografia) → pontos (x,y) → costmap
```

Se o QR não decodificar (longe/ângulo), o YOLO ainda localiza o cubo; o robô se aproxima e lê de perto.

## 3. Passo a passo

### Fase 0 – Ambiente (Windows)
- Python 3.10/3.11, venv, `opencv-contrib-python`, `ultralytics`, `numpy`, `pyzbar`.
- GoPro via **GoPro Webcam** (USB). Fixar resolução (1280x720), **desligar estabilização e lente "SuperView"** — usar modo *Linear*: distorção menor e calibração estável.

### Fase 1 – Calibração da câmera (obrigatório para medir distâncias)
- Tabuleiro ChArUco impresso, ~30 fotos em ângulos variados → `camera_matrix`, `dist_coeffs` (salvar em YAML).
- Sem isso, nenhuma coordenada para a garra será confiável.

### Fase 2 – Reconhecimento de cubos SEM IA (primeira entrega)
- Detectar e decodificar QR → ID + 4 cantos → desenhar contorno.
- `solvePnP` → distância e posição. Validar com régua (erro alvo < 1 cm a 40 cm).

### Fase 3 – Dataset + YOLO-seg
- Coletar 500–1500 imagens com a **própria GoPro**, na iluminação real, várias alturas/ângulos, cubos parciais, fundo da arena.
- Anotar polígonos no **Roboflow** ou **CVAT** (classes: `cubo`, `barrier_tape`, …).
- Treinar `yolo11n-seg` (GPU se tiver; senão Google Colab). Métrica: mAP50-95 de máscara.
- Exportar para ONNX/TensorRT para rodar no robô.

### Fase 4 – Fita zebrada / parede virtual
- Segmentação da fita + homografia câmera→chão (calibrada com marcas no piso).
- Publicar distância mínima e pontos da fita como obstáculos.

### Fase 5 – Migração para ROS 2
Escrever a lógica como **biblioteca Python pura** desde já (sem depender de janelas/webcam), e o ROS vira só um "invólucro":

| Nó | Publica |
|---|---|
| `camera_node` (`usb_cam`/`v4l2_camera`) | `/camera/image_raw`, `/camera/camera_info` |
| `cube_detector` | `/vision/cubes` (ID + pose, `vision_msgs/Detection3DArray`) + TF |
| `tape_detector` | `/vision/barrier_tape` (`PointCloud2`) → camada do Nav2 costmap |
| `arm_planner` (MoveIt 2 / IK própria) | consome pose do cubo via TF |

Calibração ROS: `camera_calibration` gera o mesmo `camera_info`.

## 4. Estrutura do repositório

```
vision/
  calibration/   # calibração da câmera
  cubes/         # detecção + ID + pose dos cubos
  tape/          # fita zebrada (futuro)
  common/        # captura, utilidades
datasets/        # (fora do git, usar Roboflow/DVC)
models/          # pesos treinados (fora do git)
ros2_ws/         # pacotes ROS 2 (futuro)
```

## 5. Dicas que fazem diferença
- Câmera no braço ("eye-in-hand") dá pose muito melhor para pegar; câmera fixa é melhor para fita. Se só tiver uma, priorize a do braço e faça varredura.
- Calibração mão-olho (`cv2.calibrateHandEye`) para converter câmera → base do braço.
- Iluminação da competição varia: grave dataset com luz forte, fraca e reflexo.
- Latência da GoPro como webcam é alta (~100–200 ms); para o robô final, considere uma câmera USB UVC global-shutter ou Intel RealSense (dá profundidade).
