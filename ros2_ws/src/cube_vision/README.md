# cube_vision

Nó ROS 2 que detecta os cubos (AprilTag 36h11) e publica o ID e a pose 3D do centro de cada um.

**Estado:** escrito sem acesso a uma máquina com ROS 2. Só as conversões foram testadas (no Windows,
sem ROS). O build e a execução do nó ainda precisam ser testados no Ubuntu do robô.

## Instalação (Ubuntu, ROS 2 Humble ou Jazzy)

O detector precisa de OpenCV >= 4.8, mais novo que o do apt nessas versões:

```bash
sudo apt install ros-$ROS_DISTRO-usb-cam ros-$ROS_DISTRO-vision-msgs ros-$ROS_DISTRO-tf2-ros
pip3 install "opencv-contrib-python>=4.8,<4.12" "numpy<2"

cd ros2_ws
colcon build --packages-select cube_vision
source install/setup.bash
export PYTHONPATH=$(realpath ..):$PYTHONPATH   # raiz do repositório, onde está o pacote vision/
```

## Execução

Use a mesma resolução da calibração (1920x1080):

```bash
ros2 launch cube_vision cube_vision.launch.py video_device:=/dev/video0 publish_debug:=true
ros2 topic echo /vision/cubes
```

| Tópico | Tipo | Conteúdo |
|---|---|---|
| `/camera/image_raw`, `/camera/camera_info` | entrada | imagem e calibração, publicadas pelo `usb_cam` |
| `/vision/cubes` | `vision_msgs/Detection3DArray` | ID (`cubo_3`) e pose do centro, no frame `camera_optical_frame` |
| `/vision/cubes/debug` | `sensor_msgs/Image` | imagem com contornos e eixos, se `publish_debug:=true` |
| TF `camera_optical_frame` → `cube_<id>` | | pose de cada cubo |

## Calibração

`config/camera_info.yaml` é gerado a partir da calibração, na raiz do repositório:

```bash
python -m vision.tools.export_camera_info camera_calib_webcam.yaml \
    ros2_ws/src/cube_vision/config/camera_info.yaml
```

Rode `colcon build` de novo depois de alterar esse arquivo.

## Coordenadas do robô

O nó publica no frame da câmera. Para obter as coordenadas em `base_link`, publique a posição da câmera
no robô com um `static_transform_publisher` (`base_link` → `camera_optical_frame`), medida com a câmera
já na posição final.
