"""Câmera (usb_cam) + detector de cubos.

  ros2 launch cube_vision cube_vision.launch.py video_device:=/dev/video0 publish_debug:=true
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    info = os.path.join(get_package_share_directory("cube_vision"), "config", "camera_info.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("video_device", default_value="/dev/video0"),
        DeclareLaunchArgument("publish_debug", default_value="false"),
        Node(
            package="usb_cam", executable="usb_cam_node_exe", namespace="camera", name="camera",
            parameters=[{
                "video_device": LaunchConfiguration("video_device"),
                "image_width": 1920,
                "image_height": 1080,
                "framerate": 30.0,
                "pixel_format": "mjpeg2rgb",
                "camera_name": "camera",
                "frame_id": "camera_optical_frame",
                "camera_info_url": "file://" + info,
            }],
        ),
        Node(
            package="cube_vision", executable="cube_detector", namespace="camera",
            parameters=[{
                "tag_size": 0.032,
                "cube_size": 0.042,
                "publish_debug": ParameterValue(LaunchConfiguration("publish_debug"), value_type=bool),
            }],
        ),
    ])
