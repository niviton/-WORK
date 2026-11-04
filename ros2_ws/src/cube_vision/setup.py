from glob import glob

from setuptools import setup

package_name = "cube_vision"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="Detecta os cubos com AprilTag 36h11 e publica ID e pose 3D.",
    license="MIT",
    entry_points={
        "console_scripts": ["cube_detector = cube_vision.cube_detector_node:main"],
    },
)
