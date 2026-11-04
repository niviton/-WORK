"""Nó ROS 2: imagem -> ID e pose 3D dos cubos.

Assina:   image_raw (sensor_msgs/Image), camera_info (sensor_msgs/CameraInfo)
Publica:  /vision/cubes (vision_msgs/Detection3DArray), no frame óptico da câmera
          TF <frame da câmera> -> cube_<id>
          /vision/cubes/debug (sensor_msgs/Image), só se publish_debug = true

As poses usam o horário da imagem (header.stamp): quem consulta a TF no mesmo horário
compensa o atraso da câmera.
"""
import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster
from vision.cubes.detector import CubeDetector
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from cube_vision.conversions import cube_pose, image_to_bgr


class CubeDetectorNode(Node):
    def __init__(self):
        super().__init__("cube_detector")
        self.tag_size = self.declare_parameter("tag_size", 0.032).value
        self.cube_size = self.declare_parameter("cube_size", 0.042).value
        self.smooth = self.declare_parameter("smooth", 0.6).value
        self.hold = self.declare_parameter("hold_frames", 5).value
        self.publish_debug = self.declare_parameter("publish_debug", False).value

        self.detector = None
        self.pub = self.create_publisher(Detection3DArray, "/vision/cubes", 10)
        self.debug_pub = self.create_publisher(Image, "/vision/cubes/debug", 1)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(CameraInfo, "camera_info", self.on_info, qos_profile_sensor_data)
        self.create_subscription(Image, "image_raw", self.on_image, qos_profile_sensor_data)
        self.get_logger().info("Aguardando camera_info...")

    def on_info(self, msg):
        if self.detector is not None:
            return
        K = np.array(msg.k, float).reshape(3, 3)
        if K[0, 0] == 0:
            self.get_logger().warn("camera_info sem calibração (fx = 0); ignorando", once=True)
            return
        self.detector = CubeDetector(K, np.array(msg.d, float), self.tag_size,
                                     smooth=self.smooth, hold=self.hold)
        self.get_logger().info(f"Calibração recebida: {msg.width}x{msg.height}")

    def on_image(self, msg):
        if self.detector is None:
            return
        try:
            frame = image_to_bgr(msg)
        except ValueError as e:
            self.get_logger().error(str(e), throttle_duration_sec=5.0)
            return
        cubes = self.detector.detect(frame)

        out = Detection3DArray()
        out.header = msg.header
        for cube in cubes:
            center, quat = cube_pose(cube.rvec, cube.tvec, self.cube_size)
            out.detections.append(self.to_detection(msg.header, cube.id, center, quat))
            self.tf.sendTransform(self.to_transform(msg.header, cube.id, center, quat))
        self.pub.publish(out)

        if self.publish_debug:
            self.debug_pub.publish(self.to_debug(msg.header, frame, cubes))

    def to_detection(self, header, cube_id, center, quat):
        det = Detection3D()
        det.header = header
        det.id = cube_id
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = f"cubo_{cube_id}"
        hyp.hypothesis.score = 1.0
        for pose in (hyp.pose.pose, det.bbox.center):
            pose.position.x, pose.position.y, pose.position.z = map(float, center)
            o = pose.orientation
            o.x, o.y, o.z, o.w = map(float, quat)
        det.results.append(hyp)
        det.bbox.size.x = det.bbox.size.y = det.bbox.size.z = float(self.cube_size)
        return det

    @staticmethod
    def to_transform(header, cube_id, center, quat):
        t = TransformStamped()
        t.header = header
        t.child_frame_id = f"cube_{cube_id}"
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(float, center)
        r = t.transform.rotation
        r.x, r.y, r.z, r.w = map(float, quat)
        return t

    def to_debug(self, header, frame, cubes):
        for c in cubes:
            cv2.polylines(frame, [c.corners.astype(int)], True, (0, 255, 0), 2)
            cv2.drawFrameAxes(frame, self.detector.K, self.detector.dist, c.rvec, c.tvec, 0.03)
            org = tuple(int(v) for v in c.corners[0] - [0, 10])
            cv2.putText(frame, f"cubo_{c.id} {c.distance * 100:.1f}cm", org,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        img = Image()
        img.header = header
        img.height, img.width = frame.shape[:2]
        img.encoding = "bgr8"
        img.step = frame.shape[1] * 3
        img.data = frame.tobytes()
        return img


def main():
    rclpy.init()
    node = CubeDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
