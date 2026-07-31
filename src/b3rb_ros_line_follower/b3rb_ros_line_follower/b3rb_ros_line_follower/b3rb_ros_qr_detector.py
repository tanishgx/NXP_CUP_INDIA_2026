# Copyright 2024-2026 NXP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np

# Pyzbar is used as a fallback detector when OpenCV's built-in
# QRCodeDetector fails to find/decode a code in a frame.
try:
    from pyzbar import pyzbar
except ImportError:
    pyzbar = None


class QRDetector(Node):
    """
    ROS 2 Node that processes raw camera images to scan for QR codes.
    It publishes the detected QR code payload on the `/qr_detection` topic.
    """

    def __init__(self):
        super().__init__('qr_detector')

        # --- Declare tunable parameters (override via launch file or
        # `ros2 param set` at runtime, no code changes/rebuild needed) ---
        self.declare_parameter('camera_topic', '/camera/image_raw/compressed')
        self.declare_parameter('detection_topic', '/qr_detection')
        self.declare_parameter('roi_x_start', 0.0)
        self.declare_parameter('roi_x_end', 1.0)
        self.declare_parameter('roi_y_start', 0.0)
        self.declare_parameter('roi_y_end', 0.6)
        self.declare_parameter('use_adaptive_threshold', True)
        # Minimum seconds before the same QR payload can be published again.
        # Prevents flooding /qr_detection while the robot sits in front of
        # the same sign across many frames.
        self.declare_parameter('repeat_suppress_seconds', 1.5)

        camera_topic = self.get_parameter('camera_topic').value
        detection_topic = self.get_parameter('detection_topic').value

        self._last_data = None
        self._last_publish_time = 0.0

        if pyzbar is None:
            self.get_logger().warn(
                "pyzbar not installed - falling back to OpenCV-only detection. "
                "Install with: pip install pyzbar"
            )

        # Subscription for camera images.
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            camera_topic,
            self.camera_image_callback,
            10)

        # Publisher for QR code detection results.
        self.publisher_qr = self.create_publisher(
            String,
            detection_topic,
            10)

        self.get_logger().info(
            f"QR Detector Node started. Subscribed to {camera_topic}, "
            f"publishing on {detection_topic}."
        )

    def camera_image_callback(self, message):
        """Processes incoming camera frames to detect QR codes."""
        # Convert compressed image message to OpenCV format
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            self.get_logger().debug("Failed to decode incoming image.")
            return

        qr_data = self.detect_qr_code(image)
        if qr_data is None:
            return

        # --- Debounce: skip re-publishing the same payload too soon ---
        now = time.monotonic()
        suppress_seconds = self.get_parameter('repeat_suppress_seconds').value
        if (qr_data == self._last_data and
                (now - self._last_publish_time) < suppress_seconds):
            return

        self._last_data = qr_data
        self._last_publish_time = now

        msg = String()
        msg.data = qr_data
        self.publisher_qr.publish(msg)
        self.get_logger().info(f"Published QR Data: {qr_data}")

    def preprocess_image(self, image):
        """
        Crop to the region of interest, convert to grayscale, and
        (optionally) apply adaptive thresholding.

        Cropping cuts down how many pixels the detector has to scan
        (faster). Grayscale removes color-channel noise. Adaptive
        thresholding binarizes the image using a locally-computed
        threshold per region, which helps a lot when the track has
        uneven lighting (shadows/glare across the frame) rather than
        one uniform brightness level.
        """
        h, w = image.shape[:2]

        x1 = int(self.get_parameter('roi_x_start').value * w)
        x2 = int(self.get_parameter('roi_x_end').value * w)
        y1 = int(self.get_parameter('roi_y_start').value * h)
        y2 = int(self.get_parameter('roi_y_end').value * h)

        roi = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        if self.get_parameter('use_adaptive_threshold').value:
            gray = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=25,
                C=5)

        return gray

    def detect_qr_code(self, image):
        """
        Detect and decode QR code in the image.

        Runs the (fast) OpenCV detector first on the preprocessed frame.
        If that fails to find anything, falls back to pyzbar, which
        tends to be more robust to skewed/angled or partially obscured
        codes, at the cost of speed.
        """
        processed = self.preprocess_image(image)

        # --- Method 1: OpenCV built-in QR detector (fast, tried first) ---
        try:
            detector = cv2.QRCodeDetector()
            data, bbox, straight_qrcode = detector.detectAndDecode(processed)
            if bbox is not None and data != "":
                return data
        except Exception as e:
            self.get_logger().debug(f"OpenCV QR Detection failed: {e}")

        # --- Method 2: pyzbar fallback (slower, more robust) ---
        if pyzbar is not None:
            try:
                decoded_objects = pyzbar.decode(processed)
                for obj in decoded_objects:
                    return obj.data.decode('utf-8')
            except Exception as e:
                self.get_logger().debug(f"pyzbar QR Detection failed: {e}")

        return None


def main(args=None):
    rclpy.init(args=args)
    node = QRDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
