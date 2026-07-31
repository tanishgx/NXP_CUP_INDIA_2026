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

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np
import os

# HINT: TensorFlow/Keras can be heavy and might not be installed by default.
# We use tflite-runtime for lightweight inference (pre-approved for NXP Cup).
# Install it using: pip install tflite-runtime
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    Interpreter = None

# Sign classification labels (must match model training order)
CLASS_LABELS = ["TURN_LEFT", "TURN_RIGHT", "GO_STRAIGHT", "STOP_SIGN"]

# HINT: HSV color ranges for traffic sign detection
# Red signs (typically stop, prohibitory)
RED_LOWER_1 = np.array([0, 120, 100])
RED_UPPER_1 = np.array([10, 255, 255])
RED_LOWER_2 = np.array([170, 120, 100])
RED_UPPER_2 = np.array([180, 255, 255])
# Blue signs (typically mandatory, directional)
BLUE_LOWER = np.array([100, 150, 100])
BLUE_UPPER = np.array([130, 255, 255])
# Yellow/Orange signs (typically warning)
YELLOW_LOWER = np.array([20, 100, 100])
YELLOW_UPPER = np.array([35, 255, 255])

class ObjectRecognizer(Node):
    """
    ROS 2 Node that processes raw camera images to recognize traffic sign boards.
    It publishes the detected sign type/labels on the `/sign_board_detection` topic.
    """
    def __init__(self):
        super().__init__('object_recognizer')

        # Subscription for camera images.
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            10)

        # Publisher for sign board detection results.
        self.publisher_sign = self.create_publisher(
            String,
            '/sign_board_detection',
            10)

        # Attempt to load the pre-trained TFLite model (model.tflite) located in the same directory.
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        if Interpreter is not None:
            try:
                dir_path = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(dir_path, 'model.tflite')
                if os.path.exists(model_path):
                    self.interpreter = Interpreter(model_path=model_path)
                    self.interpreter.allocate_tensors()
                    self.input_details = self.interpreter.get_input_details()
                    self.output_details = self.interpreter.get_output_details()
                    self.get_logger().info(f"Loaded TFLite model from {model_path}")
                else:
                    self.get_logger().warn(f"Model file not found at {model_path}")
            except Exception as e:
                self.get_logger().error(f"Failed to load TFLite model: {e}")
        else:
            self.get_logger().warn("tflite-runtime not installed. Running in OpenCV mode.")

        # Frame rate limiting to avoid overwhelming the CPU
        self.last_process_time = 0
        self.process_interval = 0.1  # Process at 10Hz max

        self.get_logger().info("Object Recognizer Node started. Waiting for images...")

    def camera_image_callback(self, message):
        """Processes incoming camera frames to classify traffic signs."""
        # Frame rate limiting to avoid overwhelming the CPU
        current_time = self.get_clock().now().nanoseconds / 1e9
        if current_time - self.last_process_time < self.process_interval:
            return
        self.last_process_time = current_time

        # Convert compressed image message to OpenCV format
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            self.get_logger().debug("Failed to decode image")
            return

        sign_detected = self.classify_sign(image)

        if sign_detected is not None:
            msg = String()
            msg.data = sign_detected
            self.publisher_sign.publish(msg)
            self.get_logger().info(f"Detected Sign Board: {sign_detected}")

    def classify_sign(self, image):
        """
        Classify traffic sign boards.
        
        OPTIMIZATION HINTS:
        - If TensorFlow is installed, you can pre-process the image (e.g. crop the sign region, 
          resize to 150x150, normalize, expand dimensions) and feed it into `self.model.predict()`.
        - Alternatively, you can use classic Computer Vision techniques:
          1. Color Segmentation: Convert to HSV and threshold for specific sign colors.
          2. Shape Detection: Find contours and approximate polygons.
          3. Template Matching: Match regions of interest against template images of sign boards.
        """
        # --- Method 1: TFLite Model Inference (Primary) ---
        if self.interpreter is not None:
            try:
                # Resize image to match model input dimensions (e.g., 150x150)
                input_shape = self.input_details[0]['shape']
                input_height = input_shape[1]
                input_width = input_shape[2]
                resized_image = cv2.resize(image, (input_width, input_height))
                
                # Add batch dimension and normalize
                image_array = np.expand_dims(resized_image, axis=0).astype(np.float32) / 255.0
                
                # Run inference
                self.interpreter.set_tensor(self.input_details[0]['index'], image_array)
                self.interpreter.invoke()
                output = self.interpreter.get_tensor(self.output_details[0]['index'])
                
                # Parse predictions based on model's classification classes
                class_idx = np.argmax(output[0])
                confidence = output[0][class_idx]
                
                # Only return detection if confidence is above threshold
                if confidence > 0.5 and class_idx < len(CLASS_LABELS):
                    self.get_logger().debug(
                        f"TFLite prediction: {CLASS_LABELS[class_idx]} (confidence: {confidence:.2f})")
                    return CLASS_LABELS[class_idx]
            except Exception as e:
                self.get_logger().debug(f"TFLite inference failed: {e}")

        # --- Method 2: OpenCV Color + Shape Detection (Fallback) ---
        return self.classify_sign_cv(image)

    def classify_sign_cv(self, image):
        """
        Fallback classification using OpenCV color segmentation and shape detection.
        
        HINTS for optimization:
        - Color Segmentation: Convert to HSV and threshold for specific sign colors.
        - Shape Detection: Find contours and approximate polygons.
        - Template Matching: Match regions of interest against template images.
        """
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            height, width = image.shape[:2]

            # Create color masks for traffic signs
            # Red signs (stop, prohibitory) - red wraps around hue 0/180
            red_mask1 = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
            red_mask2 = cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)

            # Blue signs (mandatory, directional)
            blue_mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)

            # Yellow/Orange signs (warning)
            yellow_mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)

            # Combine all sign color masks
            combined_mask = cv2.bitwise_or(red_mask, cv2.bitwise_or(blue_mask, yellow_mask))

            # Morphological operations to clean up noise
            kernel = np.ones((5, 5), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

            # Find contours in the combined mask
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best_sign = None
            best_area = 0

            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter small contours (noise)
                if area < 500 or area > (height * width * 0.5):
                    continue

                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)

                # Approximate polygon to detect shape
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                num_vertices = len(approx)

                # Determine sign type based on color and shape
                sign_type = None

                # Check which color mask this contour belongs to
                mask_val = combined_mask[y + h // 2, x + w // 2]

                if mask_val > 0:
                    # Arrow detection: elongated shape with pointed end
                    if 0.3 < aspect_ratio < 0.7 and num_vertices >= 7:
                        # Check if more of the shape is on left or right side of center
                        if x + w // 2 < width // 2:
                            sign_type = "TURN_LEFT"
                        else:
                            sign_type = "TURN_RIGHT"
                    
                    # Circular/octagonal sign (stop sign)
                    elif num_vertices >= 7 and 0.8 < aspect_ratio < 1.2:
                        # Check for red color dominance
                        red_count = cv2.countNonZero(red_mask[y:y+h, x:x+w])
                        if red_count > (area * 0.3):
                            sign_type = "STOP_SIGN"
                    
                    # Triangular sign or directional arrow
                    elif num_vertices == 3 and area > best_area:
                        # Triangle could be yield or directional
                        if x + w // 2 < width // 3:
                            sign_type = "TURN_LEFT"
                        elif x + w // 2 > width * 2 // 3:
                            sign_type = "TURN_RIGHT"
                        else:
                            sign_type = "GO_STRAIGHT"
                    
                    # Rectangular sign with arrow-like features
                    elif num_vertices == 4 and aspect_ratio < 0.8:
                        # Tall rectangle often indicates straight ahead
                        sign_type = "GO_STRAIGHT"

                # Update best detection (largest valid sign wins)
                if sign_type is not None and area > best_area:
                    best_sign = sign_type
                    best_area = area

            if best_sign is not None:
                self.get_logger().debug(
                    f"OpenCV detection: {best_sign} (area: {best_area:.0f})")
                return best_sign

        except Exception as e:
            self.get_logger().debug(f"OpenCV detection failed: {e}")

        return None

def main(args=None):
    rclpy.init(args=args)
    node = ObjectRecognizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
