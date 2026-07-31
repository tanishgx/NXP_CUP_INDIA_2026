# Copyright 2024-2026 NXP
# Copyright 2016 Open Source Robotics Foundation, Inc.
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
import numpy as np
import cv2
import math
from synapse_msgs.msg import EdgeVectors

QOS_PROFILE_DEFAULT = 10
PI = math.pi

RED_COLOR = (0, 0, 255)
BLUE_COLOR = (255, 0, 0)
GREEN_COLOR = (0, 255, 0)
YELLOW_COLOR = (0, 255, 255)

# ---------------------------------------------------------------------------
# TUNABLE PARAMETERS
# ---------------------------------------------------------------------------
# How much of the image (from the bottom) is analyzed.
VECTOR_IMAGE_HEIGHT_PERCENTAGE = 0.225
VECTOR_MAGNITUDE_MINIMUM = 2.25

# Switch between the simple grayscale threshold and the more robust
# HSV/LAB based threshold. LAB's L-channel is generally the most stable
# across changes in scene lighting/white balance for isolating black stripes.
USE_LAB_THRESHOLD = True
THRESHOLD_BLACK = 60          # LAB L-channel threshold (0-255)
THRESHOLD_BLACK_GRAY = 25     # fallback grayscale threshold

# Enable Inverse Perspective Mapping (birds-eye view) before extracting
# vectors. This makes lane geometry closer to linear/parallel, which makes
# steering math and curve fitting far more reliable than a raw camera view.
USE_IPM = True

# Region of interest mask (fractions of image width/height) to reject the
# buggy's own chassis/sky/etc. Expressed as a trapezoid: (top_w, bottom_w)
# fractions of width, centered, applied to the *cropped* lower region.
ROI_TOP_WIDTH_FRACTION = 0.55
ROI_BOTTOM_WIDTH_FRACTION = 1.0


class EdgeVectorsPublisher(Node):
    """
    ROS 2 Node that processes raw camera images to detect the lane edges
    (left/right bounds). It publishes the detected boundaries as
    synapse_msgs/EdgeVectors.
    """

    def __init__(self):
        super().__init__('edge_vectors_publisher')

        # Subscription for camera images.
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            QOS_PROFILE_DEFAULT)

        # Publisher for edge vectors.
        self.publisher_edge_vectors = self.create_publisher(
            EdgeVectors,
            '/edge_vectors',
            QOS_PROFILE_DEFAULT)

        # Publisher for thresh image (for debugging thresholding/segmentation).
        self.publisher_thresh_image = self.create_publisher(
            CompressedImage,
            "/debug_images/thresh_image",
            QOS_PROFILE_DEFAULT)

        # Publisher for vector image (for debugging vector drawing).
        self.publisher_vector_image = self.create_publisher(
            CompressedImage,
            "/debug_images/vector_image",
            QOS_PROFILE_DEFAULT)

        # Publisher for the warped (birds-eye) debug image.
        self.publisher_warp_image = self.create_publisher(
            CompressedImage,
            "/debug_images/warp_image",
            QOS_PROFILE_DEFAULT)

        self.image_height = 0
        self.image_width = 0
        self.lower_image_height = 0
        self.upper_image_height = 0

        # Perspective transform matrices, computed lazily once we know
        # the incoming image size.
        self.warp_matrix = None
        self.warp_matrix_inv = None
        self.warp_size = None

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------
    def publish_debug_image(self, publisher, image):
        """Helper function to publish OpenCV debug images to ROS topics."""
        message = CompressedImage()
        _, encoded_data = cv2.imencode('.jpg', image)
        message.format = "jpeg"
        message.data = encoded_data.tobytes()
        publisher.publish(message)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def get_vector_angle_in_radians(self, vector):
        """Calculates the slope angle of a vector in radians."""
        if ((vector[0][0] - vector[1][0]) == 0):  # Prevent division by zero
            theta = PI / 2
        else:
            slope = (vector[1][1] - vector[0][1]) / (vector[0][0] - vector[1][0])
            theta = math.atan(slope)
        return theta

    def build_ipm_matrices(self, width, height):
        """
        Builds the perspective transform (and its inverse) that maps the
        cropped, lower "road" region of the camera image into a birds-eye
        (top-down) rectangle.

        NOTE: the source trapezoid below is a reasonable starting point for
        a forward-facing camera looking down at a track. You will likely
        need to tune these four source points against your own camera's
        mount angle/FOV using the /debug_images/warp_image topic — lane
        edges should look roughly parallel and vertical once tuned.
        """
        src = np.float32([
            [width * 0.15, height],        # bottom-left
            [width * 0.85, height],        # bottom-right
            [width * 0.62, 0],             # top-right
            [width * 0.38, 0],             # top-left
        ])
        dst = np.float32([
            [width * 0.25, height],
            [width * 0.75, height],
            [width * 0.75, 0],
            [width * 0.25, 0],
        ])
        self.warp_matrix = cv2.getPerspectiveTransform(src, dst)
        self.warp_matrix_inv = cv2.getPerspectiveTransform(dst, src)
        self.warp_size = (width, height)

    def apply_roi_mask(self, binary_image):
        """
        Masks out everything outside a trapezoid region of interest, to
        reject the buggy's own chassis and any noise near the image edges.
        """
        h, w = binary_image.shape[:2]
        top_w = w * ROI_TOP_WIDTH_FRACTION
        bottom_w = w * ROI_BOTTOM_WIDTH_FRACTION
        polygon = np.array([[
            (int((w - bottom_w) / 2), h),
            (int((w + bottom_w) / 2), h),
            (int((w + top_w) / 2), 0),
            (int((w - top_w) / 2), 0),
        ]], dtype=np.int32)

        mask = np.zeros_like(binary_image)
        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(binary_image, mask)

    # ------------------------------------------------------------------
    # Thresholding
    # ------------------------------------------------------------------
    def threshold_image(self, image):
        """
        Isolates the black lane boundary stripes from the road surface.

        Uses the LAB color space's L (lightness) channel by default, which
        is far more robust to color-cast/white-balance shifts than raw BGR
        grayscale, while still cheap to compute. Falls back to a simple
        grayscale threshold if USE_LAB_THRESHOLD is False.
        """
        if USE_LAB_THRESHOLD:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            # Slight blur to reduce speckle noise before thresholding.
            l_channel = cv2.GaussianBlur(l_channel, (5, 5), 0)
            thresh = cv2.threshold(
                l_channel, THRESHOLD_BLACK, 255, cv2.THRESH_BINARY_INV)[1]
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(
                gray, THRESHOLD_BLACK_GRAY, 255, cv2.THRESH_BINARY_INV)[1]

        # Morphological cleanup: close small gaps in the stripe, then
        # remove isolated speckle noise.
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        return thresh

    # ------------------------------------------------------------------
    # Vector extraction (unchanged core logic from the provided baseline)
    # ------------------------------------------------------------------
    def compute_vectors_from_image(self, image, thresh):
        """
        Analyzes the binary threshold image and extracts left and right
        lane edge vectors via contour finding.
        """
        contours = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

        vectors = []
        for i in range(len(contours)):
            coordinates = contours[i][:, 0, :]

            min_y_value = np.min(coordinates[:, 1])
            max_y_value = np.max(coordinates[:, 1])

            min_y_coords = np.array(coordinates[coordinates[:, 1] == min_y_value])
            max_y_coords = np.array(coordinates[coordinates[:, 1] == max_y_value])

            min_y_coord = min_y_coords[0]
            max_y_coord = max_y_coords[0]

            magnitude = np.linalg.norm(min_y_coord - max_y_coord)
            if (magnitude > VECTOR_MAGNITUDE_MINIMUM):
                rover_point = [self.image_width / 2, self.lower_image_height]
                middle_point = (min_y_coord + max_y_coord) / 2
                distance = np.linalg.norm(middle_point - rover_point)

                angle = self.get_vector_angle_in_radians([min_y_coord, max_y_coord])
                if angle > 0:
                    min_y_coord[0] = np.max(min_y_coords[:, 0])
                else:
                    max_y_coord[0] = np.max(max_y_coords[:, 0])

                vectors.append([list(min_y_coord), list(max_y_coord), distance])

                cv2.line(image, tuple(min_y_coord), tuple(max_y_coord), BLUE_COLOR, 2)

        return vectors, image

    def unwarp_point(self, point):
        """Maps a single (x, y) point from warped/birds-eye space back to
        the original cropped-image space using the inverse perspective
        matrix."""
        pt = np.array([[point]], dtype=np.float32)  # shape (1,1,2)
        unwarped = cv2.perspectiveTransform(pt, self.warp_matrix_inv)
        return [int(unwarped[0, 0, 0]), int(unwarped[0, 0, 1])]

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def process_image_for_edge_vectors(self, image):
        """
        Preprocesses the frame (color-space thresholding + ROI + optional
        birds-eye warp) and extracts lane vectors.
        """
        self.image_height, self.image_width, _ = image.shape
        self.lower_image_height = int(self.image_height * VECTOR_IMAGE_HEIGHT_PERCENTAGE)
        self.upper_image_height = int(self.image_height - self.lower_image_height)

        # 1. Crop to the lower region of interest close to the buggy.
        crop_top = self.image_height - self.lower_image_height
        image_cropped = image[crop_top:].copy()

        crop_h, crop_w = image_cropped.shape[:2]

        # 2. Optionally warp the cropped region into a birds-eye view so
        #    lane edges become close to parallel/vertical, which makes the
        #    vector math and any future curve-fitting far more linear.
        if USE_IPM:
            if self.warp_matrix is None or self.warp_size != (crop_w, crop_h):
                self.build_ipm_matrices(crop_w, crop_h)
            working_image = cv2.warpPerspective(
                image_cropped, self.warp_matrix, (crop_w, crop_h))
        else:
            working_image = image_cropped

        # 3. Threshold to isolate the black lane stripes.
        thresh = self.threshold_image(working_image)

        # 4. Mask out chassis/off-track noise with a region-of-interest.
        thresh = self.apply_roi_mask(thresh)

        # 5. Extract raw candidate vectors from contours.
        vectors, debug_img = self.compute_vectors_from_image(working_image.copy(), thresh)

        # 6. Sort by distance to buggy (closest first).
        vectors = sorted(vectors, key=lambda x: x[2])

        # 7. Split into left / right halves.
        half_width = crop_w / 2
        vectors_left = [v for v in vectors if ((v[0][0] + v[1][0]) / 2) < half_width]
        vectors_right = [v for v in vectors if ((v[0][0] + v[1][0]) / 2) >= half_width]

        final_vectors = []
        for side_vectors in [vectors_left, vectors_right]:
            if len(side_vectors) > 0:
                best_vector = side_vectors[0]
                cv2.line(debug_img, tuple(best_vector[0]), tuple(best_vector[1]), GREEN_COLOR, 2)

                p0 = best_vector[0]
                p1 = best_vector[1]

                if USE_IPM:
                    # Map the chosen vector's endpoints back out of
                    # birds-eye space into the original cropped-image space
                    # so downstream nodes (and the rest of your teammates'
                    # code) keep working with familiar camera-frame
                    # coordinates.
                    p0 = self.unwarp_point(p0)
                    p1 = self.unwarp_point(p1)

                # Shift back into full (uncropped) image coordinates.
                p0[1] += self.upper_image_height
                p1[1] += self.upper_image_height
                final_vectors.append([p0, p1])

        # Publish debug topics.
        self.publish_debug_image(self.publisher_thresh_image, thresh)
        self.publish_debug_image(self.publisher_vector_image, debug_img)
        if USE_IPM:
            self.publish_debug_image(self.publisher_warp_image, working_image)

        return final_vectors

    def camera_image_callback(self, message):
        """Processes incoming camera frames and publishes detected EdgeVectors."""
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        vectors = self.process_image_for_edge_vectors(image)

        vectors_message = EdgeVectors()
        vectors_message.image_height = image.shape[0]
        vectors_message.image_width = image.shape[1]
        vectors_message.vector_count = 0

        # Vector 1 (usually representing Left boundary)
        if len(vectors) > 0:
            vectors_message.vector_1[0].x = float(vectors[0][0][0])
            vectors_message.vector_1[0].y = float(vectors[0][0][1])
            vectors_message.vector_1[1].x = float(vectors[0][1][0])
            vectors_message.vector_1[1].y = float(vectors[0][1][1])
            vectors_message.vector_count += 1

        # Vector 2 (usually representing Right boundary)
        if len(vectors) > 1:
            vectors_message.vector_2[0].x = float(vectors[1][0][0])
            vectors_message.vector_2[0].y = float(vectors[1][0][1])
            vectors_message.vector_2[1].x = float(vectors[1][1][0])
            vectors_message.vector_2[1].y = float(vectors[1][1][1])
            vectors_message.vector_count += 1

        self.publisher_edge_vectors.publish(vectors_message)


def main(args=None):
    rclpy.init(args=args)
    node = EdgeVectorsPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
