"""Pool table detection using color segmentation and contour detection."""

import cv2
import numpy as np
from typing import Optional, Tuple
from ..utils.image_utils import convert_to_hsv, create_mask_from_hsv, morphological_close, morphological_open


class TableDetector:
    """Detects pool table boundaries in GamePigeon screenshots."""

    def __init__(
        self,
        hsv_lower: Tuple[int, int, int] = (35, 40, 40),
        hsv_upper: Tuple[int, int, int] = (85, 255, 255),
        min_area: int = 50000
    ):
        """Initialize table detector.

        Args:
            hsv_lower: Lower HSV bound for green felt
            hsv_upper: Upper HSV bound for green felt
            min_area: Minimum contour area to consider as table
        """
        self.hsv_lower = hsv_lower
        self.hsv_upper = hsv_upper
        self.min_area = min_area

    def detect(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Detect pool table in image.

        Args:
            image: Input BGR image

        Returns:
            Tuple of (bounding_box, table_mask) where:
                - bounding_box: [x, y, width, height]
                - table_mask: Binary mask of table region
            Returns None if table not detected
        """
        # Convert to HSV for color-based detection
        hsv = convert_to_hsv(image)

        # Create mask for green felt
        mask = create_mask_from_hsv(hsv, self.hsv_lower, self.hsv_upper)

        # Apply morphological operations to clean up mask
        mask = morphological_close(mask, kernel_size=15)
        mask = morphological_open(mask, kernel_size=5)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Check if area is large enough
        area = cv2.contourArea(largest_contour)
        if area < self.min_area:
            return None

        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        bounding_box = np.array([x, y, w, h])

        return bounding_box, mask

    def get_table_corners(self, bounding_box: np.ndarray) -> np.ndarray:
        """Get four corners of table from bounding box.

        Args:
            bounding_box: [x, y, width, height]

        Returns:
            Array of corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        x, y, w, h = bounding_box

        corners = np.array([
            [x, y],  # Top-left
            [x + w, y],  # Top-right
            [x + w, y + h],  # Bottom-right
            [x, y + h]  # Bottom-left
        ])

        return corners

    def get_table_boundaries(self, bounding_box: np.ndarray) -> dict:
        """Get table boundary information.

        Args:
            bounding_box: [x, y, width, height]

        Returns:
            Dictionary with boundary information:
                - 'left': x coordinate of left edge
                - 'right': x coordinate of right edge
                - 'top': y coordinate of top edge
                - 'bottom': y coordinate of bottom edge
                - 'width': table width
                - 'height': table height
                - 'center': [center_x, center_y]
        """
        x, y, w, h = bounding_box

        return {
            'left': x,
            'right': x + w,
            'top': y,
            'bottom': y + h,
            'width': w,
            'height': h,
            'center': np.array([x + w // 2, y + h // 2])
        }

    def is_point_on_table(self, point: np.ndarray, bounding_box: np.ndarray) -> bool:
        """Check if a point is within table boundaries.

        Args:
            point: Point coordinates [x, y]
            bounding_box: Table bounding box [x, y, width, height]

        Returns:
            True if point is on table
        """
        x, y, w, h = bounding_box
        px, py = point

        return (x <= px <= x + w) and (y <= py <= y + h)
