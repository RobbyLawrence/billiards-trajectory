"""Cue stick detection using Hough Line Transform and edge detection."""

import cv2
import numpy as np
from typing import Optional, Tuple
from ..utils.image_utils import convert_to_gray
from ..utils.geometry import distance_point_to_line, normalize_vector


class CueDetector:
    """Detects cue stick direction using line detection."""

    def __init__(
        self,
        rho: float = 1.0,
        theta: float = np.pi / 180,
        threshold: int = 100,
        min_line_length: int = 50,
        max_line_gap: int = 10,
        proximity_threshold: float = 100.0
    ):
        """Initialize cue detector.

        Args:
            rho: Distance resolution in pixels for Hough transform
            theta: Angle resolution in radians for Hough transform
            threshold: Minimum votes for line detection
            min_line_length: Minimum line length
            max_line_gap: Maximum gap between line segments
            proximity_threshold: Max distance from cue ball to consider line as cue
        """
        self.rho = rho
        self.theta = theta
        self.threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.proximity_threshold = proximity_threshold

    def detect(
        self,
        image: np.ndarray,
        cue_ball_center: np.ndarray
    ) -> Optional[Tuple[np.ndarray, float]]:
        """Detect cue stick direction from cue ball.

        Args:
            image: Input BGR image
            cue_ball_center: Cue ball center coordinates [x, y]

        Returns:
            Tuple of (direction_vector, angle_degrees) if detected, None otherwise
            Direction vector is normalized and points in the shot direction
        """
        # Convert to grayscale
        gray = convert_to_gray(image)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines using Hough Line Transform (Probabilistic)
        lines = cv2.HoughLinesP(
            edges,
            rho=self.rho,
            theta=self.theta,
            threshold=self.threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        if lines is None:
            return None

        # Find line closest to cue ball
        best_line = None
        min_distance = float('inf')

        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_start = np.array([x1, y1], dtype=float)
            line_end = np.array([x2, y2], dtype=float)

            # Calculate distance from cue ball to line
            dist = distance_point_to_line(cue_ball_center, line_start, line_end)

            # Check if line is close enough to cue ball
            if dist < min_distance and dist < self.proximity_threshold:
                min_distance = dist
                best_line = (line_start, line_end)

        if best_line is None:
            return None

        # Calculate direction vector from the best line
        line_start, line_end = best_line
        direction = line_end - line_start

        # Determine which end of the line is closer to cue ball
        dist_to_start = np.linalg.norm(cue_ball_center - line_start)
        dist_to_end = np.linalg.norm(cue_ball_center - line_end)

        # Direction should point AWAY from cue ball (shot direction)
        if dist_to_start < dist_to_end:
            # Cue ball is closer to start, so direction is start -> end
            direction = line_end - line_start
        else:
            # Cue ball is closer to end, so direction is end -> start
            direction = line_start - line_end

        # Normalize direction vector
        direction_normalized = normalize_vector(direction)

        # Calculate angle in degrees (0 = right, 90 = down)
        angle_rad = np.arctan2(direction_normalized[1], direction_normalized[0])
        angle_deg = np.degrees(angle_rad)

        return direction_normalized, angle_deg

    def detect_from_aiming_line(
        self,
        image: np.ndarray,
        cue_ball_center: np.ndarray,
        line_color_hsv_range: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None
    ) -> Optional[Tuple[np.ndarray, float]]:
        """Detect cue direction from GamePigeon's aiming line (if visible).

        GamePigeon often shows a white/yellow aiming guide line.

        Args:
            image: Input BGR image
            cue_ball_center: Cue ball center coordinates [x, y]
            line_color_hsv_range: Optional HSV range for aiming line color

        Returns:
            Tuple of (direction_vector, angle_degrees) if detected, None otherwise
        """
        if line_color_hsv_range is None:
            # Default to detecting white/bright yellow aiming lines
            # White: high V, low S
            line_color_hsv_range = ((0, 0, 200), (180, 50, 255))

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower, upper = line_color_hsv_range

        # Create mask for aiming line color
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        # Find lines in the mask
        edges = cv2.Canny(mask, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=30,
            maxLineGap=5
        )

        if lines is None:
            return None

        # Find line starting from or passing through cue ball
        best_line = None
        min_distance = float('inf')

        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_start = np.array([x1, y1], dtype=float)
            line_end = np.array([x2, y2], dtype=float)

            # Check distance to cue ball
            dist = min(
                np.linalg.norm(cue_ball_center - line_start),
                np.linalg.norm(cue_ball_center - line_end),
                distance_point_to_line(cue_ball_center, line_start, line_end)
            )

            if dist < min_distance:
                min_distance = dist
                best_line = (line_start, line_end)

        if best_line is None or min_distance > 50:
            return None

        # Calculate direction
        line_start, line_end = best_line
        dist_to_start = np.linalg.norm(cue_ball_center - line_start)
        dist_to_end = np.linalg.norm(cue_ball_center - line_end)

        if dist_to_start < dist_to_end:
            direction = line_end - line_start
        else:
            direction = line_start - line_end

        direction_normalized = normalize_vector(direction)
        angle_rad = np.arctan2(direction_normalized[1], direction_normalized[0])
        angle_deg = np.degrees(angle_rad)

        return direction_normalized, angle_deg

    def visualize_detection(
        self,
        image: np.ndarray,
        cue_ball_center: np.ndarray,
        direction: np.ndarray,
        length: int = 200
    ) -> np.ndarray:
        """Draw detected cue direction on image for debugging.

        Args:
            image: Input image
            cue_ball_center: Cue ball center
            direction: Direction vector (normalized)
            length: Length of arrow to draw

        Returns:
            Image with visualization
        """
        vis_image = image.copy()

        # Calculate end point
        end_point = cue_ball_center + direction * length
        end_point = end_point.astype(int)
        center = cue_ball_center.astype(int)

        # Draw arrow
        cv2.arrowedLine(
            vis_image,
            tuple(center),
            tuple(end_point),
            (0, 255, 255),  # Yellow
            3,
            tipLength=0.3
        )

        return vis_image
