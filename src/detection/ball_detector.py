"""Ball detection and classification using Hough Circle Transform."""

import cv2
import numpy as np
from typing import List, Dict, Tuple
from ..utils.image_utils import convert_to_gray, convert_to_hsv


class Ball:
    """Represents a detected pool ball."""

    def __init__(self, center: np.ndarray, radius: float, ball_type: str, color: Tuple[int, int, int]):
        """Initialize ball.

        Args:
            center: Ball center coordinates [x, y]
            radius: Ball radius in pixels
            ball_type: Type of ball ('cue', 'solid', 'stripe', 'eight')
            color: BGR color tuple
        """
        self.center = center
        self.radius = radius
        self.ball_type = ball_type
        self.color = color

    def __repr__(self):
        return f"Ball(center={self.center}, radius={self.radius:.1f}, type={self.ball_type})"


class BallDetector:
    """Detects and classifies pool balls using Hough Circle Transform."""

    def __init__(
        self,
        dp: float = 1.2,
        min_dist: int = 30,
        param1: int = 50,
        param2: int = 30,
        min_radius: int = 10,
        max_radius: int = 40,
        cue_ball_threshold: int = 200,
        black_ball_threshold: int = 80
    ):
        """Initialize ball detector.

        Args:
            dp: Inverse ratio of accumulator resolution
            min_dist: Minimum distance between circle centers
            param1: Canny edge detector higher threshold
            param2: Accumulator threshold for circle detection
            min_radius: Minimum ball radius
            max_radius: Maximum ball radius
            cue_ball_threshold: Minimum V value for white cue ball
            black_ball_threshold: Maximum V value for black 8-ball
        """
        self.dp = dp
        self.min_dist = min_dist
        self.param1 = param1
        self.param2 = param2
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.cue_ball_threshold = cue_ball_threshold
        self.black_ball_threshold = black_ball_threshold

    def detect(self, image: np.ndarray, table_mask: Optional[np.ndarray] = None) -> List[Ball]:
        """Detect all balls in image.

        Args:
            image: Input BGR image
            table_mask: Optional binary mask of table region

        Returns:
            List of detected Ball objects
        """
        # Convert to grayscale for Hough Circle detection
        gray = convert_to_gray(image)

        # Apply mask if provided
        if table_mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=table_mask)

        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )

        if circles is None:
            return []

        # Convert to integer coordinates
        circles = np.round(circles[0, :]).astype(int)

        # Classify each detected ball
        balls = []
        for (x, y, r) in circles:
            ball_type, color = self._classify_ball(image, x, y, r)
            ball = Ball(
                center=np.array([x, y], dtype=float),
                radius=float(r),
                ball_type=ball_type,
                color=color
            )
            balls.append(ball)

        return balls

    def _classify_ball(self, image: np.ndarray, x: int, y: int, radius: int) -> Tuple[str, Tuple[int, int, int]]:
        """Classify ball type based on color.

        Args:
            image: Input BGR image
            x: Ball center x coordinate
            y: Ball center y coordinate
            radius: Ball radius

        Returns:
            Tuple of (ball_type, color)
        """
        # Create circular mask for ball
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), radius - 2, 255, -1)

        # Get mean color in BGR and HSV
        mean_bgr = cv2.mean(image, mask=mask)[:3]
        hsv = convert_to_hsv(image)
        mean_hsv = cv2.mean(hsv, mask=mask)[:3]

        # Extract HSV components
        h, s, v = mean_hsv

        # Classify based on color characteristics
        # Cue ball: High brightness (V) and low saturation (S)
        if v > self.cue_ball_threshold and s < 50:
            return 'cue', (255, 255, 255)

        # Eight ball: Very dark (low V)
        if v < self.black_ball_threshold:
            return 'eight', (0, 0, 0)

        # For colored balls, check saturation
        # Solids have more saturation, stripes have mixed areas
        if s > 80:
            # Likely a solid ball - use dominant color
            return 'solid', tuple(int(c) for c in mean_bgr)
        else:
            # Might be a striped ball or partially visible
            return 'stripe', tuple(int(c) for c in mean_bgr)

    def get_cue_ball(self, balls: List[Ball]) -> Optional[Ball]:
        """Find the cue ball from detected balls.

        Args:
            balls: List of detected balls

        Returns:
            Cue ball if found, None otherwise
        """
        for ball in balls:
            if ball.ball_type == 'cue':
                return ball
        return None

    def filter_balls_in_region(
        self,
        balls: List[Ball],
        region_min: np.ndarray,
        region_max: np.ndarray
    ) -> List[Ball]:
        """Filter balls within a rectangular region.

        Args:
            balls: List of balls
            region_min: Minimum corner [x, y]
            region_max: Maximum corner [x, y]

        Returns:
            Filtered list of balls
        """
        filtered = []
        for ball in balls:
            if (region_min[0] <= ball.center[0] <= region_max[0] and
                region_min[1] <= ball.center[1] <= region_max[1]):
                filtered.append(ball)
        return filtered

    def get_ball_positions(self, balls: List[Ball]) -> np.ndarray:
        """Get array of ball center positions.

        Args:
            balls: List of balls

        Returns:
            Nx2 array of ball positions
        """
        if not balls:
            return np.array([])
        return np.array([ball.center for ball in balls])

    def get_ball_radii(self, balls: List[Ball]) -> np.ndarray:
        """Get array of ball radii.

        Args:
            balls: List of balls

        Returns:
            Array of ball radii
        """
        if not balls:
            return np.array([])
        return np.array([ball.radius for ball in balls])
