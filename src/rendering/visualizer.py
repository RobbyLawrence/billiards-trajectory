"""Visualization of detected elements and predicted trajectories."""

import cv2
import numpy as np
from typing import List, Dict, Optional
from ..detection.ball_detector import Ball
from ..physics.trajectory import TrajectoryPoint


class TrajectoryVisualizer:
    """Draws detected elements and trajectories on pool table image."""

    def __init__(self):
        """Initialize visualizer with default colors and styles."""
        self.colors = {
            'cue_trajectory': (0, 255, 0),  # Green
            'ball_trajectory': (0, 255, 255),  # Yellow
            'collision_point': (0, 0, 255),  # Red
            'cushion_point': (255, 0, 255),  # Magenta
            'cue_direction': (0, 255, 255),  # Yellow
            'ball_outline': (255, 255, 255),  # White
            'cue_ball': (255, 255, 255),  # White
            'text': (255, 255, 255)  # White
        }

        self.line_thickness = 2
        self.point_radius = 5
        self.ball_outline_thickness = 2

    def draw_balls(
        self,
        image: np.ndarray,
        balls: List[Ball],
        highlight_cue: bool = True
    ) -> np.ndarray:
        """Draw detected balls on image.

        Args:
            image: Input image
            balls: List of detected balls
            highlight_cue: Whether to highlight cue ball differently

        Returns:
            Image with balls drawn
        """
        result = image.copy()

        for ball in balls:
            center = tuple(ball.center.astype(int))
            radius = int(ball.radius)

            # Draw ball circle with its detected color
            if ball.ball_type == 'cue' and highlight_cue:
                color = self.colors['cue_ball']
                cv2.circle(result, center, radius, color, self.ball_outline_thickness)
            else:
                # Draw filled circle with ball color
                cv2.circle(result, center, radius, ball.color, -1)
                # Draw outline
                cv2.circle(result, center, radius, self.colors['ball_outline'], 1)

            # Add label
            label = ball.ball_type[0].upper()
            cv2.putText(
                result,
                label,
                (center[0] - 5, center[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                self.colors['text'],
                1
            )

        return result

    def draw_cue_direction(
        self,
        image: np.ndarray,
        cue_ball_center: np.ndarray,
        direction: np.ndarray,
        length: int = 150
    ) -> np.ndarray:
        """Draw cue direction arrow.

        Args:
            image: Input image
            cue_ball_center: Cue ball center position
            direction: Direction vector (normalized)
            length: Arrow length in pixels

        Returns:
            Image with cue direction drawn
        """
        result = image.copy()

        start = tuple(cue_ball_center.astype(int))
        end = tuple((cue_ball_center + direction * length).astype(int))

        cv2.arrowedLine(
            result,
            start,
            end,
            self.colors['cue_direction'],
            self.line_thickness + 1,
            tipLength=0.2
        )

        return result

    def draw_trajectory(
        self,
        image: np.ndarray,
        trajectory: List[TrajectoryPoint],
        color: Optional[tuple] = None,
        draw_events: bool = True,
        dashed: bool = True
    ) -> np.ndarray:
        """Draw ball trajectory path.

        Args:
            image: Input image
            trajectory: List of trajectory points
            color: Line color (uses default if None)
            draw_events: Whether to mark collision/cushion events
            dashed: Whether to draw dashed line

        Returns:
            Image with trajectory drawn
        """
        if len(trajectory) < 2:
            return image

        result = image.copy()

        if color is None:
            color = self.colors['cue_trajectory']

        # Draw path between trajectory points
        for i in range(len(trajectory) - 1):
            pt1 = tuple(trajectory[i].position.astype(int))
            pt2 = tuple(trajectory[i + 1].position.astype(int))

            if dashed:
                self._draw_dashed_line(result, pt1, pt2, color, self.line_thickness)
            else:
                cv2.line(result, pt1, pt2, color, self.line_thickness, cv2.LINE_AA)

        # Draw event markers
        if draw_events:
            for point in trajectory:
                pos = tuple(point.position.astype(int))

                if point.event_type == 'collision':
                    cv2.circle(result, pos, self.point_radius, self.colors['collision_point'], -1)
                    cv2.circle(result, pos, self.point_radius + 2, (255, 255, 255), 1)
                elif point.event_type == 'cushion':
                    cv2.circle(result, pos, self.point_radius, self.colors['cushion_point'], -1)
                    cv2.circle(result, pos, self.point_radius + 2, (255, 255, 255), 1)

        return result

    def draw_all_trajectories(
        self,
        image: np.ndarray,
        cue_trajectory: List[TrajectoryPoint],
        other_trajectories: Dict[int, List[TrajectoryPoint]]
    ) -> np.ndarray:
        """Draw all ball trajectories.

        Args:
            image: Input image
            cue_trajectory: Cue ball trajectory
            other_trajectories: Dictionary of other ball trajectories

        Returns:
            Image with all trajectories drawn
        """
        result = image.copy()

        # Draw other ball trajectories first
        for ball_idx, trajectory in other_trajectories.items():
            result = self.draw_trajectory(
                result,
                trajectory,
                color=self.colors['ball_trajectory'],
                draw_events=True,
                dashed=True
            )

        # Draw cue ball trajectory on top
        result = self.draw_trajectory(
            result,
            cue_trajectory,
            color=self.colors['cue_trajectory'],
            draw_events=True,
            dashed=False
        )

        return result

    def draw_legend(
        self,
        image: np.ndarray,
        position: tuple = (10, 30)
    ) -> np.ndarray:
        """Draw legend explaining visualization elements.

        Args:
            image: Input image
            position: Top-left position for legend

        Returns:
            Image with legend drawn
        """
        result = image.copy()
        x, y = position

        # Background rectangle for legend
        legend_items = [
            ("Cue ball path", self.colors['cue_trajectory']),
            ("Other ball paths", self.colors['ball_trajectory']),
            ("Ball collision", self.colors['collision_point']),
            ("Cushion bounce", self.colors['cushion_point'])
        ]

        # Calculate legend size
        line_height = 25
        legend_height = len(legend_items) * line_height + 20
        legend_width = 220

        # Draw semi-transparent background
        overlay = result.copy()
        cv2.rectangle(
            overlay,
            (x - 5, y - 20),
            (x + legend_width, y + legend_height - 20),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.7, result, 0.3, 0, result)

        # Draw legend items
        for idx, (label, color) in enumerate(legend_items):
            y_pos = y + idx * line_height

            if "path" in label:
                # Draw line sample
                cv2.line(
                    result,
                    (x, y_pos),
                    (x + 30, y_pos),
                    color,
                    2,
                    cv2.LINE_AA
                )
                text_x = x + 40
            else:
                # Draw circle sample
                cv2.circle(result, (x + 15, y_pos), 5, color, -1)
                cv2.circle(result, (x + 15, y_pos), 7, (255, 255, 255), 1)
                text_x = x + 40

            cv2.putText(
                result,
                label,
                (text_x, y_pos + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.colors['text'],
                1,
                cv2.LINE_AA
            )

        return result

    def _draw_dashed_line(
        self,
        image: np.ndarray,
        pt1: tuple,
        pt2: tuple,
        color: tuple,
        thickness: int,
        dash_length: int = 10
    ):
        """Draw a dashed line between two points.

        Args:
            image: Image to draw on
            pt1: Start point (x, y)
            pt2: End point (x, y)
            color: Line color
            thickness: Line thickness
            dash_length: Length of each dash
        """
        dist = np.linalg.norm(np.array(pt2) - np.array(pt1))
        dashes = int(dist / dash_length)

        for i in range(0, dashes, 2):
            start = (
                int(pt1[0] + (pt2[0] - pt1[0]) * i / dashes),
                int(pt1[1] + (pt2[1] - pt1[1]) * i / dashes)
            )
            end = (
                int(pt1[0] + (pt2[0] - pt1[0]) * min(i + 1, dashes) / dashes),
                int(pt1[1] + (pt2[1] - pt1[1]) * min(i + 1, dashes) / dashes)
            )
            cv2.line(image, start, end, color, thickness, cv2.LINE_AA)

    def create_full_visualization(
        self,
        image: np.ndarray,
        balls: List[Ball],
        cue_direction: Optional[np.ndarray],
        cue_trajectory: List[TrajectoryPoint],
        other_trajectories: Dict[int, List[TrajectoryPoint]],
        show_legend: bool = True
    ) -> np.ndarray:
        """Create complete visualization with all elements.

        Args:
            image: Original input image
            balls: Detected balls
            cue_direction: Cue direction vector (if detected)
            cue_trajectory: Cue ball trajectory
            other_trajectories: Other ball trajectories
            show_legend: Whether to include legend

        Returns:
            Fully annotated image
        """
        result = image.copy()

        # Draw balls
        result = self.draw_balls(result, balls)

        # Draw cue direction
        cue_ball = next((b for b in balls if b.ball_type == 'cue'), None)
        if cue_ball is not None and cue_direction is not None:
            result = self.draw_cue_direction(result, cue_ball.center, cue_direction)

        # Draw trajectories
        result = self.draw_all_trajectories(result, cue_trajectory, other_trajectories)

        # Draw legend
        if show_legend:
            result = self.draw_legend(result)

        return result
