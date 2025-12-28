"""Geometry utility functions for vector math and intersections."""

import numpy as np
from typing import Tuple, Optional


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length.

    Args:
        vector: Input vector as numpy array

    Returns:
        Normalized vector
    """
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        return vector
    return vector / magnitude


def dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate dot product of two vectors.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Dot product
    """
    return np.dot(v1, v2)


def distance_point_to_line(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
    """Calculate perpendicular distance from point to line.

    Args:
        point: Point coordinates [x, y]
        line_start: Line start point [x, y]
        line_end: Line end point [x, y]

    Returns:
        Distance from point to line
    """
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len = np.linalg.norm(line_vec)

    if line_len == 0:
        return np.linalg.norm(point_vec)

    line_unit = line_vec / line_len
    proj_length = np.dot(point_vec, line_unit)
    proj_point = line_start + proj_length * line_unit

    return np.linalg.norm(point - proj_point)


def line_circle_intersection(
    line_start: np.ndarray,
    line_direction: np.ndarray,
    circle_center: np.ndarray,
    circle_radius: float
) -> Optional[Tuple[np.ndarray, float]]:
    """Find intersection of line ray with circle.

    Args:
        line_start: Starting point of ray [x, y]
        line_direction: Direction vector (should be normalized) [dx, dy]
        circle_center: Center of circle [x, y]
        circle_radius: Radius of circle

    Returns:
        Tuple of (intersection_point, distance) if intersection exists, None otherwise
    """
    # Vector from line start to circle center
    to_center = circle_center - line_start

    # Project onto line direction
    proj_length = np.dot(to_center, line_direction)

    # If projection is behind the ray, no forward intersection
    if proj_length < 0:
        return None

    # Find closest point on ray to circle center
    closest_point = line_start + proj_length * line_direction

    # Distance from circle center to closest point
    dist_to_center = np.linalg.norm(circle_center - closest_point)

    # Check if ray intersects circle
    if dist_to_center > circle_radius:
        return None

    # Calculate distance along ray to intersection
    chord_half = np.sqrt(circle_radius**2 - dist_to_center**2)
    intersection_dist = proj_length - chord_half

    # If intersection is behind start point
    if intersection_dist < 0:
        intersection_dist = proj_length + chord_half

    intersection_point = line_start + intersection_dist * line_direction

    return intersection_point, intersection_dist


def circles_intersect(
    center1: np.ndarray,
    radius1: float,
    center2: np.ndarray,
    radius2: float
) -> bool:
    """Check if two circles intersect.

    Args:
        center1: Center of first circle [x, y]
        radius1: Radius of first circle
        center2: Center of second circle [x, y]
        radius2: Radius of second circle

    Returns:
        True if circles intersect
    """
    distance = np.linalg.norm(center2 - center1)
    return distance < (radius1 + radius2)


def reflect_vector(incident: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Reflect a vector across a surface normal.

    Args:
        incident: Incident vector
        normal: Surface normal (should be normalized)

    Returns:
        Reflected vector
    """
    return incident - 2 * np.dot(incident, normal) * normal


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate angle between two vectors in radians.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Angle in radians
    """
    v1_norm = normalize_vector(v1)
    v2_norm = normalize_vector(v2)
    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    return np.arccos(cos_angle)


def point_in_rect(point: np.ndarray, rect_min: np.ndarray, rect_max: np.ndarray) -> bool:
    """Check if point is inside rectangle.

    Args:
        point: Point coordinates [x, y]
        rect_min: Rectangle minimum corner [x, y]
        rect_max: Rectangle maximum corner [x, y]

    Returns:
        True if point is inside rectangle
    """
    return (rect_min[0] <= point[0] <= rect_max[0] and
            rect_min[1] <= point[1] <= rect_max[1])
