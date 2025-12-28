"""Collision detection and physics for pool balls."""

import numpy as np
from typing import Tuple, Optional
from ..utils.geometry import circles_intersect, normalize_vector, reflect_vector
from .constants import COEFFICIENT_OF_RESTITUTION


def detect_ball_collision(
    pos1: np.ndarray,
    radius1: float,
    pos2: np.ndarray,
    radius2: float
) -> bool:
    """Detect if two balls are colliding.

    Args:
        pos1: Position of first ball [x, y]
        radius1: Radius of first ball
        pos2: Position of second ball [x, y]
        radius2: Radius of second ball

    Returns:
        True if balls are colliding
    """
    return circles_intersect(pos1, radius1, pos2, radius2)


def calculate_collision_response(
    pos1: np.ndarray,
    vel1: np.ndarray,
    mass1: float,
    pos2: np.ndarray,
    vel2: np.ndarray,
    mass2: float,
    restitution: float = COEFFICIENT_OF_RESTITUTION
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate velocities after elastic collision between two balls.

    Uses 2D elastic collision physics with conservation of momentum.

    Args:
        pos1: Position of first ball [x, y]
        vel1: Velocity of first ball [vx, vy]
        mass1: Mass of first ball
        pos2: Position of second ball [x, y]
        vel2: Velocity of second ball [vx, vy]
        mass2: Mass of second ball
        restitution: Coefficient of restitution (0-1)

    Returns:
        Tuple of (new_vel1, new_vel2) after collision
    """
    # Vector from ball 1 to ball 2
    collision_normal = pos2 - pos1
    distance = np.linalg.norm(collision_normal)

    if distance == 0:
        # Balls are at same position, no collision response
        return vel1, vel2

    # Normalize collision normal
    collision_normal = collision_normal / distance

    # Relative velocity
    relative_vel = vel1 - vel2

    # Velocity along collision normal
    vel_along_normal = np.dot(relative_vel, collision_normal)

    # If balls are separating, no collision
    if vel_along_normal < 0:
        return vel1, vel2

    # Calculate impulse
    impulse = (-(1 + restitution) * vel_along_normal) / (1/mass1 + 1/mass2)

    # Apply impulse to velocities
    impulse_vector = impulse * collision_normal
    new_vel1 = vel1 + impulse_vector / mass1
    new_vel2 = vel2 - impulse_vector / mass2

    return new_vel1, new_vel2


def find_ball_collision_time(
    pos: np.ndarray,
    vel: np.ndarray,
    radius: float,
    other_pos: np.ndarray,
    other_radius: float,
    max_time: float = 10.0
) -> Optional[Tuple[float, np.ndarray]]:
    """Find time and position when moving ball will collide with stationary ball.

    Args:
        pos: Current position of moving ball [x, y]
        vel: Velocity of moving ball [vx, vy]
        radius: Radius of moving ball
        other_pos: Position of stationary ball [x, y]
        other_radius: Radius of stationary ball
        max_time: Maximum time to check

    Returns:
        Tuple of (collision_time, collision_position) if collision occurs, None otherwise
    """
    # Relative position
    rel_pos = pos - other_pos
    speed = np.linalg.norm(vel)

    if speed < 1e-6:
        return None

    # Direction of motion
    direction = vel / speed

    # Solve quadratic equation for collision time
    # |pos + vel*t - other_pos|^2 = (radius + other_radius)^2

    a = speed**2
    b = 2 * np.dot(rel_pos, vel)
    c = np.dot(rel_pos, rel_pos) - (radius + other_radius)**2

    discriminant = b**2 - 4*a*c

    if discriminant < 0:
        return None

    # Two solutions - we want the earlier one (smaller t)
    t1 = (-b - np.sqrt(discriminant)) / (2*a)
    t2 = (-b + np.sqrt(discriminant)) / (2*a)

    # Use the earlier positive time
    collision_time = min(t for t in [t1, t2] if t >= 0) if any(t >= 0 for t in [t1, t2]) else None

    if collision_time is None or collision_time > max_time:
        return None

    # Calculate collision position
    collision_pos = pos + vel * collision_time

    return collision_time, collision_pos


def reflect_off_cushion(
    velocity: np.ndarray,
    wall_normal: np.ndarray,
    restitution: float = COEFFICIENT_OF_RESTITUTION
) -> np.ndarray:
    """Calculate velocity after bouncing off cushion.

    Args:
        velocity: Incident velocity [vx, vy]
        wall_normal: Wall normal vector (should be normalized)
        restitution: Coefficient of restitution

    Returns:
        Reflected velocity
    """
    # Reflect velocity
    reflected = reflect_vector(velocity, wall_normal)

    # Apply energy loss
    return reflected * restitution


def find_cushion_collision(
    pos: np.ndarray,
    vel: np.ndarray,
    radius: float,
    table_bounds: dict,
    max_time: float = 10.0
) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
    """Find time and position when ball collides with table cushion.

    Args:
        pos: Current ball position [x, y]
        vel: Ball velocity [vx, vy]
        radius: Ball radius
        table_bounds: Dictionary with 'left', 'right', 'top', 'bottom' boundaries
        max_time: Maximum time to check

    Returns:
        Tuple of (collision_time, collision_position, wall_normal) if collision occurs
    """
    speed = np.linalg.norm(vel)
    if speed < 1e-6:
        return None

    min_time = max_time
    collision_wall = None

    # Check each wall
    walls = {
        'left': (table_bounds['left'] + radius, np.array([1.0, 0.0])),
        'right': (table_bounds['right'] - radius, np.array([-1.0, 0.0])),
        'top': (table_bounds['top'] + radius, np.array([0.0, 1.0])),
        'bottom': (table_bounds['bottom'] - radius, np.array([0.0, -1.0]))
    }

    # Left/Right walls (vertical)
    if vel[0] < 0:  # Moving left
        t = (walls['left'][0] - pos[0]) / vel[0]
        if 0 < t < min_time:
            min_time = t
            collision_wall = 'left'
    elif vel[0] > 0:  # Moving right
        t = (walls['right'][0] - pos[0]) / vel[0]
        if 0 < t < min_time:
            min_time = t
            collision_wall = 'right'

    # Top/Bottom walls (horizontal)
    if vel[1] < 0:  # Moving up
        t = (walls['top'][0] - pos[1]) / vel[1]
        if 0 < t < min_time:
            min_time = t
            collision_wall = 'top'
    elif vel[1] > 0:  # Moving down
        t = (walls['bottom'][0] - pos[1]) / vel[1]
        if 0 < t < min_time:
            min_time = t
            collision_wall = 'bottom'

    if collision_wall is None:
        return None

    collision_pos = pos + vel * min_time
    wall_normal = walls[collision_wall][1]

    return min_time, collision_pos, wall_normal
