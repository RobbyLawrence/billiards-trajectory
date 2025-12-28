"""Trajectory calculation and simulation for pool balls."""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from .collision import (
    find_ball_collision_time,
    find_cushion_collision,
    calculate_collision_response,
    reflect_off_cushion
)
from .constants import (
    BALL_MASS_KG,
    MIN_VELOCITY,
    MAX_BOUNCES,
    TIME_STEP,
    DEFAULT_CUE_VELOCITY
)
from ..detection.ball_detector import Ball


@dataclass
class TrajectoryPoint:
    """Represents a point along a ball's trajectory."""
    position: np.ndarray
    velocity: np.ndarray
    time: float
    event_type: str  # 'start', 'collision', 'cushion', 'stop', 'pocket'


class TrajectorySimulator:
    """Simulates ball trajectories with collisions."""

    def __init__(
        self,
        table_bounds: dict,
        ball_radius: float,
        max_bounces: int = MAX_BOUNCES,
        min_velocity: float = MIN_VELOCITY,
        time_step: float = TIME_STEP
    ):
        """Initialize trajectory simulator.

        Args:
            table_bounds: Dictionary with table boundaries
            ball_radius: Radius of balls in pixels
            max_bounces: Maximum cushion bounces to simulate
            min_velocity: Minimum velocity threshold
            time_step: Simulation time step
        """
        self.table_bounds = table_bounds
        self.ball_radius = ball_radius
        self.max_bounces = max_bounces
        self.min_velocity = min_velocity
        self.time_step = time_step

    def simulate_cue_shot(
        self,
        cue_ball: Ball,
        direction: np.ndarray,
        initial_velocity: float,
        other_balls: List[Ball]
    ) -> Tuple[List[TrajectoryPoint], Dict[int, List[TrajectoryPoint]]]:
        """Simulate a cue ball shot and resulting trajectories.

        Args:
            cue_ball: The cue ball
            direction: Shot direction (normalized vector)
            initial_velocity: Initial velocity magnitude
            other_balls: List of other balls on table

        Returns:
            Tuple of:
                - List of trajectory points for cue ball
                - Dictionary mapping ball index to trajectory points for other balls
        """
        # Initialize cue ball trajectory
        cue_ball_trajectory = [TrajectoryPoint(
            position=cue_ball.center.copy(),
            velocity=direction * initial_velocity,
            time=0.0,
            event_type='start'
        )]

        # Track moving balls: {ball_index: (ball, current_velocity)}
        moving_balls = {-1: (cue_ball, direction * initial_velocity)}
        other_ball_trajectories = {}

        current_time = 0.0
        bounce_count = 0
        max_simulation_time = 30.0  # Maximum 30 seconds

        while current_time < max_simulation_time and bounce_count < self.max_bounces:
            # Find next collision event
            next_event = self._find_next_event(moving_balls, other_balls)

            if next_event is None:
                # No more events, balls have stopped
                break

            event_time, event_type, event_data = next_event
            current_time += event_time

            # Update positions of all moving balls
            for ball_idx, (ball, velocity) in list(moving_balls.items()):
                new_pos = ball.center + velocity * event_time

                # Update ball position
                if ball_idx == -1:  # Cue ball
                    cue_ball.center = new_pos
                else:
                    other_balls[ball_idx].center = new_pos

            # Handle event
            if event_type == 'cushion':
                ball_idx, wall_normal = event_data
                ball, velocity = moving_balls[ball_idx]

                # Reflect velocity
                new_velocity = reflect_off_cushion(velocity, wall_normal)
                moving_balls[ball_idx] = (ball, new_velocity)

                # Add trajectory point
                if ball_idx == -1:
                    cue_ball_trajectory.append(TrajectoryPoint(
                        position=ball.center.copy(),
                        velocity=new_velocity,
                        time=current_time,
                        event_type='cushion'
                    ))

                bounce_count += 1

            elif event_type == 'ball_collision':
                ball1_idx, ball2_idx = event_data

                # Get balls and velocities
                ball1, vel1 = moving_balls.get(ball1_idx, (
                    cue_ball if ball1_idx == -1 else other_balls[ball1_idx],
                    np.array([0.0, 0.0])
                ))
                ball2, vel2 = moving_balls.get(ball2_idx, (
                    other_balls[ball2_idx],
                    np.array([0.0, 0.0])
                ))

                # Calculate collision response
                new_vel1, new_vel2 = calculate_collision_response(
                    ball1.center, vel1, BALL_MASS_KG,
                    ball2.center, vel2, BALL_MASS_KG
                )

                # Update velocities
                if np.linalg.norm(new_vel1) > self.min_velocity:
                    moving_balls[ball1_idx] = (ball1, new_vel1)
                else:
                    moving_balls.pop(ball1_idx, None)

                if np.linalg.norm(new_vel2) > self.min_velocity:
                    moving_balls[ball2_idx] = (ball2, new_vel2)
                    # Start tracking this ball's trajectory
                    if ball2_idx not in other_ball_trajectories:
                        other_ball_trajectories[ball2_idx] = [TrajectoryPoint(
                            position=ball2.center.copy(),
                            velocity=new_vel2,
                            time=current_time,
                            event_type='collision'
                        )]
                else:
                    moving_balls.pop(ball2_idx, None)

                # Add trajectory point for cue ball
                if ball1_idx == -1:
                    cue_ball_trajectory.append(TrajectoryPoint(
                        position=ball1.center.copy(),
                        velocity=new_vel1,
                        time=current_time,
                        event_type='collision'
                    ))

            # Remove stopped balls
            for ball_idx in list(moving_balls.keys()):
                ball, velocity = moving_balls[ball_idx]
                if np.linalg.norm(velocity) < self.min_velocity:
                    moving_balls.pop(ball_idx)

            # Break if no more moving balls
            if not moving_balls:
                break

        # Add final stop points
        for ball_idx, (ball, velocity) in moving_balls.items():
            if ball_idx == -1:
                cue_ball_trajectory.append(TrajectoryPoint(
                    position=ball.center.copy(),
                    velocity=np.array([0.0, 0.0]),
                    time=current_time,
                    event_type='stop'
                ))

        return cue_ball_trajectory, other_ball_trajectories

    def _find_next_event(
        self,
        moving_balls: Dict[int, Tuple[Ball, np.ndarray]],
        all_balls: List[Ball]
    ) -> Optional[Tuple[float, str, any]]:
        """Find the next collision or cushion event.

        Returns:
            Tuple of (time_to_event, event_type, event_data) or None
        """
        min_time = float('inf')
        next_event = None

        # Check each moving ball
        for ball_idx, (ball, velocity) in moving_balls.items():
            # Check cushion collisions
            cushion_result = find_cushion_collision(
                ball.center, velocity, self.ball_radius, self.table_bounds
            )

            if cushion_result is not None:
                t, pos, normal = cushion_result
                if t < min_time:
                    min_time = t
                    next_event = (t, 'cushion', (ball_idx, normal))

            # Check ball-to-ball collisions
            for other_idx, other_ball in enumerate(all_balls):
                # Skip if it's the same ball
                if ball_idx == other_idx:
                    continue

                # Skip if other ball is also moving (will be checked separately)
                if other_idx in moving_balls:
                    continue

                collision_result = find_ball_collision_time(
                    ball.center, velocity, self.ball_radius,
                    other_ball.center, self.ball_radius
                )

                if collision_result is not None:
                    t, pos = collision_result
                    if t < min_time:
                        min_time = t
                        next_event = (t, 'ball_collision', (ball_idx, other_idx))

        return next_event if next_event is not None else None

    def get_path_points(self, trajectory: List[TrajectoryPoint], num_points: int = 50) -> np.ndarray:
        """Get smoothed path points from trajectory for visualization.

        Args:
            trajectory: List of trajectory points
            num_points: Number of points to generate

        Returns:
            Nx2 array of path points
        """
        if len(trajectory) < 2:
            return np.array([trajectory[0].position]) if trajectory else np.array([])

        # Extract positions
        positions = np.array([tp.position for tp in trajectory])

        # Interpolate between positions
        t = np.linspace(0, len(positions) - 1, num_points)
        x = np.interp(t, np.arange(len(positions)), positions[:, 0])
        y = np.interp(t, np.arange(len(positions)), positions[:, 1])

        return np.column_stack([x, y])
