"""Main entry point for GamePigeon Pool Predictor."""

import argparse
import cv2
import yaml
import sys
from pathlib import Path
from typing import Optional

from detection.table_detector import TableDetector
from detection.ball_detector import BallDetector
from detection.cue_detector import CueDetector
from physics.trajectory import TrajectorySimulator
from physics.constants import DEFAULT_CUE_VELOCITY
from rendering.visualizer import TrajectoryVisualizer
from rendering.pdf_generator import PDFGenerator
from utils.image_utils import resize_image, preprocess_image


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file (uses default if None)

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Use default config
        config_path = Path(__file__).parent.parent / "config" / "detection_params.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def process_screenshot(
    image_path: str,
    output_path: str,
    config: dict,
    velocity: Optional[float] = None,
    show_debug: bool = False
) -> bool:
    """Process GamePigeon screenshot and generate trajectory prediction.

    Args:
        image_path: Path to input screenshot
        output_path: Path to output PDF
        config: Configuration dictionary
        velocity: Optional custom initial velocity
        show_debug: Whether to show debug visualizations

    Returns:
        True if successful, False otherwise
    """
    # Load image
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return False

    # Preprocess image
    print("Preprocessing image...")
    image = resize_image(
        image,
        config['preprocessing']['max_width'],
        config['preprocessing']['max_height']
    )
    image = preprocess_image(image, config['preprocessing']['gaussian_blur_kernel'])

    # Detect table
    print("Detecting pool table...")
    table_detector = TableDetector(
        hsv_lower=tuple(config['table']['hsv_lower']),
        hsv_upper=tuple(config['table']['hsv_upper']),
        min_area=config['table']['min_area']
    )

    table_result = table_detector.detect(image)

    if table_result is None:
        print("Error: Could not detect pool table in image")
        print("Try adjusting the HSV color range in config/detection_params.yaml")
        return False

    table_bbox, table_mask = table_result
    table_bounds = table_detector.get_table_boundaries(table_bbox)
    print(f"Table detected: {table_bounds['width']}x{table_bounds['height']} pixels")

    # Detect balls
    print("Detecting balls...")
    ball_detector = BallDetector(
        dp=config['balls']['dp'],
        min_dist=config['balls']['min_dist'],
        param1=config['balls']['param1'],
        param2=config['balls']['param2'],
        min_radius=config['balls']['min_radius'],
        max_radius=config['balls']['max_radius'],
        cue_ball_threshold=config['balls']['cue_ball_value_threshold'],
        black_ball_threshold=config['balls']['black_ball_value_threshold']
    )

    balls = ball_detector.detect(image, table_mask)
    print(f"Detected {len(balls)} balls")

    if len(balls) == 0:
        print("Error: No balls detected")
        print("Try adjusting the Hough Circle parameters in config/detection_params.yaml")
        return False

    # Find cue ball
    cue_ball = ball_detector.get_cue_ball(balls)

    if cue_ball is None:
        print("Error: Could not detect cue ball")
        print("Using first detected ball as cue ball")
        cue_ball = balls[0]
        cue_ball.ball_type = 'cue'

    print(f"Cue ball at position: ({cue_ball.center[0]:.1f}, {cue_ball.center[1]:.1f})")

    # Detect cue direction
    print("Detecting cue direction...")
    cue_detector = CueDetector(
        rho=config['cue']['rho'],
        theta=config['cue']['theta'],
        threshold=config['cue']['threshold'],
        min_line_length=config['cue']['min_line_length'],
        max_line_gap=config['cue']['max_line_gap'],
        proximity_threshold=config['cue']['proximity_threshold']
    )

    # Try to detect cue from aiming line first
    cue_result = cue_detector.detect_from_aiming_line(image, cue_ball.center)

    # Fallback to edge-based detection
    if cue_result is None:
        cue_result = cue_detector.detect(image, cue_ball.center)

    if cue_result is None:
        print("Warning: Could not detect cue direction automatically")
        print("Using default direction (right)")
        import numpy as np
        cue_direction = np.array([1.0, 0.0])
        cue_angle = 0.0
    else:
        cue_direction, cue_angle = cue_result
        print(f"Cue direction detected: {cue_angle:.1f}°")

    # Simulate trajectory
    print("Simulating ball trajectories...")
    simulator = TrajectorySimulator(
        table_bounds=table_bounds,
        ball_radius=cue_ball.radius,
        max_bounces=config['simulation']['max_bounces'],
        min_velocity=config['simulation']['min_velocity'],
        time_step=config['simulation']['time_step']
    )

    # Use custom velocity or default
    initial_velocity = velocity if velocity is not None else config['simulation']['initial_velocity']

    # Get other balls (excluding cue ball)
    other_balls = [b for b in balls if b != cue_ball]

    cue_trajectory, other_trajectories = simulator.simulate_cue_shot(
        cue_ball,
        cue_direction,
        initial_velocity,
        other_balls
    )

    print(f"Simulation complete: {len(cue_trajectory)} trajectory points")
    print(f"Predicted {len(other_trajectories)} ball-to-ball collisions")

    # Visualize
    print("Creating visualization...")
    visualizer = TrajectoryVisualizer()

    annotated_image = visualizer.create_full_visualization(
        image=image,
        balls=balls,
        cue_direction=cue_direction,
        cue_trajectory=cue_trajectory,
        other_trajectories=other_trajectories,
        show_legend=True
    )

    # Show debug window if requested
    if show_debug:
        cv2.imshow("Trajectory Prediction", annotated_image)
        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Generate PDF
    print(f"Generating PDF: {output_path}")
    pdf_generator = PDFGenerator()

    metadata = {
        'balls_detected': len(balls),
        'cue_angle': cue_angle,
        'collisions': len(other_trajectories)
    }

    pdf_generator.create_pdf(
        output_path=output_path,
        annotated_image=annotated_image,
        original_image=image,
        title="GamePigeon 8 Ball - Trajectory Prediction",
        metadata=metadata
    )

    print(f"Success! Output saved to: {output_path}")
    return True


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GamePigeon 8 Ball Trajectory Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main input.png output.pdf
  python -m src.main screenshot.png result.pdf --velocity 8.0
  python -m src.main image.png output.pdf --config custom_config.yaml --debug
        """
    )

    parser.add_argument(
        'input',
        type=str,
        help='Path to input GamePigeon screenshot'
    )

    parser.add_argument(
        'output',
        type=str,
        help='Path to output PDF file'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to custom configuration YAML file'
    )

    parser.add_argument(
        '--velocity',
        type=float,
        default=None,
        help='Initial cue ball velocity (default: 5.0 m/s)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Show debug visualization window'
    )

    args = parser.parse_args()

    # Validate input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Process screenshot
    success = process_screenshot(
        image_path=args.input,
        output_path=args.output,
        config=config,
        velocity=args.velocity,
        show_debug=args.debug
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
