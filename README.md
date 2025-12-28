# billiards-trajectory
This project aims to output a PDF with vector trajectories drawn on a user-provided PDF.
# GamePigeon 8 Ball Trajectory Predictor

A Python tool that analyzes GamePigeon 8 ball screenshots and predicts ball trajectories using computer vision and physics simulation.

## Features

- **Automatic Ball Detection**: Uses Hough Circle Transform to detect pool balls
- **Table Detection**: Identifies pool table boundaries using color segmentation
- **Cue Direction Detection**: Automatically detects the shot direction
- **Physics Simulation**: Calculates ball trajectories with collision detection
- **PDF Output**: Generates annotated PDF with predicted ball paths

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Navigate to the project directory:
```bash
cd gamepigeon-pool-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Process a GamePigeon screenshot and generate trajectory prediction:

```bash
python -m src.main <input_screenshot> <output_pdf>
```

Example:
```bash
python -m src.main examples/sample_screenshots/shot1.png output.pdf
```

### Advanced Options

**Custom Initial Velocity:**
```bash
python -m src.main input.png output.pdf --velocity 8.0
```

**Custom Configuration:**
```bash
python -m src.main input.png output.pdf --config my_config.yaml
```

**Debug Mode** (shows visualization window):
```bash
python -m src.main input.png output.pdf --debug
```

### Command-Line Arguments

- `input`: Path to GamePigeon screenshot (required)
- `output`: Path to output PDF file (required)
- `--config`: Path to custom configuration YAML file (optional)
- `--velocity`: Initial cue ball velocity in m/s (default: 5.0)
- `--debug`: Show debug visualization window before generating PDF

## Configuration

The detection and simulation parameters can be customized in `config/detection_params.yaml`:

### Table Detection
- `hsv_lower/upper`: HSV color range for green felt
- `min_area`: Minimum contour area for table

### Ball Detection
- `dp`: Hough Circle accumulator resolution
- `min_dist`: Minimum distance between ball centers
- `param1/param2`: Hough Circle sensitivity parameters
- `min_radius/max_radius`: Expected ball radius range

### Cue Detection
- `threshold`: Line detection sensitivity
- `min_line_length`: Minimum cue stick length
- `proximity_threshold`: Maximum distance from cue ball

### Physics Simulation
- `max_bounces`: Maximum cushion bounces to simulate
- `min_velocity`: Velocity threshold for stopping
- `initial_velocity`: Default cue ball speed

## How It Works

1. **Image Preprocessing**: Resizes and applies noise reduction
2. **Table Detection**: Identifies green felt area using HSV color filtering
3. **Ball Detection**:
   - Applies Hough Circle Transform to find circular objects
   - Classifies balls by color (cue ball, solids, stripes, 8-ball)
4. **Cue Detection**:
   - Attempts to detect GamePigeon's aiming line
   - Falls back to edge detection and Hough Line Transform
5. **Physics Simulation**:
   - Calculates linear trajectories from cue ball
   - Detects ball-to-ball and ball-to-cushion collisions
   - Applies elastic collision physics
6. **Visualization**:
   - Draws detected balls and cue direction
   - Renders predicted trajectories
   - Marks collision and cushion bounce points
7. **PDF Generation**: Creates PDF with annotated image and metadata

## Output

The generated PDF includes:

- **Page 1**: Annotated screenshot with:
  - Detected balls (outlined and labeled)
  - Cue direction arrow
  - Predicted trajectories (green for cue ball, yellow for other balls)
  - Collision points (red dots)
  - Cushion bounce points (magenta dots)
  - Legend explaining visualization elements
  - Metadata (balls detected, shot angle, predicted collisions)

- **Page 2** (optional): Original screenshot for comparison

## Limitations

- **No Spin Physics**: Does not simulate english, follow, or draw shots
- **Basic Friction**: Uses simplified friction model
- **Ball Occlusion**: Partially hidden balls may not be detected
- **Lighting Sensitivity**: Requires decent screenshot quality
- **Cue Detection**: May fail if cue stick is faint or off-screen

## Troubleshooting

### "Could not detect pool table"
- Adjust `hsv_lower` and `hsv_upper` in config file
- Ensure screenshot shows the full table with green felt visible
- Try increasing contrast of the screenshot

### "No balls detected"
- Adjust Hough Circle parameters (`param1`, `param2`, `min_dist`)
- Check that balls are clearly visible in screenshot
- Modify `min_radius` and `max_radius` to match ball size in pixels

### "Could not detect cue direction"
- Tool will use default direction (right)
- Try capturing screenshot with GamePigeon's aiming line visible
- Manually adjust detection parameters in config

### Poor trajectory predictions
- Increase `initial_velocity` if trajectories are too short
- Decrease if trajectories extend too far
- Adjust `COEFFICIENT_OF_RESTITUTION` in `src/physics/constants.py`

## Project Structure

```
gamepigeon-pool-predictor/
├── README.md
├── requirements.txt
├── config/
│   └── detection_params.yaml
├── src/
│   ├── main.py                   # CLI entry point
│   ├── detection/                # Computer vision modules
│   │   ├── table_detector.py
│   │   ├── ball_detector.py
│   │   └── cue_detector.py
│   ├── physics/                  # Physics simulation
│   │   ├── constants.py
│   │   ├── trajectory.py
│   │   └── collision.py
│   ├── rendering/                # Visualization
│   │   ├── visualizer.py
│   │   └── pdf_generator.py
│   └── utils/                    # Helper functions
│       ├── image_utils.py
│       └── geometry.py
└── examples/
    └── sample_screenshots/
```

## Technical Details

### Computer Vision
- **OpenCV** for image processing and feature detection
- **Hough Circle Transform** for robust ball detection
- **HSV color space** for table and ball classification
- **Canny edge detection** and **Hough Line Transform** for cue detection

### Physics Simulation
- **2D elastic collision** with momentum conservation
- **Coefficient of restitution** for energy loss
- **Ray-casting** for trajectory calculation
- **Event-based simulation** for accurate collision timing

### Libraries Used
- `opencv-python`: Computer vision and image processing
- `numpy`: Numerical computations and array operations
- `Pillow`: Image format conversion
- `reportlab`: PDF generation
- `PyYAML`: Configuration file parsing

## Future Enhancements

Potential improvements for future versions:

- [ ] Spin physics (english, follow, draw)
- [ ] Advanced friction modeling
- [ ] Pocket detection and sink predictions
- [ ] Multiple shot suggestions
- [ ] Web interface
- [ ] Real-time video analysis
- [ ] Machine learning for improved detection
- [ ] 3D visualization

## License

This project is for educational purposes. GamePigeon is a trademark of Vitalii Zlotskii.

## Contributing

Contributions welcome! Please feel free to submit issues and pull requests.

## Support

For bugs and feature requests, please open an issue on the project repository.

---

**Note**: This tool provides estimated trajectories based on simplified physics. Actual GamePigeon gameplay may differ due to game-specific physics, spin effects, and other factors not modeled in this basic simulation.
