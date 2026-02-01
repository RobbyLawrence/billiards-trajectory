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
