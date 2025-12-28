"""Physical constants for pool ball simulation."""

# Ball properties
BALL_RADIUS_METERS = 0.0286  # Standard pool ball radius in meters (57.15mm diameter)
BALL_MASS_KG = 0.170  # Standard pool ball mass in kilograms

# Table properties (standard 8-foot table)
TABLE_WIDTH_METERS = 1.118  # ~44 inches
TABLE_LENGTH_METERS = 2.237  # ~88 inches

# Pocket properties
POCKET_RADIUS_METERS = 0.055  # Approximate pocket opening radius

# Physical constants
COEFFICIENT_OF_RESTITUTION = 0.95  # Energy retention on collision (0-1)
COEFFICIENT_OF_FRICTION = 0.2  # Sliding friction coefficient
ROLLING_RESISTANCE = 0.01  # Rolling resistance coefficient

# Simulation parameters
MAX_BOUNCES = 10  # Maximum cushion bounces to simulate
MIN_VELOCITY = 0.1  # Minimum velocity before ball is considered stopped (m/s)
TIME_STEP = 0.01  # Simulation time step in seconds
DEFAULT_CUE_VELOCITY = 5.0  # Default initial cue ball velocity (m/s)

# Pocket positions (normalized coordinates 0-1)
# Assuming standard 6-pocket table layout
POCKET_POSITIONS = [
    (0.0, 0.0),  # Top-left corner
    (0.5, 0.0),  # Top-middle
    (1.0, 0.0),  # Top-right corner
    (0.0, 1.0),  # Bottom-left corner
    (0.5, 1.0),  # Bottom-middle
    (1.0, 1.0),  # Bottom-right corner
]
