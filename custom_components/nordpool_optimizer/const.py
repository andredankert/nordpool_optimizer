"""Common constants for integration."""

from enum import Enum

DOMAIN = "nordpool_optimizer"


class OptimizerStates(Enum):
    """Standard numeric identifiers for optimizer states."""

    Ok = 0
    Idle = 1
    Warning = 2
    Error = 3
    Unknown = 4


# Optimization modes
CONF_MODE = "mode"
CONF_MODE_ABSOLUTE = "absolute"
CONF_MODE_DAILY = "daily"
CONF_MODE_LIST = [CONF_MODE_ABSOLUTE, CONF_MODE_DAILY]

# Device configuration
CONF_DEVICE_NAME = "device_name"
CONF_PRICES_ENTITY = "prices_entity"
CONF_DURATION = "duration"

# Mode-specific configuration
CONF_PRICE_THRESHOLD = "price_threshold"  # For absolute mode
CONF_SLOT_TYPE = "slot_type"  # For daily mode
CONF_SLOT_TYPE_CONSECUTIVE = "consecutive"
CONF_SLOT_TYPE_SEPARATE = "separate"
CONF_SLOT_TYPE_LIST = [CONF_SLOT_TYPE_CONSECUTIVE, CONF_SLOT_TYPE_SEPARATE]

# Optional time window
CONF_TIME_WINDOW = "time_window"
CONF_TIME_WINDOW_ENABLED = "time_window_enabled"

# Price graph entity configuration
CONF_ENABLE_PRICE_GRAPH = "enable_price_graph"
CONF_GRAPH_HOURS_AHEAD = "graph_hours_ahead"
GRAPH_HOURS_OPTIONS = [12, 24, 48]
DEFAULT_GRAPH_HOURS = 24

# Device colors for multi-device visualization
DEVICE_COLORS = [
    "#4CAF50",  # Green
    "#2196F3",  # Blue
    "#FF9800",  # Orange
    "#9C27B0",  # Purple
    "#F44336",  # Red
    "#00BCD4",  # Cyan
    "#795548",  # Brown
    "#607D8B",  # Blue Grey
    "#E91E63",  # Pink
    "#FFC107",  # Amber
]

# File reader for testing
NAME_FILE_READER = "file_reader"
PATH_FILE_READER = "config/config_entry-nordpool_optimizer.json"