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

# File reader for testing
NAME_FILE_READER = "file_reader"
PATH_FILE_READER = "config/config_entry-nordpool_optimizer.json"