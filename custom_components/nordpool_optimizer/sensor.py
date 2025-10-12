"""Timer entity implementation for Nordpool Optimizer."""

from __future__ import annotations

import datetime as dt
import logging
import time
from typing import Dict, Set

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN, UnitOfTime
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from . import NordpoolOptimizer, NordpoolOptimizerEntity
from .const import (
    CONF_GRAPH_HOURS_AHEAD,
    DEFAULT_GRAPH_HOURS,
    DEVICE_COLORS,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

# Global setup state tracking to prevent infinite loops
_setup_in_progress: Set[str] = set()
_setup_retry_counts: Dict[str, int] = {}
_setup_last_attempt: Dict[str, float] = {}
_setup_locks_created: Dict[str, float] = {}

# Configuration constants
MAX_SETUP_RETRIES = 3
SETUP_RETRY_DELAY = 30.0  # seconds - extended cooldown to prevent infinite loops
SETUP_TIMEOUT = 60.0  # seconds - increased to accommodate longer delays


def _get_setup_key(config_entry: ConfigEntry) -> str:
    """Get a unique key for tracking setup state."""
    return f"{DOMAIN}_{config_entry.entry_id}"


def _cleanup_stale_setup_locks() -> None:
    """Clean up setup locks that have timed out."""
    current_time = time.time()
    stale_keys = []

    for key, created_time in _setup_locks_created.items():
        if current_time - created_time > SETUP_TIMEOUT:
            stale_keys.append(key)

    for key in stale_keys:
        _LOGGER.warning("Cleaning up stale setup lock for %s (timeout after %.1fs)",
                       key, SETUP_TIMEOUT)
        _setup_in_progress.discard(key)
        _setup_retry_counts.pop(key, None)
        _setup_last_attempt.pop(key, None)
        _setup_locks_created.pop(key, None)


def _can_start_setup(setup_key: str) -> bool:
    """Check if setup can be started based on retry logic."""
    current_time = time.time()

    # Clean up any stale locks first
    _cleanup_stale_setup_locks()

    # Check if setup is already in progress
    if setup_key in _setup_in_progress:
        _LOGGER.debug("Setup already in progress for %s", setup_key)
        return False

    # Check retry count
    retry_count = _setup_retry_counts.get(setup_key, 0)
    if retry_count >= MAX_SETUP_RETRIES:
        _LOGGER.warning("Maximum setup retries (%d) exceeded for %s",
                       MAX_SETUP_RETRIES, setup_key)
        return False

    # Check retry delay
    last_attempt = _setup_last_attempt.get(setup_key, 0)
    if current_time - last_attempt < SETUP_RETRY_DELAY:
        time_left = SETUP_RETRY_DELAY - (current_time - last_attempt)
        _LOGGER.debug("Setup retry delay active for %s (%.1fs remaining)",
                     setup_key, time_left)
        return False

    return True


def _start_setup(setup_key: str) -> None:
    """Mark setup as started."""
    current_time = time.time()
    _setup_in_progress.add(setup_key)
    _setup_last_attempt[setup_key] = current_time
    _setup_locks_created[setup_key] = current_time
    retry_count = _setup_retry_counts.get(setup_key, 0)
    _setup_retry_counts[setup_key] = retry_count + 1
    _LOGGER.debug("Started setup for %s (attempt %d/%d)",
                 setup_key, retry_count + 1, MAX_SETUP_RETRIES)


def _finish_setup(setup_key: str, success: bool = True) -> None:
    """Mark setup as finished."""
    _setup_in_progress.discard(setup_key)
    _setup_locks_created.pop(setup_key, None)

    if success:
        # Clear retry count on success but keep last attempt timestamp for cooldown
        _setup_retry_counts.pop(setup_key, None)
        # Keep last_attempt timestamp to enforce cooldown period even after success
        _setup_last_attempt[setup_key] = time.time()
        _LOGGER.debug("Setup completed successfully for %s (30s cooldown active)", setup_key)
    else:
        _LOGGER.debug("Setup failed for %s (will retry if attempts remain)", setup_key)


async def async_setup_entry(
    hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities
):
    """Create timer entity for platform."""
    setup_key = _get_setup_key(config_entry)

    # Check if we can start setup (prevents infinite loops)
    if not _can_start_setup(setup_key):
        _LOGGER.debug("Skipping setup for %s (already in progress or rate limited)", setup_key)
        return False

    _start_setup(setup_key)

    try:
        optimizer: NordpoolOptimizer = hass.data[DOMAIN][config_entry.entry_id]
        entities = []

        # Always create timer entity for this device
        entities.append(
            NordpoolOptimizerTimerEntity(
                optimizer,
                entity_description=SensorEntityDescription(
                    key="timer",
                    # Remove device class to allow custom formatting
                ),
            )
        )

        # Create price graph entity only once (for the first optimizer setup)
        # Check if price graph entity already exists
        all_entity_ids = hass.states.async_entity_ids("sensor")
        existing_entity_ids = [entity_id for entity_id in all_entity_ids
                              if entity_id.startswith("sensor.nordpool_optimizer_price_graph")]

        # Also check if any domain data already has a graph entity flag
        domain_data = hass.data.get(DOMAIN, {})
        graph_entity_exists = any(
            hasattr(opt, '_has_graph_entity') and opt._has_graph_entity
            for opt in domain_data.values()
            if hasattr(opt, '_has_graph_entity')
        )

        if not existing_entity_ids and not graph_entity_exists:
            # Create the global price graph entity with configured hours
            graph_hours = config_entry.data.get(CONF_GRAPH_HOURS_AHEAD, DEFAULT_GRAPH_HOURS)
            graph_entity = NordpoolOptimizerPriceGraphEntity(hass, graph_hours)
            entities.append(graph_entity)

            # Mark that we've created the graph entity
            optimizer._has_graph_entity = True
            _LOGGER.debug("Created price graph entity for nordpool optimizer")
        else:
            # Graph entity already exists, but skip discovery update during setup to prevent loops
            # The auto-registration will happen naturally through the price graph entity's update cycle
            _LOGGER.debug("Price graph entity already exists for %s, auto-registration will occur on next update", setup_key)

        async_add_entities(entities)
        _finish_setup(setup_key, success=True)
        return True

    except Exception as e:
        _LOGGER.exception("Failed to setup entities for %s: %s", setup_key, e)
        _finish_setup(setup_key, success=False)
        return False


class NordpoolOptimizerTimerEntity(NordpoolOptimizerEntity, SensorEntity):
    """Main timer entity showing countdown/runtime with numeric state."""

    def __init__(
        self,
        optimizer: NordpoolOptimizer,
        entity_description: SensorEntityDescription,
    ) -> None:
        """Initialize the timer entity."""
        super().__init__(optimizer)
        self.entity_description = entity_description
        self._attr_name = optimizer.device_name
        self._attr_unique_id = (
            f"nordpool_optimizer_{optimizer.device_name}"
            .lower()
            .replace(" ", "_")
            .replace(".", "")
        )

    @property
    def native_value(self) -> int | None:
        """Return the numeric state in minutes."""
        if not self._optimizer.is_valid:
            return None

        now = dt_util.now()

        # Check if we're in an active optimization period
        if self._optimizer.is_currently_optimal(now):
            # Return positive minutes remaining in optimization period
            end_time = self._optimizer.get_current_period_end(now)
            if end_time:
                remaining = end_time - now
                minutes = max(0, int(remaining.total_seconds() / 60))
                _LOGGER.debug("Current optimal period, remaining: %s seconds = %s minutes",
                             remaining.total_seconds(), minutes)
                return minutes
            return 0
        else:
            # Return negative minutes until next optimization period
            next_start = self._optimizer.get_next_optimal_start(now)
            if next_start:
                countdown = next_start - now
                minutes = -max(0, int(countdown.total_seconds() / 60))
                _LOGGER.debug("Next optimal start in: %s seconds = %s minutes",
                             countdown.total_seconds(), minutes)
                return minutes
            return 0

    @property
    def state(self) -> str:
        """Return formatted timer display."""
        value = self.native_value
        if value is None:
            return STATE_UNAVAILABLE

        if value == 0:
            if self._optimizer.is_currently_optimal(dt_util.now()):
                return "Starting"
            else:
                return "Off"

        # Convert minutes to hours:minutes format
        abs_value = abs(value)
        hours = abs_value // 60
        minutes = abs_value % 60

        if value > 0:
            # Positive = runtime remaining
            return f"{hours:02d}h{minutes:02d}m"
        else:
            # Negative = countdown to start
            return f"-{hours:02d}h{minutes:02d}m"

    @property
    def unit_of_measurement(self) -> str | None:
        """Return None to prevent unit display since we have custom formatting."""
        return None

    @property
    def icon(self) -> str:
        """Return dynamic icon based on state."""
        value = self.native_value
        if value is None:
            return "mdi:timer-off"
        elif value > 0:
            return "mdi:timer"  # Running
        elif value < 0:
            return "mdi:timer-outline"  # Countdown
        else:
            return "mdi:timer-off"  # Off/starting

    @property
    def extra_state_attributes(self) -> dict:
        """Return additional state attributes."""
        now = dt_util.now()
        next_start = self._optimizer.get_next_optimal_start(now)
        current_end = self._optimizer.get_current_period_end(now)

        attributes = {
            "device_name": self._optimizer.device_name,
            "mode": self._optimizer.mode,
            "duration_hours": self._optimizer.duration,
            "currently_optimal": self._optimizer.is_currently_optimal(now),
            "price_sensor": self._optimizer.price_sensor_id,
            "minutes_value": self.native_value,  # For automation use
        }

        # Only add datetime attributes if they are valid datetime objects
        if next_start and isinstance(next_start, dt.datetime):
            attributes["next_optimal_start"] = next_start
        if current_end and isinstance(current_end, dt.datetime):
            attributes["current_period_end"] = current_end

        # Add mode-specific attributes
        if hasattr(self._optimizer, 'price_threshold'):
            attributes["price_threshold"] = self._optimizer.price_threshold
        if hasattr(self._optimizer, 'slot_type'):
            attributes["slot_type"] = self._optimizer.slot_type
        if hasattr(self._optimizer, 'time_window'):
            attributes["time_window"] = self._optimizer.time_window

        return attributes

    async def async_added_to_hass(self) -> None:
        """Register with optimizer for updates."""
        await super().async_added_to_hass()
        self._optimizer.register_output_listener_entity(self, "timer")


class NordpoolOptimizerPriceGraphEntity(SensorEntity):
    """Price graph entity showing future prices with optimal periods for all devices."""

    def __init__(self, hass: HomeAssistant, hours_ahead: int = DEFAULT_GRAPH_HOURS) -> None:
        """Initialize the price graph entity."""
        self._hass = hass
        self._hours_ahead = hours_ahead
        self._attr_name = "Nordpool Price Graph"
        self._attr_unique_id = "nordpool_optimizer_price_graph"
        self._attr_icon = "mdi:chart-line"
        self._attr_device_class = SensorDeviceClass.MONETARY
        self._attr_native_unit_of_measurement = None  # Will be set from price data

    @property
    def native_value(self) -> float | None:
        """Return current price as sensor state."""
        # Get current price from any available optimizer
        optimizers = self._get_all_optimizers()
        if not optimizers:
            return None

        for optimizer in optimizers:
            if optimizer._prices_entity.valid:
                current_price = optimizer._prices_entity.current_price_attr
                if current_price is not None:
                    return current_price

        return None

    @property
    def native_unit_of_measurement(self) -> str | None:
        """Return unit of measurement from price data."""
        optimizers = self._get_all_optimizers()
        if not optimizers:
            return None

        for optimizer in optimizers:
            if optimizer._prices_entity.valid and optimizer._prices_entity._np:
                # Try to get unit from price entity attributes
                return optimizer._prices_entity._np.attributes.get("unit", "kr/kWh")

        return "kr/kWh"  # Default fallback

    @property
    def state(self) -> str | None:
        """Return formatted current price."""
        value = self.native_value
        if value is None:
            return STATE_UNAVAILABLE
        return f"{value:.3f}"

    @property
    def extra_state_attributes(self) -> dict:
        """Return price graph data for chart visualization."""
        optimizers = self._get_all_optimizers()
        if not optimizers:
            return {}

        # Get price data from any optimizer with valid data
        price_entity = None
        for optimizer in optimizers:
            if optimizer._prices_entity.valid:
                price_entity = optimizer._prices_entity
                break

        if not price_entity:
            return {"error": "No valid price data available"}

        # Build price array with optimal period flags
        now = dt_util.now()

        # Smart time range selection based on actual data availability
        all_prices = price_entity._all_prices
        if not all_prices:
            # Fallback if no price data
            start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = start_time + dt.timedelta(days=1) - dt.timedelta(microseconds=1)
        else:
            # Check if we have sufficient tomorrow's data (at least 12 hours coverage)
            tomorrow_start = now.replace(hour=0, minute=0, second=0, microsecond=0) + dt.timedelta(days=1)
            tomorrow_end = tomorrow_start + dt.timedelta(days=1)

            # Count hours of tomorrow data available
            tomorrow_hours = set()
            for price_data in all_prices:
                price_time = price_data["start"]
                if not isinstance(price_time, dt.datetime):
                    price_time = dt_util.parse_datetime(price_time)

                if price_time and tomorrow_start <= price_time < tomorrow_end:
                    # Add the hour to our set
                    tomorrow_hours.add(price_time.hour)

            # Require at least 12 hours of tomorrow data for sufficient coverage
            has_sufficient_tomorrow_data = len(tomorrow_hours) >= 12
            _LOGGER.debug("Tomorrow data check: %d hours available, sufficient: %s",
                         len(tomorrow_hours), has_sufficient_tomorrow_data)

            if has_sufficient_tomorrow_data:
                # Sufficient tomorrow data available: show today 00:00 to tomorrow 23:59
                start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
                end_time = start_time + dt.timedelta(days=2) - dt.timedelta(microseconds=1)
            else:
                # Insufficient tomorrow data: show yesterday 00:00 to today 23:59
                today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                start_time = today_start - dt.timedelta(days=1)
                end_time = today_start + dt.timedelta(days=1) - dt.timedelta(microseconds=1)
        prices_ahead = []
        optimal_periods = []

        # Get all prices within time range
        all_prices = price_entity._all_prices
        if not all_prices:
            return {"error": "No price data available"}

        # Build unique price data with optimal device info
        price_map = {}  # Use dict to avoid duplicates by timestamp

        for price_data in all_prices:
            price_time = price_data["start"]
            if isinstance(price_time, str):
                price_time = dt_util.parse_datetime(price_time)

            if not price_time:
                continue

            if start_time <= price_time < end_time:
                time_key = price_time.isoformat()

                # Only process each timestamp once
                if time_key not in price_map:
                    # Check which devices are optimal at this time
                    optimal_devices = []
                    for optimizer in optimizers:
                        if optimizer.is_currently_optimal(price_time):
                            optimal_devices.append(optimizer.device_name)

                    price_map[time_key] = {
                        "time": time_key,
                        "price": price_data["value"],
                        "optimal_devices": optimal_devices
                    }

        # Convert map back to list, sorted by time
        prices_ahead = sorted(price_map.values(), key=lambda x: x["time"])

        # Collect optimal periods from all devices with row-based positioning
        device_color_map = {}
        device_row_map = {}
        device_periods = []
        color_index = 0
        row_index = 0

        for optimizer in optimizers:
            device_name = optimizer.device_name

            # Assign color and row to device
            if device_name not in device_color_map:
                device_color_map[device_name] = DEVICE_COLORS[color_index % len(DEVICE_COLORS)]
                device_row_map[device_name] = row_index
                color_index += 1
                row_index += 1

            # Collect periods for this device from persistent storage
            device_optimal_periods = []

            # Include persistent periods that fall within our time window
            if hasattr(optimizer, '_persistent_periods'):
                for period in optimizer._persistent_periods:
                    # Include periods that overlap with our display time window
                    if period.start_time < end_time and period.end_time > start_time:
                        device_optimal_periods.append({
                            "start": period.start_time.isoformat(),
                            "end": period.end_time.isoformat(),
                            "average_price": period.average_price,
                            "status": period.status
                        })
            else:
                # Fallback to current_optimal_periods for backwards compatibility
                for period in optimizer._current_optimal_periods:
                    if period.start_time < end_time and period.end_time > now:
                        device_optimal_periods.append({
                            "start": period.start_time.isoformat(),
                            "end": period.end_time.isoformat(),
                            "average_price": period.average_price,
                            "status": "active" if period.start_time <= now < period.end_time else "planned"
                        })

            # Add device periods if any exist
            if device_optimal_periods:
                # Calculate y_position for chart annotation (negative values below chart)
                y_position = -0.05 - (device_row_map[device_name] * 0.08)  # Stagger rows

                device_periods.append({
                    "device": device_name,
                    "row": device_row_map[device_name],
                    "periods": device_optimal_periods,
                    "color": device_color_map[device_name],
                    "y_position": y_position
                })

                # Also add to legacy optimal_periods for backward compatibility
                for period_data in device_optimal_periods:
                    optimal_periods.append({
                        "device": device_name,
                        "start": period_data["start"],
                        "end": period_data["end"],
                        "average_price": period_data["average_price"],
                        "status": period_data.get("status", "unknown"),
                        "color": device_color_map[device_name],
                        "row": device_row_map[device_name]
                    })

        return {
            "prices_ahead": prices_ahead,
            "optimal_periods": optimal_periods,  # Legacy format
            "device_periods": device_periods,    # New row-based format
            "device_colors": device_color_map,
            "device_rows": device_row_map,
            "hours_ahead": self._hours_ahead,
            "unit": self.native_unit_of_measurement,
            "last_updated": now.isoformat(),
            "current_time": now.isoformat(),  # Current time for vertical line marker
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "tomorrow_data_available": has_tomorrow_data if all_prices else False
            },
            "total_devices": len(optimizers),
            "chart_layout": {
                "row_height": 0.08,
                "row_spacing": 0.05,
                "base_offset": -0.05
            }
        }

    def _get_all_optimizers(self) -> list[NordpoolOptimizer]:
        """Get all optimizer instances and auto-register with new ones."""
        optimizers = []
        domain_data = self._hass.data.get(DOMAIN, {})

        for config_entry_id, optimizer in domain_data.items():
            if isinstance(optimizer, NordpoolOptimizer):
                optimizers.append(optimizer)

                # Check if any setup is in progress before auto-registering to prevent loops
                setup_key = f"{DOMAIN}_{config_entry_id}"
                if setup_key in _setup_in_progress:
                    _LOGGER.debug("Skipping auto-registration for %s (setup in progress)", optimizer.device_name)
                    continue

                # Auto-register with new optimizers that haven't been registered yet
                graph_listener_key = f"graph_{self.unique_id}"
                if graph_listener_key not in optimizer._output_listeners:
                    optimizer.register_output_listener_entity(self, graph_listener_key)
                    _LOGGER.debug("Auto-registered graph entity with new optimizer: %s", optimizer.device_name)

        return optimizers

    @property
    def should_poll(self) -> bool:
        """No need to poll, updates are triggered by optimizers."""
        return False

    def update_callback(self) -> None:
        """Called when any optimizer updates."""
        self.schedule_update_ha_state()

    def force_discovery_update(self) -> None:
        """Force an immediate discovery update to pick up new devices."""
        # This will trigger _get_all_optimizers which will auto-register new devices
        self.schedule_update_ha_state()
        _LOGGER.debug("Forced discovery update for price graph entity")

    async def async_added_to_hass(self) -> None:
        """Register with all optimizers for updates."""
        await super().async_added_to_hass()

        # Register with all existing optimizers
        optimizers = self._get_all_optimizers()
        for optimizer in optimizers:
            optimizer.register_output_listener_entity(self, f"graph_{self.unique_id}")

    async def async_will_remove_from_hass(self) -> None:
        """Cleanup when entity is removed."""
        # Note: In a full implementation, we'd need to unregister from optimizers
        # but the current optimizer cleanup handles this automatically
        await super().async_will_remove_from_hass()