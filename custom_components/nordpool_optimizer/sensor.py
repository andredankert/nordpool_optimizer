"""Timer entity implementation for Nordpool Optimizer."""

from __future__ import annotations

import datetime as dt
import logging

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
    DEVICE_COLORS,
    DOMAIN,
    DEFAULT_GRAPH_HOURS,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities
):
    """Create timer entity for platform."""

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
        # Create the global price graph entity
        graph_entity = NordpoolOptimizerPriceGraphEntity(hass, DEFAULT_GRAPH_HOURS)
        entities.append(graph_entity)

        # Mark that we've created the graph entity
        optimizer._has_graph_entity = True
        _LOGGER.debug("Created price graph entity for nordpool optimizer")
    else:
        # Graph entity already exists, trigger discovery update
        _LOGGER.debug("Price graph entity already exists, triggering discovery update for new device")

        # Find the existing graph entity and trigger discovery
        for entity_id in existing_entity_ids:
            entity_state = hass.states.get(entity_id)
            if entity_state:
                # The auto-registration will happen on the next update
                # We can't directly access the entity object here, but the next update cycle will pick it up
                _LOGGER.debug("Will auto-register with existing price graph entity: %s", entity_id)

    async_add_entities(entities)
    return True


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
        end_time = now + dt.timedelta(hours=self._hours_ahead)
        prices_ahead = []
        optimal_periods = []

        # Get all prices within time range
        all_prices = price_entity._all_prices
        if not all_prices:
            return {"error": "No price data available"}

        # Filter prices for the specified time range
        for price_data in all_prices:
            price_time = price_data["start"]
            if isinstance(price_time, str):
                price_time = dt_util.parse_datetime(price_time)

            if not price_time:
                continue

            if now <= price_time < end_time:
                # Check which devices are optimal at this time
                optimal_devices = []
                for optimizer in optimizers:
                    if optimizer.is_currently_optimal(price_time):
                        optimal_devices.append(optimizer.device_name)

                prices_ahead.append({
                    "time": price_time.isoformat(),
                    "price": price_data["value"],
                    "optimal_devices": optimal_devices
                })

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

            # Collect periods for this device
            device_optimal_periods = []
            for period in optimizer._current_optimal_periods:
                if period.start_time < end_time and period.end_time > now:
                    device_optimal_periods.append({
                        "start": period.start_time.isoformat(),
                        "end": period.end_time.isoformat(),
                        "average_price": period.average_price
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