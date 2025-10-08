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
from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities
):
    """Create timer entity for platform."""

    optimizer: NordpoolOptimizer = hass.data[DOMAIN][config_entry.entry_id]
    entities = []

    entities.append(
        NordpoolOptimizerTimerEntity(
            optimizer,
            entity_description=SensorEntityDescription(
                key="timer",
                device_class=SensorDeviceClass.DURATION,
                native_unit_of_measurement=UnitOfTime.MINUTES,
            ),
        )
    )

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
                return max(0, int(remaining.total_seconds() / 60))
            return 0
        else:
            # Return negative minutes until next optimization period
            next_start = self._optimizer.get_next_optimal_start(now)
            if next_start:
                countdown = next_start - now
                return -max(0, int(countdown.total_seconds() / 60))
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
        }

        if next_start:
            attributes["next_optimal_start"] = next_start
        if current_end:
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