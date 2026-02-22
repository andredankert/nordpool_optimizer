"""Timer entity implementation for Nordpool Optimizer."""

from __future__ import annotations

import datetime as dt
import logging
import pickle
from pathlib import Path

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN, UnitOfTime
from homeassistant.core import HomeAssistant
from homeassistant.helpers.event import async_track_time_change
from homeassistant.util import dt as dt_util

from . import NordpoolOptimizer, NordpoolOptimizerEntity
from .const import (
    CONF_GRAPH_HOURS_AHEAD,
    DEFAULT_GRAPH_HOURS,
    DEVICE_COLORS,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities
):
    """Create timer entity for platform."""
    optimizer: NordpoolOptimizer = hass.data[DOMAIN][config_entry.entry_id]
    entities = []

    # Create timer entity for this device
    entities.append(
        NordpoolOptimizerTimerEntity(
            optimizer,
            entity_description=SensorEntityDescription(
                key="timer",
            ),
        )
    )

    # Create price graph entity only once (singleton across all optimizers)
    all_entity_ids = hass.states.async_entity_ids("sensor")
    existing_graph_ids = [eid for eid in all_entity_ids
                         if eid.startswith("sensor.nordpool_optimizer_price_graph")]

    domain_data = hass.data.get(DOMAIN, {})
    graph_entity_exists = any(
        hasattr(opt, '_has_graph_entity') and opt._has_graph_entity
        for opt in domain_data.values()
        if hasattr(opt, '_has_graph_entity')
    )

    if not existing_graph_ids and not graph_entity_exists:
        graph_hours = config_entry.data.get(CONF_GRAPH_HOURS_AHEAD, DEFAULT_GRAPH_HOURS)
        graph_entity = NordpoolOptimizerPriceGraphEntity(hass, graph_hours)
        entities.append(graph_entity)
        optimizer._has_graph_entity = True
        _LOGGER.debug("Created price graph entity for nordpool optimizer")

    async_add_entities(entities)
    return True


async def async_unload_entry(
    hass: HomeAssistant, config_entry: ConfigEntry
) -> bool:
    """Unload entities when config entry is unloaded."""
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
        # Check if optimizer is cleaned up or not initialized
        if not self._optimizer or not hasattr(self._optimizer, 'is_valid'):
            return None
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
        # Check if optimizer exists and is initialized
        if not self._optimizer or not hasattr(self._optimizer, 'is_valid'):
            return STATE_UNAVAILABLE
            
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
        # If optimizer already has valid data, trigger immediate update
        # Otherwise schedule update which will happen after optimizer updates
        if self._optimizer.is_valid:
            # Optimizer already has data, update immediately
            self.schedule_update_ha_state()
        else:
            # Optimizer doesn't have data yet, will update when it gets data
            # But still schedule an update to show current state (unavailable)
            self.schedule_update_ha_state()


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
        self._minutely_update = None
        # Price history buffer: retains prices across midnight when Nordpool
        # drops yesterday's raw_today.  Keyed by ISO timestamp string.
        self._price_history: dict[str, dict] = {}
        self._price_cache_file = (
            Path(hass.config.config_dir)
            / "custom_components" / "nordpool_optimizer" / "cache"
            / "price_history.pkl"
        )
        self._load_price_history()

    @property
    def native_value(self) -> float | None:
        """Return current price as sensor state."""
        # Get current price from any available optimizer
        try:
            optimizers = self._get_all_optimizers()
            if not optimizers:
                return None

            for optimizer in optimizers:
                if hasattr(optimizer, '_prices_entity') and optimizer._prices_entity.valid:
                    current_price = optimizer._prices_entity.current_price_attr
                    if current_price is not None:
                        return current_price
        except Exception:
            # If there's an error getting optimizers, return None (will show unavailable)
            return None

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
        try:
            value = self.native_value
            if value is None:
                return STATE_UNAVAILABLE
            return f"{value:.3f}"
        except Exception:
            return STATE_UNAVAILABLE

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

        # Accumulate prices into our history buffer so yesterday's data
        # survives past midnight (the Nordpool entity drops raw_today at midnight).
        live_prices = price_entity._all_prices
        for price_data in (live_prices or []):
            price_time = price_data["start"]
            if not isinstance(price_time, dt.datetime):
                price_time = dt_util.parse_datetime(price_time)
            if price_time:
                key = price_time.isoformat()
                self._price_history[key] = {
                    "start": price_time,
                    "value": price_data["value"],
                }

        # Prune entries older than 3 days to keep memory bounded
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff = today_start - dt.timedelta(days=2)
        self._price_history = {
            k: v for k, v in self._price_history.items()
            if v["start"] >= cutoff
        }

        # Persist to disk so yesterday's prices survive restarts
        self._save_price_history()

        # Display window: yesterday + today + tomorrow
        start_time = today_start - dt.timedelta(days=1)
        end_time = today_start + dt.timedelta(days=2)

        # Keep the full window for period filtering (past completed periods
        # should always show, even if price data doesn't go back that far)
        period_window_start = start_time
        period_window_end = end_time

        # Build all_prices from history buffer (contains yesterday even after midnight)
        all_prices = sorted(self._price_history.values(), key=lambda p: p["start"])

        if all_prices:
            _LOGGER.debug("Price history buffer: %d entries from %s to %s, "
                         "display window: %s to %s",
                         len(all_prices),
                         all_prices[0]["start"].strftime('%Y-%m-%d %H:%M'),
                         all_prices[-1]["start"].strftime('%Y-%m-%d %H:%M'),
                         start_time.strftime('%Y-%m-%d %H:%M'),
                         end_time.strftime('%Y-%m-%d %H:%M'))
        else:
            _LOGGER.debug("No price data in history buffer")

        if not all_prices:
            return {"error": "No price data available"}

        prices_ahead = []
        optimal_periods = []

        available_start = all_prices[0]["start"]
        available_end = all_prices[-1]["start"]

        _LOGGER.debug("Available price data spans: %s to %s",
                     available_start.strftime('%Y-%m-%d %H:%M'),
                     available_end.strftime('%Y-%m-%d %H:%M'))

        # If we need data before what's available, adjust the start time
        if start_time < available_start:
            old_start = start_time
            start_time = available_start
            _LOGGER.debug("Adjusted start time from %s to %s (data not available)",
                         old_start.strftime('%Y-%m-%d %H:%M'),
                         start_time.strftime('%Y-%m-%d %H:%M'))

        # Build unique price data with optimal device info
        price_map = {}  # Use dict to avoid duplicates by timestamp

        _LOGGER.debug("Filtering prices: total available=%d, time range %s to %s",
                     len(all_prices), start_time.strftime('%Y-%m-%d %H:%M'), end_time.strftime('%Y-%m-%d %H:%M'))

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

        _LOGGER.debug("Price filtering result: %d prices included in range", len(prices_ahead))

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

            # Include persistent periods that fall within the full display window
            # (not narrowed by price data availability)
            if hasattr(optimizer, '_persistent_periods'):
                for period in optimizer._persistent_periods:
                    if period.start_time < period_window_end and period.end_time > period_window_start:
                        device_optimal_periods.append({
                            "start": period.start_time.isoformat(),
                            "end": period.end_time.isoformat(),
                            "average_price": period.average_price,
                            "status": period.status
                        })
            else:
                # Fallback to current_optimal_periods for backwards compatibility
                for period in optimizer._current_optimal_periods:
                    if period.start_time < period_window_end and period.end_time > now:
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
            },
            "total_devices": len(optimizers),
            "chart_layout": {
                "row_height": 0.08,
                "row_spacing": 0.05,
                "base_offset": -0.05
            }
        }

    # ------------------------------------------------------------------
    # Price history persistence
    # ------------------------------------------------------------------

    def _load_price_history(self) -> None:
        """Load price history from disk cache."""
        if not self._price_cache_file.exists():
            return
        try:
            with open(self._price_cache_file, "rb") as fh:
                data = pickle.load(fh)
            if not isinstance(data, dict):
                return
            # Re-key and validate entries
            now = dt_util.now()
            cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0) - dt.timedelta(days=2)
            for key, entry in data.items():
                if isinstance(entry, dict) and isinstance(entry.get("start"), dt.datetime):
                    if entry["start"] >= cutoff:
                        self._price_history[key] = entry
            _LOGGER.debug("Loaded %d price history entries from cache", len(self._price_history))
        except Exception:
            _LOGGER.debug("Could not load price history cache, starting fresh")

    def _save_price_history(self) -> None:
        """Save price history to disk cache."""
        try:
            self._price_cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._price_cache_file.with_suffix(".tmp")
            with open(tmp, "wb") as fh:
                pickle.dump(self._price_history, fh)
            tmp.replace(self._price_cache_file)
        except Exception:
            _LOGGER.debug("Could not save price history cache")

    def _get_all_optimizers(self) -> list[NordpoolOptimizer]:
        """Get all optimizer instances and auto-register with new ones."""
        optimizers = []
        domain_data = self._hass.data.get(DOMAIN, {})

        for config_entry_id, optimizer in domain_data.items():
            if isinstance(optimizer, NordpoolOptimizer):
                optimizers.append(optimizer)

                # Auto-register with new optimizers that haven't been registered yet
                graph_listener_key = f"graph_{self.unique_id}"
                existing_listener = optimizer._output_listeners.get(graph_listener_key)
                # Only register if not already registered with this exact entity instance
                if existing_listener is not self:
                    optimizer.register_output_listener_entity(self, graph_listener_key)
                    if existing_listener is None:
                        _LOGGER.debug("Auto-registered graph entity with optimizer: %s", optimizer.device_name)
                    else:
                        _LOGGER.debug("Re-registered graph entity with optimizer: %s (replaced old listener)", optimizer.device_name)

        return optimizers

    @property
    def should_poll(self) -> bool:
        """No need to poll, updates are triggered by optimizers."""
        return False

    def update_callback(self) -> None:
        """Called when any optimizer updates."""
        self.schedule_update_ha_state()

    def minutely_time_update(self, _) -> None:
        """Update current time marker every minute for live chart updates."""
        # Only update the state to refresh time-sensitive attributes like current_time
        # This doesn't recalculate price data, just updates the time marker
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

        # Register minutely updates for live time marker updates
        self._minutely_update = async_track_time_change(
            self.hass, self.minutely_time_update, second=0
        )
        
        # Schedule immediate update to ensure entity shows current state
        self.schedule_update_ha_state()

    async def async_will_remove_from_hass(self) -> None:
        """Cleanup when entity is removed."""
        # Cleanup minutely time update listener
        if self._minutely_update:
            self._minutely_update()
            self._minutely_update = None

        # Note: In a full implementation, we'd need to unregister from optimizers
        # but the current optimizer cleanup handles this automatically
        await super().async_will_remove_from_hass()