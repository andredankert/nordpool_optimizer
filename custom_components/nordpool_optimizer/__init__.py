"""Main package for nordpool optimizer."""

from __future__ import annotations

import datetime as dt
import logging

from homeassistant.config_entries import SOURCE_IMPORT, ConfigEntry
from homeassistant.const import (
    ATTR_UNIT_OF_MEASUREMENT,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
    Platform,
)
from homeassistant.core import HomeAssistant, HomeAssistantError
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_time_change,
)
from homeassistant.util import dt as dt_util

from .config_flow import NordpoolOptimizerConfigFlow
from .const import (
    CONF_DEVICE_NAME,
    CONF_DURATION,
    CONF_MODE,
    CONF_MODE_ABSOLUTE,
    CONF_MODE_DAILY,
    CONF_PRICE_THRESHOLD,
    CONF_PRICES_ENTITY,
    CONF_SLOT_TYPE,
    CONF_SLOT_TYPE_CONSECUTIVE,
    CONF_SLOT_TYPE_SEPARATE,
    CONF_TIME_WINDOW,
    CONF_TIME_WINDOW_ENABLED,
    DOMAIN,
    NAME_FILE_READER,
    PATH_FILE_READER,
    OptimizerStates,
)
from .helpers import get_np_from_file

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [Platform.SENSOR]


async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Set up this integration using UI."""
    config_entry.async_on_unload(config_entry.add_update_listener(async_reload_entry))

    if DOMAIN not in hass.data:
        hass.data[DOMAIN] = {}

    if config_entry.entry_id not in hass.data[DOMAIN]:
        optimizer = NordpoolOptimizer(hass, config_entry)
        await optimizer.async_setup()
        hass.data[DOMAIN][config_entry.entry_id] = optimizer

    if config_entry is not None:
        if config_entry.source == SOURCE_IMPORT:
            hass.async_create_task(
                hass.config_entries.async_remove(config_entry.entry_id)
            )
            return False

    await hass.config_entries.async_forward_entry_setups(config_entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unloading a config_flow entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        optimizer = hass.data[DOMAIN].pop(entry.entry_id)
        optimizer.cleanup()
    return unload_ok


async def async_reload_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> None:
    """Reload the config entry."""
    await async_unload_entry(hass, config_entry)
    await async_setup_entry(hass, config_entry)


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate old entry."""
    _LOGGER.debug(
        "Attempting migrating configuration from version %s.%s",
        config_entry.version,
        config_entry.minor_version,
    )

    installed_version = NordpoolOptimizerConfigFlow.VERSION
    installed_minor_version = NordpoolOptimizerConfigFlow.MINOR_VERSION

    if config_entry.version > installed_version:
        _LOGGER.warning(
            "Downgrading major version from %s to %s is not allowed",
            config_entry.version,
            installed_version,
        )
        return False

    if (
        config_entry.version == installed_version
        and config_entry.minor_version > installed_minor_version
    ):
        _LOGGER.warning(
            "Downgrading minor version from %s.%s to %s.%s is not allowed",
            config_entry.version,
            config_entry.minor_version,
            installed_version,
            installed_minor_version,
        )
        return False

    # Future migration logic can be added here
    return True


class NordpoolOptimizer:
    """Optimizer base class."""

    _hourly_update = None

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        """Initialize optimizer."""
        self._hass = hass
        self._config = config_entry
        self._state_change_listeners = []

        # Configuration
        self.device_name = self._config.data[CONF_DEVICE_NAME]
        self.mode = self._config.data[CONF_MODE]
        self.duration = self._config.data[CONF_DURATION]

        # Mode-specific configuration
        self.price_threshold = self._config.data.get(CONF_PRICE_THRESHOLD)
        self.slot_type = self._config.data.get(CONF_SLOT_TYPE, CONF_SLOT_TYPE_CONSECUTIVE)

        # Time window configuration
        self.time_window_enabled = self._config.data.get(CONF_TIME_WINDOW_ENABLED, False)
        self.time_window = self._config.data.get(CONF_TIME_WINDOW, "")

        # Input entities
        self._prices_entity = PricesEntity(self._config.data[CONF_PRICES_ENTITY])

        # Output entities
        self._output_listeners: dict[str, NordpoolOptimizerEntity] = {}

        # State variables
        self._last_update = None
        self._optimizer_status = NordpoolOptimizerStatus()
        self._current_optimal_periods: list[OptimalPeriod] = []

    def as_dict(self):
        """For diagnostics serialization."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_") or k in ["_config", "_optimizer_status"]
        }

    async def async_setup(self):
        """Post initialization setup."""
        # Ensure an update is done on every hour
        self._hourly_update = async_track_time_change(
            self._hass, self.scheduled_update, minute=0, second=0
        )

    @property
    def is_valid(self) -> bool:
        """Check if optimizer has valid data."""
        return self._prices_entity.valid and self._optimizer_status.status == OptimizerStates.Ok

    @property
    def price_sensor_id(self) -> str:
        """Entity id of source sensor."""
        return self._prices_entity.unique_id

    @property
    def optimizer_status(self) -> NordpoolOptimizerStatus:
        """Current optimizer status."""
        return self._optimizer_status

    def cleanup(self):
        """Cleanup by removing event listeners."""
        for listener in self._state_change_listeners:
            listener()

    def register_output_listener_entity(
        self, entity: NordpoolOptimizerEntity, conf_key=""
    ) -> None:
        """Register output entity."""
        if conf_key in self._output_listeners:
            _LOGGER.warning(
                'An output listener with key "%s" is overriding previous entity "%s"',
                conf_key,
                self._output_listeners.get(conf_key).entity_id,
            )
        self._output_listeners[conf_key] = entity

    def get_device_info(self) -> DeviceInfo:
        """Get device info to group entities."""
        return DeviceInfo(
            identifiers={(DOMAIN, self._config.entry_id)},
            name=self.device_name,
            manufacturer="Nordpool",
            entry_type=DeviceEntryType.SERVICE,
            model="Optimizer",
        )

    def scheduled_update(self, _):
        """Scheduled updates callback."""
        _LOGGER.debug("Scheduled callback")
        self.update()

    def update(self):
        """Optimizer update call function."""
        _LOGGER.debug("Updating optimizer for %s", self.device_name)

        # Update price data
        price_updated = self._prices_entity.update(self._hass)
        _LOGGER.debug("Price entity update result: %s, valid: %s", price_updated, self._prices_entity.valid)

        if not price_updated or not self._prices_entity.valid:
            _LOGGER.warning("Price entity %s failed to update or is invalid", self._prices_entity.unique_id)
            self.set_unavailable()
            self._optimizer_status.status = OptimizerStates.Error
            self._optimizer_status.running_text = "No valid Price data"
            return

        # Validate configuration
        _LOGGER.debug("Configuration check - duration: %s, mode: %s, price_threshold: %s",
                     self.duration, self.mode, getattr(self, 'price_threshold', 'N/A'))

        if not self.duration or self.duration <= 0:
            _LOGGER.warning("Aborting update since no valid Duration: %s", self.duration)
            self._optimizer_status.status = OptimizerStates.Error
            self._optimizer_status.running_text = "No valid Duration data"
            return

        # Mode-specific validation
        if self.mode == CONF_MODE_ABSOLUTE and (not self.price_threshold):
            _LOGGER.warning("Aborting update since no price threshold for absolute mode: %s", self.price_threshold)
            self._optimizer_status.status = OptimizerStates.Error
            self._optimizer_status.running_text = "No valid price threshold"
            return

        # Set status to OK
        self._optimizer_status.status = OptimizerStates.Ok
        self._optimizer_status.running_text = "ok"

        # Calculate optimal periods based on mode
        now = dt_util.now()
        if self.mode == CONF_MODE_ABSOLUTE:
            self._current_optimal_periods = self._calculate_absolute_periods(now)
        elif self.mode == CONF_MODE_DAILY:
            self._current_optimal_periods = self._calculate_daily_periods(now)

        self._last_update = now

        # Notify all listeners
        for listener in self._output_listeners.values():
            listener.update_callback()

    def _calculate_absolute_periods(self, now: dt.datetime) -> list[OptimalPeriod]:
        """Calculate optimal periods for absolute mode."""
        periods = []

        # Look ahead up to 48 hours for price data
        end_time = now + dt.timedelta(hours=48)

        # Check each hour to see if it meets the threshold
        current_hour = now.replace(minute=0, second=0, microsecond=0)

        while current_hour < end_time:
            # Check if this hour is within time window (if enabled)
            if self.time_window_enabled and not self._is_in_time_window(current_hour):
                current_hour += dt.timedelta(hours=1)
                continue

            # Get price group for the duration starting at this hour
            period_end = current_hour + dt.timedelta(hours=self.duration)
            price_group = self._prices_entity.get_prices_group(current_hour, period_end)

            if price_group.valid and price_group.average <= self.price_threshold:
                periods.append(OptimalPeriod(
                    start_time=current_hour,
                    end_time=period_end,
                    average_price=price_group.average
                ))

            current_hour += dt.timedelta(hours=1)

        return periods

    def _calculate_daily_periods(self, now: dt.datetime) -> list[OptimalPeriod]:
        """Calculate optimal periods for daily mode."""
        periods = []

        # Calculate for today and tomorrow
        for day_offset in [0, 1]:
            day_start = (now + dt.timedelta(days=day_offset)).replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + dt.timedelta(days=1)

            if self.slot_type == CONF_SLOT_TYPE_CONSECUTIVE:
                period = self._find_consecutive_daily_period(day_start, day_end)
            else:
                period = self._find_separate_daily_periods(day_start, day_end)

            if period:
                if isinstance(period, list):
                    periods.extend(period)
                else:
                    periods.append(period)

        return periods

    def _find_consecutive_daily_period(self, day_start: dt.datetime, day_end: dt.datetime) -> OptimalPeriod | None:
        """Find the cheapest consecutive period in a day."""
        best_period = None
        best_price = float('inf')

        # Try each possible starting hour
        current_hour = day_start
        while current_hour + dt.timedelta(hours=self.duration) <= day_end:
            # Check if this hour is within time window (if enabled)
            if self.time_window_enabled and not self._is_in_time_window(current_hour):
                current_hour += dt.timedelta(hours=1)
                continue

            period_end = current_hour + dt.timedelta(hours=self.duration)
            price_group = self._prices_entity.get_prices_group(current_hour, period_end)

            if price_group.valid and price_group.average < best_price:
                best_price = price_group.average
                best_period = OptimalPeriod(
                    start_time=current_hour,
                    end_time=period_end,
                    average_price=price_group.average
                )

            current_hour += dt.timedelta(hours=1)

        return best_period

    def _find_separate_daily_periods(self, day_start: dt.datetime, day_end: dt.datetime) -> list[OptimalPeriod]:
        """Find the cheapest separate hours in a day."""
        # Get all hourly prices for the day
        hourly_prices = []
        current_hour = day_start

        while current_hour < day_end:
            # Check if this hour is within time window (if enabled)
            if self.time_window_enabled and not self._is_in_time_window(current_hour):
                current_hour += dt.timedelta(hours=1)
                continue

            hour_end = current_hour + dt.timedelta(hours=1)
            price_group = self._prices_entity.get_prices_group(current_hour, hour_end)

            if price_group.valid:
                hourly_prices.append({
                    'hour': current_hour,
                    'price': price_group.average
                })

            current_hour += dt.timedelta(hours=1)

        # Sort by price and take the cheapest hours
        hourly_prices.sort(key=lambda x: x['price'])
        cheapest_hours = hourly_prices[:min(self.duration, len(hourly_prices))]

        # Create periods for each hour
        periods = []
        for hour_data in cheapest_hours:
            periods.append(OptimalPeriod(
                start_time=hour_data['hour'],
                end_time=hour_data['hour'] + dt.timedelta(hours=1),
                average_price=hour_data['price']
            ))

        return periods

    def _is_in_time_window(self, time: dt.datetime) -> bool:
        """Check if time is within the configured time window."""
        if not self.time_window_enabled or not self.time_window:
            return True

        try:
            # Parse time window like "22:00-06:00"
            start_str, end_str = self.time_window.split('-')
            start_hour, start_min = map(int, start_str.split(':'))
            end_hour, end_min = map(int, end_str.split(':'))

            time_minutes = time.hour * 60 + time.minute
            start_minutes = start_hour * 60 + start_min
            end_minutes = end_hour * 60 + end_min

            if start_minutes <= end_minutes:
                # Normal time range (e.g., "08:00-17:00")
                return start_minutes <= time_minutes < end_minutes
            else:
                # Overnight time range (e.g., "22:00-06:00")
                return time_minutes >= start_minutes or time_minutes < end_minutes

        except (ValueError, AttributeError):
            _LOGGER.warning("Invalid time window format: %s", self.time_window)
            return True

    def is_currently_optimal(self, now: dt.datetime) -> bool:
        """Check if current time is in an optimal period."""
        for period in self._current_optimal_periods:
            if period.start_time <= now < period.end_time:
                return True
        return False

    def get_current_period_end(self, now: dt.datetime) -> dt.datetime | None:
        """Get end time of current optimal period."""
        for period in self._current_optimal_periods:
            if period.start_time <= now < period.end_time:
                return period.end_time
        return None

    def get_next_optimal_start(self, now: dt.datetime) -> dt.datetime | None:
        """Get start time of next optimal period."""
        future_periods = [p for p in self._current_optimal_periods if p.start_time > now]
        if future_periods:
            return min(future_periods, key=lambda p: p.start_time).start_time
        return None

    def set_unavailable(self) -> None:
        """Set output state to unavailable."""
        _LOGGER.debug("Setting output states to unavailable")
        for listener in self._output_listeners.values():
            listener.update_callback()


class PricesEntity:
    """Representation for Nordpool state."""

    def __init__(self, unique_id: str) -> None:
        """Initialize state tracker."""
        self._unique_id = unique_id
        self._np = None

    def as_dict(self):
        """For diagnostics serialization."""
        return self.__dict__

    @property
    def unique_id(self) -> str:
        """Get the unique id."""
        return self._unique_id

    @property
    def valid(self) -> bool:
        """Get if data is valid."""
        return self._np is not None

    @property
    def _all_prices(self):
        if np_prices := self._np.attributes.get("raw_today"):
            # For Nordpool format
            if self._np.attributes["tomorrow_valid"]:
                np_prices += self._np.attributes["raw_tomorrow"]
            return np_prices
        elif e_prices := self._np.attributes.get("prices"):  # noqa: RET505
            # For ENTSO-e format
            e_prices = [
                {"start": dt_util.parse_datetime(ep["time"]), "value": ep["price"]}
                for ep in e_prices
            ]
            return e_prices  # noqa: RET504
        return []

    @property
    def current_price_attr(self):
        """Get the current price attribute."""
        if self._np is not None:
            if current := self._np.attributes.get("current_price"):
                # For Nordpool format
                return current
            else:  # noqa: RET505
                # For general, find in list
                now = dt_util.now()
                for price in self._all_prices:
                    if (
                        price["start"] < now
                        and price["start"] + dt.timedelta(hours=1) > now
                    ):
                        return price["value"]
        return None

    def update(self, hass: HomeAssistant) -> bool:
        """Update price in storage."""
        if self._unique_id == NAME_FILE_READER:
            np = get_np_from_file(PATH_FILE_READER)
        else:
            np = hass.states.get(self._unique_id)

        if np is None:
            _LOGGER.warning("Got empty data from Nordpool entity %s ", self._unique_id)
        elif "today" not in np.attributes and "prices_today" not in np.attributes:
            _LOGGER.warning(
                "No values for today in Nordpool entity %s ", self._unique_id
            )
        else:
            _LOGGER.debug(
                "Nordpool sensor %s was updated successfully", self._unique_id
            )
            self._np = np

        return self._np is not None

    def get_prices_group(
        self, start: dt.datetime, end: dt.datetime
    ) -> NordpoolPricesGroup:
        """Get a range of prices from NP given the start and end datetimes."""
        started = False
        selected = []
        for p in self._all_prices:
            if p["start"] > start - dt.timedelta(hours=1):
                started = True
            if p["start"] > end:
                break
            if started:
                selected.append(p)
        return NordpoolPricesGroup(selected)


class NordpoolPricesGroup:
    """A slice of Nordpool prices with helper functions."""

    def __init__(self, prices) -> None:
        """Initialize price group."""
        self._prices = prices

    def __str__(self) -> str:
        """Get string representation of class."""
        return f"start_time={self.start_time.strftime('%Y-%m-%d %H:%M')} average={self.average} len(_prices)={len(self._prices)}"

    def __repr__(self) -> str:
        """Get string representation for debugging."""
        return type(self).__name__ + f" ({self.__str__()})"

    @property
    def valid(self) -> bool:
        """Is the price group valid."""
        return len(self._prices) > 0

    @property
    def average(self) -> float:
        """The average price of the price group."""
        if not self.valid:
            return float('inf')
        return sum([p["value"] for p in self._prices]) / len(self._prices)

    @property
    def start_time(self) -> dt.datetime:
        """The start time of first price in group."""
        if not self.valid:
            return None
        return self._prices[0]["start"]


class OptimalPeriod:
    """Represents an optimal time period."""

    def __init__(self, start_time: dt.datetime, end_time: dt.datetime, average_price: float):
        """Initialize optimal period."""
        self.start_time = start_time
        self.end_time = end_time
        self.average_price = average_price

    def __str__(self) -> str:
        """String representation."""
        return f"OptimalPeriod({self.start_time.strftime('%H:%M')}-{self.end_time.strftime('%H:%M')}, {self.average_price:.3f})"


class NordpoolOptimizerStatus:
    """Status for the overall optimizer."""

    def __init__(self) -> None:
        """Initiate status."""
        self.status = OptimizerStates.Unknown
        self.running_text = ""


class NordpoolOptimizerEntity(Entity):
    """Base class for nordpool optimizer entities."""

    def __init__(
        self,
        optimizer: NordpoolOptimizer,
    ) -> None:
        """Initialize entity."""
        self._optimizer = optimizer
        self._attr_device_info = optimizer.get_device_info()

    def as_dict(self):
        """For diagnostics serialization."""
        return {
            k: v
            for k, v in self.__dict__.items()
            if not (
                k.startswith("_")
                or k in ["hass", "platform", "registry_entry", "device_entry"]
            )
        }

    @property
    def should_poll(self):
        """No need to poll. Coordinator notifies entity of updates."""
        return False

    def update_callback(self) -> None:
        """Call from optimizer that new data available."""
        self.schedule_update_ha_state()