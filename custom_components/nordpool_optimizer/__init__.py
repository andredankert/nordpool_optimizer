"""Main package for nordpool optimizer."""

from __future__ import annotations

import datetime as dt
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

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
    CONF_NETWORK_FEE,
    CONF_PRICE_THRESHOLD,
    CONF_PRICES_ENTITY,
    CONF_PROVIDER_FEE,
    CONF_SLOT_TYPE,
    CONF_SLOT_TYPE_CONSECUTIVE,
    CONF_SLOT_TYPE_SEPARATE,
    CONF_TAX_PERCENTAGE,
    CONF_TIME_WINDOW,
    CONF_TIME_WINDOW_ENABLED,
    DEFAULT_NETWORK_FEE,
    DEFAULT_PROVIDER_FEE,
    DEFAULT_TAX_PERCENTAGE,
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

    # Always create a new optimizer instance
    # During reload, the old one should have been removed by async_unload_entry
    if config_entry.entry_id in hass.data.get(DOMAIN, {}):
        _LOGGER.warning("Optimizer already exists for entry %s, cleaning up old instance", config_entry.entry_id)
        old_optimizer = hass.data[DOMAIN].pop(config_entry.entry_id)
        old_optimizer.cleanup()
    
    # Create fresh optimizer instance
    _LOGGER.debug("Creating new optimizer instance for entry: %s", config_entry.entry_id)
    optimizer = NordpoolOptimizer(hass, config_entry)
    await optimizer.async_setup()
    hass.data[DOMAIN][config_entry.entry_id] = optimizer

    if config_entry is not None:
        if config_entry.source == SOURCE_IMPORT:
            hass.async_create_task(
                hass.config_entries.async_remove(config_entry.entry_id)
            )
            return False

    # Before setting up platforms, ensure any existing entities are removed
    # This prevents "already been setup" errors during reload
    from homeassistant.helpers import entity_registry as er
    entity_registry = er.async_get(hass)
    
    # Find and remove any existing entities for this config entry
    existing_entities = [
        entity_id for entity_id, entity in entity_registry.entities.items()
        if entity.config_entry_id == config_entry.entry_id
    ]
    if existing_entities:
        _LOGGER.debug("Removing %d existing entities for entry %s before setup", 
                     len(existing_entities), config_entry.entry_id)
        for entity_id in existing_entities:
            try:
                entity_registry.async_remove(entity_id)
            except Exception as e:
                _LOGGER.debug("Failed to remove entity %s: %s", entity_id, e)
    
    # Small delay to ensure entity removal completes
    import asyncio
    await asyncio.sleep(0.1)
    
    # Now set up platforms
    await hass.config_entries.async_forward_entry_setups(config_entry, PLATFORMS)
    
    # Trigger update immediately after entities are set up
    # This ensures entities have data right away instead of showing unavailable
    if config_entry.entry_id in hass.data[DOMAIN]:
        optimizer = hass.data[DOMAIN][config_entry.entry_id]
        
        async def _trigger_update_immediately():
            """Trigger update immediately to ensure entities have data."""
            try:
                # Very short delay just to let entities finish async_added_to_hass
                import asyncio
                await asyncio.sleep(0.1)
                
                # Check if optimizer still exists
                if config_entry.entry_id not in hass.data.get(DOMAIN, {}):
                    return
                
                optimizer = hass.data[DOMAIN][config_entry.entry_id]
                _LOGGER.info("Triggering immediate update for %s", optimizer.device_name)
                
                # Run update in executor
                await hass.async_add_executor_job(optimizer.update, True)
                
                # Immediately notify all registered entities
                _LOGGER.info("Notifying %d entities after update for %s", len(optimizer._output_listeners), optimizer.device_name)
                for listener in list(optimizer._output_listeners.values()):
                    try:
                        listener.schedule_update_ha_state()
                    except Exception as e:
                        _LOGGER.warning("Failed to notify entity: %s", e)
                
            except Exception as e:
                _LOGGER.error("Failed to update optimizer: %s", e, exc_info=True)
                # Notify entities even on error
                try:
                    if config_entry.entry_id in hass.data.get(DOMAIN, {}):
                        optimizer = hass.data[DOMAIN][config_entry.entry_id]
                        for listener in list(optimizer._output_listeners.values()):
                            listener.schedule_update_ha_state()
                except Exception:
                    pass
        
        # Schedule immediately (don't wait)
        hass.async_create_task(_trigger_update_immediately())
    
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unloading a config_flow entry."""
    _LOGGER.debug("Unloading optimizer entry: %s", entry.entry_id)
    
    from homeassistant.config_entries import ConfigEntryState
    
    # Clean up optimizer first (before unloading platforms)
    if DOMAIN in hass.data and entry.entry_id in hass.data[DOMAIN]:
        optimizer = hass.data[DOMAIN].pop(entry.entry_id)
        optimizer.cleanup()
        _LOGGER.debug("Cleaned up optimizer for entry: %s", entry.entry_id)
    
    # Only try to unload platforms if entry is actually loaded
    # If entry was never loaded or is in a bad state, skip platform unload
    if entry.state != ConfigEntryState.LOADED:
        _LOGGER.debug("Entry %s is in state %s, skipping platform unload", entry.entry_id, entry.state)
        return True  # Return True since we cleaned up the optimizer
    
    # Try to unload platforms, but don't fail if it errors
    try:
        unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
        if not unload_ok:
            _LOGGER.warning("Failed to unload platforms for entry %s, but continuing", entry.entry_id)
        return unload_ok
    except ValueError as e:
        # "Config entry was never loaded!" - this is OK, just means platforms weren't set up
        if "never loaded" in str(e).lower():
            _LOGGER.debug("Platforms were never loaded for entry %s, this is OK", entry.entry_id)
            return True
        _LOGGER.warning("Error unloading platforms for entry %s: %s", entry.entry_id, e)
        return False
    except Exception as e:
        _LOGGER.warning("Error unloading platforms for entry %s: %s", entry.entry_id, e)
        return False


async def async_reload_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> None:
    """Reload the config entry - completely remove old and create new."""
    _LOGGER.info("Reloading optimizer entry: %s (device: %s)", 
                 config_entry.entry_id, config_entry.data.get(CONF_DEVICE_NAME, 'unknown'))
    
    from homeassistant.config_entries import ConfigEntryState
    
    # Clean up optimizer first (always do this)
    if DOMAIN in hass.data and config_entry.entry_id in hass.data[DOMAIN]:
        optimizer = hass.data[DOMAIN].pop(config_entry.entry_id)
        optimizer.cleanup()
        _LOGGER.debug("Cleaned up optimizer for entry: %s", config_entry.entry_id)
    
    # Note: async_update_reload_and_abort calls async_unload_entry internally
    # Our async_unload_entry now handles the "never loaded" case gracefully
    # So we don't need to do anything special here - just set up fresh
    
    # Small delay to ensure cleanup completes
    import asyncio
    await asyncio.sleep(0.2)
    
    # Now set up fresh - this will create a completely new optimizer instance
    # async_setup_entry will handle creating the optimizer and setting up platforms
    try:
        await async_setup_entry(hass, config_entry)
        _LOGGER.info("Reload completed successfully for optimizer entry: %s", config_entry.entry_id)
    except Exception as e:
        _LOGGER.error("Failed to set up entry after reload: %s", e, exc_info=True)
        # Re-raise to let Home Assistant handle it
        raise



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
    _minutely_update = None

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        """Initialize optimizer."""
        self._hass = hass
        self._config = config_entry
        self._state_change_listeners = []
        self._is_cleaned_up = False  # Flag to prevent updates after cleanup

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
        self._prices_entity = PricesEntity(self._config.data[CONF_PRICES_ENTITY], hass, self)

        # Output entities
        self._output_listeners: dict[str, NordpoolOptimizerEntity] = {}

        # State variables
        self._last_update = None
        self._optimizer_status = NordpoolOptimizerStatus()
        self._current_optimal_periods: list[OptimalPeriod] = []

        # Persistent period storage
        self._persistent_periods: list[PersistentOptimalPeriod] = []
        self._period_cache_file = Path(hass.config.config_dir) / "custom_components" / "nordpool_optimizer" / "cache" / f"periods_{self.device_name.replace(' ', '_').lower()}.pkl"

        # Configuration change tracking
        self._cached_fee_settings = self._get_initial_fee_settings()
        self._last_config_hash = self._get_config_hash()

    def as_dict(self):
        """For diagnostics serialization."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_") or k in ["_config", "_optimizer_status"]
        }

    def _get_initial_fee_settings(self) -> dict[str, float]:
        """Get initial fee settings safely during setup."""
        try:
            # Try to get fee settings from any config entry for this domain
            for entry in self._hass.config_entries.async_entries(DOMAIN):
                options = entry.options
                return {
                    "tax_percentage": options.get(CONF_TAX_PERCENTAGE, DEFAULT_TAX_PERCENTAGE),
                    "provider_fee": options.get(CONF_PROVIDER_FEE, DEFAULT_PROVIDER_FEE),
                    "network_fee": options.get(CONF_NETWORK_FEE, DEFAULT_NETWORK_FEE),
                }
        except Exception:
            # If anything goes wrong, return defaults
            pass

        # Fallback to defaults
        return {
            "tax_percentage": DEFAULT_TAX_PERCENTAGE,
            "provider_fee": DEFAULT_PROVIDER_FEE,
            "network_fee": DEFAULT_NETWORK_FEE,
        }

    def refresh_fee_settings(self) -> None:
        """Refresh cached fee settings from config entries."""
        self._cached_fee_settings = self._get_initial_fee_settings()
        _LOGGER.debug("Refreshed fee settings for %s: %s", self.device_name, self._cached_fee_settings)

    async def async_setup(self):
        """Post initialization setup."""
        # Try to load cached price data first for immediate availability
        cache_loaded = self._prices_entity.load_cache()

        # Load period cache to preserve completed past periods (needed for spacing).
        # Future planned periods will be cleared and recalculated by update().
        periods_cache_loaded = self.load_period_cache()
        if periods_cache_loaded:
            now = dt_util.now()
            self._persistent_periods = [
                p for p in self._persistent_periods
                if p.status == "completed" and p.end_time < now
            ]
            _LOGGER.debug("Loaded %d completed past periods for %s", len(self._persistent_periods), self.device_name)

        # Always fetch fresh price data to ensure tomorrow's prices are included.
        # If the entity isn't ready yet, PricesEntity.update() keeps cached data as fallback.
        try:
            self.update(force_price_update=True)
        except Exception as e:
            _LOGGER.warning("Failed to fetch initial price data for %s: %s", self.device_name, e)
            if cache_loaded:
                _LOGGER.debug("Falling back to cached price data for %s", self.device_name)
                self.update(force_price_update=False)

        if periods_cache_loaded:
            _LOGGER.debug("Loaded persistent periods cache for %s", self.device_name)
        else:
            _LOGGER.debug("No valid periods cache found for %s", self.device_name)

        # Ensure an update is done on every hour for optimization calculations
        self._hourly_update = async_track_time_change(
            self._hass, self.scheduled_update, minute=0, second=0
        )

        # Update display every minute for live countdown
        self._minutely_update = async_track_time_change(
            self._hass, self.minutely_display_update, second=0
        )

        # Recalculate immediately when tomorrow's prices become available
        self._state_change_listeners.append(
            async_track_state_change_event(
                self._hass, [self._prices_entity.unique_id], self._on_price_entity_change
            )
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
        _LOGGER.debug("Cleaning up optimizer for %s", self.device_name)
        self._is_cleaned_up = True
        for listener in self._state_change_listeners:
            listener()
        if self._hourly_update:
            self._hourly_update()
        if self._minutely_update:
            self._minutely_update()
        # Clear output listeners to prevent stale references
        self._output_listeners.clear()
        _LOGGER.debug("Cleanup completed for %s", self.device_name)

    def register_output_listener_entity(
        self, entity: NordpoolOptimizerEntity, conf_key=""
    ) -> None:
        """Register output entity."""
        if conf_key in self._output_listeners:
            existing_entity = self._output_listeners.get(conf_key)
            # If it's the same entity instance, skip re-registration silently
            if existing_entity is entity:
                return
            # Also check if it's the same entity by unique_id (for graph entity which is global)
            existing_unique_id = getattr(existing_entity, 'unique_id', None) or getattr(existing_entity, '_attr_unique_id', None)
            new_unique_id = getattr(entity, 'unique_id', None) or getattr(entity, '_attr_unique_id', None)
            if existing_unique_id and new_unique_id and existing_unique_id == new_unique_id:
                # Same logical entity, just update the reference silently
                self._output_listeners[conf_key] = entity
                return
            # Log warning only if it's actually a different entity
            existing_id = (getattr(existing_entity, 'entity_id', None) or 
                          getattr(existing_entity, 'unique_id', None) or 
                          str(existing_entity))
            _LOGGER.warning(
                'An output listener with key "%s" is overriding previous entity "%s"',
                conf_key,
                existing_id,
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

    def minutely_display_update(self, _):
        """Minutely display update callback - only updates entity displays, no recalculation."""
        _LOGGER.debug("Minutely display update for %s", self.device_name)

        # Only notify listeners to update their display, don't recalculate optimization
        for listener in self._output_listeners.values():
            listener.update_callback()

    async def _on_price_entity_change(self, event):
        """Recalculate when tomorrow's prices become available."""
        new_state = event.data.get("new_state")
        old_state = event.data.get("old_state")
        if not new_state or not old_state:
            return
        new_valid = new_state.attributes.get("tomorrow_valid", False)
        old_valid = old_state.attributes.get("tomorrow_valid", False)
        if new_valid and not old_valid:
            _LOGGER.info("Tomorrow prices now available for %s, recalculating", self.device_name)
            await self._hass.async_add_executor_job(self.update, True)

    def update(self, force_price_update: bool = True):
        """Optimizer update call function."""
        # Prevent updates after cleanup
        if self._is_cleaned_up:
            _LOGGER.debug("Skipping update for %s - optimizer has been cleaned up", self.device_name)
            return
        
        try:
            _LOGGER.info("Starting optimizer update for %s (mode: %s, duration: %s)", 
                         self.device_name, self.mode, self.duration)

            # Update price data (unless we're using cached data)
            if force_price_update:
                price_updated = self._prices_entity.update(self._hass)
                _LOGGER.debug("Price entity update result: %s, valid: %s", price_updated, self._prices_entity.valid)
            else:
                # Using cached data, just check if valid
                price_updated = self._prices_entity.valid
                _LOGGER.debug("Using cached price data, valid: %s", self._prices_entity.valid)

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

            # Update period statuses and calculate new periods incrementally
            now = dt_util.now()
            self._update_period_statuses(now)

            # Check for configuration changes and handle them
            if self._check_configuration_changed():
                self._handle_configuration_change(now)

            # Clear ALL future periods before recalculation (not just planned ones)
            # This ensures we recalculate with current configuration
            initial_period_count = len(self._persistent_periods)
            self._clear_future_periods(now)
            # Also clear any active/planned periods that might conflict with new config
            # Keep only completed periods that are fully in the past
            self._persistent_periods = [
                period for period in self._persistent_periods
                if period.status == "completed" and period.end_time < now
            ]
            cleared_count = initial_period_count - len(self._persistent_periods)
            if cleared_count > 0:
                _LOGGER.debug("Cleared %d periods total (including active/planned) for %s before recalculation",
                             cleared_count, self.device_name)

            # Calculate ALL optimal periods from now until end of available data
            new_periods = self._calculate_all_periods(now)

            # Add new periods to persistent storage
            if new_periods:
                # Apply merging logic for absolute mode (daily mode keeps separate periods per day)
                if self.mode == CONF_MODE_ABSOLUTE:
                    merged_periods = self._merge_consecutive_periods(new_periods)
                    _LOGGER.debug("Merged %d new periods into %d consecutive periods for %s",
                                len(new_periods), len(merged_periods), self.device_name)
                else:
                    merged_periods = new_periods

                # Add all merged periods to persistent storage
                persistent_new_periods = [
                    PersistentOptimalPeriod.from_optimal_period(period, self.device_name)
                    for period in merged_periods
                ]
                self._persistent_periods.extend(persistent_new_periods)
                _LOGGER.debug("Added %d periods for %s", len(merged_periods), self.device_name)

            # Clean up old periods and save to cache
            self._cleanup_old_periods()
            self.save_period_cache()

            # Update current_optimal_periods for backwards compatibility
            self._update_current_optimal_periods()

            self._last_update = now

            # Notify all listeners
            _LOGGER.debug("Notifying %d listeners for %s", len(self._output_listeners), self.device_name)
            for listener in self._output_listeners.values():
                listener.update_callback()
            
            _LOGGER.info("Completed optimizer update for %s", self.device_name)
                
        except Exception as e:
            _LOGGER.error("Error during optimizer update for %s: %s", self.device_name, e, exc_info=True)
            # Set error state but still notify listeners so they can show error
            self._optimizer_status.status = OptimizerStates.Error
            self._optimizer_status.running_text = f"Update error: {str(e)[:50]}"
            # Still notify listeners so they can update their state
            for listener in self._output_listeners.values():
                try:
                    listener.update_callback()
                except Exception:
                    pass  # Ignore errors in listener callbacks

    def _calculate_absolute_periods(self, now: dt.datetime) -> list[OptimalPeriod]:
        """Calculate optimal periods for absolute mode using 15-minute granularity."""
        periods = []

        # Look ahead up to 48 hours, capped at actual data availability
        data_end = self._prices_entity.data_end
        end_time = min(now + dt.timedelta(hours=48), data_end) if data_end else now + dt.timedelta(hours=48)

        step = self._prices_entity.price_interval
        expected_count = self.duration * self._prices_entity.entries_per_hour

        # Round current time down to nearest price interval boundary
        interval_minutes = max(1, int(step.total_seconds() / 60))
        rounded_minutes = (now.minute // interval_minutes) * interval_minutes
        current_slot = now.replace(minute=rounded_minutes, second=0, microsecond=0)

        duration_timedelta = dt.timedelta(hours=self.duration)

        while current_slot < end_time:
            # Check if this slot is within time window (if enabled)
            if self.time_window_enabled and not self._is_in_time_window(current_slot):
                current_slot += step
                continue

            # Get price group for the duration starting at this slot
            period_end = current_slot + duration_timedelta
            price_group = self._prices_entity.get_prices_group(current_slot, period_end)

            if price_group.valid and price_group.count >= expected_count and price_group.average <= self.price_threshold:
                periods.append(OptimalPeriod(
                    start_time=current_slot,
                    end_time=period_end,
                    average_price=price_group.average
                ))

            current_slot += step

        return periods

    def _calculate_daily_periods(self, now: dt.datetime) -> list[OptimalPeriod]:
        """Calculate optimal periods for daily mode."""
        if self.slot_type == CONF_SLOT_TYPE_CONSECUTIVE:
            return self._calculate_daily_consecutive(now)
        else:
            return self._calculate_daily_separate(now)

    def _calculate_daily_consecutive(self, now: dt.datetime) -> list[OptimalPeriod]:
        """Find cheapest consecutive slots across full price range with min spacing.

        Searches the entire available price window (up to 48h) instead of per
        calendar day, so slots can cross midnight (e.g. 22:00-02:00).  A minimum
        gap between slots prevents back-to-back runs.
        """
        periods = []
        duration_td = dt.timedelta(hours=self.duration)
        min_gap = dt.timedelta(hours=max(0, 24 - self.duration - 7))

        step = self._prices_entity.price_interval
        interval_minutes = max(1, int(step.total_seconds() / 60))
        rounded_minutes = (now.minute // interval_minutes) * interval_minutes
        earliest = now.replace(minute=rounded_minutes, second=0, microsecond=0)
        # Cap at actual data availability to avoid phantom slots at the edge
        data_end = self._prices_entity.data_end
        search_end = min(now + dt.timedelta(hours=48), data_end) if data_end else now + dt.timedelta(hours=48)

        _LOGGER.info(
            "[%s] daily_consecutive: duration=%sh, gap=%sh, interval=%dmin, "
            "entries_per_hour=%d, data_end=%s, search_end=%s, price_count=%d",
            self.device_name, self.duration, min_gap.total_seconds() / 3600,
            interval_minutes, self._prices_entity.entries_per_hour,
            data_end.strftime('%Y-%m-%d %H:%M') if data_end else 'None',
            search_end.strftime('%Y-%m-%d %H:%M'),
            len(self._prices_entity._all_prices),
        )

        # Respect spacing from recently completed slots:
        # At this point self._persistent_periods only contains completed past
        # periods (cleared by update() before calling _calculate_all_periods).
        if self._persistent_periods:
            last_end = max(p.end_time for p in self._persistent_periods)
            spacing_boundary = last_end + min_gap
            if spacing_boundary > earliest:
                earliest = spacing_boundary
                _LOGGER.info(
                    "[%s] Spacing from last completed slot pushes earliest to %s",
                    self.device_name, earliest.strftime('%Y-%m-%d %H:%M'))

        while earliest + duration_td <= search_end:
            period = self._find_consecutive_daily_period(earliest, search_end)
            if not period:
                break
            _LOGGER.info(
                "[%s] Selected slot: %s - %s, avg price: %.4f",
                self.device_name,
                period.start_time.strftime('%Y-%m-%d %H:%M'),
                period.end_time.strftime('%Y-%m-%d %H:%M'),
                period.average_price,
            )
            periods.append(period)
            earliest = period.end_time + min_gap

        return periods

    def _calculate_daily_separate(self, now: dt.datetime) -> list[OptimalPeriod]:
        """Calculate separate daily periods (per calendar day)."""
        periods = []
        for day_offset in [0, 1]:
            day_start = (now + dt.timedelta(days=day_offset)).replace(
                hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + dt.timedelta(days=1)
            result = self._find_separate_daily_periods(day_start, day_end)
            if result:
                periods.extend(result)
        return periods

    def _find_consecutive_daily_period(self, day_start: dt.datetime, day_end: dt.datetime) -> OptimalPeriod | None:
        """Find the cheapest consecutive period matching price data resolution."""
        best_period = None
        best_price = float('inf')
        candidates = []  # collect top candidates for logging

        step = self._prices_entity.price_interval
        expected_count = self.duration * self._prices_entity.entries_per_hour

        # Align start to price interval boundary
        interval_seconds = int(step.total_seconds())
        start_ts = int(day_start.timestamp())
        remainder = start_ts % interval_seconds
        if remainder:
            current_slot = day_start + dt.timedelta(seconds=interval_seconds - remainder)
        else:
            current_slot = day_start

        duration_timedelta = dt.timedelta(hours=self.duration)
        evaluated = 0
        skipped_count = 0
        skipped_window = 0

        while current_slot + duration_timedelta <= day_end:
            # Check if this slot is within time window (if enabled)
            if self.time_window_enabled and not self._is_in_time_window(current_slot):
                current_slot += step
                skipped_window += 1
                continue

            period_end = current_slot + duration_timedelta
            price_group = self._prices_entity.get_prices_group(current_slot, period_end)
            evaluated += 1

            if not price_group.valid or price_group.count < expected_count:
                skipped_count += 1
                current_slot += step
                continue

            avg = price_group.average
            candidates.append((current_slot, period_end, avg, price_group.count))

            if avg < best_price:
                best_price = avg
                best_period = OptimalPeriod(
                    start_time=current_slot,
                    end_time=period_end,
                    average_price=avg,
                )

            current_slot += step

        # Log top 5 cheapest candidates
        candidates.sort(key=lambda c: c[2])
        top = candidates[:5]
        _LOGGER.info(
            "[%s] Search %s→%s: evaluated=%d, valid=%d, skipped_count=%d, skipped_window=%d, expected_count=%d",
            self.device_name,
            day_start.strftime('%m-%d %H:%M'), day_end.strftime('%m-%d %H:%M'),
            evaluated, len(candidates), skipped_count, skipped_window, expected_count,
        )
        for i, (s, e, avg, cnt) in enumerate(top):
            _LOGGER.info(
                "[%s]   #%d  %s - %s  avg=%.4f  count=%d",
                self.device_name, i + 1,
                s.strftime('%m-%d %H:%M'), e.strftime('%m-%d %H:%M'), avg, cnt,
            )

        return best_period

    def _find_separate_daily_periods(self, day_start: dt.datetime, day_end: dt.datetime) -> list[OptimalPeriod]:
        """Find the cheapest separate 15-minute slots in a day."""
        # Get all 15-minute prices for the day
        slot_prices = []
        current_slot = day_start

        while current_slot < day_end:
            # Check if this slot is within time window (if enabled)
            if self.time_window_enabled and not self._is_in_time_window(current_slot):
                current_slot += dt.timedelta(minutes=15)
                continue

            slot_end = current_slot + dt.timedelta(minutes=15)
            price_group = self._prices_entity.get_prices_group(current_slot, slot_end)

            if price_group.valid:
                slot_prices.append({
                    'slot': current_slot,
                    'price': price_group.average
                })

            current_slot += dt.timedelta(minutes=15)

        # Sort by price and take the cheapest slots
        # Duration in hours * 4 slots per hour = total slots needed
        total_slots_needed = int(self.duration * 4)
        slot_prices.sort(key=lambda x: x['price'])
        cheapest_slots = slot_prices[:min(total_slots_needed, len(slot_prices))]

        _LOGGER.debug("Separate mode for %s: need %d slots (%.1f hours), found %d cheapest slots",
                     self.device_name, total_slots_needed, self.duration, len(cheapest_slots))

        # Create periods for each 15-minute slot
        periods = []
        for slot_data in cheapest_slots:
            periods.append(OptimalPeriod(
                start_time=slot_data['slot'],
                end_time=slot_data['slot'] + dt.timedelta(minutes=15),
                average_price=slot_data['price']
            ))
            _LOGGER.debug("Selected separate slot for %s: %s-%s, price: %.3f",
                         self.device_name, slot_data['slot'].strftime('%H:%M'),
                         (slot_data['slot'] + dt.timedelta(minutes=15)).strftime('%H:%M'),
                         slot_data['price'])

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


    def _clear_future_periods(self, now: dt.datetime) -> None:
        """Clear future planned periods before recalculation."""
        initial_count = len(self._persistent_periods)

        # Remove all future planned periods (keep completed and active ones)
        self._persistent_periods = [
            period for period in self._persistent_periods
            if period.status in ["completed", "active"] or period.start_time <= now
        ]

        removed_count = initial_count - len(self._persistent_periods)
        if removed_count > 0:
            _LOGGER.debug("Cleared %d future periods for %s before recalculation",
                         removed_count, self.device_name)

    def _calculate_all_periods(self, now: dt.datetime) -> list[OptimalPeriod]:
        """Calculate ALL optimal periods from now until end of available data."""
        if self.mode == CONF_MODE_ABSOLUTE:
            return self._calculate_absolute_periods(now)
        elif self.mode == CONF_MODE_DAILY:
            return self._calculate_daily_periods(now)
        else:
            return []

    def _update_period_statuses(self, now: dt.datetime) -> None:
        """Update status of all persistent periods based on current time."""
        for period in self._persistent_periods:
            period.update_status(now)

    def _get_config_hash(self) -> str:
        """Generate a hash of the current configuration to detect changes."""
        import hashlib

        # Include all configuration parameters that affect optimization
        config_data = {
            'mode': self.mode,
            'duration': self.duration,
            'price_threshold': self.price_threshold,
            'slot_type': self.slot_type,
            'time_window_enabled': self.time_window_enabled,
            'time_window': self.time_window,
        }

        # Include global fee settings in configuration hash (using cached values)
        config_data.update({
            'global_tax_percentage': self._cached_fee_settings['tax_percentage'],
            'global_provider_fee': self._cached_fee_settings['provider_fee'],
            'global_network_fee': self._cached_fee_settings['network_fee'],
        })

        config_str = str(sorted(config_data.items()))
        return hashlib.md5(config_str.encode()).hexdigest()

    def _check_configuration_changed(self) -> bool:
        """Check if configuration has changed since last calculation."""
        current_hash = self._get_config_hash()
        if current_hash != self._last_config_hash:
            _LOGGER.debug("Configuration change detected for %s", self.device_name)
            self._last_config_hash = current_hash
            return True
        return False

    def _handle_configuration_change(self, now: dt.datetime) -> None:
        """Handle configuration changes by invalidating future periods."""
        _LOGGER.info("Configuration changed for %s, invalidating future periods", self.device_name)

        # Mark all future (planned) periods as cancelled
        for period in self._persistent_periods:
            if period.status == "planned" and period.start_time > now:
                period.status = "cancelled"
                _LOGGER.debug("Cancelled future period: %s", period)


        # Save the updated periods
        self.save_period_cache()

    def _merge_consecutive_periods(self, periods: list[OptimalPeriod]) -> list[OptimalPeriod]:
        """Merge consecutive or overlapping periods into single periods."""
        if not periods:
            return []

        # Sort periods by start time
        sorted_periods = sorted(periods, key=lambda p: p.start_time)
        merged = []
        current = sorted_periods[0]

        for next_period in sorted_periods[1:]:
            # Check if periods are consecutive or overlapping
            if current.end_time >= next_period.start_time:
                # Merge periods - extend current to include next period
                # Calculate weighted average price for merged period
                current_duration = (current.end_time - current.start_time).total_seconds() / 3600
                next_duration = (next_period.end_time - next_period.start_time).total_seconds() / 3600
                total_duration = current_duration + next_duration

                if total_duration > 0:
                    weighted_price = (
                        current.average_price * current_duration +
                        next_period.average_price * next_duration
                    ) / total_duration
                else:
                    weighted_price = current.average_price

                # Create merged period
                current = OptimalPeriod(
                    start_time=current.start_time,
                    end_time=max(current.end_time, next_period.end_time),
                    average_price=weighted_price
                )
            else:
                # Periods are not consecutive - add current and start new
                merged.append(current)
                current = next_period

        # Add the last period
        merged.append(current)

        _LOGGER.debug("Merged %d periods into %d consecutive periods", len(periods), len(merged))
        return merged

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

    def load_period_cache(self) -> bool:
        """Load persistent periods from cache file."""
        if not self._period_cache_file or not self._period_cache_file.exists():
            _LOGGER.debug("No period cache file found at %s", self._period_cache_file)
            return False

        try:
            # Check file size first - if it's suspiciously small or zero, skip
            file_size = self._period_cache_file.stat().st_size
            if file_size == 0:
                _LOGGER.debug("Period cache file is empty, skipping: %s", self._period_cache_file)
                return False
            
            with open(self._period_cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            # Check if cache is for our device and is recent enough (max 72 hours old)
            if (cache_data.get('device_id') != self.device_name or
                not self._is_period_cache_valid(cache_data.get('timestamp'))):
                _LOGGER.debug("Period cache invalid or too old for device %s", self.device_name)
                return False

            # Load persistent periods and update their status
            self._persistent_periods = cache_data.get('periods', [])

            # Update period statuses based on current time
            now = dt_util.now()
            for period in self._persistent_periods:
                period.update_status(now)

            # Update current_optimal_periods for backwards compatibility
            self._update_current_optimal_periods()

            _LOGGER.debug("Successfully loaded %d persistent periods for %s",
                         len(self._persistent_periods), self.device_name)
            return True

        except (pickle.PickleError, KeyError, OSError, EOFError, ValueError) as e:
            # If cache is corrupted, delete it and continue without cache
            _LOGGER.warning("Failed to load period cache file (corrupted?): %s. Removing corrupted cache.", e)
            try:
                self._period_cache_file.unlink()
            except OSError:
                pass  # Ignore errors deleting corrupted cache
            return False

    def save_period_cache(self) -> None:
        """Save persistent periods to cache file."""
        if not self._period_cache_file:
            return

        try:
            cache_data = {
                'device_id': self.device_name,
                'timestamp': dt_util.now().isoformat(),
                'periods': self._persistent_periods,
            }

            # Ensure cache directory exists
            self._period_cache_file.parent.mkdir(parents=True, exist_ok=True)

            # Use atomic write: write to temp file first, then rename
            temp_file = self._period_cache_file.with_suffix('.tmp')
            # Write to temp file
            with open(temp_file, 'wb') as f:
                pickle.dump(cache_data, f)
            # Ensure temp file was written successfully before renaming
            if not temp_file.exists():
                raise OSError(f"Temp file was not created: {temp_file}")
            # Atomic rename (works on most filesystems)
            temp_file.replace(self._period_cache_file)
            # Verify the rename succeeded
            if not self._period_cache_file.exists():
                raise OSError(f"Period cache file was not created after rename: {self._period_cache_file}")

            _LOGGER.debug("Saved %d persistent periods to cache for %s",
                         len(self._persistent_periods), self.device_name)

        except (OSError, pickle.PickleError) as e:
            _LOGGER.warning("Failed to save period cache file: %s", e)
            # Clean up temp file if it exists
            try:
                temp_file = self._period_cache_file.with_suffix('.tmp')
                if temp_file.exists():
                    temp_file.unlink()
            except OSError:
                pass

    def _is_period_cache_valid(self, timestamp_str: str) -> bool:
        """Check if period cache timestamp is recent enough."""
        if not timestamp_str:
            return False

        try:
            cache_time = dt.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            if cache_time.tzinfo is None:
                cache_time = cache_time.replace(tzinfo=dt_util.UTC)

            # Cache is valid for 72 hours (periods can span multiple days)
            max_age = dt.timedelta(hours=72)
            return dt_util.now() - cache_time <= max_age

        except (ValueError, TypeError):
            return False

    def _update_current_optimal_periods(self) -> None:
        """Update current_optimal_periods from persistent periods for backwards compatibility."""
        now = dt_util.now()
        # Only include active and planned periods in current list
        active_periods = [
            p.to_optimal_period() for p in self._persistent_periods
            if p.status in ["planned", "active"] and p.end_time > now
        ]
        self._current_optimal_periods = active_periods

    def _cleanup_old_periods(self) -> None:
        """Remove old completed periods to prevent cache from growing too large."""
        cutoff_time = dt_util.now() - dt.timedelta(days=7)  # Keep 7 days of history
        initial_count = len(self._persistent_periods)

        # Keep only periods that are not completed or are recent
        self._persistent_periods = [
            p for p in self._persistent_periods
            if p.status != "completed" or p.end_time > cutoff_time
        ]

        removed_count = initial_count - len(self._persistent_periods)
        if removed_count > 0:
            _LOGGER.debug("Cleaned up %d old periods for %s", removed_count, self.device_name)

    def set_unavailable(self) -> None:
        """Set output state to unavailable."""
        _LOGGER.debug("Setting output states to unavailable")
        for listener in self._output_listeners.values():
            listener.update_callback()


class PricesEntity:
    """Representation for Nordpool state."""

    def __init__(self, unique_id: str, hass: HomeAssistant = None, optimizer: 'NordpoolOptimizer' = None) -> None:
        """Initialize state tracker."""
        self._unique_id = unique_id
        self._np = None
        self._hass = hass
        self._optimizer = optimizer
        self._cache_file = None
        if hass:
            self._cache_file = Path(hass.config.config_dir) / "nordpool_optimizer_cache.pkl"

    def as_dict(self):
        """For diagnostics serialization."""
        return self.__dict__

    def _get_global_fee_settings(self) -> dict[str, float]:
        """Get global fee settings from optimizer's cached values."""
        if self._optimizer and hasattr(self._optimizer, '_cached_fee_settings'):
            return self._optimizer._cached_fee_settings

        # Fallback to defaults if optimizer not available
        return {
            "tax_percentage": DEFAULT_TAX_PERCENTAGE,
            "provider_fee": DEFAULT_PROVIDER_FEE,
            "network_fee": DEFAULT_NETWORK_FEE,
        }

    def _apply_global_fees(self, base_price: float) -> float:
        """Apply global fees to a base price.

        Formula: new_price = (1 + tax/100) * (price + provider_fee) + network_fee
        """
        fees = self._get_global_fee_settings()

        # Apply the formula
        tax_multiplier = 1 + fees["tax_percentage"] / 100
        price_with_provider_fee = base_price + fees["provider_fee"]
        final_price = tax_multiplier * price_with_provider_fee + fees["network_fee"]

        return final_price

    @property
    def unique_id(self) -> str:
        """Get the unique id."""
        return self._unique_id

    @property
    def valid(self) -> bool:
        """Get if data is valid."""
        return self._np is not None

    @property
    def price_interval(self) -> dt.timedelta:
        """Detect interval between price entries (e.g. 15min or 1h)."""
        prices = self._all_prices
        if len(prices) >= 2:
            return prices[1]["start"] - prices[0]["start"]
        return dt.timedelta(hours=1)

    @property
    def entries_per_hour(self) -> int:
        """Number of price entries per hour (4 for 15-min, 1 for hourly)."""
        interval_seconds = self.price_interval.total_seconds()
        return max(1, int(3600 / interval_seconds))

    @property
    def data_end(self) -> dt.datetime | None:
        """End of available price data (last entry start + interval)."""
        prices = self._all_prices
        if prices:
            return prices[-1]["start"] + self.price_interval
        return None

    @property
    def _all_prices(self):
        base_prices = []

        if np_prices := self._np.attributes.get("raw_today"):
            # For Nordpool format – copy to avoid mutating the original list
            base_prices = list(np_prices)
            if self._np.attributes["tomorrow_valid"]:
                base_prices += self._np.attributes["raw_tomorrow"]
        elif e_prices := self._np.attributes.get("prices"):  # noqa: RET505
            # For ENTSO-e format
            base_prices = [
                {"start": dt_util.parse_datetime(ep["time"]), "value": ep["price"]}
                for ep in e_prices
            ]

        # Apply global fees to all prices
        fee_adjusted_prices = []
        for price_data in base_prices:
            adjusted_price_data = price_data.copy()
            adjusted_price_data["value"] = self._apply_global_fees(price_data["value"])
            fee_adjusted_prices.append(adjusted_price_data)

        return fee_adjusted_prices

    @property
    def current_price_attr(self):
        """Get the current price attribute with global fees applied."""
        if self._np is not None:
            if current := self._np.attributes.get("current_price"):
                # For Nordpool format - apply fees to base price
                return self._apply_global_fees(current)
            else:  # noqa: RET505
                # For general, find in list (fees already applied in _all_prices)
                now = dt_util.now()
                for price in self._all_prices:
                    if (
                        price["start"] < now
                        and price["start"] + dt.timedelta(hours=1) > now
                    ):
                        return price["value"]
        return None

    def load_cache(self) -> bool:
        """Load price data from cache file using pickle."""
        if not self._cache_file or not self._cache_file.exists():
            _LOGGER.debug("No cache file found at %s", self._cache_file)
            return False

        try:
            # Check file size first - if it's suspiciously small or zero, skip
            file_size = self._cache_file.stat().st_size
            if file_size == 0:
                _LOGGER.debug("Cache file is empty, skipping: %s", self._cache_file)
                return False
            
            with open(self._cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            # Check if cache is for our entity and is recent enough (max 6 hours old)
            if (cache_data.get('entity_id') != self._unique_id or
                not self._is_cache_valid(cache_data.get('timestamp'))):
                _LOGGER.debug("Cache invalid or too old for entity %s", self._unique_id)
                return False

            # Create a mock state object from cached data (datetime objects preserved by pickle)
            class MockState:
                def __init__(self, attributes):
                    self.attributes = attributes

            self._np = MockState(cache_data['attributes'])
            _LOGGER.debug("Successfully loaded cached price data for %s", self._unique_id)
            return True

        except (pickle.PickleError, KeyError, OSError, EOFError, ValueError) as e:
            # If cache is corrupted, delete it and continue without cache
            _LOGGER.warning("Failed to load cache file (corrupted?): %s. Removing corrupted cache.", e)
            try:
                self._cache_file.unlink()
            except OSError:
                pass  # Ignore errors deleting corrupted cache
            return False

    def save_cache(self) -> None:
        """Save current price data to cache file using pickle."""
        if not self._cache_file or not self._np:
            return

        try:
            cache_data = {
                'entity_id': self._unique_id,
                'timestamp': dt_util.now().isoformat(),
                'attributes': dict(self._np.attributes)
            }

            # Ensure cache directory exists
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)

            # Use atomic write: write to temp file first, then rename
            temp_file = self._cache_file.with_suffix('.tmp')
            # Write to temp file
            with open(temp_file, 'wb') as f:
                pickle.dump(cache_data, f)
            # Ensure temp file was written successfully before renaming
            if not temp_file.exists():
                raise OSError(f"Temp file was not created: {temp_file}")
            # Atomic rename (works on most filesystems)
            temp_file.replace(self._cache_file)
            # Verify the rename succeeded
            if not self._cache_file.exists():
                raise OSError(f"Cache file was not created after rename: {self._cache_file}")

            _LOGGER.debug("Saved price data cache for %s", self._unique_id)

        except (OSError, pickle.PickleError) as e:
            _LOGGER.warning("Failed to save cache file: %s", e)
            # Clean up temp file if it exists
            try:
                temp_file = self._cache_file.with_suffix('.tmp')
                if temp_file.exists():
                    temp_file.unlink()
            except OSError:
                pass

    def _is_cache_valid(self, timestamp_str: str) -> bool:
        """Check if cache timestamp is recent enough."""
        if not timestamp_str:
            return False

        try:
            cache_time = dt_util.parse_datetime(timestamp_str)
            if not cache_time:
                return False

            # Cache is valid if less than 6 hours old
            age = dt_util.now() - cache_time
            return age.total_seconds() < 6 * 3600

        except (ValueError, TypeError):
            return False

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
            # Save to cache when successfully updated
            self.save_cache()

        return self._np is not None

    def get_prices_group(
        self, start: dt.datetime, end: dt.datetime
    ) -> NordpoolPricesGroup:
        """Get a range of prices from NP given the start and end datetimes."""
        selected = []
        for p in self._all_prices:
            price_start = p["start"]
            # Include price slots that start within our time range [start, end)
            if start <= price_start < end:
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
    def count(self) -> int:
        """Number of price data points in the group."""
        return len(self._prices)

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


@dataclass
class PersistentOptimalPeriod:
    """Represents an optimal time period with persistence and status tracking."""
    start_time: dt.datetime
    end_time: dt.datetime
    average_price: float
    status: str  # 'planned', 'active', 'completed', 'cancelled'
    calculation_time: dt.datetime  # When this period was calculated
    device_id: str

    def to_optimal_period(self) -> OptimalPeriod:
        """Convert to legacy OptimalPeriod for backwards compatibility."""
        return OptimalPeriod(self.start_time, self.end_time, self.average_price)

    @classmethod
    def from_optimal_period(cls, period: OptimalPeriod, device_id: str, status: str = "planned") -> "PersistentOptimalPeriod":
        """Create from legacy OptimalPeriod."""
        return cls(
            start_time=period.start_time,
            end_time=period.end_time,
            average_price=period.average_price,
            status=status,
            calculation_time=dt_util.now(),
            device_id=device_id
        )

    def update_status(self, now: dt.datetime) -> None:
        """Update status based on current time."""
        if self.status == "cancelled":
            return  # Don't change cancelled status

        if now >= self.end_time:
            self.status = "completed"
        elif now >= self.start_time:
            self.status = "active"
        else:
            self.status = "planned"

    def __str__(self) -> str:
        """String representation."""
        return f"PersistentOptimalPeriod({self.start_time.strftime('%H:%M')}-{self.end_time.strftime('%H:%M')}, {self.average_price:.3f}, {self.status})"


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