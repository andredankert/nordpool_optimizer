"""Config flow for Nordpool Optimizer integration."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import ATTR_NAME, ATTR_UNIT_OF_MEASUREMENT
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

from .const import (
    CONF_DEVICE_NAME,
    CONF_DURATION,
    CONF_GRAPH_HOURS_AHEAD,
    CONF_MODE,
    CONF_MODE_ABSOLUTE,
    CONF_MODE_DAILY,
    CONF_MODE_LIST,
    CONF_NETWORK_FEE,
    CONF_PRICE_THRESHOLD,
    CONF_PRICES_ENTITY,
    CONF_PROVIDER_FEE,
    CONF_SLOT_TYPE,
    CONF_SLOT_TYPE_CONSECUTIVE,
    CONF_SLOT_TYPE_LIST,
    CONF_SLOT_TYPE_SEPARATE,
    CONF_TAX_PERCENTAGE,
    CONF_TIME_WINDOW,
    CONF_TIME_WINDOW_ENABLED,
    DEFAULT_GRAPH_HOURS,
    DEFAULT_NETWORK_FEE,
    DEFAULT_PROVIDER_FEE,
    DEFAULT_TAX_PERCENTAGE,
    DOMAIN,
    GRAPH_HOURS_OPTIONS,
    NAME_FILE_READER,
    PATH_FILE_READER,
)
from .helpers import get_np_from_file

_LOGGER = logging.getLogger(__name__)

ENTOSOE_DOMAIN = None
try:
    from ..entsoe.const import DOMAIN as ENTOSOE_DOMAIN
except ImportError:
    _LOGGER.warning("Could not import ENTSO-e integration")

NORDPOOL_DOMAIN = None
try:
    from ..nordpool import DOMAIN as NORDPOOL_DOMAIN
except ImportError:
    _LOGGER.warning("Could not import Nord Pool integration")


class NordpoolOptimizerConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Nordpool Optimizer config flow."""

    VERSION = 3
    MINOR_VERSION = 0
    data = None
    options = None

    @staticmethod
    def async_get_options_flow(config_entry):
        """Create the options flow."""
        return NordpoolOptimizerOptionsFlow(config_entry)

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle initial user step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            self.data = user_input

            # Get available price entities
            price_entities = self._get_price_entities()
            if self.data[CONF_PRICES_ENTITY] not in price_entities:
                errors["base"] = "invalid_price_entity"
            else:
                # Get unit of measurement from price entity
                try:
                    if self.data[CONF_PRICES_ENTITY] == NAME_FILE_READER:
                        np_entity = get_np_from_file(PATH_FILE_READER)
                    else:
                        np_entity = self.hass.states.get(self.data[CONF_PRICES_ENTITY])

                    self.options = {
                        ATTR_UNIT_OF_MEASUREMENT: np_entity.attributes.get(
                            ATTR_UNIT_OF_MEASUREMENT
                        )
                    }
                except (IndexError, KeyError, AttributeError):
                    _LOGGER.warning("Could not extract currency from price entity")
                    self.options = {}

                # Set unique ID based on device name and price entity
                await self.async_set_unique_id(
                    f"{self.data[CONF_DEVICE_NAME]}_{self.data[CONF_PRICES_ENTITY]}"
                    .lower()
                    .replace(" ", "_")
                )
                self._abort_if_unique_id_configured()

                return await self.async_step_mode()

        # Get available price entities
        price_entities = self._get_price_entities()

        data_schema = vol.Schema(
            {
                vol.Required(CONF_DEVICE_NAME): str,
                vol.Required(CONF_PRICES_ENTITY): vol.In(price_entities),
            }
        )

        return self.async_show_form(
            step_id="user",
            data_schema=data_schema,
            errors=errors,
        )

    async def async_step_mode(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle mode selection step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            self.data.update(user_input)

            if self.data[CONF_MODE] == CONF_MODE_ABSOLUTE:
                return await self.async_step_absolute()
            else:
                return await self.async_step_daily()

        data_schema = vol.Schema(
            {
                vol.Required(CONF_MODE): vol.In(CONF_MODE_LIST),
                vol.Required(CONF_DURATION, default=3): vol.All(
                    int, vol.Range(min=1, max=12)
                ),
            }
        )

        return self.async_show_form(
            step_id="mode",
            data_schema=data_schema,
            errors=errors,
            description_placeholders={
                "device_name": self.data[CONF_DEVICE_NAME],
            },
        )

    async def async_step_absolute(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle absolute mode configuration."""
        errors: dict[str, str] = {}

        if user_input is not None:
            self.data.update(user_input)
            return await self.async_step_time_window()

        data_schema = vol.Schema(
            {
                vol.Required(CONF_PRICE_THRESHOLD, default=0.5): vol.All(
                    vol.Coerce(float), vol.Range(min=-10.0, max=10.0)
                ),
            }
        )

        return self.async_show_form(
            step_id="absolute",
            data_schema=data_schema,
            errors=errors,
            description_placeholders={
                "device_name": self.data[CONF_DEVICE_NAME],
                "duration": str(self.data[CONF_DURATION]),
            },
        )

    async def async_step_daily(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle daily mode configuration."""
        errors: dict[str, str] = {}

        if user_input is not None:
            self.data.update(user_input)
            return await self.async_step_time_window()

        data_schema = vol.Schema(
            {
                vol.Required(CONF_SLOT_TYPE, default=CONF_SLOT_TYPE_CONSECUTIVE): vol.In(
                    CONF_SLOT_TYPE_LIST
                ),
            }
        )

        return self.async_show_form(
            step_id="daily",
            data_schema=data_schema,
            errors=errors,
            description_placeholders={
                "device_name": self.data[CONF_DEVICE_NAME],
                "duration": str(self.data[CONF_DURATION]),
            },
        )

    async def async_step_time_window(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle optional time window configuration."""
        errors: dict[str, str] = {}

        if user_input is not None:
            self.data.update(user_input)

            # Validate time window format if enabled
            if self.data.get(CONF_TIME_WINDOW_ENABLED) and self.data.get(
                CONF_TIME_WINDOW
            ):
                try:
                    start_str, end_str = self.data[CONF_TIME_WINDOW].split("-")
                    # Validate time format
                    start_hour, start_min = map(int, start_str.split(":"))
                    end_hour, end_min = map(int, end_str.split(":"))

                    if not (0 <= start_hour <= 23 and 0 <= start_min <= 59):
                        errors["time_window"] = "invalid_start_time"
                    elif not (0 <= end_hour <= 23 and 0 <= end_min <= 59):
                        errors["time_window"] = "invalid_end_time"
                    else:
                        return await self.async_step_graph_settings()
                except (ValueError, AttributeError):
                    errors["time_window"] = "invalid_time_format"
            else:
                # No time window or disabled
                return await self.async_step_graph_settings()

        data_schema = vol.Schema(
            {
                vol.Optional(CONF_TIME_WINDOW_ENABLED, default=False): bool,
                vol.Optional(CONF_TIME_WINDOW, default="22:00-06:00"): str,
            }
        )

        return self.async_show_form(
            step_id="time_window",
            data_schema=data_schema,
            errors=errors,
            description_placeholders={
                "device_name": self.data[CONF_DEVICE_NAME],
                "mode": self.data[CONF_MODE],
            },
        )

    async def async_step_graph_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle graph settings configuration."""
        errors: dict[str, str] = {}

        if user_input is not None:
            self.data.update(user_input)
            return self.async_create_entry(
                title=self.data[CONF_DEVICE_NAME],
                data=self.data,
                options=self.options,
            )

        data_schema = vol.Schema(
            {
                vol.Optional(CONF_GRAPH_HOURS_AHEAD, default=DEFAULT_GRAPH_HOURS): vol.In(
                    GRAPH_HOURS_OPTIONS
                ),
            }
        )

        return self.async_show_form(
            step_id="graph_settings",
            data_schema=data_schema,
            errors=errors,
            description_placeholders={
                "device_name": self.data[CONF_DEVICE_NAME],
            },
        )

    def _get_price_entities(self) -> dict[str, str]:
        """Get available price entities."""
        price_entities = {}

        # Add file reader for testing
        price_entities[NAME_FILE_READER] = "File Reader (for testing)"

        try:
            # Look for Nordpool entities
            if NORDPOOL_DOMAIN:
                for entity_id in self.hass.states.async_entity_ids():
                    if entity_id.startswith(f"sensor.{NORDPOOL_DOMAIN}"):
                        state = self.hass.states.get(entity_id)
                        if state and "raw_today" in state.attributes:
                            price_entities[entity_id] = (
                                state.attributes.get("friendly_name") or entity_id
                            )

            # Look for ENTSO-e entities
            if ENTOSOE_DOMAIN:
                for entity_id in self.hass.states.async_entity_ids():
                    if entity_id.startswith(f"sensor.{ENTOSOE_DOMAIN}"):
                        state = self.hass.states.get(entity_id)
                        if state and "prices" in state.attributes:
                            price_entities[entity_id] = (
                                state.attributes.get("friendly_name") or entity_id
                            )

            # Fallback to any sensor with price data
            for entity_id in self.hass.states.async_entity_ids():
                if entity_id.startswith("sensor.") and entity_id not in price_entities:
                    state = self.hass.states.get(entity_id)
                    if state and ("raw_today" in state.attributes or "prices" in state.attributes):
                        price_entities[entity_id] = (
                            state.attributes.get("friendly_name") or entity_id
                        )

        except Exception as e:
            _LOGGER.exception("Error getting price entities: %s", e)

        return price_entities

    async def async_step_reconfigure(self, user_input: dict[str, Any] | None = None):
        """Handle reconfiguration of the config entry."""
        if user_input is not None:
            config_entry = self._get_reconfigure_entry()
            # Update entry data in-place; the update listener
            # (async_on_config_update) applies it to the live optimizer
            # without destroying entities.
            self.hass.config_entries.async_update_entry(
                config_entry,
                data={**config_entry.data, **user_input},
            )
            return self.async_abort(reason="reconfigure_successful")

        config_entry = self._get_reconfigure_entry()

        # Build schema with current values
        data_schema = vol.Schema(
            {
                vol.Required(
                    CONF_DEVICE_NAME, default=config_entry.data[CONF_DEVICE_NAME]
                ): str,
                vol.Required(
                    CONF_MODE, default=config_entry.data[CONF_MODE]
                ): vol.In(CONF_MODE_LIST),
                vol.Required(
                    CONF_DURATION, default=config_entry.data[CONF_DURATION]
                ): vol.All(int, vol.Range(min=1, max=12)),
            }
        )

        # Add mode-specific fields
        if config_entry.data[CONF_MODE] == CONF_MODE_ABSOLUTE:
            data_schema = data_schema.extend(
                {
                    vol.Required(
                        CONF_PRICE_THRESHOLD,
                        default=config_entry.data.get(CONF_PRICE_THRESHOLD, 0.5),
                    ): vol.All(vol.Coerce(float), vol.Range(min=-10.0, max=10.0)),
                }
            )
        else:
            data_schema = data_schema.extend(
                {
                    vol.Required(
                        CONF_SLOT_TYPE,
                        default=config_entry.data.get(
                            CONF_SLOT_TYPE, CONF_SLOT_TYPE_CONSECUTIVE
                        ),
                    ): vol.In(CONF_SLOT_TYPE_LIST),
                }
            )

        # Add time window fields
        data_schema = data_schema.extend(
            {
                vol.Optional(
                    CONF_TIME_WINDOW_ENABLED,
                    default=config_entry.data.get(CONF_TIME_WINDOW_ENABLED, False),
                ): bool,
                vol.Optional(
                    CONF_TIME_WINDOW,
                    default=config_entry.data.get(CONF_TIME_WINDOW, "22:00-06:00"),
                ): str,
            }
        )

        # Add graph settings
        data_schema = data_schema.extend(
            {
                vol.Optional(
                    CONF_GRAPH_HOURS_AHEAD,
                    default=config_entry.data.get(CONF_GRAPH_HOURS_AHEAD, DEFAULT_GRAPH_HOURS),
                ): vol.In(GRAPH_HOURS_OPTIONS),
            }
        )

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=data_schema,
        )


class NordpoolOptimizerOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for global fee settings."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage global fee options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        # Get current option values or defaults
        current_options = self.config_entry.options

        data_schema = vol.Schema(
            {
                vol.Optional(
                    CONF_TAX_PERCENTAGE,
                    default=current_options.get(CONF_TAX_PERCENTAGE, DEFAULT_TAX_PERCENTAGE),
                ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=100.0)),
                vol.Optional(
                    CONF_PROVIDER_FEE,
                    default=current_options.get(CONF_PROVIDER_FEE, DEFAULT_PROVIDER_FEE),
                ): vol.All(vol.Coerce(float), vol.Range(min=-10.0, max=10.0)),
                vol.Optional(
                    CONF_NETWORK_FEE,
                    default=current_options.get(CONF_NETWORK_FEE, DEFAULT_NETWORK_FEE),
                ): vol.All(vol.Coerce(float), vol.Range(min=-10.0, max=10.0)),
            }
        )

        return self.async_show_form(
            step_id="init",
            data_schema=data_schema,
            description_placeholders={
                "formula": "new_price = (1 + tax/100) × (price + provider_fee) + network_fee"
            },
        )