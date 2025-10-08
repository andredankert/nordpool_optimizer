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
    CONF_MODE,
    CONF_MODE_ABSOLUTE,
    CONF_MODE_DAILY,
    CONF_MODE_LIST,
    CONF_PRICE_THRESHOLD,
    CONF_PRICES_ENTITY,
    CONF_SLOT_TYPE,
    CONF_SLOT_TYPE_CONSECUTIVE,
    CONF_SLOT_TYPE_LIST,
    CONF_SLOT_TYPE_SEPARATE,
    CONF_TIME_WINDOW,
    CONF_TIME_WINDOW_ENABLED,
    DOMAIN,
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
                        return self.async_create_entry(
                            title=self.data[CONF_DEVICE_NAME],
                            data=self.data,
                            options=self.options,
                        )
                except (ValueError, AttributeError):
                    errors["time_window"] = "invalid_time_format"
            else:
                # No time window or disabled
                return self.async_create_entry(
                    title=self.data[CONF_DEVICE_NAME],
                    data=self.data,
                    options=self.options,
                )

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

    def _get_price_entities(self) -> dict[str, str]:
        """Get available price entities."""
        price_entities = {}

        # Add file reader for testing
        price_entities[NAME_FILE_READER] = "File Reader (for testing)"

        # Look for Nordpool entities
        if NORDPOOL_DOMAIN:
            for entity_id, state in self.hass.states.async_all().items():
                if (
                    entity_id.startswith(f"sensor.{NORDPOOL_DOMAIN}")
                    and "raw_today" in state.attributes
                ):
                    price_entities[entity_id] = (
                        state.attributes.get("friendly_name") or entity_id
                    )

        # Look for ENTSO-e entities
        if ENTOSOE_DOMAIN:
            for entity_id, state in self.hass.states.async_all().items():
                if (
                    entity_id.startswith(f"sensor.{ENTOSOE_DOMAIN}")
                    and "prices" in state.attributes
                ):
                    price_entities[entity_id] = (
                        state.attributes.get("friendly_name") or entity_id
                    )

        # Fallback to any sensor with price data
        for entity_id, state in self.hass.states.async_all().items():
            if entity_id.startswith("sensor.") and (
                "raw_today" in state.attributes or "prices" in state.attributes
            ):
                if entity_id not in price_entities:
                    price_entities[entity_id] = (
                        state.attributes.get("friendly_name") or entity_id
                    )

        return price_entities

    async def async_step_reconfigure(self, user_input: dict[str, Any] | None = None):
        """Handle reconfiguration of the config entry."""
        if user_input is not None:
            return self.async_update_reload_and_abort(
                self._get_reconfigure_entry(),
                data_updates=user_input,
            )

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

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=data_schema,
        )