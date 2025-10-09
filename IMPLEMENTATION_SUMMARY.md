# Nordpool Optimizer Implementation Summary

## Overview
This document summarizes the complete transformation from "Nordpool Planner" to "Nordpool Optimizer" and all subsequent improvements.

## Phase 1: Complete Architecture Transformation

### Original → New Architecture
- **From**: 6 entities per device (binary sensors, number sliders, separate sensors)
- **To**: 1 clean timer entity per device showing countdown/runtime

### Renaming & Structure
- **Directory**: `nordpool_planner` → `nordpool_optimizer`
- **Domain**: Changed throughout 47+ file locations
- **Classes**: `NordpoolPlanner` → `NordpoolOptimizer`
- **Version**: Bumped to 3.0.0 to reflect major rewrite

### New Optimization Logic
**Replaced complex accept_cost/accept_rate system with two clear modes:**

#### Mode 1: Absolute Price Threshold
- **Input**: Price threshold (e.g. 0.5 SEK/kWh) + Duration (e.g. 3h)
- **Logic**: Run immediately when average price for specified duration drops below threshold
- **Behavior**: Opportunistic - can run multiple times or not at all per day

#### Mode 2: Daily Guaranteed Slot
- **Input**: Daily duration (e.g. 4h) + Slot type
- **Logic**: Find cheapest X hours within 24h period
- **Options**:
  - **Consecutive**: Single block (e.g. 14:00-18:00)
  - **Separate**: Individual cheapest hours (can be non-consecutive)

### Optional Time Windows
- **Format**: "22:00-06:00" (single input field)
- **Logic**: Restrict optimization to specific hours
- **Midnight spanning**: Supports overnight periods (22:00-06:00 = night hours)

## Phase 2: Timer Entity Implementation

### Single Entity Design
- **Entity Type**: Custom sensor with formatted display
- **State Display**: "-01h14m" (countdown) or "+02h12m" (runtime remaining)
- **Numeric State**: Minutes for automation (negative = off, positive = on)
- **Icon**: Dynamic based on state (countdown/running/off)

### Dashboard Integration
- **Before**: Complex settings with 6 entities per device
- **After**: Clean single entity: "Electric Heater: -01h14m"
- **Automation**: Simple numeric conditions (< 0 = off, > 0 = on)

## Phase 3: Device-Based Configuration

### Multi-Step Setup Dialog
1. **Device Info**: Name + price sensor selection
2. **Mode Selection**: Absolute vs Daily + duration
3. **Mode Configuration**: Price threshold OR slot type
4. **Time Window**: Optional hour restrictions

### Multiple Device Support
- Each "Add Entry" creates independent optimizer
- Each device gets own timer entity with different settings
- Clean dashboard with one entity per optimized device

## Phase 4: Recent Improvements (Current Session)

### Issue 1: Minutely Display Updates
**Problem**: Timer only updated hourly, showing static countdown
**Solution**:
- **Minutely Timer**: Added separate timer for display updates every minute
- **Hourly Optimization**: Kept for price data and optimization calculations
- **Live Countdown**: Real-time updates "-01h14m" → "-01h13m" → "-01h12m"

**Implementation**:
```python
# In async_setup():
self._minutely_update = async_track_time_change(
    self._hass, self.minutely_display_update, second=0
)

def minutely_display_update(self, _):
    """Light display-only update every minute"""
    for listener in self._output_listeners.values():
        listener.update_callback()
```

### Issue 2: Consecutive Period Merging
**Problem**: Separate periods like 14:00-15:00, 15:00-16:00, 16:00-17:00 showing as disconnected
**Solution**:
- **Smart Merging**: Automatically merge consecutive/overlapping periods
- **Weighted Pricing**: Merged periods use weighted average pricing
- **Clean Display**: Single period 14:00-17:00 instead of 3 separate ones

**Implementation**:
```python
def _merge_consecutive_periods(self, periods: list[OptimalPeriod]) -> list[OptimalPeriod]:
    """Merge consecutive or overlapping periods into single periods"""
    # Sort by start time, merge touching/overlapping periods
    # Calculate weighted average price for merged periods
```

## Current Code Structure

### Key Files
- **`__init__.py`**: Core optimization logic, timer management
- **`sensor.py`**: Timer entity display and state management
- **`config_flow.py`**: Device-based setup wizard
- **`const.py`**: Configuration constants for new modes
- **`helpers.py`**: Price data parsing utilities

### Key Classes
- **`NordpoolOptimizer`**: Main coordinator with optimization logic
- **`NordpoolOptimizerTimerEntity`**: Single timer entity implementation
- **`OptimalPeriod`**: Represents time periods with pricing
- **`PricesEntity`**: Price data abstraction (reused from original)

## Configuration Examples

### Example 1: Electric Heater
- **Mode**: Daily slot, 4h consecutive, 22:00-08:00 window
- **Result**: Find cheapest 4 consecutive hours between 22:00-08:00 daily

### Example 2: EV Charger
- **Mode**: Absolute threshold, 6h, below 0.3 SEK/kWh
- **Result**: Run immediately when 6h average price drops below 0.3

### Example 3: Water Heater
- **Mode**: Daily slot, 2h separate slots, no time restriction
- **Result**: Find 2 cheapest individual hours per day

## User Experience Improvements

### Dashboard
- **Before**: 18 entities for 3 devices (6 each)
- **After**: 3 clean timer entities showing live countdown

### Automation
- **Simple conditions**: `state_attr('sensor.heater', 'minutes_value') > 0`
- **Clear logic**: Negative = wait, Positive = run
- **Live feedback**: Minute-by-minute updates

### Setup Experience
- **Guided wizard**: Step-by-step device configuration
- **Clear modes**: Absolute vs Daily with explanations
- **Flexible options**: Time windows, slot types, thresholds

## Technical Benefits

### Performance
- **Efficient updates**: Heavy calculations hourly, light display updates minutely
- **Smart merging**: Reduces period complexity and state changes
- **Clean state management**: Single entity per device reduces overhead

### Maintainability
- **Clear architecture**: Device-centric instead of feature-centric
- **Reduced complexity**: 1 entity type instead of 4 different types
- **Better separation**: Configuration vs display logic clearly separated

### User Adoption
- **Intuitive interface**: Timer display everyone understands
- **Predictable behavior**: Clear modes without conflicting parameters
- **Real-time feedback**: Live countdown builds user confidence

## Repository Status
- **Repository**: https://github.com/andredankert/nordpool_optimizer
- **Version**: 3.0.0 (major rewrite)
- **HACS**: Compatible with direct repository download
- **Installation**: Via HACS custom repository or manual copy

## Future Enhancement Opportunities
- **Multiple time windows**: Support for multiple daily windows
- **Price forecasting**: ML-based price prediction for better optimization
- **Load balancing**: Coordinate multiple devices to avoid peak usage
- **Cost reporting**: Track actual savings from optimization
- **Seasonal adjustments**: Different strategies for winter/summer

---

*This implementation transforms a complex price analysis tool into an intuitive, user-friendly device optimization system with live feedback and intelligent period management.*