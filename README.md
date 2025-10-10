# Nordpool Optimizer - Smart Device Scheduling for Home Assistant

Transform your electricity consumption with intelligent price-based device optimization. Nordpool Optimizer automatically schedules your high-consumption devices during the cheapest electricity periods, saving money and reducing environmental impact.

![Nordpool Optimizer Dashboard](https://img.shields.io/badge/Home%20Assistant-Integration-blue?logo=home-assistant)
![Version](https://img.shields.io/badge/Version-3.0.0-green)
![HACS](https://img.shields.io/badge/HACS-Compatible-orange)

## 🌟 Features

### ⚡ Smart Device Optimization
- **Individual device timers** showing live countdown/runtime
- **Two optimization modes**: Absolute price thresholds or daily cheapest slots
- **Flexible time windows** for overnight scheduling
- **Real-time updates** with minute-by-minute countdown

### 📊 Advanced Visualization
- **Multi-device price graph** showing all optimal periods
- **Row-based period display** preventing visual overlap
- **Color-coded devices** with proper legend
- **Future price visualization** for planning ahead

### 🔧 Easy Setup & Management
- **Device-centric configuration** - one setup per device
- **Guided setup wizard** with step-by-step configuration
- **Auto-discovery** of new devices without restart
- **Persistent caching** for instant startup

## 📈 What's New in v3.0

### Complete Architecture Rewrite
- **From 6 entities per device** → **1 clean timer entity**
- **From complex accept_cost/rate** → **Clear optimization modes**
- **From binary sensors** → **Intuitive countdown timers**

### New Optimization Modes

#### 1. Absolute Price Threshold
- Set a price limit (e.g., 0.5 SEK/kWh) and duration (e.g., 3 hours)
- Device runs immediately when average price drops below threshold
- Opportunistic - can run multiple times or not at all per day

#### 2. Daily Guaranteed Slot
- Ensure device runs for specified duration every day
- **Consecutive**: Single block (e.g., 14:00-18:00)
- **Separate**: Individual cheapest hours (non-consecutive)

### Price Graph Entity
- **Unified visualization** of all devices and future prices
- **Row-based periods** preventing overlap confusion
- **Automatic device discovery** - new devices appear instantly
- **Chart-ready data** for ApexCharts, Chart.js, etc.

## 🚀 Installation

### Prerequisites
Install a Nordpool price sensor first:
- **Recommended**: [Custom Nordpool Integration](https://github.com/custom-components/nordpool)
- **Alternative**: [ENTSO-e Integration](https://github.com/JaccoR/hass-entso-e)
- **Built-in**: [Home Assistant Nordpool](https://www.home-assistant.io/integrations/nordpool/) (limited testing)

### Option 1: HACS (Recommended)
1. Go to **HACS** → **Integrations**
2. Click **⋮** → **Custom Repositories**
3. Add: `https://github.com/andredankert/nordpool_optimizer`
4. Category: **Integration** → **Add**
5. Install **Nordpool Optimizer**
6. **Restart Home Assistant**

### Option 2: Manual Installation
1. Download and extract to `<config_dir>/custom_components/nordpool_optimizer/`
2. Restart Home Assistant

## ⚙️ Configuration

### Adding Devices
1. Go to **Settings** → **Devices & Services**
2. Click **+ Add Integration**
3. Search for **Nordpool Optimizer**
4. Follow the guided setup:
   - **Device Info**: Name + price sensor selection
   - **Mode Selection**: Absolute vs Daily + duration
   - **Mode Configuration**: Price threshold OR slot type
   - **Time Window**: Optional hour restrictions

### Configuration Examples

#### Electric Heater (Overnight Heating)
- **Mode**: Daily slot, 4h consecutive
- **Time Window**: 22:00-08:00
- **Result**: Find cheapest 4 consecutive hours between 22:00-08:00 daily

#### EV Charger (Opportunistic Charging)
- **Mode**: Absolute threshold, 6h duration
- **Price Threshold**: 0.3 SEK/kWh
- **Result**: Charge immediately when 6h average price drops below 0.3

#### Water Heater (Flexible Timing)
- **Mode**: Daily slot, 2h separate slots
- **Time Window**: None
- **Result**: Find 2 cheapest individual hours per day

## 🎯 Using the Entities

### Timer Entity Display
Each device gets one clean timer entity showing:
- **Countdown**: `-01h14m` (time until optimal period)
- **Runtime**: `+02h12m` (time remaining in optimal period)
- **Status**: `Off`, `Starting`, or time display

### Entity Attributes
All attributes are automation-ready:
```yaml
# Check if currently optimal (boolean)
{{ state_attr('sensor.dehumidifier', 'currently_optimal') }}

# Get minutes value for numeric comparisons
{{ state_attr('sensor.dehumidifier', 'minutes_value') }}

# Get next start time (datetime)
{{ state_attr('sensor.dehumidifier', 'next_optimal_start') }}
```

### Example Automation
```yaml
automation:
  - alias: "Dehumidifier follows optimal periods"
    trigger:
      - platform: state
        entity_id: sensor.dehumidifier
        attribute: currently_optimal
    action:
      - choose:
          - conditions:
              - condition: template
                value_template: "{{ state_attr('sensor.dehumidifier', 'currently_optimal') }}"
            sequence:
              - service: switch.turn_on
                target:
                  entity_id: switch.dehumidifier
          - conditions:
              - condition: template
                value_template: "{{ not state_attr('sensor.dehumidifier', 'currently_optimal') }}"
              - condition: state
                entity_id: input_boolean.dehumidifier_forced_run
                state: 'off'
            sequence:
              - service: switch.turn_off
                target:
                  entity_id: switch.dehumidifier
```

## 📊 Price Graph Visualization

### Automatic Graph Entity
Nordpool Optimizer automatically creates a `sensor.nordpool_price_graph` entity containing:
- **Future price data** for next 24 hours
- **All device optimal periods** with color coding
- **Row-based layout** preventing visual overlap
- **Chart-ready data structure**

### Quick Visualization
Add this simple card to see the data:
```yaml
type: entities
title: "Price Graph Data"
entities:
  - sensor.nordpool_price_graph
```

### ApexCharts Integration

#### Install ApexCharts Card
1. **HACS** → **Frontend** → Search "**ApexCharts Card**" → **Download**
2. Restart Home Assistant + clear browser cache (Ctrl+F5)

#### Simple Chart Configuration
```yaml
type: custom:apexcharts-card
header:
  title: "Nordpool Future Prices"
  show: true
series:
  - entity: sensor.nordpool_price_graph
    data_generator: |
      const now = new Date().getTime();
      const prices = entity.attributes.prices_ahead || [];
      return prices
        .filter(item => new Date(item.time).getTime() >= now)
        .map(item => [new Date(item.time).getTime(), item.price]);
    name: "Price (kr/kWh)"
    type: line
    color: "#2196F3"
span:
  start: hour
graph_span: 24h
apex_config:
  chart:
    height: 350
  xaxis:
    type: datetime
    min: new Date().getTime()
  yaxis:
    title:
      text: "Price (kr/kWh)"
```

#### Multi-Device Chart with Periods
```yaml
type: custom:apexcharts-card
header:
  title: "Nordpool Prices & Device Periods"
  show: true
series:
  - entity: sensor.nordpool_price_graph
    data_generator: |
      const now = new Date().getTime();
      const prices = entity.attributes.prices_ahead || [];
      return prices
        .filter(item => new Date(item.time).getTime() >= now)
        .map(item => [new Date(item.time).getTime(), item.price]);
    name: "Price (kr/kWh)"
    type: line
    color: "#2196F3"
    stroke_width: 2
  - entity: sensor.nordpool_price_graph
    data_generator: |
      const now = new Date().getTime();
      const periods = entity.attributes.device_periods || [];
      if (periods.length === 0) return [];
      const device = periods[0];
      const result = [];
      device.periods.forEach(period => {
        const startTime = new Date(period.start).getTime();
        const endTime = new Date(period.end).getTime();
        if (endTime >= now) {
          result.push([startTime, device.y_position]);
          result.push([endTime, device.y_position]);
          result.push([null, null]);
        }
      });
      return result;
    name: "Dehumidifier"
    type: line
    color: "#4CAF50"
    stroke_width: 6
    opacity: 0.8
  - entity: sensor.nordpool_price_graph
    data_generator: |
      const now = new Date().getTime();
      const periods = entity.attributes.device_periods || [];
      if (periods.length < 2) return [];
      const device = periods[1];
      const result = [];
      device.periods.forEach(period => {
        const startTime = new Date(period.start).getTime();
        const endTime = new Date(period.end).getTime();
        if (endTime >= now) {
          result.push([startTime, device.y_position]);
          result.push([endTime, device.y_position]);
          result.push([null, null]);
        }
      });
      return result;
    name: "Device 2"
    type: line
    color: "#FF9800"
    stroke_width: 6
    opacity: 0.8
span:
  start: hour
graph_span: 24h
apex_config:
  chart:
    height: 350
  xaxis:
    type: datetime
    min: new Date().getTime()
  yaxis:
    title:
      text: "Price (kr/kWh)"
```

### Dynamic Auto-Updating Chart

For automatically updating charts when devices are added:

#### 1. Install config-template-card
**HACS** → **Frontend** → Search "**config template card**" → **Download**

#### 2. Add to configuration.yaml
```yaml
template:
  - trigger:
      - platform: state
        entity_id: sensor.nordpool_price_graph
        attribute: device_periods
      - platform: homeassistant
        event: start
    sensor:
      - name: "Apex Card Config"
        state: "ok"
        attributes:
          device_count: >
            {% set periods = state_attr('sensor.nordpool_price_graph', 'device_periods') or [] %}
            {{ periods | length }}
          devices: >
            {% set periods = state_attr('sensor.nordpool_price_graph', 'device_periods') or [] %}
            {{ periods | map(attribute='device') | list }}
```

#### 3. Dynamic Dashboard Card
```yaml
type: custom:config-template-card
variables:
  PERIODS: states['sensor.nordpool_price_graph'].attributes.device_periods || []
card:
  type: custom:apexcharts-card
  header:
    title: "Nordpool Prices & Device Periods"
    show: true
  series: >
    ${[
      {
        entity: 'sensor.nordpool_price_graph',
        data_generator: `
          const now = new Date().getTime();
          const prices = entity.attributes.prices_ahead || [];
          return prices
            .filter(item => new Date(item.time).getTime() >= now)
            .map(item => [new Date(item.time).getTime(), item.price]);
        `,
        name: 'Price (kr/kWh)',
        type: 'line',
        color: '#2196F3',
        stroke_width: 2
      },
      ...PERIODS.map((device, index) => ({
        entity: 'sensor.nordpool_price_graph',
        data_generator: `
          const now = new Date().getTime();
          const periods = entity.attributes.device_periods || [];
          const device = periods[${index}];
          if (!device) return [];
          const result = [];
          device.periods.forEach(period => {
            const startTime = new Date(period.start).getTime();
            const endTime = new Date(period.end).getTime();
            if (endTime >= now) {
              result.push([startTime, device.y_position]);
              result.push([endTime, device.y_position]);
              result.push([null, null]);
            }
          });
          return result;
        `,
        name: device.device,
        type: 'line',
        color: device.color,
        stroke_width: 6,
        opacity: 0.8
      }))
    ]}
  span:
    start: hour
  graph_span: 24h
  apex_config:
    chart:
      height: 350
    xaxis:
      type: datetime
      min: ${new Date().getTime()}
    yaxis:
      title:
        text: "Price (kr/kWh)"
```

This approach automatically adds new devices to the chart without manual configuration updates!

## 🔧 Technical Features

### Persistent Price Caching
- **Startup optimization**: Instant price data availability after restart
- **6-hour cache validity**: Fresh data without constant API calls
- **Pickle serialization**: Preserves datetime objects correctly
- **Graceful fallback**: Uses live data if cache invalid

### Auto-Discovery
- **New devices**: Automatically detected and included in price graph
- **No restart required**: Changes appear immediately
- **Smart registration**: Prevents duplicate listeners

### Performance Optimizations
- **Minutely display updates**: Live countdown without heavy calculations
- **Hourly optimization**: Price analysis and period calculation
- **Consecutive period merging**: Reduces complexity and state changes
- **Efficient caching**: Minimal storage with maximum performance

## 🎛️ Dashboard Integration

### Before (Complex)
- 18 entities for 3 devices (6 each)
- Multiple binary sensors, sliders, and separate sensors
- Confusing state management

### After (Clean)
- 3 timer entities showing live countdown
- 1 price graph entity for all devices
- Simple automation conditions

### Example Dashboard Layout
```yaml
type: entities
title: "Device Optimization"
entities:
  - entity: sensor.dehumidifier
    name: "Dehumidifier"
  - entity: sensor.ev_charger
    name: "EV Charger"
  - entity: sensor.water_heater
    name: "Water Heater"
```

## 🏠 Home Assistant Integration

### Native Features
- **Device grouping**: All entities grouped per optimizer device
- **Area assignment**: Assign optimizers to house areas
- **Automation templates**: Use in any automation or script
- **History tracking**: Full state history and statistics

### Automation Examples
```yaml
# Simple on/off automation
- alias: "Device follows optimizer"
  trigger:
    - platform: template
      value_template: "{{ state_attr('sensor.device', 'currently_optimal') }}"
  action:
    - service: "switch.turn_{{ 'on' if trigger.to_state.state else 'off' }}"
      target:
        entity_id: switch.device

# Advanced with forced run override
- alias: "Smart device control"
  trigger:
    - platform: state
      entity_id: sensor.device
      attribute: currently_optimal
  condition:
    - condition: or
      conditions:
        - condition: template
          value_template: "{{ state_attr('sensor.device', 'currently_optimal') }}"
        - condition: state
          entity_id: input_boolean.device_forced_run
          state: 'on'
  action:
    - service: switch.turn_on
      target:
        entity_id: switch.device
```

## 📚 Migration from v2.x

### Breaking Changes
- **Configuration method**: GUI-only configuration (no YAML)
- **Entity structure**: Single timer entity instead of multiple sensors
- **Optimization modes**: New absolute/daily system replaces accept_cost/accept_rate

### Migration Steps
1. **Remove old integration** from Settings → Devices & Services
2. **Delete old YAML configuration** if present
3. **Install new version** following installation instructions
4. **Reconfigure devices** using new guided setup
5. **Update automations** to use new entity attributes

### Benefits After Migration
- **Cleaner interface**: 1 entity instead of 6 per device
- **Better performance**: Instant startup with caching
- **More intuitive**: Timer display everyone understands
- **Future-proof**: Active development and new features

## ❓ FAQ

### Q: Why is my timer showing "unavailable"?
**A:** Check that your Nordpool price sensor has valid data and the cache is working. Look for debug logs in Home Assistant.

### Q: Can I have different time windows for different devices?
**A:** Yes! Each device has its own configuration including optional time windows.

### Q: How do I add a manual override?
**A:** Create an `input_boolean` helper and use it in your automation conditions as shown in the examples.

### Q: Does it work with time-of-use tariffs?
**A:** Yes, as long as your price sensor reflects the actual cost including tariffs.

### Q: Can I see historical optimization performance?
**A:** Use Home Assistant's history and statistics features on the timer entities to track activation patterns.

## 🐛 Troubleshooting

### Common Issues

**Entity shows "Unknown"**
- Verify Nordpool sensor has valid price data
- Check that duration and mode settings are valid
- Restart Home Assistant to refresh cache

**Graph not updating**
- Clear browser cache (Ctrl+F5)
- Verify ApexCharts card is properly installed
- Check entity attributes in Developer Tools

**New device not appearing in graph**
- Wait for next update cycle (up to 1 hour)
- Check debug logs for auto-registration messages
- Restart integration if auto-discovery fails

### Debug Logging
Enable debug logging in `configuration.yaml`:
```yaml
logger:
  logs:
    custom_components.nordpool_optimizer: debug
```

## 📈 Performance & Savings

### Real-World Results
Users report 20-40% electricity cost savings for high-consumption devices like:
- **Electric heating**: Overnight optimization during cheap periods
- **EV charging**: Opportunistic charging when prices drop
- **Heat pumps**: Thermal mass utilization during price valleys
- **Water heaters**: Storage heating during optimal windows

### Environmental Impact
- **Load balancing**: Reduces peak demand stress
- **Renewable utilization**: Aligns consumption with wind/solar production
- **Grid stability**: Distributed smart consumption helps overall efficiency

## 🤝 Contributing

We welcome contributions! Please:
1. **Fork the repository**
2. **Create a feature branch**
3. **Add tests for new functionality**
4. **Submit a pull request**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on [nordpool_diff](https://github.com/jpulakka/nordpool_diff) by jpulakka
- Inspired by [nordpool_planner](https://github.com/dala318/nordpool_planner) by dala318
- Thanks to the Home Assistant community for testing and feedback

---

**⚡ Start optimizing your electricity consumption today!**

Transform your high-consumption devices into smart, cost-aware appliances that automatically run during the cheapest periods. Save money, reduce environmental impact, and gain insight into your energy usage patterns.