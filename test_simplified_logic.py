#!/usr/bin/env python3
"""
Test the simplified full recalculation logic
"""

import datetime as dt

# Sample price data that should trigger dehumidifier (Oct 19, 2025)
CHEAP_MORNING_PRICES = [
    ("2025-10-19T00:30:00+02:00", 0.471),  # Should trigger
    ("2025-10-19T00:45:00+02:00", 0.469),  # Should trigger
    ("2025-10-19T01:00:00+02:00", 0.496),  # Should trigger
    ("2025-10-19T01:15:00+02:00", 0.482),  # Should trigger
    ("2025-10-19T01:30:00+02:00", 0.471),  # Should trigger
    ("2025-10-19T01:45:00+02:00", 0.468),  # Should trigger
    ("2025-10-19T02:00:00+02:00", 0.513),  # Above threshold
    ("2025-10-19T02:15:00+02:00", 0.493),  # Should trigger
    ("2025-10-19T02:30:00+02:00", 0.482),  # Should trigger
]

def parse_time(time_str: str) -> dt.datetime:
    return dt.datetime.fromisoformat(time_str)

def calculate_absolute_periods_simple(prices, threshold=0.5, duration_hours=1):
    """Simulate the simplified absolute mode calculation"""
    periods = []

    for time_str, price in prices:
        start_time = parse_time(time_str)
        end_time = start_time + dt.timedelta(hours=duration_hours)

        # For simplicity, assume each slot represents a 1-hour period
        # In reality, it would aggregate 4 15-minute slots
        if price <= threshold:
            periods.append({
                'start': start_time,
                'end': end_time,
                'price': price
            })

    return periods

def merge_consecutive_periods(periods):
    """Simulate consecutive period merging"""
    if not periods:
        return []

    merged = []
    current = periods[0]

    for next_period in periods[1:]:
        # Check if periods are consecutive (end time equals next start time)
        if current['end'] >= next_period['start']:
            # Merge periods
            current = {
                'start': current['start'],
                'end': max(current['end'], next_period['end']),
                'price': (current['price'] + next_period['price']) / 2  # Simplified average
            }
        else:
            # Not consecutive - add current and start new
            merged.append(current)
            current = next_period

    # Add the last period
    merged.append(current)
    return merged

def test_simplified_logic():
    """Test the simplified recalculation logic"""
    print("Testing simplified full recalculation logic...")
    print()

    # Test dehumidifier absolute mode
    print("=== DEHUMIDIFIER (Absolute Mode, 1h duration, 0.5 threshold) ===")

    periods = calculate_absolute_periods_simple(CHEAP_MORNING_PRICES, threshold=0.5, duration_hours=1)
    print(f"Found {len(periods)} individual periods under threshold:")

    for period in periods:
        print(f"  {period['start'].strftime('%H:%M')}-{period['end'].strftime('%H:%M')}: {period['price']:.3f}")

    print()

    # Test merging
    merged = merge_consecutive_periods(periods)
    print(f"After merging consecutive periods: {len(merged)} periods")

    for period in merged:
        duration = period['end'] - period['start']
        hours = duration.total_seconds() / 3600
        print(f"  {period['start'].strftime('%H:%M')}-{period['end'].strftime('%H:%M')}: {hours:.1f}h avg {period['price']:.3f}")

    print()

    # Expected result analysis
    print("=== ANALYSIS ===")
    if merged:
        total_duration = sum((p['end'] - p['start']).total_seconds() / 3600 for p in merged)
        print(f"Total optimal hours: {total_duration:.1f}h")

        # Check if it covers the expected cheap morning periods
        covers_morning = any(
            p['start'] <= parse_time("2025-10-19T01:00:00+02:00") < p['end']
            for p in merged
        )
        print(f"Covers cheap morning hours (01:00): {covers_morning}")
    else:
        print("❌ NO PERIODS FOUND - This would be the bug!")

    print()
    print("Expected: Dehumidifier should have long merged periods covering all cheap slots")
    print("This would fix the missing 01:00-05:00 period issue!")

if __name__ == "__main__":
    test_simplified_logic()