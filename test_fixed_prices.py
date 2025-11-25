#!/usr/bin/env python3
"""
Test the fixed get_prices_group logic
"""

import datetime as dt

# Sample price data from user (Oct 16, 2025)
PRICE_DATA = [
    ("2025-10-16T02:30:00+02:00", 0.507),
    ("2025-10-16T02:45:00+02:00", 0.472),
    ("2025-10-16T03:00:00+02:00", 0.492),
    ("2025-10-16T03:15:00+02:00", 0.472),
    ("2025-10-16T03:30:00+02:00", 0.472),
    ("2025-10-16T03:45:00+02:00", 0.388),  # Cheapest!
    ("2025-10-16T04:00:00+02:00", 0.478),
    ("2025-10-16T04:15:00+02:00", 0.495),
    ("2025-10-16T04:30:00+02:00", 0.502),
    ("2025-10-16T04:45:00+02:00", 0.511),
]

def parse_time(time_str: str) -> dt.datetime:
    return dt.datetime.fromisoformat(time_str)

def fixed_get_prices_group(start: dt.datetime, end: dt.datetime, prices):
    """Fixed logic - only include slots within time range"""
    selected = []
    for time_str, price in prices:
        price_start = parse_time(time_str)
        if start <= price_start < end:
            selected.append((time_str, price))
    return selected

def test_dehumidifier_fixed():
    """Test dehumidifier with fixed price group logic"""
    threshold = 0.5

    print("Testing FIXED dehumidifier logic...")
    print()

    # Test the problematic period that should work
    start_time = parse_time("2025-10-16T03:45:00+02:00")
    end_time = start_time + dt.timedelta(hours=1)  # 1-hour duration

    selected = fixed_get_prices_group(start_time, end_time, PRICE_DATA)
    avg_price = sum(price for _, price in selected) / len(selected) if selected else float('inf')

    print(f"Period {start_time.strftime('%H:%M')}-{end_time.strftime('%H:%M')}:")
    print(f"  Selected slots: {len(selected)}")
    for time_str, price in selected:
        slot_time = parse_time(time_str)
        print(f"    {slot_time.strftime('%H:%M')}: {price:.3f}")
    print(f"  Average: {avg_price:.3f}")
    print(f"  Meets threshold (<= 0.5): {avg_price <= threshold}")
    print()

    # Test all potential 1-hour periods with cheap slots
    cheap_starts = ["03:00", "03:15", "03:30", "03:45", "04:00"]

    print("All 1-hour periods that should trigger dehumidifier:")
    for start_str in cheap_starts:
        start_time = parse_time(f"2025-10-16T{start_str}:00+02:00")
        end_time = start_time + dt.timedelta(hours=1)

        selected = fixed_get_prices_group(start_time, end_time, PRICE_DATA)
        avg_price = sum(price for _, price in selected) / len(selected) if selected else float('inf')

        if avg_price <= threshold:
            print(f"  ✅ {start_time.strftime('%H:%M')}-{end_time.strftime('%H:%M')}: avg {avg_price:.3f}")
        else:
            print(f"  ❌ {start_time.strftime('%H:%M')}-{end_time.strftime('%H:%M')}: avg {avg_price:.3f}")

if __name__ == "__main__":
    test_dehumidifier_fixed()