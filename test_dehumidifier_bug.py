#!/usr/bin/env python3
"""
Test script to debug dehumidifier absolute mode issue
"""

import datetime as dt
from typing import List, Dict

# Sample price data from user (Oct 16, 2025) - tonight's data
PRICE_DATA = [
    ("2025-10-16T02:30:00+02:00", 0.507),
    ("2025-10-16T02:45:00+02:00", 0.472),
    ("2025-10-16T03:00:00+02:00", 0.492),
    ("2025-10-16T03:15:00+02:00", 0.472),
    ("2025-10-16T03:30:00+02:00", 0.472),
    ("2025-10-16T03:45:00+02:00", 0.388),  # Cheapest - should trigger!
    ("2025-10-16T04:00:00+02:00", 0.478),
    ("2025-10-16T04:15:00+02:00", 0.495),
    ("2025-10-16T04:30:00+02:00", 0.502),
    ("2025-10-16T04:45:00+02:00", 0.511),
    ("2025-10-16T05:00:00+02:00", 0.507),
]

def parse_time(time_str: str) -> dt.datetime:
    """Parse ISO time string to datetime"""
    return dt.datetime.fromisoformat(time_str)

def simulate_get_prices_group(start: dt.datetime, end: dt.datetime, prices: List[tuple]) -> List[tuple]:
    """Simulate the get_prices_group logic"""
    started = False
    selected = []

    for time_str, price in prices:
        p_start = parse_time(time_str)

        # This is the suspicious logic from the original function
        if p_start > start - dt.timedelta(hours=1):
            started = True
        if p_start > end:
            break
        if started:
            selected.append((time_str, price))

    return selected

def calculate_average(selected_prices: List[tuple]) -> float:
    """Calculate average price"""
    if not selected_prices:
        return float('inf')
    return sum(price for _, price in selected_prices) / len(selected_prices)

def test_dehumidifier_periods():
    """Test dehumidifier absolute mode with 1-hour duration, 0.5 threshold"""
    threshold = 0.5
    duration_hours = 1

    print("Testing dehumidifier absolute mode (1-hour periods, threshold 0.5)...")
    print()

    # Test periods starting at 15-minute boundaries
    test_starts = [
        "2025-10-16T03:00:00+02:00",
        "2025-10-16T03:15:00+02:00",
        "2025-10-16T03:30:00+02:00",
        "2025-10-16T03:45:00+02:00",  # Should work - includes 0.388
        "2025-10-16T04:00:00+02:00",
    ]

    for start_str in test_starts:
        start_time = parse_time(start_str)
        end_time = start_time + dt.timedelta(hours=duration_hours)

        # Simulate what get_prices_group returns
        selected = simulate_get_prices_group(start_time, end_time, PRICE_DATA)
        avg_price = calculate_average(selected)

        meets_threshold = avg_price <= threshold

        print(f"Period {start_time.strftime('%H:%M')}-{end_time.strftime('%H:%M')}:")
        print(f"  Selected slots: {len(selected)}")
        for time_str, price in selected:
            slot_time = parse_time(time_str)
            print(f"    {slot_time.strftime('%H:%M')}: {price:.3f}")
        print(f"  Average: {avg_price:.3f}")
        print(f"  Meets threshold (<= 0.5): {meets_threshold}")
        print()

def test_correct_logic():
    """Test what the logic SHOULD be"""
    print("Testing CORRECT logic (simple time range check)...")
    print()

    start_time = parse_time("2025-10-16T03:45:00+02:00")
    end_time = start_time + dt.timedelta(hours=1)

    # Correct logic: include all slots within the time range
    selected = []
    for time_str, price in PRICE_DATA:
        slot_time = parse_time(time_str)
        # Include if slot starts within our period
        if start_time <= slot_time < end_time:
            selected.append((time_str, price))

    avg_price = calculate_average(selected)

    print(f"Period {start_time.strftime('%H:%M')}-{end_time.strftime('%H:%M')} (CORRECT):")
    print(f"  Selected slots: {len(selected)}")
    for time_str, price in selected:
        slot_time = parse_time(time_str)
        print(f"    {slot_time.strftime('%H:%M')}: {price:.3f}")
    print(f"  Average: {avg_price:.3f}")
    print(f"  Meets threshold (<= 0.5): {avg_price <= 0.5}")

if __name__ == "__main__":
    test_dehumidifier_periods()
    test_correct_logic()