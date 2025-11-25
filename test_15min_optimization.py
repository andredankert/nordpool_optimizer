#!/usr/bin/env python3
"""
Test script to verify 15-minute granularity optimization
"""

import datetime as dt
from typing import List, Dict

# Sample price data from user (Oct 16, 2025)
PRICE_DATA = [
    ("2025-10-16T00:00:00+02:00", 0.576),
    ("2025-10-16T00:15:00+02:00", 0.95),
    ("2025-10-16T00:30:00+02:00", 0.921),
    ("2025-10-16T00:45:00+02:00", 0.784),
    ("2025-10-16T01:00:00+02:00", 0.895),
    ("2025-10-16T01:15:00+02:00", 0.827),
    ("2025-10-16T01:30:00+02:00", 0.717),
    ("2025-10-16T01:45:00+02:00", 0.717),
    ("2025-10-16T02:00:00+02:00", 0.621),
    ("2025-10-16T02:15:00+02:00", 0.62),
    ("2025-10-16T02:30:00+02:00", 0.508),
    ("2025-10-16T02:45:00+02:00", 0.472),
    ("2025-10-16T03:00:00+02:00", 0.493),
    ("2025-10-16T03:15:00+02:00", 0.472),
    ("2025-10-16T03:30:00+02:00", 0.472),
    ("2025-10-16T03:45:00+02:00", 0.389),  # Cheapest slot!
    ("2025-10-16T04:00:00+02:00", 0.478),
    ("2025-10-16T04:15:00+02:00", 0.495),
    ("2025-10-16T04:30:00+02:00", 0.502),
    ("2025-10-16T04:45:00+02:00", 0.511),
    ("2025-10-16T05:00:00+02:00", 0.507),
    ("2025-10-16T05:15:00+02:00", 0.653),
    ("2025-10-16T05:30:00+02:00", 0.688),
    ("2025-10-16T05:45:00+02:00", 0.8),
    ("2025-10-16T06:00:00+02:00", 0.609),
]

def parse_time(time_str: str) -> dt.datetime:
    """Parse ISO time string to datetime"""
    return dt.datetime.fromisoformat(time_str)

def find_best_consecutive_4h(prices: List[tuple]) -> tuple:
    """Find best consecutive 4-hour window (16 slots)"""
    best_start_idx = 0
    best_avg = float('inf')

    # Need 16 consecutive slots for 4 hours
    for i in range(len(prices) - 15):  # -15 because we need 16 slots total
        window_prices = [prices[j][1] for j in range(i, i + 16)]
        avg_price = sum(window_prices) / 16

        if avg_price < best_avg:
            best_avg = avg_price
            best_start_idx = i

    start_time = parse_time(prices[best_start_idx][0])
    end_time = start_time + dt.timedelta(hours=4)

    return start_time, end_time, best_avg

def find_cheapest_16_slots(prices: List[tuple]) -> List[tuple]:
    """Find 16 cheapest individual slots"""
    sorted_prices = sorted(prices, key=lambda x: x[1])
    return sorted_prices[:16]

def main():
    print("Testing 15-minute granularity optimization...")
    print(f"Total price slots: {len(PRICE_DATA)}")
    print()

    # Test consecutive mode (like AC device)
    start_time, end_time, avg_price = find_best_consecutive_4h(PRICE_DATA)
    print(f"CONSECUTIVE MODE (4-hour window):")
    print(f"Best window: {start_time.strftime('%H:%M')}-{end_time.strftime('%H:%M')}")
    print(f"Average price: {avg_price:.3f}")

    # Show what this window includes
    start_slot = 0
    for i, (time_str, _) in enumerate(PRICE_DATA):
        if parse_time(time_str) == start_time:
            start_slot = i
            break

    print("Window includes these 15-min slots:")
    for i in range(start_slot, min(start_slot + 16, len(PRICE_DATA))):
        time_str, price = PRICE_DATA[i]
        time_obj = parse_time(time_str)
        print(f"  {time_obj.strftime('%H:%M')}: {price:.3f}")

    print()

    # Test separate mode
    cheapest_slots = find_cheapest_16_slots(PRICE_DATA)
    print(f"SEPARATE MODE (16 cheapest slots):")
    print("Cheapest 16 individual 15-min slots:")
    for time_str, price in cheapest_slots:
        time_obj = parse_time(time_str)
        print(f"  {time_obj.strftime('%H:%M')}: {price:.3f}")

    avg_separate = sum(price for _, price in cheapest_slots) / len(cheapest_slots)
    print(f"Average price of separate slots: {avg_separate:.3f}")

    print()
    print("ANALYSIS:")
    print(f"Previous AC result (00:00-04:00): {(0.576 + 0.95 + 0.921 + 0.784 + 0.895 + 0.827 + 0.717 + 0.717 + 0.621 + 0.62 + 0.508 + 0.472 + 0.493 + 0.472 + 0.472 + 0.389) / 16:.3f}")
    print(f"New optimal consecutive window should be better!")

    # Check if cheapest slot (03:45 = 0.389) is included
    cheapest_time = parse_time("2025-10-16T03:45:00+02:00")
    includes_cheapest = start_time <= cheapest_time < end_time
    print(f"Optimal window includes cheapest slot (03:45): {includes_cheapest}")

if __name__ == "__main__":
    main()