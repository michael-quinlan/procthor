#!/usr/bin/env python3
"""Debug script to identify why 4br-2ba houses fail."""

import sys
import random
import traceback
from collections import Counter

# Add procthor to path
sys.path.insert(0, '/Users/michaelqunlan/.workspaces/version-clean/procthor')

from procthor.generation import HouseGenerator
from procthor.generation.hallway_room_specs import HALLWAY_SPECS

# Get the 4br-2ba spec
spec_4br = None
for spec in HALLWAY_SPECS:
    bedroom_count = sum(1 for room_id, room_type in spec.room_type_map.items() if room_type == "Bedroom")
    if bedroom_count == 4:
        spec_4br = spec
        break

if not spec_4br:
    print("ERROR: Could not find 4br spec")
    sys.exit(1)

print(f"Testing 4br-2ba spec: {spec_4br.room_type_map}")
print("-" * 60)

# Track failures
failure_reasons = Counter()
warning_types = Counter()
success_count = 0
attempt_count = 0

# Create generator
house_generator = HouseGenerator(
    split="train",
    seed=random.randint(0, 999999),
    room_spec=spec_4br,
)

# Try to generate 50 houses
target = 50
attempts_per_house = 20

for i in range(target):
    house_generator.room_spec = spec_4br
    house_generator.partial_house = None
    
    for attempt in range(attempts_per_house):
        attempt_count += 1
        try:
            house, _ = house_generator.sample()
            
            # Check for warnings
            warnings = house.data.get("metadata", {}).get("warnings", [])
            if warnings:
                for w in warnings:
                    warning_types[w] += 1
                continue  # Retry
            
            # Success!
            success_count += 1
            print(f"House {i+1}: SUCCESS after {attempt+1} attempts")
            break
            
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)[:50]
            failure_reasons[f"{error_type}: {error_msg}"] += 1
            
    else:
        print(f"House {i+1}: FAILED after {attempts_per_house} attempts")

print("\n" + "=" * 60)
print(f"RESULTS: {success_count}/{target} houses generated ({success_count/target*100:.1f}%)")
print(f"Total attempts: {attempt_count}")
print(f"Average attempts per success: {attempt_count/max(success_count,1):.1f}")

print("\n--- Failure Reasons (exceptions) ---")
for reason, count in failure_reasons.most_common(10):
    print(f"  {count:4d}x  {reason}")

print("\n--- Warning Types (validation failures) ---")
for warning, count in warning_types.most_common(10):
    print(f"  {count:4d}x  {warning}")

# Cleanup
if house_generator.controller:
    house_generator.controller.stop()
