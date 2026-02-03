#!/usr/bin/env python3
"""Diagnose bedroom adjacency - why aren't they connected to Hallway/LivingRoom?"""

import sys
sys.path.insert(0, "/Users/michaelqunlan/.workspaces/version-clean/procthor")

from collections import Counter
from procthor.generation.hallway_room_specs import HALLWAY_SPECS
from procthor.generation.floorplan_generation import generate_floorplan, get_room_adjacencies, InvalidFloorplan
from procthor.generation.interior_boundaries import sample_interior_boundary
from procthor.constants import OUTDOOR_ROOM_ID
import numpy as np

# Get 4br-2ba spec
spec_4br = HALLWAY_SPECS[2]
print(f"Testing spec: {spec_4br.room_type_map}")
print("=" * 70)

NUM_HOUSES = 20
bedroom_adjacency_stats = Counter()
bedroom_can_connect_stats = {"can_connect": 0, "cannot_connect": 0}

for house_num in range(NUM_HOUSES):
    # Generate floorplan only
    interior_boundary = sample_interior_boundary(
        num_rooms=len(spec_4br.room_type_map),
        average_room_size=6,  # Default value
    )
    
    try:
        floorplan = generate_floorplan(
            interior_boundary=interior_boundary,
            room_spec=spec_4br,
        )
    except InvalidFloorplan:
        print(f"House {house_num + 1}: Failed to generate floorplan")
        continue
    
    # Pad floorplan like the actual generation does
    floorplan = np.pad(floorplan, pad_width=1, mode="constant", constant_values=OUTDOOR_ROOM_ID)
    
    # Get adjacencies from floorplan
    adjacencies = get_room_adjacencies(floorplan)
    
    print(f"\nHouse {house_num + 1}:")
    
    # For each bedroom, what rooms is it adjacent to (shares walls with)?
    bedrooms = [rid for rid, rtype in spec_4br.room_type_map.items() if rtype == "Bedroom"]
    
    for bedroom_id in bedrooms:
        adjacent_rooms = adjacencies.get(bedroom_id, set())
        adjacent_types = [spec_4br.room_type_map.get(r, f"exterior({r})") for r in adjacent_rooms]
        
        # Check if bedroom shares a wall with Hallway or LivingRoom
        can_connect_to = [t for t in adjacent_types if t in {"Hallway", "LivingRoom"}]
        
        for t in adjacent_types:
            bedroom_adjacency_stats[t] += 1
        
        if can_connect_to:
            bedroom_can_connect_stats["can_connect"] += 1
            status = "OK"
        else:
            bedroom_can_connect_stats["cannot_connect"] += 1
            status = "NO HALLWAY/LIVING ADJACENT!"
        
        print(f"  Bedroom {bedroom_id} shares walls with: {adjacent_types} [{status}]")

print(f"\n{'='*70}")
print("SUMMARY: What rooms do bedrooms share walls with?")
print("=" * 70)
for room_type, count in bedroom_adjacency_stats.most_common():
    print(f"  {room_type}: {count}")

print(f"\nBedrooms that CAN connect to Hallway/LivingRoom: {bedroom_can_connect_stats['can_connect']}")
print(f"Bedrooms that CANNOT connect: {bedroom_can_connect_stats['cannot_connect']}")

total = bedroom_can_connect_stats['can_connect'] + bedroom_can_connect_stats['cannot_connect']
pct = 100 * bedroom_can_connect_stats['cannot_connect'] / total if total > 0 else 0
print(f"Failure rate: {pct:.1f}%")
print(f"\nThis means {pct:.1f}% of bedrooms have NO shared wall with Hallway or LivingRoom,")
print("so they CANNOT get a door to those rooms!")
