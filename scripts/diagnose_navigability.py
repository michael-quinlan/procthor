#!/usr/bin/env python3
"""Diagnose why rooms are not navigable in 4br-2ba houses."""

import sys
sys.path.insert(0, "/Users/michaelqunlan/.workspaces/version-clean/procthor")

import random
from collections import Counter
from shapely.geometry import Point, Polygon

from procthor.generation import HouseGenerator, PROCTHOR_INITIALIZATION
from procthor.generation.hallway_room_specs import HALLWAY_SPECS
from ai2thor.controller import Controller

# Get 4br-2ba spec
spec_4br = HALLWAY_SPECS[2]  # 4br-2ba
print(f"Testing spec: {spec_4br.room_type_map}")
print("=" * 70)

# Create controller and generator
controller = Controller(**PROCTHOR_INITIALIZATION)
house_generator = HouseGenerator(
    split="train",
    seed=None,  # Random seed for variety
    controller=controller,
)
house_generator.room_spec = spec_4br

# Test multiple houses
NUM_HOUSES = 10
disconnected_rooms_total = 0
total_private_rooms = 0

try:
    for house_num in range(NUM_HOUSES):
        print(f"\n{'='*70}")
        print(f"HOUSE {house_num + 1}")
        print("=" * 70)
        
        house_generator.partial_house = None
        house, _ = house_generator.sample()
        
        # Check door connectivity
        doors = house.data.get("doors", [])
        door_connections = {}
        for door in doors:
            room0 = door.get("room0")
            room1 = door.get("room1")
            if room0 and room1:
                r0_id = int(room0.split("|")[1]) if "|" in room0 else room0
                r1_id = int(room1.split("|")[1]) if "|" in room1 else room1
                door_connections.setdefault(r0_id, set()).add(r1_id)
                door_connections.setdefault(r1_id, set()).add(r0_id)
        
        # Find disconnected bedrooms/bathrooms
        private_rooms = ["Bedroom", "Bathroom"]
        disconnected = []
        
        for room_id in house.rooms.keys():
            room_type = spec_4br.room_type_map.get(room_id, "Unknown")
            if room_type in private_rooms:
                total_private_rooms += 1
                connections = door_connections.get(room_id, set())
                if len(connections) == 0:
                    disconnected.append((room_id, room_type))
                    disconnected_rooms_total += 1
        
        print(f"Door connections:")
        for room_id in sorted(house.rooms.keys()):
            room_type = spec_4br.room_type_map.get(room_id, "Unknown")
            connections = door_connections.get(room_id, set())
            conn_types = [spec_4br.room_type_map.get(c, "?") for c in connections]
            status = "NO DOORS!" if len(connections) == 0 else ""
            print(f"  Room {room_id} ({room_type:12s}) -> {list(connections)} {status}")
        
        if disconnected:
            print(f"\n  *** DISCONNECTED ROOMS: {disconnected}")
        else:
            print(f"\n  All rooms have door connections!")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"Total private rooms across {NUM_HOUSES} houses: {total_private_rooms}")
    print(f"Disconnected rooms (no doors): {disconnected_rooms_total}")
    print(f"Disconnection rate: {100*disconnected_rooms_total/total_private_rooms:.1f}%")

finally:
    controller.stop()
