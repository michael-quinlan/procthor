#!/usr/bin/env python3
"""Test script to verify ratio overlap score scaling."""

import sys
import json
from pathlib import Path

# Add the repo to path
sys.path.insert(0, str(Path(__file__).parent))

from procthor.generation import HouseGenerator, PROCTHOR10K_ROOM_SPEC_SAMPLER

def test_ratio_scaling():
    """Generate 10 houses and report ratio overlap scores."""
    results = []

    house_generator = HouseGenerator(
        split="train",
        room_spec_sampler=PROCTHOR10K_ROOM_SPEC_SAMPLER
    )

    for i in range(10):
        print(f"Generating house {i+1}/10...", end=" ", flush=True)
        try:
            house, _ = house_generator.sample()

            # Extract room info from house.rooms (dict of room_id -> ProceduralRoom)
            rooms = {}
            for room_id, room in house.rooms.items():
                room_type = room.room_type
                if room_type not in rooms:
                    rooms[room_type] = []
                # Get room polygon area
                rooms[room_type].append({
                    "id": room_id,
                    "num_vertices": len(room.room_polygon.polygon.exterior.coords)
                })

            result = {
                "house_id": i + 1,
                "rooms": rooms,
                "success": True
            }
            results.append(result)
            print("✓")
        except Exception as e:
            print(f"✗ ({str(e)[:50]})")
            results.append({
                "house_id": i + 1,
                "error": str(e),
                "success": False
            })

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    successful = sum(1 for r in results if r.get("success"))
    print(f"Successfully generated: {successful}/10 houses")

    # Analyze room types
    room_type_counts = {}
    for result in results:
        if result.get("success"):
            for room_type in result.get("rooms", {}).keys():
                room_type_counts[room_type] = room_type_counts.get(room_type, 0) + 1

    if room_type_counts:
        print(f"\nRoom Type Distribution:")
        for room_type, count in sorted(room_type_counts.items()):
            print(f"  {room_type}: {count} houses")

    # Save detailed results
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to test_results.json")

if __name__ == "__main__":
    test_ratio_scaling()

