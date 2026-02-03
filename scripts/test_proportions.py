#!/usr/bin/env python
"""
Test script to analyze room proportion variance between grid cells and polygon areas.

This helps identify the root cause of proportion validation failures.
"""

import sys
import random
from typing import Dict, List, Tuple

# Add the project to path
sys.path.insert(0, '/Users/michaelqunlan/.workspaces/version-clean/procthor')

from procthor.generation import HouseGenerator
from procthor.generation.hallway_room_specs import HALLWAY_ROOM_SPEC_SAMPLER


def get_polygon_areas(house) -> Dict[str, float]:
    """Extract polygon areas from generated house."""
    areas = {}
    for room in house.rooms.values():
        room_type = room.room_type
        area_m2 = room.room_polygon.polygon.area
        if room_type not in areas:
            areas[room_type] = []
        areas[room_type].append(area_m2)
    return areas


def check_proportions(areas: Dict[str, List[float]]) -> Tuple[bool, List[str]]:
    """Check if proportions are valid, return (passed, reasons)."""
    issues = []
    
    living_areas = areas.get("LivingRoom", [])
    bedroom_areas = areas.get("Bedroom", [])
    hallway_areas = areas.get("Hallway", [])
    
    if not living_areas:
        issues.append("No LivingRoom found")
        return False, issues
    
    living_area = sum(living_areas)  # Total living room area
    max_bedroom = max(bedroom_areas) if bedroom_areas else 0
    max_hallway = max(hallway_areas) if hallway_areas else 0
    min_bedroom = min(bedroom_areas) if bedroom_areas else float('inf')
    
    # Rule 1: LivingRoom should be >= largest bedroom
    if bedroom_areas and living_area < max_bedroom:
        diff = max_bedroom - living_area
        pct = (diff / max_bedroom) * 100
        issues.append(f"Living ({living_area:.1f}m²) < MaxBedroom ({max_bedroom:.1f}m²) by {pct:.0f}%")
    
    # Rule 2: Hallway should be <= smallest bedroom
    if hallway_areas and bedroom_areas and max_hallway > min_bedroom:
        diff = max_hallway - min_bedroom
        pct = (diff / min_bedroom) * 100
        issues.append(f"Hallway ({max_hallway:.1f}m²) > MinBedroom ({min_bedroom:.1f}m²) by {pct:.0f}%")
    
    # Rule 3: LivingRoom should be >= largest hallway
    if hallway_areas and living_area < max_hallway:
        diff = max_hallway - living_area
        pct = (diff / living_area) * 100
        issues.append(f"Living ({living_area:.1f}m²) < Hallway ({max_hallway:.1f}m²) by {pct:.0f}%")
    
    return len(issues) == 0, issues


def main(n_houses: int = 20, seed: int = None):
    print(f"Testing room proportions on {n_houses} houses...")
    print("=" * 80)
    
    if seed is not None:
        random.seed(seed)
    
    passed = 0
    failed = 0
    all_issues = []
    
    # Statistics for variance analysis
    living_ratios = []
    bedroom_ratios = []
    hallway_ratios = []
    
    for i in range(n_houses):
        house_seed = random.randint(0, 999999)
        room_spec = HALLWAY_ROOM_SPEC_SAMPLER.sample()
        
        try:
            generator = HouseGenerator(
                split="train",
                seed=house_seed,
                room_spec=room_spec
            )
            house, _ = generator.sample()
            
            if generator.controller:
                generator.controller.stop()
            
            areas = get_polygon_areas(house)
            is_valid, issues = check_proportions(areas)
            
            # Print results
            status = "✓ PASS" if is_valid else "✗ FAIL"
            print(f"\nHouse {i+1} (seed={house_seed}): {status}")
            
            # Print room areas
            for room_type, room_areas in sorted(areas.items()):
                total = sum(room_areas)
                if len(room_areas) > 1:
                    print(f"  {room_type}: {total:.1f}m² ({len(room_areas)} rooms @ {[f'{a:.1f}' for a in room_areas]})")
                else:
                    print(f"  {room_type}: {total:.1f}m²")
            
            if issues:
                for issue in issues:
                    print(f"  ❌ {issue}")
                all_issues.extend(issues)
                failed += 1
            else:
                passed += 1
                
        except Exception as e:
            print(f"\nHouse {i+1} (seed={house_seed}): ✗ FAILED TO GENERATE")
            print(f"  Error: {e}")
            failed += 1
    
    # Summary
    print("\n" + "=" * 80)
    print(f"SUMMARY: {passed}/{passed+failed} passed ({100*passed/(passed+failed):.0f}%)")
    
    if all_issues:
        print(f"\nMost common issues:")
        issue_counts = {}
        for issue in all_issues:
            # Extract issue type
            if "Living" in issue and "MaxBedroom" in issue:
                key = "LivingRoom < MaxBedroom"
            elif "Hallway" in issue and "MinBedroom" in issue:
                key = "Hallway > MinBedroom"
            elif "Living" in issue and "Hallway" in issue:
                key = "LivingRoom < Hallway"
            else:
                key = issue
            issue_counts[key] = issue_counts.get(key, 0) + 1
        
        for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f"  {count}x {issue}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-houses", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    main(args.num_houses, args.seed)
