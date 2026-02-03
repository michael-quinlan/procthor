#!/usr/bin/env python3
"""
Test script to analyze room proportion issues.

This script generates houses and compares:
1. Grid cell counts (what validation sees)
2. Polygon areas (what the final output has)

Goal: Understand the variance and identify root cause.
"""

import random
import sys
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Add procthor to path
sys.path.insert(0, '.')

from procthor.generation import HouseGenerator
from procthor.generation.hallway_room_specs import HALLWAY_ROOM_SPEC_SAMPLER


@dataclass
class RoomStats:
    room_id: int
    room_type: str
    grid_cells: int  # From floorplan grid
    polygon_area_m2: float  # From final polygon
    polygon_cells: float  # polygon_area / 3.61 (for comparison)
    variance_pct: float  # (polygon_cells - grid_cells) / grid_cells * 100


@dataclass 
class HouseAnalysis:
    seed: int
    rooms: List[RoomStats]
    living_room_area: float
    max_bedroom_area: float
    max_hallway_area: float
    min_bedroom_area: float
    passes_proportion_check: bool
    failure_reason: str


def analyze_house(seed: int) -> Tuple[HouseAnalysis, any]:
    """Generate a house and analyze its room proportions."""
    
    room_spec = HALLWAY_ROOM_SPEC_SAMPLER.sample()
    generator = HouseGenerator(
        split="train",
        seed=seed,
        room_spec=room_spec
    )
    
    try:
        house, _ = generator.sample()
    except Exception as e:
        return None, generator
    
    # Get room type mapping
    room_type_map = room_spec.room_type_map
    
    # Collect room stats
    rooms = []
    living_areas = []
    bedroom_areas = []
    hallway_areas = []
    
    for room_id, room in house.rooms.items():
        room_type = room_type_map.get(room_id, "Unknown")
        polygon_area = room.room_polygon.polygon.area
        
        # We don't have access to grid cells here, but we can estimate
        # based on the standard cell size (1.9m x 1.9m = 3.61 m²)
        polygon_cells = polygon_area / 3.61
        
        stats = RoomStats(
            room_id=room_id,
            room_type=room_type,
            grid_cells=0,  # Would need to instrument generate_floorplan
            polygon_area_m2=polygon_area,
            polygon_cells=polygon_cells,
            variance_pct=0  # Can't calculate without grid cells
        )
        rooms.append(stats)
        
        if room_type == "LivingRoom":
            living_areas.append(polygon_area)
        elif room_type == "Bedroom":
            bedroom_areas.append(polygon_area)
        elif room_type == "Hallway":
            hallway_areas.append(polygon_area)
    
    # Calculate proportion check
    living_area = max(living_areas) if living_areas else 0
    max_bedroom = max(bedroom_areas) if bedroom_areas else 0
    min_bedroom = min(bedroom_areas) if bedroom_areas else float('inf')
    max_hallway = max(hallway_areas) if hallway_areas else 0
    
    passes = True
    failure_reason = ""
    
    if living_area < max_bedroom:
        passes = False
        failure_reason = f"LivingRoom ({living_area:.1f}m²) < MaxBedroom ({max_bedroom:.1f}m²)"
    elif max_hallway > min_bedroom:
        passes = False
        failure_reason = f"Hallway ({max_hallway:.1f}m²) > MinBedroom ({min_bedroom:.1f}m²)"
    elif living_area < max_hallway:
        passes = False
        failure_reason = f"LivingRoom ({living_area:.1f}m²) < Hallway ({max_hallway:.1f}m²)"
    
    analysis = HouseAnalysis(
        seed=seed,
        rooms=rooms,
        living_room_area=living_area,
        max_bedroom_area=max_bedroom,
        max_hallway_area=max_hallway,
        min_bedroom_area=min_bedroom if min_bedroom != float('inf') else 0,
        passes_proportion_check=passes,
        failure_reason=failure_reason
    )
    
    return analysis, generator


def main():
    print("=" * 80)
    print("ROOM PROPORTION ANALYSIS")
    print("=" * 80)
    print()
    
    num_houses = 20
    analyses = []
    generator = None
    
    for i in range(num_houses):
        seed = random.randint(0, 999999)
        print(f"\n--- House {i+1}/{num_houses} (seed={seed}) ---")
        
        analysis, gen = analyze_house(seed)
        if gen:
            generator = gen
            
        if analysis is None:
            print("  FAILED TO GENERATE")
            continue
            
        analyses.append(analysis)
        
        # Print room breakdown
        print(f"  Rooms:")
        for r in sorted(analysis.rooms, key=lambda x: x.room_type):
            print(f"    {r.room_type:12} (id={r.room_id}): {r.polygon_area_m2:6.1f} m² ({r.polygon_area_m2 * 10.764:.0f} sqft)")
        
        print(f"  Proportions:")
        print(f"    LivingRoom:  {analysis.living_room_area:6.1f} m²")
        print(f"    MaxBedroom:  {analysis.max_bedroom_area:6.1f} m²")
        print(f"    MaxHallway:  {analysis.max_hallway_area:6.1f} m²")
        print(f"    MinBedroom:  {analysis.min_bedroom_area:6.1f} m²")
        
        if analysis.passes_proportion_check:
            print(f"  Result: ✓ PASS")
        else:
            print(f"  Result: ✗ FAIL - {analysis.failure_reason}")
    
    # Cleanup controller
    if generator and generator.controller:
        generator.controller.stop()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for a in analyses if a.passes_proportion_check)
    failed = len(analyses) - passed
    
    print(f"\nTotal: {len(analyses)} houses")
    print(f"Passed: {passed} ({100*passed/len(analyses):.0f}%)")
    print(f"Failed: {failed} ({100*failed/len(analyses):.0f}%)")
    
    if failed > 0:
        print(f"\nFailure breakdown:")
        living_lt_bed = sum(1 for a in analyses if not a.passes_proportion_check and "LivingRoom" in a.failure_reason and "Bedroom" in a.failure_reason)
        hall_gt_bed = sum(1 for a in analyses if not a.passes_proportion_check and "Hallway" in a.failure_reason and "Bedroom" in a.failure_reason)
        living_lt_hall = sum(1 for a in analyses if not a.passes_proportion_check and "LivingRoom" in a.failure_reason and "Hallway" in a.failure_reason and "Bedroom" not in a.failure_reason)
        print(f"  LivingRoom < Bedroom: {living_lt_bed}")
        print(f"  Hallway > Bedroom:    {hall_gt_bed}")
        print(f"  LivingRoom < Hallway: {living_lt_hall}")
    
    # Worst cases
    if failed > 0:
        print(f"\nWorst failures:")
        failures = [a for a in analyses if not a.passes_proportion_check]
        failures.sort(key=lambda a: a.max_bedroom_area - a.living_room_area, reverse=True)
        for a in failures[:5]:
            print(f"  Seed {a.seed}: {a.failure_reason}")
            print(f"    Living={a.living_room_area:.1f}m², MaxBed={a.max_bedroom_area:.1f}m², Hall={a.max_hallway_area:.1f}m²")
    
    # Room size distributions
    print(f"\nRoom size ranges (m²):")
    living_areas = [a.living_room_area for a in analyses]
    bedroom_areas = [a.max_bedroom_area for a in analyses]
    hallway_areas = [a.max_hallway_area for a in analyses]
    
    print(f"  LivingRoom: {min(living_areas):.1f} - {max(living_areas):.1f} (avg {sum(living_areas)/len(living_areas):.1f})")
    print(f"  MaxBedroom: {min(bedroom_areas):.1f} - {max(bedroom_areas):.1f} (avg {sum(bedroom_areas)/len(bedroom_areas):.1f})")
    print(f"  MaxHallway: {min(hallway_areas):.1f} - {max(hallway_areas):.1f} (avg {sum(hallway_areas)/len(hallway_areas):.1f})")


if __name__ == "__main__":
    main()
