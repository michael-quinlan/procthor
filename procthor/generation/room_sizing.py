"""
Room sizing constraints and automatic calculation of dims/ratios.

This module provides realistic room size targets and functions to automatically
calculate grid dimensions and room ratios based on target square footage.
"""
import math
import random
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass

# Room size targets in square meters (min, target, max)
# These are based on typical US residential construction
ROOM_SIZE_TARGETS_SQM: Dict[str, Tuple[float, float, float]] = {
    # Room type: (min_sqm, target_sqm, max_sqm)
    "Bedroom": (9.0, 14.0, 25.0),        # 97-269 sqft, target ~150 sqft
    "Bathroom": (3.5, 6.0, 12.0),        # 38-129 sqft, target ~65 sqft
    "Kitchen": (7.0, 12.0, 20.0),        # 75-215 sqft, target ~130 sqft
    "LivingRoom": (14.0, 22.0, 35.0),    # 150-377 sqft, target ~237 sqft
    "Hallway": (2.5, 5.0, 10.0),         # 27-108 sqft, target ~54 sqft
}

# Hallway shape constraints
HALLWAY_MAX_WIDTH_M = 1.8   # Hallways shouldn't be wider than 1.8m (6 ft)
HALLWAY_MIN_ASPECT_RATIO = 2.0  # Length should be at least 2x width

# Conversion constant
SQM_TO_SQFT = 10.7639


@dataclass
class RoomSizeSpec:
    """Specification for a room's size constraints."""
    room_type: str
    min_sqm: float
    target_sqm: float
    max_sqm: float
    
    @property
    def min_sqft(self) -> float:
        return self.min_sqm * SQM_TO_SQFT
    
    @property
    def target_sqft(self) -> float:
        return self.target_sqm * SQM_TO_SQFT
    
    @property
    def max_sqft(self) -> float:
        return self.max_sqm * SQM_TO_SQFT
    
    @classmethod
    def for_room_type(cls, room_type: str) -> "RoomSizeSpec":
        """Get the size spec for a room type."""
        if room_type not in ROOM_SIZE_TARGETS_SQM:
            raise ValueError(f"Unknown room type: {room_type}")
        min_sqm, target_sqm, max_sqm = ROOM_SIZE_TARGETS_SQM[room_type]
        return cls(room_type, min_sqm, target_sqm, max_sqm)


def calculate_target_house_size(
    room_type_counts: Dict[str, int],
    size_preference: str = "target"  # "min", "target", or "max"
) -> float:
    """Calculate total house size in sqm based on room counts.
    
    Args:
        room_type_counts: Dict mapping room type to count, e.g. {"Bedroom": 3, "Bathroom": 2, ...}
        size_preference: Which size to use - "min", "target", or "max"
    
    Returns:
        Total house size in square meters
    """
    total = 0.0
    for room_type, count in room_type_counts.items():
        spec = RoomSizeSpec.for_room_type(room_type)
        if size_preference == "min":
            total += spec.min_sqm * count
        elif size_preference == "max":
            total += spec.max_sqm * count
        else:
            total += spec.target_sqm * count
    return total


def calculate_dims_for_house(
    room_type_counts: Dict[str, int],
    interior_boundary_scale: float = 1.9,
    size_preference: str = "target",
    aspect_ratio_range: Tuple[float, float] = (0.8, 1.25),
    min_cells_per_room: int = 6,
    boundary_cut_buffer: float = 1.3,
) -> Tuple[int, int]:
    """Calculate grid dimensions (dims) for a house based on target room sizes.

    Args:
        room_type_counts: Dict mapping room type to count
        interior_boundary_scale: The scale factor (meters per grid cell)
        size_preference: "min", "target", or "max"
        aspect_ratio_range: Range for random aspect ratio variation
        min_cells_per_room: Minimum cells per room to ensure fitting
        boundary_cut_buffer: Extra buffer to account for boundary cuts

    Returns:
        (x_size, z_size) grid dimensions
    """
    target_area_sqm = calculate_target_house_size(room_type_counts, size_preference)

    # Account for ~15% loss due to walls/inefficiency
    effective_area_sqm = target_area_sqm * 1.15

    # Calculate total grid cells needed
    cell_area_sqm = interior_boundary_scale ** 2
    total_cells = effective_area_sqm / cell_area_sqm

    # Ensure minimum cells for room fitting (each room needs at least min_cells_per_room)
    # Also add buffer for boundary cuts which remove some cells
    total_rooms = sum(room_type_counts.values())
    min_cells_needed = total_rooms * min_cells_per_room * boundary_cut_buffer
    total_cells = max(total_cells, min_cells_needed)
    
    # Calculate side length with random aspect ratio
    aspect = random.uniform(*aspect_ratio_range)
    x_cells = math.sqrt(total_cells * aspect)
    z_cells = total_cells / x_cells

    # Round up to ensure we have enough cells, minimum 4 cells per side
    x_size = max(4, math.ceil(x_cells))
    z_size = max(4, math.ceil(z_cells))

    # Ensure total cells meets minimum
    while x_size * z_size < min_cells_needed:
        # Add one cell to the smaller dimension
        if x_size <= z_size:
            x_size += 1
        else:
            z_size += 1

    return (x_size, z_size)


def calculate_ratios_from_targets(
    room_type_counts: Dict[str, int],
) -> Dict[str, int]:
    """Calculate room ratios based on target sizes.
    
    Returns ratios that are integers (as required by LeafRoom).
    The ratios represent the relative size each room should have.
    
    Args:
        room_type_counts: Dict mapping room type to count
    
    Returns:
        Dict mapping room type to ratio (integer)
    """
    # Calculate target areas
    targets = {}
    for room_type, count in room_type_counts.items():
        spec = RoomSizeSpec.for_room_type(room_type)
        # Each instance of this room type gets the target size
        targets[room_type] = spec.target_sqm
    
    # Find smallest target to use as base unit
    min_target = min(targets.values())
    
    # Calculate ratios relative to smallest
    ratios = {}
    for room_type, target in targets.items():
        # Scale so smallest room gets ratio of ~3 (gives us granularity)
        ratio = round((target / min_target) * 3)
        ratios[room_type] = max(1, ratio)  # Minimum ratio of 1

    return ratios


def get_room_size_penalty(
    room_type: str,
    actual_area_sqm: float,
    penalty_scale: float = 0.1,
) -> float:
    """Calculate penalty for a room being outside its target size range.

    Returns a negative penalty (to be added to score).
    0 = within acceptable range
    Negative = outside range (more negative = worse)

    Args:
        room_type: The type of room
        actual_area_sqm: The actual area of the room in sqm
        penalty_scale: How much to penalize per sqm deviation
    """
    spec = RoomSizeSpec.for_room_type(room_type)

    if actual_area_sqm < spec.min_sqm:
        # Too small - penalize heavily
        deficit = spec.min_sqm - actual_area_sqm
        return -deficit * penalty_scale * 2  # Double penalty for too small
    elif actual_area_sqm > spec.max_sqm:
        # Too large - penalize moderately
        excess = actual_area_sqm - spec.max_sqm
        return -excess * penalty_scale
    else:
        # Within range - no penalty
        return 0.0


def get_hallway_shape_penalty_for_dims(
    width_m: float,
    length_m: float,
    penalty_scale: float = 0.2,
) -> float:
    """Calculate penalty for hallway not being narrow enough.

    Hallways should be elongated (long and narrow), not square.

    Args:
        width_m: The narrower dimension in meters
        length_m: The longer dimension in meters
        penalty_scale: How much to penalize
    """
    penalty = 0.0

    # Ensure width is the smaller dimension
    if width_m > length_m:
        width_m, length_m = length_m, width_m

    # Penalty for being too wide
    if width_m > HALLWAY_MAX_WIDTH_M:
        excess = width_m - HALLWAY_MAX_WIDTH_M
        penalty -= excess * penalty_scale * 2

    # Penalty for not being elongated enough
    if length_m > 0:
        aspect_ratio = length_m / width_m
        if aspect_ratio < HALLWAY_MIN_ASPECT_RATIO:
            # Too square-ish
            penalty -= (HALLWAY_MIN_ASPECT_RATIO - aspect_ratio) * penalty_scale

    return penalty


def validate_room_sizes(
    rooms: List[Dict],
    interior_boundary_scale: float,
) -> Tuple[bool, List[str]]:
    """Validate that all rooms are within acceptable size ranges.

    Args:
        rooms: List of room dicts with 'roomType' and 'floorPolygon'
        interior_boundary_scale: Scale factor for converting grid to meters

    Returns:
        (is_valid, list_of_issues)
    """
    issues = []

    for room in rooms:
        room_type = room.get("roomType")
        if room_type not in ROOM_SIZE_TARGETS_SQM:
            continue

        # Calculate room area from polygon
        poly = room.get("floorPolygon", [])
        if len(poly) < 3:
            issues.append(f"{room_type}: Invalid polygon")
            continue

        # Shoelace formula for polygon area
        n = len(poly)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += poly[i]["x"] * poly[j]["z"]
            area -= poly[j]["x"] * poly[i]["z"]
        area_sqm = abs(area) / 2.0

        spec = RoomSizeSpec.for_room_type(room_type)

        if area_sqm < spec.min_sqm:
            issues.append(
                f"{room_type} ({room.get('id', '?')}): {area_sqm:.1f} sqm "
                f"< min {spec.min_sqm:.1f} sqm"
            )
        elif area_sqm > spec.max_sqm:
            issues.append(
                f"{room_type} ({room.get('id', '?')}): {area_sqm:.1f} sqm "
                f"> max {spec.max_sqm:.1f} sqm"
            )

    return len(issues) == 0, issues

