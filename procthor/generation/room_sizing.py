"""
Room sizing constraints and automatic calculation of grid dimensions.

This module provides realistic room size targets and functions to automatically
calculate grid dimensions based on target square footage.
"""
import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple


# Room size targets in square meters (min, target, max)
# Based on typical US residential construction
ROOM_SIZE_TARGETS_SQM: Dict[str, Tuple[float, float, float]] = {
    # Room type: (min_sqm, target_sqm, max_sqm)
    "Bedroom": (9.0, 14.0, 25.0),       # 97-269 sqft, target ~150 sqft
    "Bathroom": (3.5, 6.0, 12.0),       # 38-129 sqft, target ~65 sqft
    "Kitchen": (7.0, 12.0, 20.0),       # 75-215 sqft, target ~130 sqft
    "LivingRoom": (14.0, 22.0, 35.0),   # 150-377 sqft, target ~237 sqft
    "Hallway": (2.5, 5.0, 10.0),        # 27-108 sqft, target ~54 sqft
}

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


def get_room_size_constraints(room_type: str) -> Tuple[float, float]:
    """Get min/max size constraints for a room type.

    Args:
        room_type: One of "Bedroom", "Bathroom", "Kitchen", "LivingRoom", "Hallway"

    Returns:
        (min_sqm, max_sqm) tuple
    """
    spec = RoomSizeSpec.for_room_type(room_type)
    return (spec.min_sqm, spec.max_sqm)


def calculate_target_house_size(
    room_type_counts: Dict[str, int],
    size_preference: str = "target",
) -> float:
    """Calculate total house size in sqm based on room counts.

    Args:
        room_type_counts: Dict mapping room type to count, e.g. {"Bedroom": 3, ...}
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


def calculate_grid_dims(room_spec) -> Tuple[int, int]:
    """Calculate grid dimensions for a room spec based on target room sizes.

    Counts rooms by type in the spec and calculates appropriate grid size.

    Args:
        room_spec: A RoomSpec object with spec attribute containing rooms

    Returns:
        (x_size, z_size) grid dimensions
    """
    # Count rooms by type from spec
    room_type_counts: Dict[str, int] = {}
    _count_rooms_in_spec(room_spec.spec, room_type_counts)

    return calculate_dims_for_house(room_type_counts)


def _count_rooms_in_spec(spec_items, counts: Dict[str, int]) -> None:
    """Recursively count rooms by type in a room spec."""
    for item in spec_items:
        if hasattr(item, 'room_type') and item.room_type:
            counts[item.room_type] = counts.get(item.room_type, 0) + 1
        if hasattr(item, 'children'):
            _count_rooms_in_spec(item.children, counts)


def calculate_dims_for_house(
    room_type_counts: Dict[str, int],
    interior_boundary_scale: float = 1.9,
    size_preference: str = "target",
    aspect_ratio_range: Tuple[float, float] = (0.8, 1.25),
    min_cells_per_room: int = 6,
    boundary_cut_buffer: float = 1.3,
) -> Tuple[int, int]:
    """Calculate grid dimensions for a house based on target room sizes.

    Args:
        room_type_counts: Dict mapping room type to count
        interior_boundary_scale: Scale factor (meters per grid cell)
        size_preference: "min", "target", or "max"
        aspect_ratio_range: Range for random aspect ratio variation
        min_cells_per_room: Minimum cells per room to ensure fitting
        boundary_cut_buffer: Extra buffer for boundary cuts

    Returns:
        (x_size, z_size) grid dimensions
    """
    target_area_sqm = calculate_target_house_size(room_type_counts, size_preference)

    # Account for ~15% loss due to walls/inefficiency
    effective_area_sqm = target_area_sqm * 1.15

    # Calculate total grid cells needed
    cell_area_sqm = interior_boundary_scale ** 2
    total_cells = effective_area_sqm / cell_area_sqm

    # Ensure minimum cells for room fitting
    total_rooms = sum(room_type_counts.values())
    min_cells_needed = total_rooms * min_cells_per_room * boundary_cut_buffer
    total_cells = max(total_cells, min_cells_needed)

    # Calculate side length with random aspect ratio
    aspect = random.uniform(*aspect_ratio_range)
    x_cells = math.sqrt(total_cells * aspect)
    z_cells = total_cells / x_cells

    # Round up to ensure enough cells, minimum 4 per side
    x_size = max(4, math.ceil(x_cells))
    z_size = max(4, math.ceil(z_cells))

    # Ensure total cells meets minimum
    while x_size * z_size < min_cells_needed:
        if x_size <= z_size:
            x_size += 1
        else:
            z_size += 1

    return (x_size, z_size)

