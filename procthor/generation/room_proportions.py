"""Validation for room polygon proportions.

Ensures houses have sensible room size relationships:
- LivingRoom area >= any Bedroom area
- Hallway area <= any Bedroom area  
- LivingRoom area >= Hallway area
"""

import logging
from typing import Dict, List, Tuple

from shapely.geometry import Polygon

from .house import HouseStructure
from .room_specs import RoomSpec


class InvalidRoomProportions(Exception):
    """Raised when room polygon proportions are invalid."""

    pass


def _get_polygon_areas(
    xz_poly_map: Dict[int, List[Tuple[Tuple[float, float], Tuple[float, float]]]],
    room_type_map: Dict[int, str],
) -> Dict[str, List[Tuple[int, float]]]:
    """Calculate polygon areas for each room, grouped by room type.

    Returns:
        Dict mapping room type (e.g., "LivingRoom") to list of (room_id, area) tuples.
    """
    areas_by_type: Dict[str, List[Tuple[int, float]]] = {}

    for room_id, xz_poly in xz_poly_map.items():
        room_type = room_type_map.get(room_id)
        if room_type is None:
            continue

        # Build polygon from wall loop
        floor_polygon = []
        for ((x0, z0), (x1, z1)) in xz_poly:
            floor_polygon.append((x0, z0))
        if xz_poly:
            # Add the last point to close the polygon
            floor_polygon.append((xz_poly[-1][1][0], xz_poly[-1][1][1]))

        if len(floor_polygon) < 3:
            continue

        poly = Polygon(floor_polygon)
        area = poly.area

        if room_type not in areas_by_type:
            areas_by_type[room_type] = []
        areas_by_type[room_type].append((room_id, area))

    return areas_by_type


def validate_polygon_proportions(
    house_structure: HouseStructure,
    room_spec: RoomSpec,
) -> None:
    """Validate that room polygon proportions are sensible.

    Raises InvalidRoomProportions if:
    - Any LivingRoom area < any Bedroom area
    - Any Hallway area > any Bedroom area
    - Any LivingRoom area < any Hallway area

    Args:
        house_structure: The generated house structure with xz_poly_map.
        room_spec: The room specification with room_type_map.

    Raises:
        InvalidRoomProportions: If validation fails.
    """
    areas_by_type = _get_polygon_areas(
        xz_poly_map=house_structure.xz_poly_map,
        room_type_map=room_spec.room_type_map,
    )

    living_rooms = areas_by_type.get("LivingRoom", [])
    bedrooms = areas_by_type.get("Bedroom", [])
    hallways = areas_by_type.get("Hallway", [])

    # Rule 1: LivingRoom area >= any Bedroom area
    for lr_id, lr_area in living_rooms:
        for br_id, br_area in bedrooms:
            if lr_area < br_area:
                logging.debug(
                    f"Room proportion validation failed: LivingRoom {lr_id} "
                    f"(area={lr_area:.2f}) < Bedroom {br_id} (area={br_area:.2f})"
                )
                raise InvalidRoomProportions(
                    f"LivingRoom {lr_id} (area={lr_area:.2f}) is smaller than "
                    f"Bedroom {br_id} (area={br_area:.2f})"
                )

    # Rule 2: Hallway area <= any Bedroom area
    for hw_id, hw_area in hallways:
        for br_id, br_area in bedrooms:
            if hw_area > br_area:
                logging.debug(
                    f"Room proportion validation failed: Hallway {hw_id} "
                    f"(area={hw_area:.2f}) > Bedroom {br_id} (area={br_area:.2f})"
                )
                raise InvalidRoomProportions(
                    f"Hallway {hw_id} (area={hw_area:.2f}) is larger than "
                    f"Bedroom {br_id} (area={br_area:.2f})"
                )

    # Rule 3: LivingRoom area >= Hallway area
    for lr_id, lr_area in living_rooms:
        for hw_id, hw_area in hallways:
            if lr_area < hw_area:
                logging.debug(
                    f"Room proportion validation failed: LivingRoom {lr_id} "
                    f"(area={lr_area:.2f}) < Hallway {hw_id} (area={hw_area:.2f})"
                )
                raise InvalidRoomProportions(
                    f"LivingRoom {lr_id} (area={lr_area:.2f}) is smaller than "
                    f"Hallway {hw_id} (area={hw_area:.2f})"
                )

    # Log success with room areas for debugging
    if living_rooms or bedrooms or hallways:
        logging.debug(
            f"Room proportion validation passed: "
            f"LivingRooms={[(id, f'{a:.2f}') for id, a in living_rooms]}, "
            f"Bedrooms={[(id, f'{a:.2f}') for id, a in bedrooms]}, "
            f"Hallways={[(id, f'{a:.2f}') for id, a in hallways]}"
        )

