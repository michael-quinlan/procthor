"""Room proportion validation for early rejection of invalid floorplans."""

from typing import TYPE_CHECKING, Tuple

from shapely.geometry import Polygon

if TYPE_CHECKING:
    from . import PartialHouse


def validate_room_proportions(partial_house: "PartialHouse") -> Tuple[bool, str]:
    """Validate that room proportions are sensible.
    
    Checks three rules:
    1. LivingRoom >= any Bedroom
    2. Hallway <= any Bedroom
    3. LivingRoom >= Hallway
    
    Args:
        partial_house: PartialHouse after STRUCTURE stage
        
    Returns:
        Tuple of (passed: bool, reason: str) where reason explains failure if passed=False.
    """
    # Small tolerance for floating point comparisons (0.1 m² = ~1 sqft)
    EPSILON = 0.1
    
    # Extract room areas from xz_poly_map
    room_areas = {}
    for room_id, xz_poly in partial_house.house_structure.xz_poly_map.items():
        # Convert xz_poly to Shapely polygon to calculate area
        # xz_poly is a list of ((x0, z0), (x1, z1)) wall segments
        floor_polygon = []
        for ((x0, z0), (x1, z1)) in xz_poly:
            floor_polygon.append((x0, z0))
        floor_polygon.append((x1, z1))
        
        polygon = Polygon(floor_polygon)
        area = polygon.area
        
        # Get room type from room_spec
        room_type = partial_house.room_spec.room_type_map[room_id]
        
        if room_type not in room_areas:
            room_areas[room_type] = []
        room_areas[room_type].append(area)
    
    living_areas = room_areas.get('LivingRoom', [])
    bedroom_areas = room_areas.get('Bedroom', [])
    hallway_areas = room_areas.get('Hallway', [])
    
    if not living_areas:
        return True, ""  # No living room to validate
    
    max_living = max(living_areas)
    max_bedroom = max(bedroom_areas) if bedroom_areas else 0
    min_bedroom = min(bedroom_areas) if bedroom_areas else float('inf')
    max_hallway = max(hallway_areas) if hallway_areas else 0
    
    # Rule 1: LivingRoom >= any Bedroom (with epsilon tolerance)
    if max_bedroom > 0 and max_living < max_bedroom - EPSILON:
        return False, f"Living ({max_living:.1f}m²) < Bedroom ({max_bedroom:.1f}m²)"
    
    # Rule 2: Hallway <= any Bedroom (with epsilon tolerance)
    if max_hallway > 0 and min_bedroom < float('inf') and max_hallway > min_bedroom + EPSILON:
        return False, f"Hallway ({max_hallway:.1f}m²) > Bedroom ({min_bedroom:.1f}m²)"
    
    # Rule 3: LivingRoom >= Hallway (with epsilon tolerance)
    if max_hallway > 0 and max_living < max_hallway - EPSILON:
        return False, f"Living ({max_living:.1f}m²) < Hallway ({max_hallway:.1f}m²)"
    
    return True, ""

