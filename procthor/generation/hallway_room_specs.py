"""Room specifications that use hallways for realistic circulation.

These specs ensure:
1. Hallways act as central circulation spaces
2. Bedrooms connect to hallways or public rooms, not only to other bedrooms
3. Bathrooms connect to hallways or bedrooms appropriately
4. Room sizes are based on realistic targets (using room_sizing module)
"""
from typing import Optional, Tuple
from procthor.utils.types import LeafRoom, MetaRoom, PUBLIC_ROOM_TYPES, PRIVATE_ROOM_TYPES
from procthor.generation.room_specs import RoomSpec, RoomSpecSampler
from procthor.generation.room_sizing import (
    calculate_dims_for_house,
    calculate_ratios_from_targets,
    ROOM_SIZE_TARGETS_SQM,
)


# Default interior boundary scale (meters per grid cell)
DEFAULT_INTERIOR_SCALE = 1.9


def create_hallway_room_spec(
    room_spec_id: str,
    num_bedrooms: int = 2,
    num_bathrooms: int = 1,
    dims: Optional[Tuple[int, int]] = None,
    interior_boundary_scale: float = DEFAULT_INTERIOR_SCALE,
    size_preference: str = "target",
) -> RoomSpec:
    """Create a RoomSpec with a central hallway and realistic room sizes.

    Layout concept:
    - Public zone: Kitchen + LivingRoom (open plan)
    - Circulation: Hallway connecting public to private
    - Private zone: Bedrooms + Bathroom off the hallway

    Room sizes are automatically calculated based on realistic targets
    from the room_sizing module.

    Args:
        room_spec_id: Unique ID for this room spec
        num_bedrooms: Number of bedrooms (1-4)
        num_bathrooms: Number of bathrooms (1-2)
        dims: Optional house dimensions as (x, z) in grid units.
              If None, calculated automatically from target room sizes.
        interior_boundary_scale: Meters per grid cell for size calculation
        size_preference: "min", "target", or "max" for room sizing

    Returns:
        RoomSpec object
    """
    # Count rooms by type for auto-sizing
    room_type_counts = {
        "Kitchen": 1,
        "LivingRoom": 1,
        "Hallway": 1,
        "Bedroom": num_bedrooms,
        "Bathroom": num_bathrooms,
    }

    # Get target-based ratios
    ratios = calculate_ratios_from_targets(room_type_counts)

    # Auto-calculate dims if not provided
    if dims is None:
        dims = calculate_dims_for_house(
            room_type_counts,
            interior_boundary_scale,
            size_preference,
        )

    room_id = 2  # 0 and 1 are reserved

    # Public zone: Kitchen and LivingRoom (can connect to each other)
    kitchen = LeafRoom(
        room_id=room_id,
        ratio=ratios["Kitchen"],
        room_type="Kitchen",
    )
    room_id += 1

    living_room = LeafRoom(
        room_id=room_id,
        ratio=ratios["LivingRoom"],
        room_type="LivingRoom",
    )
    room_id += 1

    # Public zone ratio is sum of kitchen + living room
    public_zone_ratio = ratios["Kitchen"] + ratios["LivingRoom"]
    public_zone = MetaRoom(ratio=public_zone_ratio, children=[kitchen, living_room])

    # Hallway: Central circulation space
    hallway = LeafRoom(
        room_id=room_id,
        ratio=ratios["Hallway"],
        room_type="Hallway",
    )
    room_id += 1

    # Private rooms: Bedrooms and bathrooms
    private_rooms = []
    private_zone_ratio = 0

    for i in range(num_bedrooms):
        bedroom = LeafRoom(
            room_id=room_id,
            ratio=ratios["Bedroom"],
            room_type="Bedroom",
            cannot_connect_only_to={"Bedroom", "Bathroom"},
            must_connect_to={"Hallway", "LivingRoom"},
        )
        private_rooms.append(bedroom)
        private_zone_ratio += ratios["Bedroom"]
        room_id += 1

    for i in range(num_bathrooms):
        bathroom = LeafRoom(
            room_id=room_id,
            ratio=ratios["Bathroom"],
            room_type="Bathroom",
            avoid_doors_from_metarooms=True,
        )
        private_rooms.append(bathroom)
        private_zone_ratio += ratios["Bathroom"]
        room_id += 1

    private_zone = MetaRoom(ratio=private_zone_ratio, children=private_rooms)

    # Combine: Hallway connects public and private zones
    total_ratio = public_zone_ratio + ratios["Hallway"] + private_zone_ratio
    house = MetaRoom(ratio=total_ratio, children=[public_zone, hallway, private_zone])

    # Capture dims for closure
    final_dims = dims

    return RoomSpec(
        room_spec_id=room_spec_id,
        sampling_weight=1,
        spec=[house],
        dims=lambda: final_dims,
    )


# Pre-defined specs for common house layouts (with auto-calculated dims)
# dims=None means dimensions are calculated automatically from target room sizes
HALLWAY_HOUSE_2BR_1BA = create_hallway_room_spec(
    "hallway-2br-1ba",
    num_bedrooms=2,
    num_bathrooms=1,
    dims=None,  # Auto-calculate
)
HALLWAY_HOUSE_3BR_2BA = create_hallway_room_spec(
    "hallway-3br-2ba",
    num_bedrooms=3,
    num_bathrooms=2,
    dims=None,  # Auto-calculate
)
HALLWAY_HOUSE_4BR_2BA = create_hallway_room_spec(
    "hallway-4br-2ba",
    num_bedrooms=4,
    num_bathrooms=2,
    dims=None,  # Auto-calculate
)


# RoomSpecSampler for hallway-based house layouts
HALLWAY_ROOM_SPEC_SAMPLER = RoomSpecSampler(
    [
        HALLWAY_HOUSE_2BR_1BA,
        HALLWAY_HOUSE_3BR_2BA,
        HALLWAY_HOUSE_4BR_2BA,
    ]
)

