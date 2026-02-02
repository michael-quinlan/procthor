"""Room specifications that use hallways for realistic circulation.

These specs ensure:
1. Hallways act as central circulation spaces
2. Bedrooms connect to hallways or public rooms
3. Bathrooms connect appropriately with avoid_doors_from_metarooms
4. Room sizes are based on realistic ratios
"""
from typing import Dict, Optional, Tuple
import random

from procthor.utils.types import LeafRoom, MetaRoom
from procthor.generation.room_specs import RoomSpec, RoomSpecSampler


# Room size ratios based on typical US residential construction
# These are relative ratios (used directly in LeafRoom)
RATIOS: Dict[str, int] = {
    "Bedroom": 4,       # ~14 sqm target
    "Bathroom": 2,      # ~6 sqm target
    "Kitchen": 3,       # ~12 sqm target
    "LivingRoom": 6,    # ~22 sqm target
    "Hallway": 2,       # ~5 sqm target (narrow circulation)
}


def create_hallway_room_spec(
    room_spec_id: str,
    num_bedrooms: int = 2,
    num_bathrooms: int = 1,
    dims: Optional[Tuple[int, int]] = None,
) -> RoomSpec:
    """Create a RoomSpec with a central hallway and realistic room sizes.

    Layout concept:
    - Public zone: Kitchen + LivingRoom (open plan)
    - Circulation: Hallway connecting public to private
    - Private zone: Bedrooms + Bathroom off the hallway

    Args:
        room_spec_id: Unique ID for this room spec
        num_bedrooms: Number of bedrooms (1-4)
        num_bathrooms: Number of bathrooms (1-2)
        dims: Optional house dimensions as (x, z) in grid units.
              If None, calculated automatically based on room count.

    Returns:
        RoomSpec object
    """
    room_id = 2  # 0 and 1 are reserved

    # Public zone: Kitchen and LivingRoom (can connect to each other)
    kitchen = LeafRoom(
        room_id=room_id,
        ratio=RATIOS["Kitchen"],
        room_type="Kitchen",
    )
    room_id += 1

    living_room = LeafRoom(
        room_id=room_id,
        ratio=RATIOS["LivingRoom"],
        room_type="LivingRoom",
    )
    room_id += 1

    # Public zone ratio is sum of kitchen + living room
    public_zone_ratio = RATIOS["Kitchen"] + RATIOS["LivingRoom"]
    public_zone = MetaRoom(ratio=public_zone_ratio, children=[kitchen, living_room])

    # Hallway: Central circulation space
    hallway = LeafRoom(
        room_id=room_id,
        ratio=RATIOS["Hallway"],
        room_type="Hallway",
    )
    room_id += 1

    # Private rooms: Bedrooms and bathrooms
    private_rooms = []
    private_zone_ratio = 0

    for i in range(num_bedrooms):
        bedroom = LeafRoom(
            room_id=room_id,
            ratio=RATIOS["Bedroom"],
            room_type="Bedroom",
        )
        private_rooms.append(bedroom)
        private_zone_ratio += RATIOS["Bedroom"]
        room_id += 1

    for i in range(num_bathrooms):
        bathroom = LeafRoom(
            room_id=room_id,
            ratio=RATIOS["Bathroom"],
            room_type="Bathroom",
            avoid_doors_from_metarooms=True,
        )
        private_rooms.append(bathroom)
        private_zone_ratio += RATIOS["Bathroom"]
        room_id += 1

    private_zone = MetaRoom(ratio=private_zone_ratio, children=private_rooms)

    # Combine: Hallway connects public and private zones
    total_ratio = public_zone_ratio + RATIOS["Hallway"] + private_zone_ratio
    house = MetaRoom(ratio=total_ratio, children=[public_zone, hallway, private_zone])

    # Auto-calculate dims if not provided
    if dims is None:
        # Base size on room count: ~4-5 cells per room
        total_rooms = 2 + 1 + num_bedrooms + num_bathrooms  # kitchen, living, hallway, beds, baths
        base_cells = total_rooms * 5
        side = int(base_cells ** 0.5) + 1
        # Add some randomness
        x_size = random.randint(side, side + 2)
        z_size = random.randint(side - 1, side + 1)
        dims = (max(5, x_size), max(4, z_size))

    final_dims = dims

    return RoomSpec(
        room_spec_id=room_spec_id,
        sampling_weight=1,
        spec=[house],
        dims=lambda d=final_dims: d,
    )


# Pre-defined specs for common house layouts
HALLWAY_SPECS = [
    create_hallway_room_spec("hallway-2br-1ba", num_bedrooms=2, num_bathrooms=1),
    create_hallway_room_spec("hallway-3br-2ba", num_bedrooms=3, num_bathrooms=2),
    create_hallway_room_spec("hallway-4br-2ba", num_bedrooms=4, num_bathrooms=2),
]

# Sampler for hallway-based room specs
HALLWAY_ROOM_SPEC_SAMPLER = RoomSpecSampler(HALLWAY_SPECS)

