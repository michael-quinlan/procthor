"""Room specifications that use hallways for realistic circulation.

These specs ensure:
1. Hallways act as central circulation spaces
2. Bedrooms connect to hallways or public rooms, not only to other bedrooms
3. Bathrooms connect to hallways or bedrooms appropriately
"""
from procthor.utils.types import LeafRoom, MetaRoom, PUBLIC_ROOM_TYPES, PRIVATE_ROOM_TYPES
from procthor.generation.room_specs import RoomSpec


def create_hallway_room_spec(
    room_spec_id: str,
    num_bedrooms: int = 2,
    num_bathrooms: int = 1,
    dims: tuple = (10, 10),
) -> RoomSpec:
    """Create a RoomSpec with a central hallway.

    Layout concept:
    - Public zone: Kitchen + LivingRoom (open plan)
    - Circulation: Hallway connecting public to private
    - Private zone: Bedrooms + Bathroom off the hallway

    Args:
        room_spec_id: Unique ID for this room spec
        num_bedrooms: Number of bedrooms (1-4)
        num_bathrooms: Number of bathrooms (1-2)
        dims: House dimensions as (x, z) in grid units

    Returns:
        RoomSpec object
    """
    room_id = 2  # 0 and 1 are reserved

    # Public zone: Kitchen and LivingRoom (can connect to each other)
    kitchen = LeafRoom(
        room_id=room_id,
        ratio=2,
        room_type="Kitchen",
    )
    room_id += 1

    living_room = LeafRoom(
        room_id=room_id,
        ratio=3,
        room_type="LivingRoom",
    )
    room_id += 1

    public_zone = MetaRoom(ratio=5, children=[kitchen, living_room])

    # Hallway: Central circulation space
    hallway = LeafRoom(
        room_id=room_id,
        ratio=1,
        room_type="Hallway",
    )
    room_id += 1

    # Private rooms: Bedrooms and bathrooms
    private_rooms = []

    for i in range(num_bedrooms):
        bedroom = LeafRoom(
            room_id=room_id,
            ratio=2,
            room_type="Bedroom",
            cannot_connect_only_to={"Bedroom", "Bathroom"},
            must_connect_to={"Hallway", "LivingRoom"},
        )
        private_rooms.append(bedroom)
        room_id += 1

    for i in range(num_bathrooms):
        bathroom = LeafRoom(
            room_id=room_id,
            ratio=1,
            room_type="Bathroom",
            avoid_doors_from_metarooms=True,
        )
        private_rooms.append(bathroom)
        room_id += 1

    private_zone = MetaRoom(ratio=num_bedrooms * 2 + num_bathrooms, children=private_rooms)

    # Combine: Hallway connects public and private zones
    house = MetaRoom(ratio=10, children=[public_zone, hallway, private_zone])

    return RoomSpec(
        room_spec_id=room_spec_id,
        sampling_weight=1,
        spec=[house],
        dims=lambda: dims,  # dims must be a callable
    )


# Pre-defined specs for common house layouts
HALLWAY_HOUSE_2BR_1BA = create_hallway_room_spec("hallway-2br-1ba", num_bedrooms=2, num_bathrooms=1, dims=(10, 10))
HALLWAY_HOUSE_3BR_2BA = create_hallway_room_spec("hallway-3br-2ba", num_bedrooms=3, num_bathrooms=2, dims=(12, 12))
HALLWAY_HOUSE_4BR_2BA = create_hallway_room_spec("hallway-4br-2ba", num_bedrooms=4, num_bathrooms=2, dims=(14, 14))


# Dict mapping name to RoomSpec
HALLWAY_ROOM_SPEC_SAMPLER = {
    "2br-1ba": HALLWAY_HOUSE_2BR_1BA,
    "3br-2ba": HALLWAY_HOUSE_3BR_2BA,
    "4br-2ba": HALLWAY_HOUSE_4BR_2BA,
}

