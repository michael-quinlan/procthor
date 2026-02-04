"""Room connectivity validation for ProcTHOR floorplans.

This module provides functions to validate that generated floorplans have
sensible room connectivity - e.g., bedrooms shouldn't only be accessible
through other bedrooms.
"""
from typing import Dict, List, Set, Tuple, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from .room_specs import RoomSpec

from procthor.utils.types import (
    LeafRoom,
    MetaRoom,
    PUBLIC_ROOM_TYPES,
    PRIVATE_ROOM_TYPES,
)


class InvalidConnectivity(Exception):
    """Raised when room connectivity constraints are violated."""
    pass


def build_room_adjacency_graph(
    doors: List[dict],
    room_type_map: Dict[int, str],
) -> Dict[int, Set[int]]:
    """Build an adjacency graph from door connections.
    
    Args:
        doors: List of door dictionaries with 'room0' and 'room1' keys.
        room_type_map: Mapping from room_id to room_type.
    
    Returns:
        Dict mapping room_id to set of adjacent room_ids.
    """
    adjacency = defaultdict(set)
    
    for door in doors:
        # Parse room IDs from door format "room|4"
        room0_str = door.get("room0", "")
        room1_str = door.get("room1", "")
        
        try:
            room0_id = int(room0_str.split("|")[1])
            room1_id = int(room1_str.split("|")[1])
        except (IndexError, ValueError):
            continue
        
        # Only add connections between interior rooms
        if room0_id in room_type_map and room1_id in room_type_map:
            adjacency[room0_id].add(room1_id)
            adjacency[room1_id].add(room0_id)
    
    return dict(adjacency)


def get_adjacent_room_types(
    room_id: int,
    adjacency: Dict[int, Set[int]],
    room_type_map: Dict[int, str],
) -> Set[str]:
    """Get the room types adjacent to a given room."""
    adjacent_ids = adjacency.get(room_id, set())
    return {room_type_map[rid] for rid in adjacent_ids if rid in room_type_map}


def can_reach_public_room(
    room_id: int,
    adjacency: Dict[int, Set[int]],
    room_type_map: Dict[int, str],
    visited: Set[int] = None,
) -> bool:
    """Check if a room can reach a public room without going only through private rooms.
    
    A room "passes" if it:
    1. Is itself a public room, OR
    2. Is directly connected to a public room, OR
    3. Can reach a public room through any path
    """
    if visited is None:
        visited = set()
    
    if room_id in visited:
        return False
    visited.add(room_id)
    
    room_type = room_type_map.get(room_id)
    
    # If this room is public, we've reached one
    if room_type in PUBLIC_ROOM_TYPES:
        return True
    
    # Check adjacent rooms
    for adjacent_id in adjacency.get(room_id, set()):
        if adjacent_id not in visited:
            if can_reach_public_room(adjacent_id, adjacency, room_type_map, visited):
                return True
    
    return False


def validate_room_constraints(
    room_spec: "RoomSpec",
    adjacency: Dict[int, Set[int]],
) -> List[str]:
    """Validate that all room connectivity constraints are satisfied.
    
    Returns:
        List of error messages (empty if all constraints are satisfied).
    """
    errors = []
    room_type_map = room_spec.room_type_map
    
    for room_id, room in room_spec.room_map.items():
        if not isinstance(room, LeafRoom):
            continue
        
        adjacent_types = get_adjacent_room_types(room_id, adjacency, room_type_map)
        
        # Check must_connect_to constraint
        if room.must_connect_to:
            if not adjacent_types.intersection(room.must_connect_to):
                errors.append(
                    f"Room {room_id} ({room.room_type}) must connect to one of "
                    f"{room.must_connect_to}, but only connects to {adjacent_types}"
                )
        
        # Check cannot_connect_only_to constraint
        if room.cannot_connect_only_to:
            if adjacent_types and adjacent_types.issubset(room.cannot_connect_only_to):
                errors.append(
                    f"Room {room_id} ({room.room_type}) cannot connect only to "
                    f"{room.cannot_connect_only_to}, but only connects to {adjacent_types}"
                )
    
    return errors


def validate_private_room_access(
    room_spec: "RoomSpec",
    adjacency: Dict[int, Set[int]],
) -> List[str]:
    """Validate that private rooms can reach public spaces.

    Returns:
        List of error messages (empty if all private rooms can reach public spaces).
    """
    errors = []
    room_type_map = room_spec.room_type_map

    for room_id, room_type in room_type_map.items():
        if room_type in PRIVATE_ROOM_TYPES:
            if not can_reach_public_room(room_id, adjacency, room_type_map):
                errors.append(
                    f"Room {room_id} ({room_type}) cannot reach any public room "
                    f"(public rooms: {PUBLIC_ROOM_TYPES})"
                )

    return errors


def validate_bathroom_public_access(
    room_spec: "RoomSpec",
    adjacency: Dict[int, Set[int]],
) -> List[str]:
    """Validate bathroom access constraints.

    Rules:
    1. At least one bathroom MUST have a door to a public room (Hallway, LivingRoom, Kitchen)
    2. Second bathroom (if present) MUST connect to EXACTLY ONE bedroom and NOTHING else (strict en-suite)

    Returns:
        List of error messages (empty if all constraints are satisfied).
    """
    errors = []
    room_type_map = room_spec.room_type_map

    # Find all bathrooms
    bathrooms = sorted([rid for rid, rtype in room_type_map.items() if rtype == "Bathroom"])

    if not bathrooms:
        return []  # No bathrooms, nothing to validate

    # Rule 1: Check if at least one bathroom is directly connected to a public room
    has_public_bathroom = False
    for bathroom_id in bathrooms:
        adjacent_ids = adjacency.get(bathroom_id, set())
        adjacent_types = {room_type_map.get(aid) for aid in adjacent_ids}
        if adjacent_types.intersection(PUBLIC_ROOM_TYPES):
            has_public_bathroom = True
            break

    if not has_public_bathroom:
        errors.append(
            "No bathroom has direct door access to a public room. "
            f"At least one bathroom should connect to {PUBLIC_ROOM_TYPES}"
        )

    # Rule 2: Second+ bathrooms must be strict en-suites (exactly 1 door to exactly 1 bedroom)
    if len(bathrooms) >= 2:
        for bathroom_id in bathrooms[1:]:  # Skip first bathroom
            adjacent_ids = adjacency.get(bathroom_id, set())
            num_connections = len(adjacent_ids)

            # Check if connected to exactly 1 room
            if num_connections != 1:
                errors.append(
                    f"En-suite bathroom (room {bathroom_id}) must have exactly 1 door, "
                    f"but has {num_connections} connections"
                )
                continue

            # Check if that 1 room is a bedroom
            connected_room_id = list(adjacent_ids)[0]
            connected_room_type = room_type_map.get(connected_room_id)
            if connected_room_type != "Bedroom":
                errors.append(
                    f"En-suite bathroom (room {bathroom_id}) must connect to a bedroom, "
                    f"but connects to {connected_room_type} (room {connected_room_id})"
                )

    return errors


def validate_hallway_connections(
    room_spec: "RoomSpec",
    adjacency: Dict[int, Set[int]],
) -> List[str]:
    """Validate hallway door connectivity rules.

    Rules:
    - Hallways must have a door to LivingRoom or Kitchen (prefer LivingRoom)
    - Hallways must have doors to at least 2 rooms

    Returns:
        List of error messages (empty if all constraints satisfied).
    """
    errors = []
    room_type_map = room_spec.room_type_map

    for room_id, room_type in room_type_map.items():
        if room_type == "Hallway":
            adjacent_ids = adjacency.get(room_id, set())
            adjacent_types = {room_type_map.get(aid) for aid in adjacent_ids}

            # Hallway must have door to LivingRoom or Kitchen
            if not adjacent_types.intersection({"LivingRoom", "Kitchen"}):
                errors.append(
                    f"Hallway (room {room_id}) has no door to LivingRoom or Kitchen"
                )

            # Hallway must connect at least 2 rooms via doors
            if len(adjacent_ids) < 2:
                errors.append(
                    f"Hallway (room {room_id}) only has doors to {len(adjacent_ids)} room(s), needs at least 2"
                )

    return errors


def validate_bedroom_access(
    room_spec: "RoomSpec",
    adjacency: Dict[int, Set[int]],
) -> List[str]:
    """Validate that bedrooms have door access to LivingRoom or Hallway.

    Returns:
        List of error messages (empty if all constraints satisfied).
    """
    errors = []
    room_type_map = room_spec.room_type_map

    for room_id, room_type in room_type_map.items():
        if room_type == "Bedroom":
            adjacent_ids = adjacency.get(room_id, set())
            adjacent_types = {room_type_map.get(aid) for aid in adjacent_ids}

            if not adjacent_types.intersection({"LivingRoom", "Hallway"}):
                errors.append(
                    f"Bedroom (room {room_id}) has no door to LivingRoom or Hallway, "
                    f"only connects to: {adjacent_types}"
                )

    return errors


def validate_no_bedroom_to_bedroom_doors(
    room_spec: "RoomSpec",
    adjacency: Dict[int, Set[int]],
) -> List[str]:
    """Validate that bedrooms don't have doors directly to other bedrooms.

    Each bedroom should be accessed independently from a hallway or living room,
    not through another bedroom.

    Returns:
        List of error messages (empty if constraint satisfied).
    """
    errors = []
    room_type_map = room_spec.room_type_map

    # Track which bedroom pairs we've already reported
    reported_pairs = set()

    for room_id, room_type in room_type_map.items():
        if room_type == "Bedroom":
            adjacent_ids = adjacency.get(room_id, set())
            for adj_id in adjacent_ids:
                if room_type_map.get(adj_id) == "Bedroom":
                    # Normalize pair to avoid duplicate reports
                    pair = (min(room_id, adj_id), max(room_id, adj_id))
                    if pair not in reported_pairs:
                        reported_pairs.add(pair)
                        errors.append(
                            f"Bedroom (room {pair[0]}) has door to another Bedroom (room {pair[1]}). "
                            f"Bedrooms should only connect to Hallway or LivingRoom."
                        )

    return errors


def validate_strict_door_rules(
    room_spec: "RoomSpec",
    adjacency: Dict[int, Set[int]],
) -> List[str]:
    """Run all strict door connectivity validations.

    Returns:
        List of all error messages (empty if all constraints satisfied).
    """
    errors = []
    errors.extend(validate_bathroom_public_access(room_spec, adjacency))
    errors.extend(validate_hallway_connections(room_spec, adjacency))
    errors.extend(validate_bedroom_access(room_spec, adjacency))
    # Note: bedroom-to-bedroom doors are prevented at placement time in doors.py
    # We don't validate here because adjacent bedrooms are fine - we just don't put doors between them
    errors.extend(validate_no_bedroom_to_bedroom_doors(room_spec, adjacency))
    return errors

