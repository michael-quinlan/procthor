"""Room connectivity validation for ProcTHOR floorplans.

This module provides functions to validate that generated floorplans have
sensible room connectivity - e.g., bedrooms shouldn't only be accessible
through other bedrooms.
"""
from typing import Dict, List, Set, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from .room_specs import RoomSpec

from procthor.utils.types import (
    LeafRoom,
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
    adjacency: Dict[int, Set[int]] = defaultdict(set)
    
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


def can_reach_public_room(
    room_id: int,
    adjacency: Dict[int, Set[int]],
    room_type_map: Dict[int, str],
    visited: Set[int] = None,
) -> bool:
    """Check if a room can reach a public room.
    
    A room "passes" if it:
    1. Is itself a public room, OR
    2. Is directly connected to a public room, OR
    3. Can reach a public room through any path
    
    Args:
        room_id: The room to check.
        adjacency: Room adjacency graph from build_room_adjacency_graph.
        room_type_map: Mapping from room_id to room_type.
        visited: Set of already visited rooms (for recursion).
    
    Returns:
        True if the room can reach a public room.
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

    Checks room-specific constraints like must_connect_to and
    cannot_connect_only_to (if defined on LeafRoom).

    Args:
        room_spec: The RoomSpec containing room definitions.
        adjacency: Room adjacency graph from build_room_adjacency_graph.

    Returns:
        List of error messages (empty if all constraints are satisfied).
    """
    errors: List[str] = []
    room_type_map = room_spec.room_type_map

    for room_id, room in room_spec.room_map.items():
        if not isinstance(room, LeafRoom):
            continue

        adjacent_ids = adjacency.get(room_id, set())
        adjacent_types = {room_type_map.get(rid) for rid in adjacent_ids if rid in room_type_map}

        # Check must_connect_to constraint (if defined)
        must_connect = getattr(room, 'must_connect_to', None)
        if must_connect:
            if not adjacent_types.intersection(must_connect):
                errors.append(
                    f"Room {room_id} ({room.room_type}) must connect to one of "
                    f"{must_connect}, but only connects to {adjacent_types}"
                )

        # Check cannot_connect_only_to constraint (if defined)
        cannot_only = getattr(room, 'cannot_connect_only_to', None)
        if cannot_only:
            if adjacent_types and adjacent_types.issubset(cannot_only):
                errors.append(
                    f"Room {room_id} ({room.room_type}) cannot connect only to "
                    f"{cannot_only}, but only connects to {adjacent_types}"
                )

    return errors


def validate_private_room_access(
    room_spec: "RoomSpec",
    adjacency: Dict[int, Set[int]],
) -> List[str]:
    """Validate that private rooms can reach public spaces.

    Args:
        room_spec: The RoomSpec containing room definitions.
        adjacency: Room adjacency graph from build_room_adjacency_graph.

    Returns:
        List of error messages (empty if all private rooms can reach public spaces).
    """
    errors: List[str] = []
    room_type_map = room_spec.room_type_map

    for room_id, room_type in room_type_map.items():
        if room_type in PRIVATE_ROOM_TYPES:
            if not can_reach_public_room(room_id, adjacency, room_type_map):
                errors.append(
                    f"Room {room_id} ({room_type}) cannot reach any public room "
                    f"(public rooms: {PUBLIC_ROOM_TYPES})"
                )

    return errors


def validate_hallway_connections(
    room_spec: "RoomSpec",
    adjacency: Dict[int, Set[int]],
) -> List[str]:
    """Validate hallway door connectivity rules.

    Rules:
    - Hallways must have doors to at least 2 rooms

    Args:
        room_spec: The RoomSpec containing room definitions.
        adjacency: Room adjacency graph from build_room_adjacency_graph.

    Returns:
        List of error messages (empty if all constraints satisfied).
    """
    errors: List[str] = []
    room_type_map = room_spec.room_type_map

    for room_id, room_type in room_type_map.items():
        if room_type == "Hallway":
            adjacent_count = len(adjacency.get(room_id, set()))
            if adjacent_count < 2:
                errors.append(
                    f"Hallway (room {room_id}) only has doors to {adjacent_count} room(s), "
                    f"needs at least 2"
                )

    return errors


def validate_strict_door_rules(
    room_spec: "RoomSpec",
    adjacency: Dict[int, Set[int]],
) -> List[str]:
    """Run all strict door connectivity validations.

    Args:
        room_spec: The RoomSpec containing room definitions.
        adjacency: Room adjacency graph from build_room_adjacency_graph.

    Returns:
        List of all error messages (empty if all constraints satisfied).
    """
    errors: List[str] = []
    errors.extend(validate_hallway_connections(room_spec, adjacency))
    return errors


def validate_connectivity(
    room_spec: "RoomSpec",
    adjacency: Dict[int, Set[int]],
) -> List[str]:
    """Validate all room connectivity constraints.

    This is a convenience function that runs all connectivity validations.

    Args:
        room_spec: The RoomSpec containing room definitions.
        adjacency: Room adjacency graph from build_room_adjacency_graph.

    Returns:
        List of error messages (empty if all constraints are satisfied).
    """
    errors: List[str] = []
    errors.extend(validate_room_constraints(room_spec, adjacency))
    errors.extend(validate_private_room_access(room_spec, adjacency))
    errors.extend(validate_strict_door_rules(room_spec, adjacency))
    return errors

