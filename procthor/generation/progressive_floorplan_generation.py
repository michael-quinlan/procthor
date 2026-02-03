"""Progressive floorplan generation.

This module implements a progressive approach to floorplan generation where rooms
are placed one at a time, with hallways growing to connect them, rather than
the all-at-once approach in floorplan_generation.py.
"""
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from procthor.constants import EMPTY_ROOM_ID, OUTDOOR_ROOM_ID
from procthor.generation.room_specs import RoomSpec
from procthor.utils.types import InvalidFloorplan, LeafRoom, MetaRoom


def _get_shared_wall_length(
    room1_cells: set,
    room2_cells: set,
) -> int:
    """Count how many cells of shared wall exist between two rooms.

    A shared wall cell is where a cell from room1 is orthogonally adjacent
    to a cell from room2.

    Args:
        room1_cells: Set of (y, x) tuples for room 1.
        room2_cells: Set of (y, x) tuples for room 2.

    Returns:
        Number of shared wall cell pairs (each pair is one grid edge).

    Example:
        >>> cells1 = {(0, 0), (0, 1), (1, 0), (1, 1)}
        >>> cells2 = {(0, 2), (1, 2)}
        >>> _get_shared_wall_length(cells1, cells2)
        2
    """
    shared_count = 0
    for (y1, x1) in room1_cells:
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (y1 + dy, x1 + dx)
            if neighbor in room2_cells:
                shared_count += 1
    return shared_count


def _get_room_cells(room_id: int, floorplan: np.ndarray) -> set:
    """Get all cells belonging to a room.

    Args:
        room_id: The room ID to find cells for.
        floorplan: The floorplan grid.

    Returns:
        Set of (y, x) tuples for all cells with the given room_id.
    """
    coords = np.argwhere(floorplan == room_id)
    return {(int(y), int(x)) for y, x in coords}


def _find_adjacent_room_ids(
    adjacent_to: List[str],
    floorplan: np.ndarray,
    room_type_map: Dict[int, str],
) -> List[int]:
    """Find room IDs for rooms of the specified types that exist in the floorplan.

    Args:
        adjacent_to: List of room types to find (e.g., ["LivingRoom", "Hallway"]).
        floorplan: The floorplan grid.
        room_type_map: Mapping of room_id to room_type.

    Returns:
        List of room IDs that match the specified types and exist in the floorplan.
    """
    result = []
    for room_id, room_type in room_type_map.items():
        if room_type in adjacent_to:
            if (floorplan == room_id).any():
                result.append(room_id)
    return result


def place_room(
    room: Union[LeafRoom, MetaRoom],
    floorplan: np.ndarray,
    adjacent_to: Optional[List[str]] = None,
    room_type_map: Optional[Dict[int, str]] = None,
    min_shared_wall: int = 1,
) -> bool:
    """Place a single room at a valid position in the floorplan.

    Finds a position where the room can be placed adjacent to specified room types,
    ensuring at least `min_shared_wall` cells of shared wall exist for door placement.

    Algorithm:
    1. If adjacent_to is specified, find all cells adjacent to those room types
    2. For each candidate position (empty cell adjacent to target rooms):
       a. Temporarily place the room (single cell initially)
       b. Calculate shared wall length with target rooms
       c. If shared wall >= min_shared_wall, accept placement
    3. If no adjacent_to constraint, find any empty cell with good placement
    4. The room is grown later by grow_rect/grow_l_shape functions

    Args:
        room: The room to place (LeafRoom or MetaRoom).
        floorplan: The floorplan grid to place the room in (modified in-place).
        adjacent_to: Optional list of room types this room must be adjacent to.
            If None, the room can be placed anywhere with empty space.
        room_type_map: Mapping of room_id to room_type. Required if adjacent_to
            is specified.
        min_shared_wall: Minimum cells of shared wall required for door placement.
            Default is 2 cells (standard door width).

    Returns:
        True if the room was placed successfully, False otherwise.

    Example:
        >>> import numpy as np
        >>> floorplan = np.zeros((5, 5), dtype=int)
        >>> floorplan[0:2, 0:3] = 2  # LivingRoom with ID 2
        >>> room = type('Room', (), {'room_id': 3, 'room_type': 'Kitchen', 'ratio': 2})()
        >>> room_type_map = {2: 'LivingRoom'}
        >>> result = place_room(room, floorplan, adjacent_to=['LivingRoom'],
        ...                     room_type_map=room_type_map, min_shared_wall=2)
        >>> result  # Should be True if valid placement found
        True
        >>> (floorplan == 3).sum() >= 1  # Room was placed
        True
    """
    rows, cols = floorplan.shape

    # Find candidate positions (empty cells)
    empty_mask = floorplan == EMPTY_ROOM_ID
    if not empty_mask.any():
        return False  # No empty cells available

    # If adjacent_to is specified, find target room IDs and prioritize adjacent cells
    target_room_ids = []
    if adjacent_to and room_type_map:
        target_room_ids = _find_adjacent_room_ids(adjacent_to, floorplan, room_type_map)
        if not target_room_ids:
            # No target rooms exist yet - can't satisfy adjacency constraint
            # Place anywhere for now (first room of its type)
            pass

    # Build candidate positions with scoring
    # Priority: cells adjacent to target rooms, then any empty cells
    candidates = []

    empty_coords = np.argwhere(empty_mask)
    for y, x in empty_coords:
        y, x = int(y), int(x)

        # Check adjacency to target rooms
        adjacent_to_target = False
        shared_wall_potential = 0

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < rows and 0 <= nx < cols:
                neighbor_id = floorplan[ny, nx]
                if neighbor_id in target_room_ids:
                    adjacent_to_target = True
                    shared_wall_potential += 1

        # Score: higher is better
        # - Adjacent to target room: +100
        # - More adjacent edges to target: +10 per edge
        # - Not on boundary: +1 (allows room to grow in more directions)
        score = 0
        if adjacent_to_target:
            score += 100 + shared_wall_potential * 10

        # Prefer interior cells (not on edge)
        if 0 < y < rows - 1 and 0 < x < cols - 1:
            score += 1

        candidates.append((score, y, x))

    if not candidates:
        return False

    # Sort by score (descending) and try placements
    candidates.sort(key=lambda c: c[0], reverse=True)

    # If we have adjacency constraints and target rooms exist, we need to find
    # a placement that can potentially achieve min_shared_wall cells
    if target_room_ids and min_shared_wall > 0:
        # Try to find a placement where we can grow to meet the shared wall requirement
        # For initial placement, we place a single cell and check if growing is feasible
        for score, y, x in candidates:
            if score >= 100:  # This cell is adjacent to a target room
                # Check if placing here could lead to sufficient shared wall
                # We need to verify that the room can grow to share min_shared_wall cells

                # For a single cell, count potential shared wall with target rooms
                potential_shared = 0
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < rows and 0 <= nx < cols:
                        if floorplan[ny, nx] in target_room_ids:
                            potential_shared += 1

                # Also check if we can grow the room to get more shared wall
                # by checking empty cells adjacent to both this cell AND target rooms
                growth_potential = 0
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < rows and 0 <= nx < cols:
                        if floorplan[ny, nx] == EMPTY_ROOM_ID:
                            # This empty cell could be part of our room
                            # Check if it's also adjacent to target rooms
                            for dy2, dx2 in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                ny2, nx2 = ny + dy2, nx + dx2
                                if 0 <= ny2 < rows and 0 <= nx2 < cols:
                                    if floorplan[ny2, nx2] in target_room_ids:
                                        growth_potential += 1
                                        break

                # If we can potentially achieve the shared wall requirement
                if potential_shared + growth_potential >= min_shared_wall:
                    # Place the room at this cell
                    floorplan[y, x] = room.room_id
                    room.min_x = x
                    room.min_y = y
                    room.max_x = x + 1
                    room.max_y = y + 1
                    return True

        # If we couldn't find a position with adjacency, fail
        if adjacent_to:
            return False

    # No adjacency constraint or couldn't satisfy it - place at best available position
    for score, y, x in candidates:
        floorplan[y, x] = room.room_id
        room.min_x = x
        room.min_y = y
        room.max_x = x + 1
        room.max_y = y + 1
        return True

    return False


def get_hallway_bounds(
    hallway_room_id: int, floorplan: np.ndarray
) -> Tuple[int, int, int, int]:
    """Get the bounding box of a hallway in the floorplan.

    Args:
        hallway_room_id: The room ID of the hallway.
        floorplan: The floorplan grid.

    Returns:
        Tuple of (min_y, max_y, min_x, max_x) - the bounding box coordinates.
        If hallway is not found, returns (0, 0, 0, 0).

    Example:
        >>> fp = np.array([
        ...     [0, 0, 0, 0],
        ...     [0, 5, 5, 0],
        ...     [0, 5, 5, 0],
        ...     [0, 0, 0, 0],
        ... ])
        >>> get_hallway_bounds(5, fp)
        (1, 3, 1, 3)
    """
    room_mask = floorplan == hallway_room_id
    if not room_mask.any():
        return (0, 0, 0, 0)

    rows = np.any(room_mask, axis=1)
    cols = np.any(room_mask, axis=0)
    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]

    min_y = int(row_indices[0])
    max_y = int(row_indices[-1]) + 1  # exclusive upper bound
    min_x = int(col_indices[0])
    max_x = int(col_indices[-1]) + 1  # exclusive upper bound

    return (min_y, max_y, min_x, max_x)


def grow_hallway(
    floorplan: np.ndarray,
    hallway_room_id: int,
    growth_cells: int = 1,
) -> bool:
    """Grow a hallway by extending its longest dimension.

    This operation extends the hallway lengthwise when more door positions
    are needed. It finds the longest dimension (horizontal or vertical) and
    attempts to extend by 1-2 cells in that direction.

    Args:
        floorplan: The floorplan grid. Modified in-place.
        hallway_room_id: The room ID of the hallway to grow.
        growth_cells: Number of cells to grow (1 or 2). Default is 1.

    Returns:
        True if the hallway was extended successfully, False if there was
        no room to grow (blocked by other rooms or boundary).

    Edge cases handled:
        - Hallway at boundary: Cannot grow beyond floorplan bounds
        - Blocked by other rooms: Only grows into EMPTY_ROOM_ID cells
        - Square hallway: Randomly chooses horizontal or vertical
        - Hallway not found: Returns False

    Example:
        >>> import numpy as np
        >>> from procthor.constants import EMPTY_ROOM_ID, OUTDOOR_ROOM_ID
        >>> # Horizontal hallway (width 3, height 1)
        >>> fp = np.array([
        ...     [1, 1, 1, 1, 1, 1],
        ...     [1, 0, 0, 0, 0, 1],
        ...     [1, 5, 5, 5, 0, 1],  # hallway_id=5, horizontal
        ...     [1, 0, 0, 0, 0, 1],
        ...     [1, 1, 1, 1, 1, 1],
        ... ])
        >>> grow_hallway(fp, 5)
        True
        >>> # Hallway should have grown right (into empty cell at [2,4])
        >>> (fp[2, 4] == 5)
        True
    """
    import random

    # Get current hallway bounds
    min_y, max_y, min_x, max_x = get_hallway_bounds(hallway_room_id, floorplan)

    # Hallway not found
    if min_y == max_y == min_x == max_x == 0:
        return False

    width = max_x - min_x   # horizontal extent
    height = max_y - min_y  # vertical extent

    # Determine longest dimension
    # If equal (square), randomly choose
    if width > height:
        primary_directions = ["right", "left"]  # extend horizontal
    elif height > width:
        primary_directions = ["down", "up"]     # extend vertical
    else:
        # Square: randomly choose
        primary_directions = random.choice([["right", "left"], ["down", "up"]])

    # Shuffle to randomize which end we try first
    random.shuffle(primary_directions)

    # Try each direction for growth
    for direction in primary_directions:
        cells_grown = 0

        for _ in range(growth_cells):
            if direction == "right":
                # Check if we can grow right
                if max_x >= floorplan.shape[1]:
                    break  # At boundary
                # Check if entire column is empty
                target_col = floorplan[min_y:max_y, max_x]
                if not (target_col == EMPTY_ROOM_ID).all():
                    break  # Blocked
                # Grow right
                floorplan[min_y:max_y, max_x] = hallway_room_id
                max_x += 1
                cells_grown += 1

            elif direction == "left":
                # Check if we can grow left
                if min_x <= 0:
                    break  # At boundary
                # Check if entire column is empty
                target_col = floorplan[min_y:max_y, min_x - 1]
                if not (target_col == EMPTY_ROOM_ID).all():
                    break  # Blocked
                # Grow left
                floorplan[min_y:max_y, min_x - 1] = hallway_room_id
                min_x -= 1
                cells_grown += 1

            elif direction == "down":
                # Check if we can grow down
                if max_y >= floorplan.shape[0]:
                    break  # At boundary
                # Check if entire row is empty
                target_row = floorplan[max_y, min_x:max_x]
                if not (target_row == EMPTY_ROOM_ID).all():
                    break  # Blocked
                # Grow down
                floorplan[max_y, min_x:max_x] = hallway_room_id
                max_y += 1
                cells_grown += 1

            elif direction == "up":
                # Check if we can grow up
                if min_y <= 0:
                    break  # At boundary
                # Check if entire row is empty
                target_row = floorplan[min_y - 1, min_x:max_x]
                if not (target_row == EMPTY_ROOM_ID).all():
                    break  # Blocked
                # Grow up
                floorplan[min_y - 1, min_x:max_x] = hallway_room_id
                min_y -= 1
                cells_grown += 1

        if cells_grown > 0:
            return True

    # If primary directions failed, try secondary (shorter dimension)
    if width > height:
        secondary_directions = ["down", "up"]
    elif height > width:
        secondary_directions = ["right", "left"]
    else:
        # Already tried both, return False
        return False

    random.shuffle(secondary_directions)

    for direction in secondary_directions:
        cells_grown = 0

        for _ in range(growth_cells):
            if direction == "right":
                if max_x >= floorplan.shape[1]:
                    break
                target_col = floorplan[min_y:max_y, max_x]
                if not (target_col == EMPTY_ROOM_ID).all():
                    break
                floorplan[min_y:max_y, max_x] = hallway_room_id
                max_x += 1
                cells_grown += 1

            elif direction == "left":
                if min_x <= 0:
                    break
                target_col = floorplan[min_y:max_y, min_x - 1]
                if not (target_col == EMPTY_ROOM_ID).all():
                    break
                floorplan[min_y:max_y, min_x - 1] = hallway_room_id
                min_x -= 1
                cells_grown += 1

            elif direction == "down":
                if max_y >= floorplan.shape[0]:
                    break
                target_row = floorplan[max_y, min_x:max_x]
                if not (target_row == EMPTY_ROOM_ID).all():
                    break
                floorplan[max_y, min_x:max_x] = hallway_room_id
                max_y += 1
                cells_grown += 1

            elif direction == "up":
                if min_y <= 0:
                    break
                target_row = floorplan[min_y - 1, min_x:max_x]
                if not (target_row == EMPTY_ROOM_ID).all():
                    break
                floorplan[min_y - 1, min_x:max_x] = hallway_room_id
                min_y -= 1
                cells_grown += 1

        if cells_grown > 0:
            return True

    return False


def can_place_door(
    room1_id: int,
    room2_id: int,
    floorplan: np.ndarray,
    min_door_width: int = 1,
) -> bool:
    """Check if a door can be placed between two rooms.

    Args:
        room1_id: First room's ID.
        room2_id: Second room's ID.
        floorplan: The floorplan grid.
        min_door_width: Minimum width for door placement in grid cells.

    Returns:
        True if a door can be placed between the rooms, False otherwise.
    """
    # TODO: Implement door placement feasibility check
    raise NotImplementedError("can_place_door not yet implemented")


def _collect_leaf_rooms(
    spec: List[Union[LeafRoom, MetaRoom]],
) -> List[LeafRoom]:
    """Recursively collect all LeafRoom objects from the spec.

    Args:
        spec: List of rooms (LeafRoom or MetaRoom).

    Returns:
        List of all LeafRoom objects found.
    """
    rooms = []
    for room in spec:
        if isinstance(room, LeafRoom):
            rooms.append(room)
        elif isinstance(room, MetaRoom):
            rooms.extend(_collect_leaf_rooms(room.children))
    return rooms


def _get_rooms_by_type(
    rooms: List[LeafRoom],
) -> Dict[str, List[LeafRoom]]:
    """Group rooms by their room type.

    Args:
        rooms: List of LeafRoom objects.

    Returns:
        Dict mapping room_type to list of rooms.
    """
    by_type: Dict[str, List[LeafRoom]] = {}
    for room in rooms:
        if room.room_type not in by_type:
            by_type[room.room_type] = []
        by_type[room.room_type].append(room)
    return by_type


def generate_floorplan_progressive(
    room_spec: RoomSpec,
    interior_boundary: np.ndarray,
    candidate_generations: int = 100,
) -> np.ndarray:
    """Generate a floorplan using progressive room placement.

    This approach places rooms one at a time, growing hallways to connect them,
    which allows for better control over room connectivity and door placement.

    The algorithm:
    1. Place LivingRoom first (no adjacency constraint)
    2. Place Kitchen adjacent to LivingRoom
    3. Place Hallway adjacent to LivingRoom
    4. For each Bedroom: place adjacent to Hallway OR LivingRoom. If placement
       fails, call grow_hallway() and retry.
    5. For each Bathroom: place adjacent to Hallway OR a Bedroom (master suite)
    6. Expand all rooms using grow_rect from floorplan_generation.py
    7. Return the floorplan

    Args:
        room_spec: Room specification for the floorplan.
        interior_boundary: Interior boundary of the floorplan (grid with
            OUTDOOR_ROOM_ID for exterior, EMPTY_ROOM_ID for interior).
        candidate_generations: Number of candidate floorplans to generate.
            The best candidate is returned.

    Returns:
        The generated floorplan as a numpy array where each cell contains
        the room_id of the room occupying that cell.

    Raises:
        InvalidFloorplan: If no valid floorplan could be generated.
    """
    from procthor.generation.floorplan_generation import grow_rect

    # Collect all leaf rooms
    all_rooms = _collect_leaf_rooms(room_spec.spec)
    rooms_by_type = _get_rooms_by_type(all_rooms)

    # Get living rooms, kitchens, hallways, bedrooms, bathrooms
    living_rooms = rooms_by_type.get("LivingRoom", [])
    kitchens = rooms_by_type.get("Kitchen", [])
    hallways = rooms_by_type.get("Hallway", [])
    bedrooms = rooms_by_type.get("Bedroom", [])
    bathrooms = rooms_by_type.get("Bathroom", [])

    # Create room_type_map for adjacency lookups
    room_type_map = room_spec.room_type_map

    best_floorplan = None
    best_room_count = 0

    for _ in range(candidate_generations):
        # Start with a copy of the boundary
        floorplan = interior_boundary.copy()

        placed_rooms: List[LeafRoom] = []
        hallway_room: Optional[LeafRoom] = None

        # 1. Place LivingRoom first (no adjacency constraint)
        for living_room in living_rooms:
            if place_room(living_room, floorplan, adjacent_to=None, room_type_map=room_type_map):
                placed_rooms.append(living_room)

        # 2. Place Kitchen adjacent to LivingRoom
        for kitchen in kitchens:
            if place_room(kitchen, floorplan, adjacent_to=["LivingRoom"], room_type_map=room_type_map):
                placed_rooms.append(kitchen)

        # 3. Place Hallway adjacent to LivingRoom
        for hallway in hallways:
            if place_room(hallway, floorplan, adjacent_to=["LivingRoom"], room_type_map=room_type_map):
                placed_rooms.append(hallway)
                hallway_room = hallway

        # 3b. Pre-grow hallway to ensure enough wall space for bedrooms
        # Grow hallway to be at least (num_bedrooms + 1) cells long
        if hallway_room is not None:
            min_hallway_cells = len(bedrooms) + 1
            for _ in range(min_hallway_cells):
                grew = grow_hallway(floorplan, hallway_room.room_id, growth_cells=1)
                if not grew:
                    break

        # 4. For each Bedroom: place adjacent to Hallway OR LivingRoom
        for bedroom in bedrooms:
            # Try placing adjacent to Hallway or LivingRoom
            placed = place_room(
                bedroom, floorplan,
                adjacent_to=["Hallway", "LivingRoom"],
                room_type_map=room_type_map
            )

            if not placed and hallway_room is not None:
                # If placement failed, try growing the hallway
                for _ in range(3):  # Max 3 hallway growth attempts
                    if grow_hallway(floorplan, hallway_room.room_id, growth_cells=1):
                        # Retry placement after hallway growth
                        placed = place_room(
                            bedroom, floorplan,
                            adjacent_to=["Hallway", "LivingRoom"],
                            room_type_map=room_type_map
                        )
                        if placed:
                            break

            if placed:
                placed_rooms.append(bedroom)

        # 5. For each Bathroom: place adjacent to Hallway OR a Bedroom
        for bathroom in bathrooms:
            placed = place_room(
                bathroom, floorplan,
                adjacent_to=["Hallway", "Bedroom"],
                room_type_map=room_type_map
            )
            if placed:
                placed_rooms.append(bathroom)

        # 6. Expand all rooms using grow_rect
        # Keep growing until no room can grow anymore
        still_growing = True
        while still_growing:
            still_growing = False
            for room in placed_rooms:
                if grow_rect(room, floorplan):
                    still_growing = True

        # Score this candidate by number of placed rooms
        room_count = len(placed_rooms)
        if room_count > best_room_count:
            best_room_count = room_count
            best_floorplan = floorplan.copy()

        # If we placed all rooms, return immediately
        if room_count == len(all_rooms):
            return floorplan

    # Return best candidate if we found one
    if best_floorplan is not None:
        return best_floorplan

    # Fall back to standard generation if progressive failed completely
    from procthor.generation.floorplan_generation import generate_floorplan
    return generate_floorplan(
        room_spec=room_spec,
        interior_boundary=interior_boundary,
        candidate_generations=candidate_generations,
    )

