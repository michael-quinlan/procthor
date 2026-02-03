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


def place_room(
    room: Union[LeafRoom, MetaRoom],
    floorplan: np.ndarray,
    target_position: Optional[Tuple[int, int]] = None,
) -> bool:
    """Place a single room at a position in the floorplan.

    Args:
        room: The room to place (LeafRoom or MetaRoom).
        floorplan: The floorplan grid to place the room in.
        target_position: Optional (y, x) position to place the room. If None,
            a suitable position will be selected automatically.

    Returns:
        True if the room was placed successfully, False otherwise.
    """
    # TODO: Implement room placement logic
    raise NotImplementedError("place_room not yet implemented")


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


def generate_floorplan_progressive(
    room_spec: RoomSpec,
    interior_boundary: np.ndarray,
    candidate_generations: int = 100,
) -> np.ndarray:
    """Generate a floorplan using progressive room placement.

    This approach places rooms one at a time, growing hallways to connect them,
    which allows for better control over room connectivity and door placement.

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
    # TODO: Implement progressive floorplan generation
    # For now, fall back to the standard generation
    from procthor.generation.floorplan_generation import generate_floorplan

    return generate_floorplan(
        room_spec=room_spec,
        interior_boundary=interior_boundary,
        candidate_generations=candidate_generations,
    )

