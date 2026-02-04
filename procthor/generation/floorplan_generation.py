#%%
"""
Things to consider:
- Hallways:
    - Our algorithm starts by placing doors between rooms
      for which connectivity was explicitly declared. Next,
      it connects any hallway to all of its adjacent public
      rooms. Unconnected private rooms are then connected,
      if possible, to an adjacent public room. Publics rooms
      with no connections are connected to an adjacent public
      room as well. Finally, our last step is a reachability test.
      We examine all rooms and if any is not reachable from
      the hallway, we use the adjacency relationships between
      rooms to find a path to the unreachable room, and create
      the necessary door(s).
"""
import random
from typing import Dict, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np

from procthor.constants import EMPTY_ROOM_ID, OUTDOOR_ROOM_ID
from procthor.generation.room_specs import RoomSpec
from procthor.utils.types import InvalidFloorplan, LeafRoom, MetaRoom


def visualize_floorplan(floorplan: np.array):
    colors = ["green", "blue", "red", "yellow", "black"]

    fig, ax = plt.subplots()
    for i, room_id in enumerate(np.unique(floorplan.flatten())):
        print(room_id)
        coords = np.argwhere(floorplan == room_id)
        ax.scatter(coords[:, 1], coords[:, 0], label=room_id, c=colors[i])
        ax.set_aspect("equal")
        ax.legend(loc=(1.04, 0))
    return fig


def select_room(
    rooms: Sequence[Union[LeafRoom, MetaRoom]]
) -> Union[LeafRoom, MetaRoom]:
    """
    From the paper:
        In SelectRoom the next room to be expanded is chosen
        on the basis of the defined size ratios r for each room.
        The chance for a room to be selected is its ratio relative to
        the total sum of ratios, defined in r. With this approach,
        variation is ensured, but the selection still respects the
        desired ratios of room areas.
    """
    total_ratio = sum(r.ratio for r in rooms)
    r = random.random() * total_ratio
    for room in rooms:
        r -= room.ratio
        if r <= 0:
            return room


def sample_initial_room_positions(
    rooms: Sequence[Union[LeafRoom, MetaRoom]], floorplan: np.ndarray
) -> None:
    """
    From the paper:
        However, placing initial positions adjacent to a
        wall does not always result in plausible results, as they
        tend to cause less regular room shapes. Therefore, we
        use a different approach. Based on the size ratio of the
        room and the total area of the building, we can estimate
        a desired area of the room. Cells positioned at least a
        specific distance (based on this estimated area) away
        from the walls are assigned a weight of 1. This results
        in much more plausible room shapes.

        This phase also deals with the defined adjacency con-
        straints. The adjacency constraints are always defined
        between two rooms, e.g. the bathroom should be next
        to the bedroom, the kitchen should be next to the living
        room, etc. When selecting the initial position of a room,
        we use the adjacency constraints to determine a list of
        rooms it should be adjacent to. We check whether these
        rooms already have an initial position. If there is, we
        alter the weights to high values in the surroundings of
        the initial positions of the rooms it should be adjacent to.
        This typically results in valid layouts; however there is a
        small chance that the algorithm grows another room in
        between the rooms that should be adjacent. To handle
        this case, we reset the generation process if some of the
        adjacency constraints were not met.

        Based on these grid weights, one cell is selected to place
        a room, and the weights around the selected cell are
        set to zero, to avoid several initial positions of different
        rooms to be too close to each other.
    """
    grid_weights = np.where(floorplan == EMPTY_ROOM_ID, 1, 0)
    for room in rooms:
        # make sure there is at least one open cell in the floorplan area.
        if (grid_weights == 0).all():
            raise InvalidFloorplan(
                "No empty cells in the floorplan to place room! This means the"
                " sampled interior boundary is too small for the room spec.\n"
                f"grid_weights:\n{grid_weights}"
                f"\nfloorplan\n{floorplan}"
            )

        # TODO: these weights could be updated by the adjacency constraints
        # and the hallways.
        # sample a grid cell by weight
        cell_idx = np.random.choice(
            grid_weights.size, p=grid_weights.ravel() / float(grid_weights.sum())
        )
        cell_y, cell_x = np.unravel_index(cell_idx, grid_weights.shape)

        # add the grid cell to the room
        room.min_x = cell_x
        room.min_y = cell_y
        room.max_x = cell_x + 1
        room.max_y = cell_y + 1

        # add the grid cell to the floorplan
        floorplan[cell_y, cell_x] = room.room_id

        # update the weights
        grid_weights[
            max(0, cell_y - 1) : min(grid_weights.shape[0], cell_y + 2),
            max(0, cell_x - 1) : min(grid_weights.shape[1], cell_x + 2),
        ] = 0


def grow_rect(room: Union[MetaRoom, LeafRoom], floorplan: np.ndarray) -> bool:
    """
    From the paper:
        The first phase of this algorithm is expanding rooms
        to rectangular shapes (GrowRect). In Fig. 3 we see an
        example of the start situation (a) and end (b) of the
        rectangular expansion phase for a building where rooms
        black, green and red have size ratios of, respectively, 8, 4
        and 2. Starting with rectangular expansion ensures two
        characteristics of real life floor plans: (i) a higher priority
        is given to obtain rectangular areas and (ii) the growth is
        done using the maximum space available, in a linear way.
        For this, all empty line intervals in the grid m to which
        the selected room can expand to are considered. The
        maximum growth, i.e. the longest line interval, which
        leads to a rectangular area is picked (randomly, if there
        are more than one candidates). A room remains available
        for selection until it can not grow more. This happens if
        there are no more directions available to grow or, in the
        rectangular expansion case, if the room has reached its
        maximum size. This condition also prevents starvation
        for lower ratio rooms, since size ratios have no relation
        with the total building area. In Fig.3 (b), all rooms have
        reached their maximum size."""
    # NOTE: check if room is already grown beyond the maximum size.
    maximum_size = room.ratio * 4
    if (room.max_x - room.min_x) * (room.max_y - room.min_y) > maximum_size:
        return False

    # NOTE: Find out how much the rectangle can grow in each direction.
    growth_sizes = {
        "right": (
            room.max_y - room.min_y
            if (
                room.max_x < floorplan.shape[1]
                and (
                    floorplan[room.min_y : room.max_y, room.max_x] == EMPTY_ROOM_ID
                ).all()
            )
            else 0
        ),
        "left": (
            room.max_y - room.min_y
            if (
                room.min_x > 0
                and (
                    floorplan[room.min_y : room.max_y, room.min_x - 1] == EMPTY_ROOM_ID
                ).all()
            )
            else 0
        ),
        "down": (
            room.max_x - room.min_x
            if (
                room.max_y < floorplan.shape[0]
                and (
                    floorplan[room.max_y, room.min_x : room.max_x] == EMPTY_ROOM_ID
                ).all()
            )
            else 0
        ),
        "up": (
            room.max_x - room.min_x
            if (
                room.min_y > 0
                and (
                    floorplan[room.min_y - 1, room.min_x : room.max_x] == EMPTY_ROOM_ID
                ).all()
            )
            else 0
        ),
    }

    max_growth_size = max(growth_sizes.values())

    # If there is no room to grow, return False
    if max_growth_size == 0:
        return False

    # NOTE: Pick a random max growth direction to grow.
    # From the paper: The maximum growth, i.e. the longest line interval, which
    # leads to a rectangular area is picked (randomly, if there are more than
    # one candidates).
    growth_direction = random.choice(
        [
            growth_direction
            for growth_direction, growth_size in growth_sizes.items()
            if growth_size == max_growth_size
        ]
    )

    # NOTE: Grow the room in the chosen direction.
    if growth_direction == "right":
        room.max_x += 1
        floorplan[room.min_y : room.max_y, room.max_x - 1] = room.room_id
    elif growth_direction == "left":
        room.min_x -= 1
        floorplan[room.min_y : room.max_y, room.min_x] = room.room_id
    elif growth_direction == "down":
        room.max_y += 1
        floorplan[room.max_y - 1, room.min_x : room.max_x] = room.room_id
    elif growth_direction == "up":
        room.min_y -= 1
        floorplan[room.min_y, room.min_x : room.max_x] = room.room_id

    return True


def grow_l_shape(room, floorplan):
    """
    From the paper:
        Of course, this first phase does not ensure that all available space
        gets assigned to a room. In the second phase, all rooms are again
        considered for further expansion, now allowing for non-rectangular
        shapes. The maximum growth line is again selected, in order to maximize
        efficient space use, i.e. to avoid narrow L-shaped edges. In this phase,
        the maximum size for each room is no longer considered, since the
        algorithm attempts to fill all the remaining empty space. Furthermore we
        included mechanisms for preventing U-shaped rooms. Fig. 3 (c)
        illustrates the result of the L-shaped growth step on the previous
        example. The final phase scans the grid for remaining empty space; this
        space is directly assigned to the room which fills most of the adjacent
        area.
    """
    # NOTE: Find out how much the rectangle can grow in each direction.
    growth_sizes = {
        "right": (
            [
                y
                for y in range(room.min_y, room.max_y)
                if (
                    floorplan[y, room.max_x] == EMPTY_ROOM_ID
                    and floorplan[y, room.max_x - 1] == room.room_id
                )
            ]
            if (room.max_x < floorplan.shape[1])
            else []
        ),
        "left": (
            [
                y
                for y in range(room.min_y, room.max_y)
                if (
                    floorplan[y, room.min_x - 1] == EMPTY_ROOM_ID
                    and floorplan[y, room.min_x] == room.room_id
                )
            ]
            if (room.min_x > 0)
            else []
        ),
        "down": (
            [
                x
                for x in range(room.min_x, room.max_x)
                if (
                    floorplan[room.max_y, x] == EMPTY_ROOM_ID
                    and floorplan[room.max_y - 1, x] == room.room_id
                )
            ]
            if (room.max_y < floorplan.shape[0])
            else []
        ),
        "up": (
            [
                x
                for x in range(room.min_x, room.max_x)
                if (
                    floorplan[room.min_y - 1, x] == EMPTY_ROOM_ID
                    and floorplan[room.min_y, x] == room.room_id
                )
            ]
            if (room.min_y > 0)
            else []
        ),
    }

    max_growth_size = max(growth_sizes.values(), key=len)
    if len(max_growth_size) == 0:
        return False

    # NOTE: Pick a random max growth direction to grow.
    growth_direction = random.choice(
        [
            growth_direction
            for growth_direction, growth_size in growth_sizes.items()
            if growth_size == max_growth_size
        ]
    )
    if growth_direction == "right":
        for y in growth_sizes["right"]:
            floorplan[y, room.max_x] = room.room_id
        room.max_x += 1
    elif growth_direction == "left":
        for y in growth_sizes["left"]:
            floorplan[y, room.min_x - 1] = room.room_id
        room.min_x -= 1
    elif growth_direction == "down":
        for x in growth_sizes["down"]:
            floorplan[room.max_y, x] = room.room_id
        room.max_y += 1
    elif growth_direction == "up":
        for x in growth_sizes["up"]:
            floorplan[room.min_y - 1, x] = room.room_id
        room.min_y -= 1

    return True


def expand_rooms(
    rooms: Sequence[Union[LeafRoom, MetaRoom]], floorplan: np.ndarray
) -> None:
    """Assign rooms from a given hierarchy to the floorplan.

    From the paper:
        Algorithm 1 outlines the expansion of rooms in our
        method. It starts with a grid m containing the initial
        positions of each room. It then picks one room at a time,
        selected from a set of available rooms (SelectRoom), and
        expands the room shape to the maximum rectangular
        space available (GrowRect). This is done until no more
        rectangular expansions are possible. At this point, the
        process resets rooms to the initial set, but now considers
        expansions that lead to L-shaped rooms (GrowLShape).
    """

    # NOTE: Initial center placement of each room
    sample_initial_room_positions(rooms, floorplan)

    # NOTE: grow rectangles
    rooms_to_grow = set(rooms)
    while rooms_to_grow:
        room = select_room(rooms_to_grow)
        can_grow = grow_rect(room, floorplan)
        if not can_grow:
            rooms_to_grow.remove(room)

    # NOTE: grow L-Shape
    rooms_to_grow = set(rooms)
    while rooms_to_grow:
        room = select_room(rooms_to_grow)
        can_grow = grow_l_shape(room, floorplan)
        if not can_grow:
            rooms_to_grow.remove(room)


def _set_ideal_ratios(
    ideal_ratios: Dict[int, float],
    rooms: Sequence[Union[MetaRoom, LeafRoom]],
    parent_sum: float = 1,
) -> None:
    """Set the ideal ratio size of each room in the floorplan.

    After calling this method, ideal ratios becomes something like:
        {
            2: 0.3,
            3: 0.1,
            4: 0.2,
            5: 0.4
        }
    where the sum of the values is 1. The value indicates the ideal absolute size
    of the room relative to the entire floorplan.
    """
    room_ratio_sum = sum([room.ratio for room in rooms])
    for room in rooms:
        ideal_ratios[room.room_id] = room.ratio / room_ratio_sum * parent_sum
        if isinstance(room, MetaRoom):
            _set_ideal_ratios(
                ideal_ratios=ideal_ratios,
                rooms=room.children,
                parent_sum=ideal_ratios[room.room_id],
            )
            del ideal_ratios[room.room_id]


def get_ratio_overlap_score(room_spec: RoomSpec, floorplan: np.ndarray) -> float:
    """Calculate the difference between the ratios in floorplan and room_spec."""
    # NOTE: Get the average ratio overlap of all rooms
    ideal_ratios = {}
    _set_ideal_ratios(ideal_ratios, rooms=room_spec.spec)

    actual_ratios = {}
    occupied_cells = (floorplan != OUTDOOR_ROOM_ID).sum()
    for room_id in room_spec.room_type_map:
        actual_ratios[room_id] = (floorplan == room_id).sum() / occupied_cells

    # NOTE: Get the average ratio overlap of all rooms
    ratio_overlap = sum(
        [min(actual_ratios[room_id], ideal_ratios[room_id]) for room_id in ideal_ratios]
    )

    return ratio_overlap


# Room types that must be rectangular (no L-shapes)
ROOM_TYPES_MUST_BE_RECTANGULAR = {"Bedroom", "Bathroom"}


def get_room_dimensions(room_id: int, floorplan: np.ndarray) -> tuple:
    """Get the width and height of a room's bounding box.

    Returns (width, height, area) of the room.
    """
    room_mask = floorplan == room_id
    if not room_mask.any():
        return (0, 0, 0)

    rows = np.any(room_mask, axis=1)
    cols = np.any(room_mask, axis=0)
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    width = col_max - col_min + 1
    height = row_max - row_min + 1
    area = room_mask.sum()

    return (width, height, area)


def is_room_rectangular(room_id: int, floorplan: np.ndarray) -> bool:
    """Check if a room in the floorplan is rectangular."""
    room_mask = floorplan == room_id
    if not room_mask.any():
        return True

    rows = np.any(room_mask, axis=1)
    cols = np.any(room_mask, axis=0)
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    actual_cells = room_mask.sum()
    bounding_box_cells = (row_max - row_min + 1) * (col_max - col_min + 1)

    return actual_cells == bounding_box_cells


def get_room_adjacencies(floorplan: np.ndarray) -> dict:
    """Get which rooms are adjacent to each other in the floorplan.

    Returns dict mapping room_id -> set of adjacent room_ids.
    """
    adjacencies = {}
    rows, cols = floorplan.shape

    for room_id in np.unique(floorplan):
        if room_id in {EMPTY_ROOM_ID, OUTDOOR_ROOM_ID}:
            continue
        adjacencies[room_id] = set()

        room_mask = floorplan == room_id
        # Check all 4 directions for adjacency
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for r in range(rows):
                for c in range(cols):
                    if room_mask[r, c]:
                        nr, nc = r + dy, c + dx
                        if 0 <= nr < rows and 0 <= nc < cols:
                            neighbor = floorplan[nr, nc]
                            if neighbor != room_id and neighbor not in {EMPTY_ROOM_ID, OUTDOOR_ROOM_ID}:
                                adjacencies[room_id].add(neighbor)

    return adjacencies


def get_rectangular_penalty(room_spec: RoomSpec, floorplan: np.ndarray) -> float:
    """Penalty for non-rectangular bedrooms/bathrooms."""
    penalty = 0.0
    for room_id, room_type in room_spec.room_type_map.items():
        if room_type in ROOM_TYPES_MUST_BE_RECTANGULAR:
            if not is_room_rectangular(room_id, floorplan):
                penalty -= 10.0
    return penalty


def get_hallway_shape_penalty(room_spec: RoomSpec, floorplan: np.ndarray) -> float:
    """Penalty for hallways that are too square (should be long and narrow)."""
    penalty = 0.0
    for room_id, room_type in room_spec.room_type_map.items():
        if room_type == "Hallway":
            width, height, area = get_room_dimensions(room_id, floorplan)
            if width == 0 or height == 0:
                continue

            # Aspect ratio: how elongated is the hallway?
            aspect_ratio = max(width, height) / min(width, height)

            # Hallways should have aspect ratio >= 2 (at least 2x longer than wide)
            # Reward narrow hallways, penalize square ones
            if aspect_ratio < 1.5:
                penalty -= 5.0  # Very square, bad
            elif aspect_ratio < 2.0:
                penalty -= 2.0  # Somewhat square
            else:
                penalty += 1.0  # Good, elongated hallway

            # Hallways should also be relatively small (not take up too much space)
            total_area = (floorplan != OUTDOOR_ROOM_ID).sum()
            hallway_ratio = area / total_area
            if hallway_ratio > 0.15:  # Hallway is more than 15% of house
                penalty -= 3.0

    return penalty


def get_hallway_connectivity_penalty(room_spec: RoomSpec, floorplan: np.ndarray) -> float:
    """Penalty if hallway doesn't connect to the LivingRoom."""
    penalty = 0.0
    adjacencies = get_room_adjacencies(floorplan)

    for room_id, room_type in room_spec.room_type_map.items():
        if room_type == "Hallway":
            adjacent_ids = adjacencies.get(room_id, set())
            adjacent_types = {room_spec.room_type_map.get(adj_id) for adj_id in adjacent_ids}

            # Hallway must connect to LivingRoom
            if "LivingRoom" not in adjacent_types:
                penalty -= 10.0  # Heavy penalty
            else:
                penalty += 2.0  # Reward good connectivity

    return penalty


def get_kitchen_livingroom_adjacency_bonus(room_spec: RoomSpec, floorplan: np.ndarray) -> float:
    """Bonus if Kitchen is adjacent to LivingRoom (open concept)."""
    bonus = 0.0
    adjacencies = get_room_adjacencies(floorplan)

    kitchen_ids = [rid for rid, rt in room_spec.room_type_map.items() if rt == "Kitchen"]

    for kitchen_id in kitchen_ids:
        adjacent_ids = adjacencies.get(kitchen_id, set())
        adjacent_types = {room_spec.room_type_map.get(adj_id) for adj_id in adjacent_ids}

        if "LivingRoom" in adjacent_types:
            bonus += 2.0  # Reward kitchen-livingroom adjacency

    return bonus


def get_bathroom_size_penalty(room_spec: RoomSpec, floorplan: np.ndarray) -> float:
    """Penalty if bathrooms are larger than bedrooms."""
    penalty = 0.0

    bathroom_areas = []
    bedroom_areas = []

    for room_id, room_type in room_spec.room_type_map.items():
        _, _, area = get_room_dimensions(room_id, floorplan)
        if room_type == "Bathroom":
            bathroom_areas.append(area)
        elif room_type == "Bedroom":
            bedroom_areas.append(area)

    if bathroom_areas and bedroom_areas:
        max_bathroom = max(bathroom_areas)
        min_bedroom = min(bedroom_areas)

        # Bathrooms should be smaller than bedrooms
        if max_bathroom >= min_bedroom:
            penalty -= 3.0

    return penalty


def get_bedroom_connectivity_penalty(room_spec: RoomSpec, floorplan: np.ndarray) -> float:
    """Penalty if bedrooms don't connect to a LivingRoom or Hallway.

    Bedrooms should be accessible from common areas, not only through
    kitchens, bathrooms, or other bedrooms.
    """
    penalty = 0.0
    adjacencies = get_room_adjacencies(floorplan)

    for room_id, room_type in room_spec.room_type_map.items():
        if room_type == "Bedroom":
            adjacent_ids = adjacencies.get(room_id, set())
            adjacent_types = {room_spec.room_type_map.get(adj_id) for adj_id in adjacent_ids}

            # Bedroom must connect to LivingRoom or Hallway
            if not adjacent_types.intersection({"LivingRoom", "Hallway"}):
                penalty -= 10.0  # Heavy penalty
            else:
                penalty += 1.0  # Small reward for good connectivity

    return penalty


# US Building Code (IRC) minimum bedroom requirements
# Minimum area: 6.5 sq meters (70 sq ft)
# Minimum dimension: 2.13 meters (7 feet)
# Grid scale is typically 3 meters per cell, so:
#   - 1 cell = 9 sq meters (exceeds 6.5 sq m minimum)
#   - 1 cell width = 3 meters (exceeds 2.13m minimum)
# For more realistic bedrooms, we require at least 2 cells area and
# prefer at least 2 cells in each dimension (6m x 6m = 36 sq m)
MIN_BEDROOM_AREA_CELLS = 2  # ~18 sq meters at 3m/cell scale
MIN_BEDROOM_DIMENSION_CELLS = 2  # ~6 meters at 3m/cell scale


def get_bedroom_size_penalty(room_spec: RoomSpec, floorplan: np.ndarray) -> float:
    """Penalty for bedrooms that don't meet minimum size requirements.

    Based on US IRC building code:
    - Minimum 6.5 sq meters (70 sq ft)
    - Minimum 2.13 meters (7 feet) in any dimension

    We use grid-cell based thresholds that exceed these minimums.
    """
    penalty = 0.0

    for room_id, room_type in room_spec.room_type_map.items():
        if room_type == "Bedroom":
            width, height, area = get_room_dimensions(room_id, floorplan)

            # Check minimum area
            if area < MIN_BEDROOM_AREA_CELLS:
                penalty -= 10.0  # Too small

            # Check minimum dimension (both width and height should be adequate)
            min_dim = min(width, height)
            if min_dim < MIN_BEDROOM_DIMENSION_CELLS:
                penalty -= 5.0  # Too narrow

            # Bonus for well-sized bedrooms (at least 4 cells = ~36 sq m)
            if area >= 4 and min_dim >= 2:
                penalty += 1.0

    return penalty


def get_room_size_constraint_penalty(
    room_spec: RoomSpec,
    floorplan: np.ndarray,
    cell_size_sqm: float = 3.6,  # Default: 1.9m scale squared
) -> float:
    """Penalty for rooms that are outside their target size constraints.

    Uses the room_sizing module to evaluate if rooms are appropriately sized.

    Args:
        room_spec: The room specification
        floorplan: The current floorplan grid
        cell_size_sqm: Area of each grid cell in square meters
    """
    from procthor.generation.room_sizing import (
        ROOM_SIZE_TARGETS_SQM,
        get_room_size_penalty,
        get_hallway_shape_penalty_for_dims,
    )

    penalty = 0.0

    for room_id, room_type in room_spec.room_type_map.items():
        if room_type not in ROOM_SIZE_TARGETS_SQM:
            continue

        width, height, area_cells = get_room_dimensions(room_id, floorplan)
        area_sqm = area_cells * cell_size_sqm

        # Get size penalty
        penalty += get_room_size_penalty(room_type, area_sqm, penalty_scale=0.05)

        # Extra penalty for hallways that aren't narrow
        if room_type == "Hallway":
            # Convert cell dimensions to approximate meters
            cell_side_m = cell_size_sqm ** 0.5
            width_m = width * cell_side_m
            height_m = height * cell_side_m
            penalty += get_hallway_shape_penalty_for_dims(
                min(width_m, height_m),
                max(width_m, height_m),
                penalty_scale=0.1
            )

    return penalty


def score_floorplan(
    room_spec: RoomSpec,
    floorplan: np.ndarray,
    cell_size_sqm: float = 3.6,
) -> float:
    """Calculate the quality of the floorplan based on the room specifications.

    Args:
        room_spec: The room specification
        floorplan: The current floorplan grid
        cell_size_sqm: Area of each grid cell in square meters (for size constraints)
    """
    score = 0.0

    # Base score: how well do room sizes match the spec?
    score += get_ratio_overlap_score(room_spec, floorplan)

    # Rule 1: Bedrooms and bathrooms must be rectangular
    score += get_rectangular_penalty(room_spec, floorplan)

    # Rule 2: Hallways should be narrow (elongated, not square)
    score += get_hallway_shape_penalty(room_spec, floorplan)

    # Rule 3: Hallways must connect to LivingRoom
    score += get_hallway_connectivity_penalty(room_spec, floorplan)

    # Rule 4: Kitchen should be adjacent to LivingRoom
    score += get_kitchen_livingroom_adjacency_bonus(room_spec, floorplan)

    # Rule 5: Bathrooms should be smaller than bedrooms
    score += get_bathroom_size_penalty(room_spec, floorplan)

    # Rule 6: Bedrooms must connect to LivingRoom or Hallway
    score += get_bedroom_connectivity_penalty(room_spec, floorplan)

    # Rule 7: Bedrooms must meet minimum size requirements (US building code)
    score += get_bedroom_size_penalty(room_spec, floorplan)

    # Rule 8: Rooms should be within target size constraints
    score += get_room_size_constraint_penalty(room_spec, floorplan, cell_size_sqm)

    return score


def validate_strict_rules(room_spec: RoomSpec, floorplan: np.ndarray) -> bool:
    """Check if floorplan passes all strict layout rules.

    Returns True if valid, False if any strict rule is violated.
    """
    adjacencies = get_room_adjacencies(floorplan)

    for room_id, room_type in room_spec.room_type_map.items():
        adjacent_ids = adjacencies.get(room_id, set())
        adjacent_types = {room_spec.room_type_map.get(adj_id) for adj_id in adjacent_ids}

        # Rule: Bedrooms must be rectangular
        if room_type == "Bedroom":
            if not is_room_rectangular(room_id, floorplan):
                return False

        # Rule: Bathrooms must be rectangular
        if room_type == "Bathroom":
            if not is_room_rectangular(room_id, floorplan):
                return False

        # Rule: Hallways must connect to LivingRoom
        if room_type == "Hallway":
            if "LivingRoom" not in adjacent_types:
                return False

        # Rule: Hallways must touch at least 2 rooms
        if room_type == "Hallway":
            if len(adjacent_ids) < 2:
                return False

        # Rule: Bedrooms must connect to LivingRoom or Hallway
        if room_type == "Bedroom":
            if not adjacent_types.intersection({"LivingRoom", "Hallway"}):
                return False

        # Rule: Bedrooms must meet minimum size
        if room_type == "Bedroom":
            width, height, area = get_room_dimensions(room_id, floorplan)
            if area < MIN_BEDROOM_AREA_CELLS:
                return False
            if min(width, height) < MIN_BEDROOM_DIMENSION_CELLS:
                return False

    # Rule: At least one bathroom must be accessible from a public area
    # (not only through bedrooms) - ensures a "guest bathroom" exists
    bathrooms = [rid for rid, rtype in room_spec.room_type_map.items() if rtype == "Bathroom"]
    if bathrooms:
        has_public_bathroom = False
        for bathroom_id in bathrooms:
            adjacent_ids = adjacencies.get(bathroom_id, set())
            adjacent_types = {room_spec.room_type_map.get(adj_id) for adj_id in adjacent_ids}
            if adjacent_types.intersection({"LivingRoom", "Hallway", "Kitchen"}):
                has_public_bathroom = True
                break
        if not has_public_bathroom:
            return False

    return True


def recursively_expand_rooms(
    rooms: Sequence[Union[LeafRoom, MetaRoom]], floorplan: np.ndarray
) -> None:
    """Assign rooms to the floorplan and expand it if it is a MetaRoom."""
    expand_rooms(rooms, floorplan)
    for room in rooms:
        if isinstance(room, MetaRoom):
            floorplan_mask = floorplan == room.room_id
            floorplan[floorplan_mask] = EMPTY_ROOM_ID
            recursively_expand_rooms(
                room.children,
                floorplan[room.min_y : room.max_y, room.min_x : room.max_x],
            )


def generate_floorplan(
    room_spec: np.ndarray,
    interior_boundary: np.ndarray,
    candidate_generations: int = 100,
    interior_boundary_scale: float = 1.9,
) -> np.ndarray:
    """Generate a floorplan for the given room spec and interior boundary.

    Args:
        room_spec: Room spec for the floorplan.
        interior_boundary: Interior boundary of the floorplan.
        candidate_generations: Number of candidate generations to generate. The
            best candidate floorplan is returned.
        interior_boundary_scale: Scale factor (meters per grid cell) for size
            constraint evaluation.
    """
    # Calculate cell size in sqm for size constraint scoring
    cell_size_sqm = interior_boundary_scale ** 2

    # NOTE: If there is only one room, the floorplan will always be the same.
    if len(room_spec.room_type_map) == 1:
        candidate_generations = 1

    best_floorplan = None
    best_score = float("-inf")
    valid_candidates = 0

    for _ in range(candidate_generations):
        floorplan = interior_boundary.copy()
        try:
            recursively_expand_rooms(rooms=room_spec.spec, floorplan=floorplan)
        except InvalidFloorplan:
            continue

        # Check strict rules - reject if any are violated
        if not validate_strict_rules(room_spec=room_spec, floorplan=floorplan):
            continue

        valid_candidates += 1
        score = score_floorplan(
            room_spec=room_spec,
            floorplan=floorplan,
            cell_size_sqm=cell_size_sqm,
        )
        if best_floorplan is None or score > best_score:
            best_floorplan = floorplan
            best_score = score

    if best_floorplan is None:
        raise InvalidFloorplan(
            "Failed to generate a valid floorplan all candidate_generations="
            f"{candidate_generations} times from the interior boundary!"
            " This means the sampled interior boundary is too small for the room"
            " spec, or the strict layout rules cannot be satisfied."
            " Try again with another interior boundary.\n"
            f"interior_boundary:\n{interior_boundary}\n, room_spec:\n{room_spec}"
        )

    return best_floorplan
