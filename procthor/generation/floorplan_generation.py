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
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np

from procthor.constants import EMPTY_ROOM_ID, OUTDOOR_ROOM_ID
from procthor.generation.room_specs import RoomSpec
from procthor.generation.treemap import (
    RoomRegion,
    allocate_room_regions,
    place_room_in_region,
    grow_room_in_region,
    get_region_adjacencies,
)
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
    rooms: Sequence[Union[LeafRoom, MetaRoom]],
    floorplan: np.ndarray,
    room_spec: Optional[RoomSpec] = None,
    interior_boundary_scale: float = 1.9,
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
        # Check bathroom size limit before growing
        if room_spec:
            room_type = room_spec.room_type_map.get(room.room_id)
            if room_type == "Bathroom" and not _bathroom_can_grow(room, floorplan, interior_boundary_scale):
                rooms_to_grow.remove(room)
                continue
        can_grow = grow_rect(room, floorplan)
        if not can_grow:
            rooms_to_grow.remove(room)

    # NOTE: grow L-Shape
    rooms_to_grow = set(rooms)
    while rooms_to_grow:
        room = select_room(rooms_to_grow)
        # Check bathroom size limit before growing
        if room_spec:
            room_type = room_spec.room_type_map.get(room.room_id)
            if room_type == "Bathroom" and not _bathroom_can_grow(room, floorplan, interior_boundary_scale):
                rooms_to_grow.remove(room)
                continue
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
    """Penalty if hallway doesn't connect to LivingRoom or Kitchen."""
    penalty = 0.0
    adjacencies = get_room_adjacencies(floorplan)

    for room_id, room_type in room_spec.room_type_map.items():
        if room_type == "Hallway":
            adjacent_ids = adjacencies.get(room_id, set())
            adjacent_types = {room_spec.room_type_map.get(adj_id) for adj_id in adjacent_ids}

            # Hallway must connect to LivingRoom
            if "LivingRoom" in adjacent_types:
                penalty += 2.0  # Best: connected to LivingRoom
            else:
                penalty -= 10.0  # Heavy penalty: no LivingRoom connection

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


# Bedroom aspect ratio constraints
# Bedrooms should not be too square (1:1) or too elongated (1:2+)
# Valid range: 1.2 to 1.5 aspect ratio
MIN_BEDROOM_ASPECT_RATIO = 1.2
MAX_BEDROOM_ASPECT_RATIO = 1.5


def get_bedroom_aspect_ratio_penalty(room_spec: RoomSpec, floorplan: np.ndarray) -> float:
    """Penalty for bedrooms with poor aspect ratios.

    Bedrooms should have a 1:1.2 to 1:1.5 aspect ratio.
    - Too square (< 1.2): penalized
    - Too elongated (> 1.5): penalized
    - Good ratio (1.2 to 1.5): rewarded
    """
    penalty = 0.0

    for room_id, room_type in room_spec.room_type_map.items():
        if room_type == "Bedroom":
            width, height, area = get_room_dimensions(room_id, floorplan)
            if width > 0 and height > 0:
                aspect_ratio = max(width, height) / min(width, height)
                if aspect_ratio < MIN_BEDROOM_ASPECT_RATIO:
                    penalty -= 5.0  # Too square
                elif aspect_ratio > MAX_BEDROOM_ASPECT_RATIO:
                    penalty -= 5.0  # Too elongated
                else:
                    penalty += 1.0  # Good ratio

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

    # Rule 9: Bedrooms should have good aspect ratios (1.2 to 1.5)
    score += get_bedroom_aspect_ratio_penalty(room_spec, floorplan)

    return score


def check_room_proportions(
    room_spec: RoomSpec,
    floorplan: np.ndarray,
    cell_size_sqm: float = 3.6,
) -> tuple:
    """Check room proportion rules.

    Returns:
        (is_valid, error_message) tuple. If valid, error_message is None.
    """
    # Rule: No bathroom should be larger than the living room
    living_room_area = None
    for room_id, room_type in room_spec.room_type_map.items():
        if room_type == "LivingRoom":
            _, _, area = get_room_dimensions(room_id, floorplan)
            living_room_area = area * cell_size_sqm
            break

    if living_room_area is not None:
        for room_id, room_type in room_spec.room_type_map.items():
            if room_type == "Bathroom":
                _, _, area = get_room_dimensions(room_id, floorplan)
                bathroom_area = area * cell_size_sqm
                if bathroom_area > living_room_area:
                    return False, f"Bathroom ({bathroom_area:.1f}m²) > LivingRoom ({living_room_area:.1f}m²)"

    return True, None


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

        # Rule: Hallways must connect to LivingRoom or Kitchen
        if room_type == "Hallway":
            if not adjacent_types.intersection({"LivingRoom", "Kitchen"}):
                return False

        # Rule: Hallways must touch at least 2 rooms
        if room_type == "Hallway":
            if len(adjacent_ids) < 2:
                return False

        # Rule: Bedrooms must have at least one non-Kitchen adjacent room
        # (doors.py allows bedroom doors to hallway, living room, or other rooms as fallback)
        if room_type == "Bedroom":
            non_kitchen_adjacent = {t for t in adjacent_types if t != "Kitchen"}
            if not non_kitchen_adjacent:
                return False

        # Rule: Bedrooms must meet minimum size
        if room_type == "Bedroom":
            width, height, area = get_room_dimensions(room_id, floorplan)
            if area < MIN_BEDROOM_AREA_CELLS:
                return False
            if min(width, height) < MIN_BEDROOM_DIMENSION_CELLS:
                return False

    # Rule: Kitchen must be adjacent to LivingRoom
    # This ensures Kitchen is reachable from the main living area
    kitchens = [rid for rid, rtype in room_spec.room_type_map.items() if rtype == "Kitchen"]
    living_rooms = [rid for rid, rtype in room_spec.room_type_map.items() if rtype == "LivingRoom"]
    for kitchen_id in kitchens:
        adjacent_ids = adjacencies.get(kitchen_id, set())
        adjacent_types = {room_spec.room_type_map.get(adj_id) for adj_id in adjacent_ids}
        if "LivingRoom" not in adjacent_types:
            logging.warning(f"Kitchen {kitchen_id} not adjacent to any LivingRoom")
            return False

    # Rule: At least one bathroom must be accessible from a public area
    # (not only through bedrooms) - ensures a "guest bathroom" exists
    bathrooms = sorted([rid for rid, rtype in room_spec.room_type_map.items() if rtype == "Bathroom"])
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

    # Rule: Second+ bathrooms must be adjacent to at least one bedroom (for en-suite door placement)
    # Note: Physical adjacency to multiple rooms is OK - door placement in doors.py
    # ensures bathrooms only get one door (enforced at line 366-378 in doors.py)
    if len(bathrooms) >= 2:
        for bathroom_id in bathrooms[1:]:  # Skip first bathroom
            adjacent_ids = adjacencies.get(bathroom_id, set())
            adjacent_types = {room_spec.room_type_map.get(adj_id) for adj_id in adjacent_ids}
            # Must have at least one adjacent bedroom (so door can be placed there)
            if "Bedroom" not in adjacent_types:
                return False

    # Rule: Check room proportions (e.g., bathroom cannot exceed living room)
    proportions_valid, _ = check_room_proportions(room_spec, floorplan)
    if not proportions_valid:
        return False

    return True


def recursively_expand_rooms(
    rooms: Sequence[Union[LeafRoom, MetaRoom]],
    floorplan: np.ndarray,
    room_spec: Optional[RoomSpec] = None,
    interior_boundary_scale: float = 1.9,
) -> None:
    """Assign rooms to the floorplan and expand it if it is a MetaRoom."""
    expand_rooms(rooms, floorplan, room_spec=room_spec, interior_boundary_scale=interior_boundary_scale)
    for room in rooms:
        if isinstance(room, MetaRoom):
            floorplan_mask = floorplan == room.room_id
            floorplan[floorplan_mask] = EMPTY_ROOM_ID
            recursively_expand_rooms(
                room.children,
                floorplan[room.min_y : room.max_y, room.min_x : room.max_x],
                room_spec=room_spec,
                interior_boundary_scale=interior_boundary_scale,
            )


def _debug_validation(room_spec: RoomSpec, floorplan: np.ndarray):
    """Debug helper to show why validation failed."""
    print("\n--- FLOORPLAN ---")
    print(floorplan)

    print("\n--- ROOM BOUNDS ---")
    for room_id, room_type in room_spec.room_type_map.items():
        room = room_spec.room_map[room_id]
        # Check if room bounds exist (they may not if room was never placed)
        if hasattr(room, 'min_y') and room.min_y is not None:
            print(f"Room {room_id} ({room_type}): ({room.min_y},{room.min_x})-({room.max_y},{room.max_x})")
        else:
            print(f"Room {room_id} ({room_type}): NOT PLACED")

    adjacencies = get_room_adjacencies(floorplan)
    print("\n--- ADJACENCIES ---")
    for room_id, neighbors in adjacencies.items():
        if room_id in room_spec.room_type_map:
            room_type = room_spec.room_type_map[room_id]
            neighbor_types = [room_spec.room_type_map.get(n, "?") for n in neighbors]
            print(f"Room {room_id} ({room_type}): {neighbor_types}")

    print("\n--- CHECKING RULES ---")
    # Check rectangularity
    for room_id, room_type in room_spec.room_type_map.items():
        room = room_spec.room_map[room_id]
        if room_type in ("Bedroom", "Bathroom"):
            # Skip if room was never placed
            if not hasattr(room, 'min_y') or room.min_y is None:
                print(f"FAIL: {room_type} {room_id} was NOT PLACED")
                continue
            is_rect = np.all(floorplan[room.min_y:room.max_y, room.min_x:room.max_x] == room_id)
            if not is_rect:
                print(f"FAIL: {room_type} {room_id} is NOT rectangular")

    # Check bedroom connections - needs at least one non-Kitchen adjacent room
    for room_id, room_type in room_spec.room_type_map.items():
        if room_type == "Bedroom":
            neighbors = adjacencies.get(room_id, set())
            neighbor_types = {room_spec.room_type_map.get(n) for n in neighbors}
            non_kitchen = {t for t in neighbor_types if t != "Kitchen"}
            if not non_kitchen:
                print(f"FAIL: Bedroom {room_id} only has Kitchen adjacent. Neighbors: {neighbor_types}")

    # Check bathroom connections
    # Note: en-suite rule is about DOORS, not physical adjacency
    # doors.py handles placing only one door to private bathrooms
    public_connected = False
    for room_id, room_type in room_spec.room_type_map.items():
        if room_type == "Bathroom":
            neighbors = adjacencies.get(room_id, set())
            neighbor_types = {room_spec.room_type_map.get(n) for n in neighbors}
            if "LivingRoom" in neighbor_types or "Kitchen" in neighbor_types or "Hallway" in neighbor_types:
                public_connected = True

    if not public_connected:
        print("FAIL: No bathroom connected to public area")
    print("--- END DEBUG ---\n")


def _count_available_edge_cells(floorplan: np.ndarray, room_id: int) -> int:
    """Count how many empty cells are adjacent to a room (available edge space)."""
    rows, cols = floorplan.shape
    room_mask = (floorplan == room_id)

    # Use a set to avoid counting the same empty cell multiple times
    available_cells = set()
    for r in range(rows):
        for c in range(cols):
            if room_mask[r, c]:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and floorplan[nr, nc] == 0:
                        available_cells.add((nr, nc))
    return len(available_cells)


def _get_hallway_room(placed_rooms: List, room_spec: "RoomSpec") -> Optional[Any]:
    """Get the hallway room object if it exists."""
    for room in placed_rooms:
        if room_spec.room_type_map.get(room.room_id) == "Hallway":
            return room
    return None


def _get_livingroom_room(placed_rooms: List, room_spec: "RoomSpec") -> Optional[Any]:
    """Get the living room object if it exists."""
    for room in placed_rooms:
        if room_spec.room_type_map.get(room.room_id) == "LivingRoom":
            return room
    return None


def _hallway_can_grow(
    hallway,
    floorplan: np.ndarray,
    interior_boundary_scale: float = 1.9
) -> bool:
    """Check if hallway can grow without exceeding size limits.

    The proportion rule requires: Hallway area <= Bedroom area.
    To ensure this, we limit hallway to the minimum bedroom target size (9.0 sqm).
    """
    if hallway is None:
        return False

    # Calculate current hallway area in cells
    hallway_cells = np.sum(floorplan == hallway.room_id)

    # Convert to square meters
    cell_area_sqm = interior_boundary_scale ** 2  # ~3.61 sqm per cell
    hallway_area_sqm = hallway_cells * cell_area_sqm

    # Max hallway size = minimum bedroom target (9.0 sqm)
    # This ensures hallway <= any bedroom
    MAX_HALLWAY_SQM = 9.0

    return hallway_area_sqm < MAX_HALLWAY_SQM


def _bathroom_can_grow(
    bathroom,
    floorplan: np.ndarray,
    interior_boundary_scale: float = 1.9
) -> bool:
    """Check if bathroom can grow without exceeding size limits.

    The max bathroom size from ROOM_SIZE_TARGETS_SQM is 8.0 sqm.
    This prevents bathrooms from growing excessively large.

    Note: Growth operations (grow_rect, grow_room_in_region) add a full row/column
    at a time. Growing adds max(width, height) cells in the worst case (e.g., a 1x3
    bathroom growing down adds 3 cells). We use this worst-case estimate.
    """
    if bathroom is None:
        return False

    # Count actual cells in the bathroom (handles L-shaped rooms correctly)
    bathroom_cells = np.sum(floorplan == bathroom.room_id)

    # Convert to square meters
    cell_area_sqm = interior_boundary_scale ** 2  # ~3.61 to ~4.84 sqm per cell
    bathroom_area_sqm = bathroom_cells * cell_area_sqm

    # Max bathroom size from ROOM_SIZE_TARGETS_SQM
    MAX_BATHROOM_SQM = 8.0

    # Already at or over limit - stop
    if bathroom_area_sqm >= MAX_BATHROOM_SQM:
        return False

    # Growth adds a full row or column. Use the LARGER dimension as worst-case
    # estimate of cells added (e.g., 1x3 bathroom growing down adds 3 cells).
    current_width = bathroom.max_x - bathroom.min_x
    current_height = bathroom.max_y - bathroom.min_y
    max_grow = max(current_width, current_height) if current_width > 0 and current_height > 0 else 1
    area_after_growth = (bathroom_cells + max_grow) * cell_area_sqm

    return area_after_growth <= MAX_BATHROOM_SQM


def _ensure_enough_edge_space(
    floorplan: np.ndarray,
    placed_rooms: List,
    room_spec: "RoomSpec",
    rooms_needing_connection: int,
    min_cells_per_room: int = 2,
    debug: bool = False
) -> bool:
    """
    Ensure hallway and living room have enough edge space for remaining rooms.
    Grows them if needed. Returns True if sufficient space exists or was created.
    """
    hallway = _get_hallway_room(placed_rooms, room_spec)
    living_room = _get_livingroom_room(placed_rooms, room_spec)

    required_cells = rooms_needing_connection * min_cells_per_room

    # Count available edge cells on hallway and living room
    hallway_available = _count_available_edge_cells(floorplan, hallway.room_id) if hallway else 0
    living_room_available = _count_available_edge_cells(floorplan, living_room.room_id) if living_room else 0
    total_available = hallway_available + living_room_available

    if debug:
        print(f"  Edge space: need {required_cells} for {rooms_needing_connection} rooms, have {total_available}")

    if total_available >= required_cells:
        return True

    # Need to grow - prefer hallway since bedrooms should connect there
    # LIMIT: Hallway growth is limited by proportion rule (hallway <= bedroom)
    max_grow_attempts = 10
    for attempt in range(max_grow_attempts):
        grew = False
        # Try growing hallway first (only if under size limit)
        if hallway and _hallway_can_grow(hallway, floorplan) and grow_l_shape(hallway, floorplan):
            grew = True
            if debug:
                print(f"  Grew hallway (attempt {attempt + 1})")

        # Also try growing living room
        if living_room and grow_l_shape(living_room, floorplan):
            grew = True
            if debug:
                print(f"  Grew living room (attempt {attempt + 1})")

        if not grew:
            break

        # Recount available space
        hallway_available = _count_available_edge_cells(floorplan, hallway.room_id) if hallway else 0
        living_room_available = _count_available_edge_cells(floorplan, living_room.room_id) if living_room else 0
        total_available = hallway_available + living_room_available

        if total_available >= required_cells:
            if debug:
                print(f"  Now have {total_available} edge cells - sufficient!")
            return True

    if debug:
        print(f"  WARNING: Could not create enough edge space ({total_available} < {required_cells})")
    return False


def _debug_print(msg):
    import sys
    print(f"[DEBUG] {msg}", flush=True)
    sys.stdout.flush()

def incremental_generate_floorplan(
    room_spec: RoomSpec,
    interior_boundary: np.ndarray,
    candidate_generations: int = 1,
    interior_boundary_scale: float = 1.9,
    debug: bool = False,
) -> np.ndarray:
    """Generate floorplan by placing rooms one at a time.

    Uses a treemap-based approach first to pre-allocate space for each room,
    which solves the problem of greedy hallway growth consuming all space.
    Falls back to the original incremental approach if treemap fails.

    Args:
        room_spec: Room spec for the floorplan.
        interior_boundary: Interior boundary of the floorplan.
        candidate_generations: Number of candidate generations to try.
        interior_boundary_scale: Scale factor (meters per grid cell).

    Returns:
        The generated floorplan as a numpy array.

    Raises:
        InvalidFloorplan: If unable to generate a valid floorplan.
    """
    print(f"[INC] Starting incremental_generate_floorplan (candidate_generations={candidate_generations})", flush=True)

    # Count bedrooms - treemap is designed for 3+ BR houses
    num_bedrooms = sum(1 for rt in room_spec.room_type_map.values() if rt == "Bedroom")
    has_hallway = "Hallway" in room_spec.room_type_map.values()

    # Try treemap approach first for 3+ BR houses with hallway
    # Use more attempts for larger houses
    if num_bedrooms >= 3 and has_hallway:
        treemap_attempts = max(5, candidate_generations * 2)  # At least 5 attempts for treemap
        print(f"[INC] Large house ({num_bedrooms}BR) - trying treemap with {treemap_attempts} attempts", flush=True)
        try:
            result = treemap_generate_floorplan(
                room_spec=room_spec,
                interior_boundary=interior_boundary,
                candidate_generations=treemap_attempts,
                interior_boundary_scale=interior_boundary_scale,
                debug=True,  # Always debug treemap for now
            )
            print(f"[INC] Treemap approach succeeded!", flush=True)
            return result
        except InvalidFloorplan as e:
            print(f"[INC] Treemap approach failed: {e}", flush=True)
            print(f"[INC] Falling back to original incremental approach", flush=True)
    else:
        print(f"[INC] Small house ({num_bedrooms}BR) - skipping treemap", flush=True)

    # Original incremental approach as fallback
    cell_size_sqm = interior_boundary_scale ** 2

    # Classify rooms by type
    living_rooms = []
    kitchens = []
    hallways = []
    bedrooms = []
    bathrooms = []

    for room_id, room_type in room_spec.room_type_map.items():
        room = room_spec.room_map[room_id]
        if room_type == "LivingRoom":
            living_rooms.append(room)
        elif room_type == "Kitchen":
            kitchens.append(room)
        elif room_type == "Hallway":
            hallways.append(room)
        elif room_type == "Bedroom":
            bedrooms.append(room)
        elif room_type == "Bathroom":
            bathrooms.append(room)

    # Order rooms for placement: LivingRoom first, Kitchen, Hallway, then private rooms
    # Interleave bedrooms and bathrooms to ensure bathrooms have space adjacent to bedrooms
    private_rooms = []
    bedroom_iter = iter(bedrooms)
    bathroom_iter = iter(bathrooms)

    # Place bedrooms and bathrooms in pairs when possible
    for i in range(max(len(bedrooms), len(bathrooms))):
        try:
            private_rooms.append(next(bedroom_iter))
        except StopIteration:
            pass
        # Add a bathroom after every 2 bedrooms or when we've placed all bedrooms
        if (i + 1) % 2 == 0 or i >= len(bedrooms) - 1:
            try:
                private_rooms.append(next(bathroom_iter))
            except StopIteration:
                pass
    # Add any remaining bathrooms
    for bathroom in bathroom_iter:
        private_rooms.append(bathroom)

    placement_order = living_rooms + kitchens + hallways + private_rooms

    best_floorplan = None
    best_score = float("-inf")

    for attempt in range(candidate_generations):
        print(f"[INC] Attempt {attempt+1}/{candidate_generations}", flush=True)
        floorplan = interior_boundary.copy()
        success = True

        # Reset room boundaries for this attempt
        for room in placement_order:
            room.min_x = None
            room.max_x = None
            room.min_y = None
            room.max_y = None

        # Place rooms one at a time
        placed_rooms = []

        # Count rooms that need connection to hallway/living room
        private_room_types = {"Bedroom", "Bathroom"}
        remaining_private = sum(1 for r in placement_order
                                if room_spec.room_type_map[r.room_id] in private_room_types)

        for room in placement_order:
            room_type = room_spec.room_type_map[room.room_id]
            print(f"[INC]   Placing room {room.room_id} ({room_type})", flush=True)

            # Before placing a bedroom/bathroom, ensure we have enough edge space
            if room_type in private_room_types and remaining_private > 0:
                _ensure_enough_edge_space(
                    floorplan=floorplan,
                    placed_rooms=placed_rooms,
                    room_spec=room_spec,
                    rooms_needing_connection=remaining_private,
                    min_cells_per_room=MIN_BEDROOM_DIMENSION_CELLS,
                    debug=True,
                )

            try:
                _place_room_incrementally(
                    room=room,
                    room_type=room_type,
                    floorplan=floorplan,
                    placed_rooms=placed_rooms,
                    room_spec=room_spec,
                )
                placed_rooms.append(room)
                print(f"[INC]   -> Placed OK", flush=True)

                # Decrement count after successful placement
                if room_type in private_room_types:
                    remaining_private -= 1

                # After placing hallway, immediately grow it to create more adjacency space
                # for future bedroom placements
                # LIMIT: Respect hallway size limit (proportion rule)
                if room_type == "Hallway":
                    growth_count = 0
                    for _ in range(5):  # Grow up to 5 times to create a longer corridor
                        if not _hallway_can_grow(room, floorplan):
                            print(f"[INC]   -> Hallway at size limit", flush=True)
                            break
                        if grow_rect(room, floorplan):
                            growth_count += 1
                        else:
                            break
                    print(f"[INC]   -> Hallway grew {growth_count} times", flush=True)

            except InvalidFloorplan as e:
                print(f"[INC]   -> Failed: {e}", flush=True)
                success = False
                break

        if not success:
            print(f"[INC] Attempt {attempt+1} failed, continuing", flush=True)
            continue

        print(f"[INC] All rooms placed, filling empty space", flush=True)
        # Fill remaining empty space - use rectangular growth for bedrooms/bathrooms
        _fill_empty_space(floorplan, placed_rooms, room_spec=room_spec, interior_boundary_scale=interior_boundary_scale)

        print(f"[INC] Validating strict rules", flush=True)
        # Check strict rules
        if not validate_strict_rules(room_spec=room_spec, floorplan=floorplan):
            print("[INC] Validation FAILED - checking why:", flush=True)
            _debug_validation(room_spec, floorplan)
            continue

        print(f"[INC] Scoring floorplan", flush=True)
        score = score_floorplan(
            room_spec=room_spec,
            floorplan=floorplan,
            cell_size_sqm=cell_size_sqm,
        )
        print(f"[INC] Score = {score}", flush=True)

        if best_floorplan is None or score > best_score:
            best_floorplan = floorplan
            best_score = score
        print(f"[INC] Best score = {best_score}", flush=True)
        print(f"[INC] Attempt complete, breaking out", flush=True)

    print(f"[INC] All attempts done, best_floorplan is {'None' if best_floorplan is None else 'set'}", flush=True)
    if best_floorplan is None:
        # Debug info for diagnosis
        import logging
        logging.debug(
            f"Incremental generation failed. Stats: attempts={candidate_generations}, "
            f"rooms={list(room_spec.room_type_map.values())}"
        )
        raise InvalidFloorplan(
            "Failed to generate valid floorplan using incremental approach after "
            f"{candidate_generations} attempts."
        )

    return best_floorplan


def treemap_generate_floorplan(
    room_spec: RoomSpec,
    interior_boundary: np.ndarray,
    candidate_generations: int = 10,
    interior_boundary_scale: float = 1.9,
    debug: bool = False,
) -> np.ndarray:
    """Generate floorplan using treemap-based space allocation.

    This algorithm pre-allocates space for each room BEFORE placement,
    preventing greedy growth by any single room (especially hallways).

    Algorithm:
    1. Use squarified treemap to partition grid into room-sized regions
    2. Place rooms within their allocated regions
    3. Grow rooms to fill their regions
    4. Fill any remaining space

    Args:
        room_spec: Room spec for the floorplan.
        interior_boundary: Interior boundary of the floorplan.
        candidate_generations: Number of attempts to try.
        interior_boundary_scale: Scale factor (meters per grid cell).
        debug: If True, print debug information.

    Returns:
        The generated floorplan as a numpy array.

    Raises:
        InvalidFloorplan: If unable to generate a valid floorplan.
    """
    if debug:
        print(f"[TREEMAP] Starting treemap_generate_floorplan (attempts={candidate_generations})")

    cell_size_sqm = interior_boundary_scale ** 2

    # Extract room ratios from room_spec
    room_ratios = {}
    for room_id in room_spec.room_type_map:
        room = room_spec.room_map.get(room_id)
        if room and hasattr(room, 'ratio'):
            room_ratios[room_id] = room.ratio
        else:
            room_ratios[room_id] = 1.0

    best_floorplan = None
    best_score = float("-inf")

    for attempt in range(candidate_generations):
        if debug:
            print(f"[TREEMAP] Attempt {attempt + 1}/{candidate_generations}")

        floorplan = interior_boundary.copy()

        # Phase 1: Allocate regions using treemap
        regions = allocate_room_regions(
            room_type_map=room_spec.room_type_map,
            room_ratios=room_ratios,
            interior_boundary=floorplan,
            margin_factor=0.0,
        )

        if debug:
            print(f"[TREEMAP]   Allocated {len(regions)} regions")
            for rid, reg in regions.items():
                rtype = room_spec.room_type_map.get(rid, "?")
                print(f"[TREEMAP]     Room {rid} ({rtype}): ({reg.min_row},{reg.min_col})-({reg.max_row},{reg.max_col}) = {reg.area} cells")

        # Phase 2: Place rooms in their allocated regions
        success = True
        placed_rooms = []

        # Order rooms for placement: public first, then hallway, then private
        room_order = []
        for room_id, room_type in room_spec.room_type_map.items():
            if room_type in {"LivingRoom", "Kitchen"}:
                room_order.insert(0, room_id)  # Public first
            elif room_type == "Hallway":
                room_order.append(room_id)  # Hallway in middle
            else:
                room_order.append(room_id)  # Private last

        for room_id in room_order:
            if room_id not in regions:
                if debug:
                    print(f"[TREEMAP]   No region for room {room_id}, skipping")
                continue

            room = room_spec.room_map.get(room_id)
            region = regions[room_id]
            room_type = room_spec.room_type_map.get(room_id, "")

            # Reset room boundaries
            room.min_x = None
            room.max_x = None
            room.min_y = None
            room.max_y = None

            # Determine minimum dimension based on room type
            min_dim = MIN_BEDROOM_DIMENSION_CELLS if room_type == "Bedroom" else 1

            # Place room in region
            if place_room_in_region(room, region, floorplan, min_dimension=min_dim, interior_boundary_scale=interior_boundary_scale):
                placed_rooms.append(room)
                if debug:
                    print(f"[TREEMAP]   Placed room {room_id} ({room_type}) at ({room.min_y},{room.min_x})-({room.max_y},{room.max_x})")
            else:
                if debug:
                    print(f"[TREEMAP]   FAILED to place room {room_id} ({room_type})")
                    # Debug: show what's in the region
                    region_data = floorplan[region.min_row:region.max_row, region.min_col:region.max_col]
                    print(f"[TREEMAP]   Region content:")
                    print(region_data)
                success = False
                break

        if not success:
            if debug:
                print(f"[TREEMAP]   Attempt {attempt + 1} failed during placement")
            continue

        # Phase 3: Grow rooms within their regions
        if debug:
            print(f"[TREEMAP]   Growing rooms within regions")

        for room in placed_rooms:
            room_id = room.room_id
            if room_id not in regions:
                continue

            region = regions[room_id]
            room_type = room_spec.room_type_map.get(room_id, "")

            # Grow until we fill the region or can't grow anymore
            # But limit hallway and bathroom size to prevent proportion check failures
            max_grow = 50
            for _ in range(max_grow):
                # Check hallway size limit before growing
                if room_type == "Hallway" and not _hallway_can_grow(room, floorplan, interior_boundary_scale):
                    break
                # Check bathroom size limit before growing
                if room_type == "Bathroom" and not _bathroom_can_grow(room, floorplan, interior_boundary_scale):
                    break
                if not grow_room_in_region(room, region, floorplan):
                    break

        # Phase 4: Fill any remaining empty space using existing L-shape growth
        if debug:
            print(f"[TREEMAP]   Filling remaining empty space")

        _fill_empty_space(floorplan, placed_rooms, room_spec=room_spec, interior_boundary_scale=interior_boundary_scale)

        # Validate
        if debug:
            print(f"[TREEMAP]   Validating strict rules")

        if not validate_strict_rules(room_spec=room_spec, floorplan=floorplan):
            if debug:
                print(f"[TREEMAP]   Validation FAILED")
                _debug_validation(room_spec, floorplan)
            continue

        # Score the floorplan
        score = score_floorplan(
            room_spec=room_spec,
            floorplan=floorplan,
            cell_size_sqm=cell_size_sqm,
        )

        if debug:
            print(f"[TREEMAP]   Score = {score}")

        if best_floorplan is None or score > best_score:
            best_floorplan = floorplan
            best_score = score
            if debug:
                print(f"[TREEMAP]   New best score!")

    if best_floorplan is None:
        raise InvalidFloorplan(
            f"Treemap generation failed after {candidate_generations} attempts."
        )

    if debug:
        print(f"[TREEMAP] Success! Best score = {best_score}")

    return best_floorplan


def _find_adjacent_empty_region(
    floorplan: np.ndarray,
    target_room_id: int,
    min_size: int = 2,
    min_dim: int = 1,
    exclude_adjacent_to: set = None,
) -> tuple:
    """Find an empty region adjacent to the target room.

    Args:
        floorplan: The current floorplan grid.
        target_room_id: The room to find space adjacent to.
        min_size: Minimum total cells (w * h >= min_size).
        min_dim: Minimum dimension in each direction (min(w, h) >= min_dim).
        exclude_adjacent_to: Set of room IDs that the region must NOT be adjacent to.

    Returns (start_row, start_col, width, height) as Python ints, or None if no suitable region found.
    """
    rows, cols = int(floorplan.shape[0]), int(floorplan.shape[1])
    if exclude_adjacent_to is None:
        exclude_adjacent_to = set()

    # Find all cells adjacent to the target room
    target_mask = floorplan == target_room_id
    candidates = []

    for r in range(rows):
        for c in range(cols):
            if floorplan[r, c] != EMPTY_ROOM_ID:
                continue

            # Check if adjacent to target room
            is_adjacent = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if target_mask[nr, nc]:
                        is_adjacent = True
                        break

            if is_adjacent:
                # Try to find a rectangular region starting from this cell
                for w in range(min_dim, cols - c + 1):
                    for h in range(min_dim, rows - r + 1):
                        region = floorplan[r:r+h, c:c+w]
                        if (region == EMPTY_ROOM_ID).all():
                            if w * h >= min_size and min(w, h) >= min_dim:
                                # Check if this region would be adjacent to excluded rooms
                                is_excluded = False
                                if exclude_adjacent_to:
                                    for rr in range(r, r + h):
                                        for cc in range(c, c + w):
                                            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                                nr, nc = rr + dr, cc + dc
                                                if 0 <= nr < rows and 0 <= nc < cols:
                                                    if floorplan[nr, nc] in exclude_adjacent_to:
                                                        is_excluded = True
                                                        break
                                            if is_excluded:
                                                break
                                        if is_excluded:
                                            break
                                if not is_excluded:
                                    candidates.append((int(r), int(c), int(w), int(h), int(w * h)))

    if not candidates:
        return None

    # Return a random candidate, weighted by size
    random.shuffle(candidates)
    candidates.sort(key=lambda x: x[4], reverse=True)
    # Pick from top candidates
    top_n = min(5, len(candidates))
    choice = random.choice(candidates[:top_n])
    return (int(choice[0]), int(choice[1]), int(choice[2]), int(choice[3]))


def _find_empty_region(
    floorplan: np.ndarray,
    min_size: int = 4,
    prefer_center: bool = True,
    min_dim: int = 1,
) -> tuple:
    """Find an empty rectangular region in the floorplan.

    Args:
        floorplan: The current floorplan grid.
        min_size: Minimum total cells (w * h >= min_size).
        prefer_center: If True, prefer regions closer to center.
        min_dim: Minimum dimension in each direction (min(w, h) >= min_dim).

    Returns (start_row, start_col, width, height) as Python ints, or None if no suitable region found.
    """
    rows, cols = int(floorplan.shape[0]), int(floorplan.shape[1])
    center_r, center_c = rows // 2, cols // 2

    candidates = []

    for r in range(rows):
        for c in range(cols):
            if floorplan[r, c] != EMPTY_ROOM_ID:
                continue

            # Find largest rectangle starting from this cell
            for w in range(min_dim, cols - c + 1):
                for h in range(min_dim, rows - r + 1):
                    region = floorplan[r:r+h, c:c+w]
                    if (region == EMPTY_ROOM_ID).all():
                        if w * h >= min_size and min(w, h) >= min_dim:
                            # Calculate distance from center for weighting
                            dist = abs(r + h/2 - center_r) + abs(c + w/2 - center_c)
                            candidates.append((int(r), int(c), int(w), int(h), int(w * h), dist))

    if not candidates:
        return None

    if prefer_center:
        # Sort by distance to center (ascending) then size (descending)
        candidates.sort(key=lambda x: (x[5], -x[4]))
    else:
        random.shuffle(candidates)
        candidates.sort(key=lambda x: x[4], reverse=True)

    # Pick from top candidates with some randomization
    top_n = min(3, len(candidates))
    choice = random.choice(candidates[:top_n])
    return (int(choice[0]), int(choice[1]), int(choice[2]), int(choice[3]))


def _place_room_incrementally(
    room,
    room_type: str,
    floorplan: np.ndarray,
    placed_rooms: list,
    room_spec: RoomSpec,
) -> None:
    """Place a single room in the floorplan.

    LivingRooms are placed near center.
    Kitchens are placed adjacent to LivingRoom.
    Hallways are placed as narrow corridors.
    Bedrooms are placed adjacent to Hallway or LivingRoom.
    Bathrooms are placed adjacent to Bedrooms (en-suite) or public areas.
    """

    # Find where to place this room based on type
    if room_type == "LivingRoom":
        # First LivingRoom goes near center
        if not placed_rooms:
            region = _find_empty_region(floorplan, min_size=4, prefer_center=True)
        else:
            # Additional LivingRooms go adjacent to existing ones
            existing_living = [r for r in placed_rooms
                             if room_spec.room_type_map.get(r.room_id) == "LivingRoom"]
            if existing_living:
                region = _find_adjacent_empty_region(
                    floorplan, existing_living[0].room_id, min_size=3
                )
            else:
                region = _find_empty_region(floorplan, min_size=3, prefer_center=True)

    elif room_type == "Kitchen":
        # Kitchen goes adjacent to LivingRoom
        living_rooms = [r for r in placed_rooms
                       if room_spec.room_type_map.get(r.room_id) == "LivingRoom"]
        if living_rooms:
            region = _find_adjacent_empty_region(
                floorplan, living_rooms[0].room_id, min_size=3
            )
        else:
            region = _find_empty_region(floorplan, min_size=3, prefer_center=False)

    elif room_type == "Hallway":
        # Hallway should connect public to private areas
        # Place adjacent to LivingRoom ONLY
        # IMPORTANT: Start with a larger hallway to provide adjacency for multiple bedrooms
        living_rooms = [r for r in placed_rooms
                       if room_spec.room_type_map.get(r.room_id) == "LivingRoom"]
        if living_rooms:
            # Start with larger min_size for hallway to have room to grow as corridor
            region = _find_adjacent_empty_region(
                floorplan, random.choice(living_rooms).room_id, min_size=4
            )
            if region is None:
                # Fallback to smaller if can't find larger space
                region = _find_adjacent_empty_region(
                    floorplan, random.choice(living_rooms).room_id, min_size=2
                )
        else:
            region = _find_empty_region(floorplan, min_size=4, prefer_center=False)
            if region is None:
                region = _find_empty_region(floorplan, min_size=2, prefer_center=False)

    elif room_type == "Bedroom":
        # Try multiple placement strategies in order of preference:
        # 1. Adjacent to Hallway (if exists)
        # 2. Grow hallway to create space
        # 3. Adjacent to LivingRoom
        # 4. Grow LivingRoom to create space
        # 5. Any empty region
        # Bedrooms require min 2x2 dimensions to pass validation
        region = None

        hallways = [r for r in placed_rooms
                   if room_spec.room_type_map.get(r.room_id) == "Hallway"]
        living_rooms = [r for r in placed_rooms
                      if room_spec.room_type_map.get(r.room_id) == "LivingRoom"]

        print(f"  [DEBUG] Placing bedroom {room.room_id}, hallways={len(hallways)}, living_rooms={len(living_rooms)}")

        # Try adjacent to hallway first
        if hallways and region is None:
            region = _find_adjacent_empty_region(
                floorplan, hallways[0].room_id, min_size=4, min_dim=MIN_BEDROOM_DIMENSION_CELLS
            )
            print(f"  [DEBUG] After hallway adjacent check: region={region}")

        # If no space adjacent to hallway, grow the hallway into empty space
        # Keep growing until we find space or can't grow anymore
        # LIMIT: Stop growing if hallway would exceed bedroom size (proportion rule)
        if region is None and hallways:
            hallway = hallways[0]
            max_grow_attempts = 10  # More aggressive growth
            for attempt in range(max_grow_attempts):
                # Check size limit before growing
                if not _hallway_can_grow(hallway, floorplan):
                    print(f"  [DEBUG] Hallway at size limit, skipping growth")
                    break
                grew = grow_rect(hallway, floorplan)
                print(f"  [DEBUG] Hallway grow attempt {attempt+1}: grew={grew}")
                if grew:
                    region = _find_adjacent_empty_region(
                        floorplan, hallway.room_id, min_size=4, min_dim=MIN_BEDROOM_DIMENSION_CELLS
                    )
                    if region is not None:
                        print(f"  [DEBUG] Found region after hallway growth: {region}")
                        break
                else:
                    print(f"  [DEBUG] Hallway can't grow anymore")
                    break

        # Try adjacent to living room
        if region is None and living_rooms:
            for lr in living_rooms:
                region = _find_adjacent_empty_region(
                    floorplan, lr.room_id, min_size=4, min_dim=MIN_BEDROOM_DIMENSION_CELLS
                )
                if region is not None:
                    print(f"  [DEBUG] Found region adjacent to living room {lr.room_id}")
                    break

        # Alternate growth strategy: try hallway first (preferred for bedroom connectivity)
        # then living room. Repeat until we find space or neither can grow.
        # LIMIT: Hallway growth is limited by proportion rule (hallway <= bedroom)
        max_total_attempts = 15
        for attempt in range(max_total_attempts):
            if region is not None:
                break

            # Try hallway growth (L-shape to reach around corners)
            # Only grow if under size limit
            hallway_grew = False
            if hallways and _hallway_can_grow(hallways[0], floorplan):
                hallway = hallways[0]
                hallway_grew = grow_rect(hallway, floorplan) or grow_l_shape(hallway, floorplan)
                if hallway_grew:
                    print(f"  [DEBUG] Hallway grew (attempt {attempt+1})")
                    region = _find_adjacent_empty_region(
                        floorplan, hallway.room_id, min_size=4, min_dim=MIN_BEDROOM_DIMENSION_CELLS
                    )
                    if region is not None:
                        print(f"  [DEBUG] Found region after hallway growth: {region}")
                        break

            # Try living room growth (L-shape to extend around)
            lr_grew = False
            if living_rooms and region is None:
                lr = living_rooms[0]
                lr_grew = grow_rect(lr, floorplan) or grow_l_shape(lr, floorplan)
                if lr_grew:
                    print(f"  [DEBUG] LivingRoom grew (attempt {attempt+1})")
                    region = _find_adjacent_empty_region(
                        floorplan, lr.room_id, min_size=4, min_dim=MIN_BEDROOM_DIMENSION_CELLS
                    )
                    if region is not None:
                        print(f"  [DEBUG] Found region after living room growth: {region}")
                        break

            # If neither grew this round, we're stuck
            if not hallway_grew and not lr_grew:
                print(f"  [DEBUG] Neither hallway nor living room could grow")
                break

        # If still no region, fail rather than placing disconnected bedroom
        if region is None:
            print(f"  [DEBUG] FAILED to place bedroom {room.room_id}")
            print(f"  [DEBUG] Current floorplan:\n{floorplan}")
            raise InvalidFloorplan(
                f"Cannot place bedroom {room.room_id} adjacent to hallway or living room"
            )

    elif room_type == "Bathroom":
        # First bathroom: place adjacent to public area (LivingRoom/Kitchen/Hallway)
        # Subsequent bathrooms: place adjacent to a bedroom (en-suite)
        bathrooms_placed = [r for r in placed_rooms
                          if room_spec.room_type_map.get(r.room_id) == "Bathroom"]

        if not bathrooms_placed:
            # First bathroom - needs public access
            public_rooms = [r for r in placed_rooms
                          if room_spec.room_type_map.get(r.room_id) in
                          {"LivingRoom", "Kitchen", "Hallway"}]
            if public_rooms:
                region = _find_adjacent_empty_region(
                    floorplan, random.choice(public_rooms).room_id, min_size=2
                )
            else:
                region = _find_empty_region(floorplan, min_size=2, prefer_center=False)
        else:
            # En-suite bathroom - adjacent to exactly ONE bedroom (strict en-suite rule)
            bedrooms = [r for r in placed_rooms
                       if room_spec.room_type_map.get(r.room_id) == "Bedroom"]
            # Find bedrooms that don't already have a connected bathroom
            available_bedrooms = []
            for bedroom in bedrooms:
                has_bathroom = False
                for bathroom in bathrooms_placed:
                    if _rooms_adjacent(bedroom.room_id, bathroom.room_id, floorplan):
                        has_bathroom = True
                        break
                if not has_bathroom:
                    available_bedrooms.append(bedroom)

            if available_bedrooms:
                # Try to find a region adjacent to ONE bedroom but NOT adjacent to others
                target_bedroom = random.choice(available_bedrooms)
                other_bedrooms = {b.room_id for b in bedrooms if b.room_id != target_bedroom.room_id}
                region = _find_adjacent_empty_region(
                    floorplan, target_bedroom.room_id, min_size=2,
                    exclude_adjacent_to=other_bedrooms
                )
                # If no exclusive region found, try without exclusion (may fail validation)
                if region is None:
                    region = _find_adjacent_empty_region(
                        floorplan, target_bedroom.room_id, min_size=2
                    )
            elif bedrooms:
                region = _find_adjacent_empty_region(
                    floorplan, random.choice(bedrooms).room_id, min_size=2
                )
            else:
                region = _find_empty_region(floorplan, min_size=2, prefer_center=False)
    else:
        region = _find_empty_region(floorplan, min_size=2, prefer_center=False)

    if region is None:
        raise InvalidFloorplan(f"No space to place {room_type} (room_id={room.room_id})")

    # Ensure all indices are Python ints (not numpy types)
    start_r, start_c, width, height = int(region[0]), int(region[1]), int(region[2]), int(region[3])

    # Calculate total grid size and number of rooms to estimate fair share
    total_cells = floorplan.size
    num_rooms = len(room_spec.room_type_map)

    # Calculate fair share per room - divide space evenly then scale by room ratio
    total_ratio = sum(r.ratio for r in room_spec.room_map.values()
                      if hasattr(r, 'ratio') and r.ratio is not None)
    if total_ratio > 0:
        room_ratio_share = room.ratio / total_ratio
    else:
        room_ratio_share = 1.0 / num_rooms

    # Target cells based on fair share - be conservative initially (use 50-70% of fair share)
    # so rooms have space to be placed, then grow later
    conservation_factor = 0.5 if num_rooms > 6 else 0.7
    target_cells = int(total_cells * room_ratio_share * conservation_factor)
    target_cells = min(target_cells, width * height)
    target_cells = max(target_cells, 2)  # At least 2 cells

    # Also limit by room type - some rooms should be smaller initially
    if room_type == "Bathroom":
        target_cells = min(target_cells, 4)  # Bathrooms are small
    elif room_type == "Hallway":
        target_cells = min(target_cells, 6)  # Hallways are narrow
    elif room_type == "Kitchen":
        target_cells = min(target_cells, 9)  # Kitchens are modest

    # Calculate dimensions that fit the target - ensure all values are ints
    if room_type == "Hallway":
        # Hallways should be narrow
        if width > height:
            used_w = int(min(width, max(target_cells, 3)))
            used_h = 1
        else:
            used_w = 1
            used_h = int(min(height, max(target_cells, 3)))
    elif room_type == "Bedroom":
        # Bedrooms must be at least 2x2 to pass validation
        min_dim = MIN_BEDROOM_DIMENSION_CELLS
        used_w = int(max(min_dim, min(width, int(target_cells ** 0.5) + 1)))
        used_h = int(max(min_dim, min(height, int(target_cells ** 0.5) + 1)))
        while used_w * used_h < target_cells and used_w < width:
            used_w += 1
        while used_w * used_h < target_cells and used_h < height:
            used_h += 1
    else:
        # Try to keep rooms somewhat square
        used_w = int(min(width, int(target_cells ** 0.5) + 1))
        used_h = int(min(height, int(target_cells ** 0.5) + 1))
        while used_w * used_h < target_cells and used_w < width:
            used_w += 1
        while used_w * used_h < target_cells and used_h < height:
            used_h += 1

    # Place the room - ensure all bounds are Python ints for grow_l_shape compatibility
    room.min_x = int(start_c)
    room.max_x = int(start_c + used_w)
    room.min_y = int(start_r)
    room.max_y = int(start_r + used_h)

    floorplan[int(start_r):int(start_r + used_h), int(start_c):int(start_c + used_w)] = room.room_id


def _rooms_adjacent(room_id1: int, room_id2: int, floorplan: np.ndarray) -> bool:
    """Check if two rooms are adjacent in the floorplan."""
    rows, cols = floorplan.shape
    mask1 = floorplan == room_id1

    for r in range(rows):
        for c in range(cols):
            if mask1[r, c]:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if floorplan[nr, nc] == room_id2:
                            return True
    return False


def _fill_empty_space(
    floorplan: np.ndarray,
    placed_rooms: list,
    room_spec: Optional[RoomSpec] = None,
    interior_boundary_scale: float = 1.9,
) -> None:
    """Fill remaining empty space by growing existing rooms.

    For bedrooms and bathrooms, only grow rectangularly to maintain their
    required rectangular shape. For other rooms, use L-shape growth.

    En-suite bathrooms (second+ bathrooms) are NOT grown to preserve the
    strict 1-bedroom adjacency rule.
    """
    # Separate rooms into rectangular-only (bedrooms/bathrooms) and L-shape allowed
    rect_only_types = {"Bedroom", "Bathroom"}
    rect_rooms = set()
    lshape_rooms = set()

    # Identify en-suite bathrooms (all bathrooms after the first are en-suite)
    # These should NOT grow because they must have exactly 1 bedroom neighbor
    bathrooms = []
    for room in placed_rooms:
        if room_spec and room_spec.room_type_map.get(room.room_id) == "Bathroom":
            bathrooms.append(room)
    ensuite_bathrooms = set(bathrooms[1:]) if len(bathrooms) > 1 else set()

    for room in placed_rooms:
        # Skip en-suite bathrooms - they shouldn't grow
        if room in ensuite_bathrooms:
            continue
        if room_spec and room_spec.room_type_map.get(room.room_id) in rect_only_types:
            rect_rooms.add(room)
        else:
            lshape_rooms.add(room)

    # First, grow rectangular rooms (bedrooms/first bathroom) with grow_rect only
    max_iterations = 500
    iteration = 0
    _debug_print(f"_fill_empty_space: starting rect growth for {len(rect_rooms)} rooms")
    while rect_rooms and iteration < max_iterations:
        iteration += 1
        if iteration % 100 == 0:
            _debug_print(f"_fill_empty_space: rect iteration {iteration}")
        room = random.choice(list(rect_rooms))

        # Check if this is a bathroom and enforce size limit
        room_type = room_spec.room_type_map.get(room.room_id) if room_spec else None
        if room_type == "Bathroom" and not _bathroom_can_grow(room, floorplan, interior_boundary_scale):
            rect_rooms.discard(room)
            continue

        can_grow = grow_rect(room, floorplan)
        if not can_grow:
            rect_rooms.discard(room)
    _debug_print(f"_fill_empty_space: rect growth done after {iteration} iterations")

    # Then, grow other rooms with L-shape growth
    # BUT limit hallway size to prevent proportion check failures
    iteration = 0
    _debug_print(f"_fill_empty_space: starting L-shape growth for {len(lshape_rooms)} rooms")
    while lshape_rooms and iteration < max_iterations:
        iteration += 1
        if iteration % 100 == 0:
            _debug_print(f"_fill_empty_space: L-shape iteration {iteration}")
        room = random.choice(list(lshape_rooms))

        # Check if this is a hallway and enforce size limit
        room_type = room_spec.room_type_map.get(room.room_id) if room_spec else None
        if room_type == "Hallway" and not _hallway_can_grow(room, floorplan, interior_boundary_scale):
            lshape_rooms.discard(room)
            continue

        # Check if this is a bathroom and enforce size limit
        if room_type == "Bathroom" and not _bathroom_can_grow(room, floorplan, interior_boundary_scale):
            lshape_rooms.discard(room)
            continue

        can_grow = grow_l_shape(room, floorplan)
        if not can_grow:
            lshape_rooms.discard(room)
    _debug_print(f"_fill_empty_space: L-shape growth done after {iteration} iterations")


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
            recursively_expand_rooms(
                rooms=room_spec.spec,
                floorplan=floorplan,
                room_spec=room_spec,
                interior_boundary_scale=interior_boundary_scale,
            )
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
