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


def get_hallway_shape_score(
    room_spec: RoomSpec,
    floorplan: np.ndarray,
    interior_boundary_scale: float = 1.9,
) -> float:
    """Score hallways based on their shape - prefer elongated hallways.

    Args:
        room_spec: The room specification.
        floorplan: The floorplan array.
        interior_boundary_scale: Meters per grid cell (default 1.9m).

    Returns:
        - Bonus if hallway has aspect ratio > 2 (elongated)
        - Penalty if hallway is too square (aspect ratio < 1.5)
        - Penalty if hallway is too narrow (width < 1.5m)
    """
    from procthor.generation.room_sizing import MIN_HALLWAY_WIDTH_M

    score = 0.0
    for room_id, room_type in room_spec.room_type_map.items():
        if room_type == "Hallway":
            width, height, area = get_room_dimensions(room_id, floorplan)
            if width == 0 or height == 0:
                continue

            # Convert grid cells to meters
            min_dim_cells = min(width, height)
            min_dim_meters = min_dim_cells * interior_boundary_scale

            # Penalty for narrow hallways (less than MIN_HALLWAY_WIDTH_M)
            # Hallways need space for multiple doors
            if min_dim_meters < MIN_HALLWAY_WIDTH_M:
                score -= 5.0  # Narrow hallway penalty

            # Aspect ratio: how elongated is the hallway?
            aspect_ratio = max(width, height) / min(width, height)

            # Hallways should have aspect ratio >= 2 (at least 2x longer than wide)
            if aspect_ratio >= 2.0:
                score += 2.0  # Good, elongated hallway
            elif aspect_ratio >= 1.5:
                score += 0.5  # Acceptable
            else:
                score -= 3.0  # Too square, penalty

            # Penalty if hallway takes up too much space
            total_area = (floorplan != OUTDOOR_ROOM_ID).sum()
            hallway_ratio = area / total_area
            if hallway_ratio > 0.15:  # Hallway is more than 15% of house
                score -= 2.0

    return score


def get_hallway_connectivity_score(room_spec: RoomSpec, floorplan: np.ndarray) -> float:
    """Score hallways based on their connectivity.

    Returns:
        - Bonus if hallway connects to multiple rooms
        - Penalty if hallway is isolated (connects to 0 or 1 room)
    """
    score = 0.0
    adjacencies = get_room_adjacencies(floorplan)

    for room_id, room_type in room_spec.room_type_map.items():
        if room_type == "Hallway":
            adjacent_ids = adjacencies.get(room_id, set())
            num_connections = len(adjacent_ids)

            # Hallway should connect multiple rooms
            if num_connections == 0:
                score -= 10.0  # Isolated hallway - heavy penalty
            elif num_connections == 1:
                score -= 5.0  # Dead-end hallway - moderate penalty
            elif num_connections == 2:
                score += 1.0  # Connects two rooms - acceptable
            else:
                score += 2.0 + (num_connections - 2) * 0.5  # Multiple connections - bonus

            # Extra bonus if hallway connects to LivingRoom
            adjacent_types = {room_spec.room_type_map.get(adj_id) for adj_id in adjacent_ids}
            if "LivingRoom" in adjacent_types:
                score += 2.0

    return score


def get_living_room_shape_score(
    room_spec: RoomSpec,
    floorplan: np.ndarray,
    interior_boundary_scale: float = 1.9,
) -> float:
    """Score living rooms based on shape quality.

    Args:
        room_spec: The room specification.
        floorplan: The floorplan array.
        interior_boundary_scale: Meters per grid cell (default 1.9m).

    Returns:
        - Bonus if aspect ratio is 1:1 to 2:1 (ideal rectangle): +5.0
        - No change if aspect ratio is 2:1 to 3:1
        - Penalty if aspect ratio > 3:1 (too narrow): -15.0
        - Penalty if width < 3m (can't fit furniture): -20.0
        - Penalty if room is fragmented (non-contiguous): -30.0
    """
    from scipy import ndimage

    score = 0.0
    for room_id, room_type in room_spec.room_type_map.items():
        if room_type == "LivingRoom":
            width, height, area = get_room_dimensions(room_id, floorplan)
            if width == 0 or height == 0:
                continue

            # Check for fragmentation (non-contiguous cells)
            room_mask = floorplan == room_id
            labeled_array, num_features = ndimage.label(room_mask)
            if num_features > 1:
                score -= 30.0  # Fragmented room - critical penalty
                continue  # Skip other checks for fragmented rooms

            # Convert grid cells to meters
            min_dim_cells = min(width, height)
            min_dim_meters = min_dim_cells * interior_boundary_scale

            # Penalty for narrow living rooms (width < 3m)
            if min_dim_meters < 3.0:
                score -= 20.0  # Too narrow to furnish - critical penalty

            # Aspect ratio scoring
            aspect_ratio = max(width, height) / min(width, height)

            if aspect_ratio <= 2.0:
                score += 5.0  # Ideal rectangle (1:1 to 2:1)
            elif aspect_ratio <= 3.0:
                pass  # Acceptable (2:1 to 3:1) - no change
            else:
                score -= 15.0  # Too narrow (> 3:1) - critical penalty

    return score


def get_door_spacing_score(room_spec: RoomSpec, floorplan: np.ndarray) -> float:
    """Score floorplan based on door spacing feasibility.

    Penalizes layouts where rooms have many shared boundaries, which increases
    the likelihood of door spacing conflicts and "Might be unable to walk
    between rooms" warnings.

    Returns:
        Score adjustment (negative for problematic layouts).
    """
    score = 0.0
    adjacencies = get_room_adjacencies(floorplan)

    # Count total adjacency pairs (each pair represents a potential door)
    adjacency_pairs = set()
    for room_id, neighbors in adjacencies.items():
        for neighbor_id in neighbors:
            pair = (min(room_id, neighbor_id), max(room_id, neighbor_id))
            adjacency_pairs.add(pair)

    num_rooms = len(room_spec.room_type_map)
    num_adjacencies = len(adjacency_pairs)

    # Penalty for high door density
    # Ideal: roughly num_rooms - 1 doors for a tree-like connectivity
    # Problematic: much more than that (mesh-like connectivity)
    expected_doors = num_rooms - 1
    excess_doors = max(0, num_adjacencies - expected_doors - 1)
    score -= excess_doors * 0.5  # Penalty for each excess door

    # Count rooms with 4+ neighbors (corner conflict prone)
    high_connectivity_rooms = sum(
        1 for neighbors in adjacencies.values() if len(neighbors) >= 4
    )
    score -= high_connectivity_rooms * 2.0  # Heavy penalty for highly-connected rooms

    # Bonus for rooms with exactly 1-2 neighbors (simpler door placement)
    simple_rooms = sum(
        1 for neighbors in adjacencies.values() if len(neighbors) <= 2
    )
    score += simple_rooms * 0.3

    return score


def get_room_proportion_score(room_spec: RoomSpec, floorplan: np.ndarray) -> float:
    """Score based on room size relationships.

    Returns:
        - Bonus if living room is the largest room (+5.0)
        - Penalty if living room is NOT largest (-25.0) - critical violation
        - Penalty if hallway is larger than any bedroom (-20.0) - critical violation
        - Bonus if hallway is smaller than all bedrooms (+5.0)
        - Bonus if kitchen+living is 40-50% of total area (+3.0)
        - Penalty if kitchen+living is <30% or >60% of total area (-15.0)
    """
    score = 0.0

    # Get room sizes (areas)
    room_sizes = {}
    total_area = (floorplan != OUTDOOR_ROOM_ID).sum()
    if total_area == 0:
        return score

    for room_id, room_type in room_spec.room_type_map.items():
        room_sizes[room_id] = (floorplan == room_id).sum()

    # Group rooms by type for comparison
    living_room_area = 0
    kitchen_area = 0
    hallway_areas = []
    bedroom_areas = []

    for room_id, room_type in room_spec.room_type_map.items():
        area = room_sizes.get(room_id, 0)
        if room_type == "LivingRoom":
            living_room_area = area
        elif room_type == "Kitchen":
            kitchen_area = area
        elif room_type == "Hallway":
            hallway_areas.append(area)
        elif room_type == "Bedroom":
            bedroom_areas.append(area)

    # Get the largest room size overall
    max_room_size = max(room_sizes.values()) if room_sizes else 0

    # Rule 1: LivingRoom should be the largest room
    if living_room_area > 0:
        if living_room_area >= max_room_size:
            score += 5.0  # LivingRoom is largest → bonus
        else:
            score -= 25.0  # LivingRoom is NOT largest → critical penalty

    # Rule 2: Hallway vs Bedroom comparison
    if hallway_areas and bedroom_areas:
        min_bedroom_area = min(bedroom_areas)
        max_hallway_area = max(hallway_areas)

        if max_hallway_area > min_bedroom_area:
            score -= 20.0  # Hallway larger than some bedroom → critical penalty
        else:
            score += 5.0  # Hallway smaller than all bedrooms → bonus

    # Rule 3: Kitchen + LivingRoom proportion of total house
    combined_area = kitchen_area + living_room_area
    if combined_area > 0:
        proportion = combined_area / total_area
        if 0.40 <= proportion <= 0.50:
            score += 3.0  # Kitchen+LivingRoom is 40-50% → bonus
        elif proportion < 0.30 or proportion > 0.60:
            score -= 15.0  # Kitchen+LivingRoom is <30% or >60% → penalty

    return score


def get_adjacency_score(room_spec: RoomSpec, floorplan: np.ndarray) -> float:
    """Score based on room adjacency relationships.

    Returns:
        - Bonus if kitchen is adjacent to living room (+5.0)
        - Penalty if kitchen NOT adjacent to living room (-15.0)
        - Penalty if kitchen is adjacent to hallway (-5.0)
        - Bonus if living room is adjacent to hallway (+3.0)
        - Bonus if hallway is adjacent to at least one bedroom (+2.0)
        - Penalty if hallway NOT adjacent to any bedroom (-10.0)
    """
    score = 0.0
    adjacencies = get_room_adjacencies(floorplan)

    # Build reverse lookup: room_type -> set of room_ids
    type_to_ids = {}
    for room_id, room_type in room_spec.room_type_map.items():
        if room_type not in type_to_ids:
            type_to_ids[room_type] = set()
        type_to_ids[room_type].add(room_id)

    kitchen_ids = type_to_ids.get("Kitchen", set())
    living_ids = type_to_ids.get("LivingRoom", set())
    hallway_ids = type_to_ids.get("Hallway", set())
    bedroom_ids = type_to_ids.get("Bedroom", set())

    # Kitchen <-> LivingRoom adjacency scoring
    for kitchen_id in kitchen_ids:
        kitchen_neighbors = adjacencies.get(kitchen_id, set())
        if kitchen_neighbors & living_ids:
            # Kitchen adjacent to LivingRoom → +5.0 bonus (open plan)
            score += 5.0
        else:
            # Kitchen NOT adjacent to LivingRoom → -15.0 penalty
            score -= 15.0

        # Kitchen adjacent to Hallway → -5.0 penalty (bad circulation)
        if kitchen_neighbors & hallway_ids:
            score -= 5.0

    # LivingRoom <-> Hallway adjacency scoring
    for living_id in living_ids:
        living_neighbors = adjacencies.get(living_id, set())
        if living_neighbors & hallway_ids:
            # LivingRoom adjacent to Hallway → +3.0 bonus
            score += 3.0

    # Hallway <-> Bedroom adjacency scoring
    for hallway_id in hallway_ids:
        hallway_neighbors = adjacencies.get(hallway_id, set())
        if hallway_neighbors & bedroom_ids:
            # Hallway adjacent to at least 1 Bedroom → +2.0 bonus
            score += 2.0
        else:
            # Hallway adjacent to 0 Bedrooms → -10.0 penalty
            score -= 10.0

    return score


def score_floorplan(room_spec: RoomSpec, floorplan: np.ndarray) -> float:
    """Calculate the quality of the floorplan based on the room specifications."""
    score = 0.0

    # Base score: how well do room sizes match the spec?
    score += get_ratio_overlap_score(room_spec, floorplan)

    # Hallway shape: prefer elongated hallways (aspect ratio > 2)
    score += get_hallway_shape_score(room_spec, floorplan)

    # Hallway connectivity: prefer hallways that connect multiple rooms
    score += get_hallway_connectivity_score(room_spec, floorplan)

    # Door spacing: penalize layouts prone to door spacing conflicts
    score += get_door_spacing_score(room_spec, floorplan)

    # Room proportions: ensure proper room size relationships
    score += get_room_proportion_score(room_spec, floorplan)

    # Living room shape: penalize narrow, fragmented, or oddly shaped living rooms
    score += get_living_room_shape_score(room_spec, floorplan)

    # Room adjacency: reward good adjacencies (kitchen-living, hallway-bedroom)
    score += get_adjacency_score(room_spec, floorplan)

    return score


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
) -> np.ndarray:
    """Generate a floorplan for the given room spec and interior boundary.

    Args:
        room_spec: Room spec for the floorplan.
        interior_boundary: Interior boundary of the floorplan.
        candidate_generations: Number of candidate generations to generate. The
            best candidate floorplan is returned.
    """
    # NOTE: If there is only one room, the floorplan will always be the same.
    if len(room_spec.room_type_map) == 1:
        candidate_generations = 1

    best_floorplan = None
    best_score = float("-inf")
    for _ in range(candidate_generations):
        floorplan = interior_boundary.copy()
        try:
            recursively_expand_rooms(rooms=room_spec.spec, floorplan=floorplan)
        except InvalidFloorplan:
            continue
        else:
            score = score_floorplan(room_spec=room_spec, floorplan=floorplan)
            if best_floorplan is None or score > best_score:
                best_floorplan = floorplan
                best_score = score

    if best_floorplan is None:
        raise InvalidFloorplan(
            "Failed to generate a valid floorplan all candidate_generations="
            f"{candidate_generations} times from the interior boundary!"
            " This means the sampled interior boundary is too small for the room"
            " spec. Try again with a another interior boundary.\n"
            f"interior_boundary:\n{interior_boundary}\n, room_spec:\n{room_spec}"
        )

    return best_floorplan
