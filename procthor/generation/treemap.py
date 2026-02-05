"""
Space-Reserving Treemap Algorithm for Floorplan Generation.

This module implements a squarified treemap algorithm to pre-allocate space
regions for each room BEFORE placement. This solves the greedy growth problem
where hallways consume too much space leaving no room for bedrooms.

Algorithm has 3 phases:
1. Space Allocation: Use treemap to partition grid into room-sized regions
2. Constrained Placement: Place rooms only within their allocated regions
3. Adjacency Refinement: Adjust regions to ensure required adjacencies

References:
- Squarified Treemap: https://www.win.tue.nl/~vanwijk/stm.pdf
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import random
import numpy as np

from procthor.constants import EMPTY_ROOM_ID, OUTDOOR_ROOM_ID


@dataclass
class RoomRegion:
    """A rectangular region allocated for a room."""
    room_id: int
    room_type: str
    min_row: int
    max_row: int
    min_col: int
    max_col: int
    target_cells: int = 0
    
    @property
    def width(self) -> int:
        return self.max_col - self.min_col
    
    @property
    def height(self) -> int:
        return self.max_row - self.min_row
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        """Return (row, col) center of region."""
        return (
            (self.min_row + self.max_row) // 2,
            (self.min_col + self.max_col) // 2
        )
    
    def contains(self, row: int, col: int) -> bool:
        """Check if point is within this region."""
        return (self.min_row <= row < self.max_row and 
                self.min_col <= col < self.max_col)


@dataclass
class TreemapNode:
    """A node in the treemap tree, representing a region to partition."""
    bounds: Tuple[int, int, int, int]  # (min_row, max_row, min_col, max_col)
    rooms: List[Tuple[int, str, float]]  # List of (room_id, room_type, weight)
    children: List["TreemapNode"] = field(default_factory=list)
    allocated_region: Optional[RoomRegion] = None
    
    @property
    def width(self) -> int:
        return self.bounds[3] - self.bounds[2]
    
    @property
    def height(self) -> int:
        return self.bounds[1] - self.bounds[0]
    
    @property
    def area(self) -> int:
        return self.width * self.height


def _worst_aspect_ratio(widths: List[float], side_length: float) -> float:
    """Calculate worst (max) aspect ratio for a row of rectangles."""
    if not widths or side_length <= 0:
        return float('inf')
    
    total = sum(widths)
    if total <= 0:
        return float('inf')
    
    # Height of row = total_area / side_length
    row_height = total / side_length if side_length > 0 else 0
    
    max_ratio = 0.0
    for w in widths:
        if row_height > 0 and w > 0:
            rect_width = w / row_height
            ratio = max(rect_width / row_height, row_height / rect_width) if row_height > 0 else float('inf')
            max_ratio = max(max_ratio, ratio)
    
    return max_ratio


def squarified_layout(
    values: List[float],
    bounds: Tuple[int, int, int, int]
) -> List[Tuple[int, int, int, int]]:
    """Partition a rectangle into sub-rectangles using squarified treemap.
    
    Args:
        values: List of areas for each sub-rectangle (will be normalized)
        bounds: (min_row, max_row, min_col, max_col) boundary
    
    Returns:
        List of (min_row, max_row, min_col, max_col) for each sub-rectangle
    """
    if not values:
        return []
    
    if len(values) == 1:
        return [bounds]
    
    min_row, max_row, min_col, max_col = bounds
    total_area = (max_row - min_row) * (max_col - min_col)
    
    if total_area <= 0:
        return [bounds] * len(values)
    
    # Normalize values to fill the available area
    total_value = sum(values)
    if total_value <= 0:
        normalized = [total_area / len(values)] * len(values)
    else:
        normalized = [v / total_value * total_area for v in values]
    
    # Sort by area (descending) for better aspect ratios
    indexed = list(enumerate(normalized))
    indexed.sort(key=lambda x: -x[1])
    
    # Layout using squarified algorithm
    results = [None] * len(values)
    _squarify_recursive(indexed, bounds, results)

    # Post-process: ensure each region has minimum 2x2 dimensions where possible
    # This helps prevent narrow bedroom allocations
    min_row, max_row, min_col, max_col = bounds
    for i, (r_min, r_max, c_min, c_max) in enumerate(results):
        width = c_max - c_min
        height = r_max - r_min

        # Try to expand narrow regions if space allows
        if width < 2 and c_max < max_col:
            results[i] = (r_min, r_max, c_min, c_min + 2)
        elif width < 2 and c_min > min_col:
            results[i] = (r_min, r_max, c_max - 2, c_max)

        # Re-read after potential width fix
        r_min, r_max, c_min, c_max = results[i]
        height = r_max - r_min

        if height < 2 and r_max < max_row:
            results[i] = (r_min, r_min + 2, c_min, c_max)
        elif height < 2 and r_min > min_row:
            results[i] = (r_max - 2, r_max, c_min, c_max)

    return results


def _squarify_recursive(
    indexed_values: List[Tuple[int, float]],
    bounds: Tuple[int, int, int, int],
    results: List
) -> None:
    """Recursive squarify algorithm."""
    if not indexed_values:
        return

    min_row, max_row, min_col, max_col = bounds
    width = max_col - min_col
    height = max_row - min_row

    if width <= 0 or height <= 0:
        # Assign all to this degenerate region
        for idx, _ in indexed_values:
            results[idx] = bounds
        return

    if len(indexed_values) == 1:
        idx, _ = indexed_values[0]
        results[idx] = bounds
        return

    # Determine layout direction (lay out along shorter side)
    if width >= height:
        side_length = height
        is_horizontal = True
    else:
        side_length = width
        is_horizontal = False

    # Build row greedily
    row = []
    remaining = list(indexed_values)
    row_areas = []

    while remaining:
        candidate = remaining[0]
        test_areas = row_areas + [candidate[1]]

        # Check if adding this improves aspect ratio
        if not row_areas or _worst_aspect_ratio(test_areas, side_length) <= _worst_aspect_ratio(row_areas, side_length):
            row.append(remaining.pop(0))
            row_areas.append(candidate[1])
        else:
            break

    if not row:
        # Force at least one item in row
        row.append(remaining.pop(0))
        row_areas.append(row[0][1])

    # Layout the row
    total_row_area = sum(row_areas)

    if is_horizontal:
        # Row is vertical strip on left
        row_width = int(round(total_row_area / height)) if height > 0 else 1
        row_width = max(1, min(row_width, width))

        current_row = min_row
        for i, (idx, area) in enumerate(row):
            if i == len(row) - 1:
                # Last one takes remaining height
                rect_height = max_row - current_row
            else:
                rect_height = int(round(area / row_width)) if row_width > 0 else 1
                rect_height = max(1, min(rect_height, max_row - current_row))

            results[idx] = (current_row, current_row + rect_height, min_col, min_col + row_width)
            current_row += rect_height

        # Recurse on remaining area
        new_bounds = (min_row, max_row, min_col + row_width, max_col)
    else:
        # Row is horizontal strip on top
        row_height = int(round(total_row_area / width)) if width > 0 else 1
        row_height = max(1, min(row_height, height))

        current_col = min_col
        for i, (idx, area) in enumerate(row):
            if i == len(row) - 1:
                rect_width = max_col - current_col
            else:
                rect_width = int(round(area / row_height)) if row_height > 0 else 1
                rect_width = max(1, min(rect_width, max_col - current_col))

            results[idx] = (min_row, min_row + row_height, current_col, current_col + rect_width)
            current_col += rect_width

        new_bounds = (min_row + row_height, max_row, min_col, max_col)

    # Recurse
    _squarify_recursive(remaining, new_bounds, results)


def allocate_room_regions(
    room_type_map: Dict[int, str],
    room_ratios: Dict[int, float],
    interior_boundary: np.ndarray,
    margin_factor: float = 0.0,
) -> Dict[int, RoomRegion]:
    """Allocate rectangular regions for each room using treemap algorithm.

    This pre-partitions the available space so each room has a reserved region
    to grow into, preventing greedy growth by any single room.

    Args:
        room_type_map: Dict mapping room_id -> room_type
        room_ratios: Dict mapping room_id -> ratio (relative size)
        interior_boundary: The floorplan grid (EMPTY_ROOM_ID for available cells)
        margin_factor: Extra margin around total bounds (0.0 = none, 0.2 = 20%)

    Returns:
        Dict mapping room_id -> RoomRegion
    """
    # Find available bounds (where EMPTY_ROOM_ID exists)
    empty_mask = (interior_boundary == EMPTY_ROOM_ID)
    if not empty_mask.any():
        return {}

    rows_with_empty = np.any(empty_mask, axis=1)
    cols_with_empty = np.any(empty_mask, axis=0)

    min_row = int(np.argmax(rows_with_empty))
    max_row = int(len(rows_with_empty) - np.argmax(rows_with_empty[::-1]))
    min_col = int(np.argmax(cols_with_empty))
    max_col = int(len(cols_with_empty) - np.argmax(cols_with_empty[::-1]))

    # Apply margin (shrink bounds slightly to allow room for adjustment)
    if margin_factor > 0:
        h_margin = int((max_row - min_row) * margin_factor / 2)
        w_margin = int((max_col - min_col) * margin_factor / 2)
        min_row += h_margin
        max_row -= h_margin
        min_col += w_margin
        max_col -= w_margin

    bounds = (min_row, max_row, min_col, max_col)
    total_width = max_col - min_col
    total_height = max_row - min_row

    # Group rooms by zone for hierarchical allocation
    public_rooms = []  # LivingRoom, Kitchen
    circulation_rooms = []  # Hallway
    private_rooms = []  # Bedrooms
    bathroom_rooms = []  # Bathrooms - need special handling for public access

    for room_id, room_type in room_type_map.items():
        ratio = room_ratios.get(room_id, 1.0)
        if room_type in {"LivingRoom", "Kitchen"}:
            public_rooms.append((room_id, room_type, ratio))
        elif room_type == "Hallway":
            circulation_rooms.append((room_id, room_type, ratio))
        elif room_type == "Bathroom":
            bathroom_rooms.append((room_id, room_type, ratio))
        else:
            private_rooms.append((room_id, room_type, ratio))

    # Calculate zone weights
    # Put bathrooms with private rooms for weight calculation but allocate at least one near hallway
    public_weight = sum(r[2] for r in public_rooms) if public_rooms else 0
    circulation_weight = sum(r[2] for r in circulation_rooms) if circulation_rooms else 0
    private_weight = sum(r[2] for r in private_rooms) + sum(r[2] for r in bathroom_rooms) if (private_rooms or bathroom_rooms) else 0

    total_weight = public_weight + circulation_weight + private_weight
    if total_weight == 0:
        return {}

    regions = {}

    # Use spine-based layout when we have a hallway and private rooms
    # This ensures bedrooms are adjacent to the hallway
    if circulation_rooms and private_rooms:
        # Calculate proportions
        public_ratio = public_weight / total_weight if total_weight > 0 else 0.3
        hallway_ratio = circulation_weight / total_weight if total_weight > 0 else 0.1
        private_ratio = private_weight / total_weight if total_weight > 0 else 0.6

        # Normalize
        total_ratio = public_ratio + hallway_ratio + private_ratio
        public_ratio /= total_ratio
        hallway_ratio /= total_ratio
        private_ratio /= total_ratio

        # Hallway should be a thin strip (2-3 cells wide)
        hallway_width = max(2, min(3, int(total_width * hallway_ratio)))

        # Calculate minimum private zone size based on bedroom count
        # Each bedroom needs at least 2x2 (4 cells), plus space for bathrooms
        num_bedrooms = len(private_rooms)
        num_bathrooms = len(bathroom_rooms)
        # Minimum width needed: 2 cols per bedroom stacked vertically, or
        # 2 * num_bedrooms cols if side by side, plus bathroom strip (2 cols)
        min_private_width = max(4, 2 * num_bedrooms + (2 if num_bathrooms > 0 else 0))
        min_private_height = max(4, 2 * num_bedrooms + (2 if num_bathrooms > 0 else 0))

        # Choose layout direction based on aspect ratio
        if total_width >= total_height:
            # Horizontal layout: Public | Hallway | Private
            # Ensure private zone has enough width for bedrooms
            max_public_width = total_width - hallway_width - min_private_width
            public_width = int((total_width - hallway_width) * (public_ratio / (public_ratio + private_ratio)))
            public_width = max(3, min(max_public_width, public_width))

            private_start = min_col + public_width + hallway_width
            private_width = max_col - private_start

            # Public zone on left
            public_bounds = (min_row, max_row, min_col, min_col + public_width)

            # Hallway in middle (vertical spine)
            hallway_bounds = (min_row, max_row, min_col + public_width, min_col + public_width + hallway_width)

            # Private zone on right
            private_bounds = (min_row, max_row, private_start, max_col)
        else:
            # Vertical layout: Public on top, Hallway, Private on bottom
            # Ensure private zone has enough height for bedrooms
            max_public_height = total_height - hallway_width - min_private_height
            public_height = int((total_height - hallway_width) * (public_ratio / (public_ratio + private_ratio)))
            public_height = max(3, min(max_public_height, public_height))

            private_start = min_row + public_height + hallway_width
            private_height = max_row - private_start

            # Public zone on top
            public_bounds = (min_row, min_row + public_height, min_col, max_col)

            # Hallway in middle (horizontal spine)
            hallway_bounds = (min_row + public_height, min_row + public_height + hallway_width, min_col, max_col)

            # Private zone on bottom
            private_bounds = (private_start, max_row, min_col, max_col)

        # Allocate public rooms
        if public_rooms:
            room_weights = [r[2] for r in public_rooms]
            room_bounds = squarified_layout(room_weights, public_bounds)
            for (room_id, room_type, ratio), room_bound in zip(public_rooms, room_bounds):
                r_min, r_max, c_min, c_max = room_bound
                target_cells = int((r_max - r_min) * (c_max - c_min) * 0.8)
                regions[room_id] = RoomRegion(
                    room_id=room_id, room_type=room_type,
                    min_row=r_min, max_row=r_max, min_col=c_min, max_col=c_max,
                    target_cells=target_cells
                )

        # Allocate hallway
        for room_id, room_type, ratio in circulation_rooms:
            r_min, r_max, c_min, c_max = hallway_bounds
            target_cells = int((r_max - r_min) * (c_max - c_min) * 0.8)
            regions[room_id] = RoomRegion(
                room_id=room_id, room_type=room_type,
                min_row=r_min, max_row=r_max, min_col=c_min, max_col=c_max,
                target_cells=target_cells
            )

        # Allocate private rooms (bedrooms and bathrooms)
        # Ensure at least one bathroom is adjacent to hallway by allocating it
        # in a strip next to the hallway
        all_private = private_rooms + bathroom_rooms

        if all_private:
            # If we have bathrooms, allocate the first one in a thin strip adjacent to hallway
            if bathroom_rooms:
                # Calculate bathroom strip size - limit to 2 cells to leave room for bedrooms
                ba_room = bathroom_rooms[0]
                ba_weight = ba_room[2]
                total_private_weight = sum(r[2] for r in all_private)

                # Calculate minimum bedroom width needed (2 cols per bedroom for stacking)
                min_bedroom_width = max(2, 2 * num_bedrooms)

                if total_width >= total_height:
                    # Horizontal layout - bathrooms on left side of private zone (adjacent to hallway)
                    private_zone_width = private_bounds[3] - private_bounds[2]

                    # Bathroom strip: max 2 cells wide to leave room for bedrooms
                    # Ensure at least min_bedroom_width remains for bedrooms
                    max_bathroom_width = max(2, private_zone_width - min_bedroom_width)
                    bathroom_width = min(2, max_bathroom_width)

                    # Bathroom strip adjacent to hallway
                    ba_bounds = (private_bounds[0], private_bounds[1], private_bounds[2], private_bounds[2] + bathroom_width)

                    # Rest of private zone for bedrooms
                    rest_bounds = (private_bounds[0], private_bounds[1], private_bounds[2] + bathroom_width, private_bounds[3])
                else:
                    # Vertical layout - bathrooms on top of private zone (adjacent to hallway)
                    private_zone_height = private_bounds[1] - private_bounds[0]

                    # Bathroom strip: max 2 cells high to leave room for bedrooms
                    max_bathroom_height = max(2, private_zone_height - min_bedroom_width)
                    bathroom_height = min(2, max_bathroom_height)

                    # Bathroom strip adjacent to hallway
                    ba_bounds = (private_bounds[0], private_bounds[0] + bathroom_height, private_bounds[2], private_bounds[3])

                    # Rest of private zone for bedrooms
                    rest_bounds = (private_bounds[0] + bathroom_height, private_bounds[1], private_bounds[2], private_bounds[3])

                # Allocate bathrooms in bathroom strip
                ba_weights = [r[2] for r in bathroom_rooms]
                ba_room_bounds = squarified_layout(ba_weights, ba_bounds)
                for (room_id, room_type, ratio), room_bound in zip(bathroom_rooms, ba_room_bounds):
                    r_min, r_max, c_min, c_max = room_bound
                    target_cells = int((r_max - r_min) * (c_max - c_min) * 0.8)
                    regions[room_id] = RoomRegion(
                        room_id=room_id, room_type=room_type,
                        min_row=r_min, max_row=r_max, min_col=c_min, max_col=c_max,
                        target_cells=target_cells
                    )

                # Allocate bedrooms in rest of private zone
                if private_rooms and rest_bounds[1] > rest_bounds[0] and rest_bounds[3] > rest_bounds[2]:
                    room_weights = [r[2] for r in private_rooms]
                    room_bounds = squarified_layout(room_weights, rest_bounds)
                    for (room_id, room_type, ratio), room_bound in zip(private_rooms, room_bounds):
                        r_min, r_max, c_min, c_max = room_bound
                        target_cells = int((r_max - r_min) * (c_max - c_min) * 0.8)
                        regions[room_id] = RoomRegion(
                            room_id=room_id, room_type=room_type,
                            min_row=r_min, max_row=r_max, min_col=c_min, max_col=c_max,
                            target_cells=target_cells
                        )
            else:
                # No bathrooms, just allocate bedrooms
                room_weights = [r[2] for r in all_private]
                room_bounds = squarified_layout(room_weights, private_bounds)
                for (room_id, room_type, ratio), room_bound in zip(all_private, room_bounds):
                    r_min, r_max, c_min, c_max = room_bound
                    target_cells = int((r_max - r_min) * (c_max - c_min) * 0.8)
                    regions[room_id] = RoomRegion(
                        room_id=room_id, room_type=room_type,
                        min_row=r_min, max_row=r_max, min_col=c_min, max_col=c_max,
                        target_cells=target_cells
                    )
    else:
        # Simple treemap layout for houses without hallway
        # Combine private rooms and bathrooms
        all_private = private_rooms + bathroom_rooms

        zone_weights = []
        zone_rooms = []

        if public_rooms:
            zone_weights.append(public_weight)
            zone_rooms.append(public_rooms)
        if circulation_rooms:
            zone_weights.append(circulation_weight)
            zone_rooms.append(circulation_rooms)
        if all_private:
            zone_weights.append(sum(r[2] for r in all_private))
            zone_rooms.append(all_private)

        zone_bounds = squarified_layout(zone_weights, bounds)

        for zone_idx, (zone_bound, rooms) in enumerate(zip(zone_bounds, zone_rooms)):
            if not rooms:
                continue

            room_weights = [r[2] for r in rooms]
            room_bounds = squarified_layout(room_weights, zone_bound)

            for (room_id, room_type, ratio), room_bound in zip(rooms, room_bounds):
                r_min, r_max, c_min, c_max = room_bound
                target_cells = int((r_max - r_min) * (c_max - c_min) * 0.8)

                regions[room_id] = RoomRegion(
                    room_id=room_id,
                    room_type=room_type,
                    min_row=r_min,
                    max_row=r_max,
                    min_col=c_min,
                    max_col=c_max,
                    target_cells=target_cells
                )

    # Validate bedroom regions - each should have at least 2x2 dimensions
    for room_id, region in regions.items():
        if region.room_type == "Bedroom":
            if region.width < 2 or region.height < 2:
                # Log warning for debugging - this indicates a layout problem
                print(f"WARNING: Bedroom {room_id} region too narrow: "
                      f"{region.width}x{region.height} (need at least 2x2)")

    return regions


def get_region_adjacencies(regions: Dict[int, RoomRegion]) -> Dict[int, Set[int]]:
    """Find which room regions are adjacent to each other.

    Two regions are adjacent if they share an edge (not just a corner).

    Returns:
        Dict mapping room_id -> set of adjacent room_ids
    """
    adjacencies = {room_id: set() for room_id in regions}

    region_list = list(regions.values())
    for i, r1 in enumerate(region_list):
        for r2 in region_list[i+1:]:
            # Check if regions share an edge
            # Horizontally adjacent: same column range, rows touch
            h_overlap = (r1.min_col < r2.max_col and r2.min_col < r1.max_col)
            v_overlap = (r1.min_row < r2.max_row and r2.min_row < r1.max_row)

            h_touch = (r1.max_row == r2.min_row or r2.max_row == r1.min_row)
            v_touch = (r1.max_col == r2.min_col or r2.max_col == r1.min_col)

            if (h_overlap and h_touch) or (v_overlap and v_touch):
                adjacencies[r1.room_id].add(r2.room_id)
                adjacencies[r2.room_id].add(r1.room_id)

    return adjacencies


def place_room_in_region(
    room,
    region: RoomRegion,
    floorplan: np.ndarray,
    min_dimension: int = 2,
) -> bool:
    """Place a room within its allocated region.

    Searches for a valid placement within the region where all cells are EMPTY_ROOM_ID.
    Tries multiple sizes and positions, starting with larger sizes.

    Args:
        room: The room object (must have min_x, max_x, min_y, max_y attributes)
        region: The allocated region for this room
        floorplan: The floorplan grid
        min_dimension: Minimum dimension for the room

    Returns:
        True if placement succeeded, False otherwise
    """
    # Try different sizes, starting from larger to smaller
    max_width = min(3, region.width)
    max_height = min(3, region.height)

    # Generate size candidates: try larger first, then smaller
    sizes = []
    for h in range(max_height, min_dimension - 1, -1):
        for w in range(max_width, min_dimension - 1, -1):
            if h >= min_dimension and w >= min_dimension:
                sizes.append((h, w))

    if not sizes:
        return False

    for init_height, init_width in sizes:
        if region.width < init_width or region.height < init_height:
            continue

        # Generate all valid positions for this size
        positions = []
        for r in range(region.min_row, region.max_row - init_height + 1):
            for c in range(region.min_col, region.max_col - init_width + 1):
                positions.append((r, c))

        if not positions:
            continue

        # Shuffle positions
        random.shuffle(positions)

        for start_row, start_col in positions:
            end_row = start_row + init_height
            end_col = start_col + init_width

            # Check if placement area is entirely empty
            placement_area = floorplan[start_row:end_row, start_col:end_col]
            if (placement_area == EMPTY_ROOM_ID).all():
                # Place the room
                room.min_y = int(start_row)
                room.max_y = int(end_row)
                room.min_x = int(start_col)
                room.max_x = int(end_col)
                floorplan[start_row:end_row, start_col:end_col] = room.room_id
                return True

    return False


def grow_room_in_region(
    room,
    region: RoomRegion,
    floorplan: np.ndarray,
) -> bool:
    """Grow a room by one cell in any direction, constrained to its region.

    Returns True if growth occurred, False if no growth possible.
    """
    room_id = room.room_id

    # Try each direction, prefer directions that keep aspect ratio reasonable
    directions = []

    current_width = room.max_x - room.min_x
    current_height = room.max_y - room.min_y

    # Calculate preference based on making room more square
    if current_width <= current_height:
        # Prefer horizontal growth
        directions = ["right", "left", "down", "up"]
    else:
        # Prefer vertical growth
        directions = ["down", "up", "right", "left"]

    random.shuffle(directions[:2])  # Shuffle within preference group
    random.shuffle(directions[2:])

    for direction in directions:
        if direction == "right":
            new_col = room.max_x
            if new_col < region.max_col:
                col_slice = floorplan[room.min_y:room.max_y, new_col]
                if (col_slice == EMPTY_ROOM_ID).all():
                    floorplan[room.min_y:room.max_y, new_col] = room_id
                    room.max_x = new_col + 1
                    return True

        elif direction == "left":
            new_col = room.min_x - 1
            if new_col >= region.min_col:
                col_slice = floorplan[room.min_y:room.max_y, new_col]
                if (col_slice == EMPTY_ROOM_ID).all():
                    floorplan[room.min_y:room.max_y, new_col] = room_id
                    room.min_x = new_col
                    return True

        elif direction == "down":
            new_row = room.max_y
            if new_row < region.max_row:
                row_slice = floorplan[new_row, room.min_x:room.max_x]
                if (row_slice == EMPTY_ROOM_ID).all():
                    floorplan[new_row, room.min_x:room.max_x] = room_id
                    room.max_y = new_row + 1
                    return True

        elif direction == "up":
            new_row = room.min_y - 1
            if new_row >= region.min_row:
                row_slice = floorplan[new_row, room.min_x:room.max_x]
                if (row_slice == EMPTY_ROOM_ID).all():
                    floorplan[new_row, room.min_x:room.max_x] = room_id
                    room.min_y = new_row
                    return True

    return False

