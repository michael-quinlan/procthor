#!/usr/bin/env python3
"""Generate a dataset of procedurally generated houses.

Example usage:
    python scripts/generate_dataset.py --num-houses 100 --output dataset.json.gz
    python scripts/generate_dataset.py --num-houses 1000 --split train --seed 42
"""

import argparse
import gzip
import json
import logging
import multiprocessing
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from tqdm import tqdm

from procthor.generation import HouseGenerator
from procthor.generation.hallway_room_specs import ALL_ROOM_SPEC_SAMPLER
from procthor.generation.validation import validate_room_proportions

# Room type colors (from generate_samples.py)
ROOM_TYPE_TO_COLOR = {
    "Kitchen": "#ffd6e7",
    "LivingRoom": "#d9f7be",
    "Bedroom": "#fff1b8",
    "Bathroom": "#bae7ff",
    "Hallway": "#d9d9d9",
}

# Conversion constant
SQM_TO_SQFT = 10.7639


def validate_final_proportions(house) -> tuple:
    """Validate that final room polygon proportions are sensible.

    This is a safety check after full generation. Early validation happens
    in HouseGenerator.sample() after the STRUCTURE stage.

    Returns:
        Tuple of (passed, reason) where reason explains failure if passed=False.
    """
    # Small tolerance for floating point comparisons (0.1 m² = ~1 sqft)
    EPSILON = 0.1

    room_areas = {}
    for room_id, room in house.rooms.items():
        room_type = room.room_type
        area = room.room_polygon.polygon.area
        if room_type not in room_areas:
            room_areas[room_type] = []
        room_areas[room_type].append(area)

    living_areas = room_areas.get('LivingRoom', [])
    bedroom_areas = room_areas.get('Bedroom', [])
    hallway_areas = room_areas.get('Hallway', [])

    if not living_areas:
        return True, ""  # No living room to validate

    max_living = max(living_areas)
    max_bedroom = max(bedroom_areas) if bedroom_areas else 0
    min_bedroom = min(bedroom_areas) if bedroom_areas else float('inf')
    max_hallway = max(hallway_areas) if hallway_areas else 0

    # Rule 1: LivingRoom >= any Bedroom (with epsilon tolerance)
    if max_bedroom > 0 and max_living < max_bedroom - EPSILON:
        return False, f"Living ({max_living:.1f}m²) < Bedroom ({max_bedroom:.1f}m²)"

    # Rule 2: Hallway <= any Bedroom (with epsilon tolerance)
    if max_hallway > 0 and min_bedroom < float('inf') and max_hallway > min_bedroom + EPSILON:
        return False, f"Hallway ({max_hallway:.1f}m²) > Bedroom ({min_bedroom:.1f}m²)"

    # Rule 3: LivingRoom >= Hallway (with epsilon tolerance)
    if max_hallway > 0 and max_living < max_hallway - EPSILON:
        return False, f"Living ({max_living:.1f}m²) < Hallway ({max_hallway:.1f}m²)"

    return True, ""



def plot_house(house, ax, title):
    """Plot a single house floorplan with doors and square footage."""
    total_sqft = 0.0
    room_areas = []  # Store for legend

    # Draw rooms
    for room_id, room in house.rooms.items():
        poly = room.room_polygon.polygon.exterior.coords
        xs = [p[0] for p in poly]
        zs = [p[1] for p in poly]
        color = ROOM_TYPE_TO_COLOR.get(room.room_type, "#cccccc")
        ax.fill(xs, zs, color)
        ax.plot(xs, zs, "#000000", linewidth=2)

        # Calculate room area (polygon.area gives sqm, convert to sqft)
        area_sqm = room.room_polygon.polygon.area
        area_sqft = area_sqm * SQM_TO_SQFT
        total_sqft += area_sqft
        room_areas.append((room.room_type, room_id, area_sqft))

        centroid = room.room_polygon.polygon.centroid
        # Show room type, ID, and sqft
        ax.text(centroid.x, centroid.y,
                f"{room.room_type}\n({room_id})\n{area_sqft:.0f} sqft",
                ha='center', va='center', fontsize=6, fontweight='bold')

    # Draw doors on top of walls
    if "doors" in house.data and house.data["doors"]:
        for door in house.data["doors"]:
            wall_id = door.get("wall0", "")
            parts = wall_id.split("|")
            if len(parts) >= 6:
                try:
                    x0, z0, x1, z1 = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])

                    if "holePolygon" in door and len(door["holePolygon"]) >= 2:
                        hole_start = float(door["holePolygon"][0]["x"])
                        hole_end = float(door["holePolygon"][1]["x"])

                        wall_len = ((x1 - x0)**2 + (z1 - z0)**2)**0.5
                        if wall_len > 0:
                            dx = (x1 - x0) / wall_len
                            dz = (z1 - z0) / wall_len

                            door_x0 = x0 + hole_start * dx
                            door_z0 = z0 + hole_start * dz
                            door_x1 = x0 + hole_end * dx
                            door_z1 = z0 + hole_end * dz

                            # Draw door as white gap with brown border (door frame)
                            ax.plot([door_x0, door_x1], [door_z0, door_z1],
                                   color='white', linewidth=8, solid_capstyle='butt', zorder=10)
                            ax.plot([door_x0, door_x1], [door_z0, door_z1],
                                   color='#8B4513', linewidth=3, solid_capstyle='butt', zorder=11)
                except (ValueError, IndexError):
                    pass

    ax.set_aspect('equal')
    # Include total sqft in title
    ax.set_title(f"{title}\nTotal: {total_sqft:.0f} sqft", fontsize=9)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Global dict for per-PID HouseGenerator instances (for multiprocessing)
_house_generators = {}


def _get_or_create_house_generator(
    split: str,
    seed: Optional[int],
    house_index: int,
    bedrooms: Optional[int],
    bathrooms: Optional[int],
    grid_size: float = 0.25,
) -> HouseGenerator:
    """Get or create a HouseGenerator for the current process.

    Each worker process gets its own HouseGenerator with its own Controller.
    """
    pid = os.getpid()
    if pid not in _house_generators:
        # Create a filtered sampler if bedroom/bathroom constraints are set
        if bedrooms is not None or bathrooms is not None:
            from procthor.generation.room_specs import RoomSpecSampler
            from procthor.generation.hallway_room_specs import (
                HALLWAY_HOUSE_2BR_1BA, HALLWAY_HOUSE_3BR_2BA, HALLWAY_HOUSE_4BR_2BA,
                NO_HALLWAY_1BR_1BA, NO_HALLWAY_2BR_1BA,
            )
            spec_map_by_bedrooms = {
                1: [(1, 1, NO_HALLWAY_1BR_1BA)],
                2: [(2, 1, HALLWAY_HOUSE_2BR_1BA), (2, 1, NO_HALLWAY_2BR_1BA)],
                3: [(3, 2, HALLWAY_HOUSE_3BR_2BA)],
                4: [(4, 2, HALLWAY_HOUSE_4BR_2BA)],
            }
            matching_specs = []
            if bedrooms in spec_map_by_bedrooms:
                for br, ba, spec in spec_map_by_bedrooms[bedrooms]:
                    if bathrooms is None or ba == bathrooms:
                        matching_specs.append(spec)
            if matching_specs:
                room_spec_sampler = RoomSpecSampler(matching_specs)
            else:
                room_spec_sampler = ALL_ROOM_SPEC_SAMPLER
        else:
            room_spec_sampler = ALL_ROOM_SPEC_SAMPLER

        # Compute seed for this worker
        worker_seed = seed + house_index if seed is not None else None

        _house_generators[pid] = HouseGenerator(
            split=split,
            seed=worker_seed,
            room_spec_sampler=room_spec_sampler,
            grid_size=grid_size,
        )

    return _house_generators[pid]


def _worker_args_wrapper(args: Tuple) -> Tuple[Optional[Dict], bool]:
    """Wrapper to unpack tuple arguments for imap_unordered."""
    return _worker_generate_house(*args)


def _worker_generate_house(
    house_index: int,
    split: str,
    seed: Optional[int],
    max_retries: int,
    save_images: bool,
    image_dir: str,
    bedrooms: Optional[int],
    bathrooms: Optional[int],
    grid_size: float = 0.25,
) -> Tuple[Optional[Dict], bool]:
    """Worker function for parallel house generation.

    Returns:
        Tuple of (house_data_dict, success_bool). house_data_dict is None if generation failed.
    """
    try:
        house_generator = _get_or_create_house_generator(
            split=split,
            seed=seed,
            house_index=house_index,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            grid_size=grid_size,
        )

        # Sample spec ONCE per house slot
        target_spec = house_generator.room_spec_sampler.sample()
        house_generator.room_spec = target_spec

        retries = 0
        while retries < max_retries:
            try:
                house_generator.partial_house = None

                # Timing instrumentation
                t_start = time.time()
                house, _ = house_generator.sample()
                t_after_sample = time.time()
                house.validate(house_generator.controller)
                t_after_validate = time.time()

                # Calculate timing
                sample_time = t_after_sample - t_start
                validate_time = t_after_validate - t_after_sample
                total_time = t_after_validate - t_start

                print(f"House {house_index}: sample={sample_time:.1f}s, validate={validate_time:.1f}s, total={total_time:.1f}s")

                # Skip houses with warnings and retry
                if house.data.get("metadata", {}).get("warnings"):
                    print(f"DEBUG: House {house_index} has warnings: {house.data.get('metadata', {}).get('warnings')}")
                    retries += 1
                    if retries >= max_retries:
                        logger.warning(f"House {house_index} failed due to warnings after {max_retries} retries")
                        return None, False
                    continue

                # Validate final room proportions
                passed, reason = validate_final_proportions(house)
                print(f"DEBUG: House {house_index} proportion check: passed={passed}, reason={reason}")
                sys.stdout.flush()
                if not passed:
                    logger.debug(f"House {house_index} failed proportion check: {reason}")
                    retries += 1
                    continue

                # Save image if requested
                if save_images:
                    os.makedirs(image_dir, exist_ok=True)
                    image_path = os.path.join(image_dir, f"house_{house_index:04d}.png")
                    save_house_image(house, image_path, house_index)

                return house.data, True
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    logger.warning(f"Failed to generate house {house_index} after {max_retries} retries: {e}")
                    return None, False

        return None, False
    except Exception as e:
        logger.error(f"Worker error for house {house_index}: {e}")
        return None, False


def save_house_image(house, image_path: str, house_num: int) -> None:
    """Save a single house floorplan as PNG.

    Args:
        house: House object with rooms and data.
        image_path: Path to save the PNG file.
        house_num: House number for the title.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    title = f"House {house_num}\n{len(house.rooms)} rooms"
    plot_house(house, ax, title)
    plt.tight_layout()
    plt.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_houses(
    num_houses: int,
    split: str,
    seed: Optional[int] = None,
    max_retries: int = 3,
    save_images: bool = False,
    image_dir: str = "./images/",
    bedrooms: Optional[int] = None,
    bathrooms: Optional[int] = None,
    grid_size: float = 0.25,
) -> List[Dict]:
    """Generate a list of house data dictionaries.

    Args:
        num_houses: Number of houses to generate.
        split: Data split to use ('train', 'val', or 'test').
        seed: Optional random seed for reproducibility.
        max_retries: Maximum retries per house on failure.
        save_images: If True, save a PNG floorplan for each house.
        image_dir: Directory to save images (if save_images is True).
        bedrooms: If set, only generate houses with exactly this many bedrooms.
        bathrooms: If set, only generate houses with exactly this many bathrooms.
        grid_size: Navigation grid size for agent pose generation (default: 0.25).

    Returns:
        List of house data dictionaries.
    """
    # Create a filtered sampler if bedroom/bathroom constraints are set
    if bedrooms is not None or bathrooms is not None:
        from procthor.generation.room_specs import RoomSpecSampler
        from procthor.generation.hallway_room_specs import (
            HALLWAY_HOUSE_2BR_1BA, HALLWAY_HOUSE_3BR_2BA, HALLWAY_HOUSE_4BR_2BA,
            NO_HALLWAY_1BR_1BA, NO_HALLWAY_2BR_1BA,
        )
        # Map bedrooms to specific room specs (bathrooms optional)
        spec_map_by_bedrooms = {
            1: [(1, 1, NO_HALLWAY_1BR_1BA)],
            2: [(2, 1, HALLWAY_HOUSE_2BR_1BA), (2, 1, NO_HALLWAY_2BR_1BA)],
            3: [(3, 2, HALLWAY_HOUSE_3BR_2BA)],
            4: [(4, 2, HALLWAY_HOUSE_4BR_2BA)],
        }

        # Find matching specs
        matching_specs = []
        if bedrooms in spec_map_by_bedrooms:
            for br, ba, spec in spec_map_by_bedrooms[bedrooms]:
                if bathrooms is None or ba == bathrooms:
                    matching_specs.append(spec)

        if matching_specs:
            room_spec_sampler = RoomSpecSampler(matching_specs)
            ba_str = f"{bathrooms}BA" if bathrooms else "any BA"
            logger.info(f"Using {bedrooms}BR/{ba_str} specs ({len(matching_specs)} options)")
        else:
            logger.warning(f"No specific spec for {bedrooms}BR/{bathrooms}BA, using ALL_ROOM_SPEC_SAMPLER")
            room_spec_sampler = ALL_ROOM_SPEC_SAMPLER
    else:
        room_spec_sampler = ALL_ROOM_SPEC_SAMPLER

    house_generator = HouseGenerator(
        split=split,
        seed=seed,
        room_spec_sampler=room_spec_sampler,
        grid_size=grid_size,
    )

    # Create image directory if needed
    if save_images:
        os.makedirs(image_dir, exist_ok=True)

    houses = []
    failed_count = 0
    house_count = 0  # Track successful houses for image numbering

    try:
        for i in tqdm(range(num_houses), desc="Generating houses", unit="house"):
            # Sample spec ONCE per house slot to avoid bias toward smaller houses
            # (larger houses fail more often, and without this fix each retry
            # would sample a new spec, biasing toward specs that succeed quickly)
            target_spec = room_spec_sampler.sample()
            house_generator.room_spec = target_spec

            retries = 0
            while retries < max_retries:
                try:
                    # Reset partial_house before each attempt to ensure fresh generation
                    house_generator.partial_house = None

                    # Timing instrumentation
                    t_start = time.time()
                    house, _ = house_generator.sample()
                    t_after_sample = time.time()
                    house.validate(house_generator.controller)
                    t_after_validate = time.time()

                    # Calculate timing
                    sample_time = t_after_sample - t_start
                    validate_time = t_after_validate - t_after_sample
                    total_time = t_after_validate - t_start

                    print(f"House {i}: sample={sample_time:.1f}s, validate={validate_time:.1f}s, total={total_time:.1f}s")

                    # Skip houses with warnings and retry
                    if house.data.get("metadata", {}).get("warnings"):
                        print(f"DEBUG: House {i} has warnings: {house.data.get('metadata', {}).get('warnings')}")
                        retries += 1
                        if retries >= max_retries:
                            logger.warning(f"House {i} failed due to warnings after {max_retries} retries")
                            failed_count += 1
                        continue

                    # Validate final room proportions (after all processing)
                    passed, reason = validate_final_proportions(house)
                    print(f"DEBUG: House {i} proportion check: passed={passed}, reason={reason}")
                    import sys; sys.stdout.flush()
                    if not passed:
                        logger.debug(f"House {i} failed proportion check: {reason}")
                        retries += 1
                        continue

                    houses.append(house.data)
                    house_count += 1

                    # Save image if requested
                    if save_images:
                        image_path = os.path.join(image_dir, f"house_{house_count:04d}.png")
                        save_house_image(house, image_path, house_count)

                    break
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        logger.warning(f"Failed to generate house {i} after {max_retries} retries: {e}")
                        failed_count += 1

        if failed_count > 0:
            logger.warning(f"Failed to generate {failed_count} houses out of {num_houses} requested.")
    finally:
        # Ensure the AI2THOR controller is properly cleaned up to avoid zombie windows
        if house_generator.controller is not None:
            house_generator.controller.stop()

    return houses


def generate_houses_parallel(
    num_houses: int,
    num_workers: int,
    split: str,
    seed: Optional[int] = None,
    max_retries: int = 3,
    save_images: bool = False,
    image_dir: str = "./images/",
    bedrooms: Optional[int] = None,
    bathrooms: Optional[int] = None,
    grid_size: float = 0.25,
) -> List[Dict]:
    """Generate houses in parallel using multiprocessing.

    Args:
        num_houses: Number of houses to generate.
        num_workers: Number of worker processes.
        split: Data split to use ('train', 'val', or 'test').
        seed: Optional random seed for reproducibility.
        max_retries: Maximum retries per house on failure.
        save_images: If True, save a PNG floorplan for each house.
        image_dir: Directory to save images (if save_images is True).
        bedrooms: If set, only generate houses with exactly this many bedrooms.
        bathrooms: If set, only generate houses with exactly this many bathrooms.
        grid_size: Navigation grid size for agent pose generation (default: 0.25).

    Returns:
        List of house data dictionaries.
    """
    # Create a filtered sampler if bedroom/bathroom constraints are set
    if bedrooms is not None or bathrooms is not None:
        from procthor.generation.room_specs import RoomSpecSampler
        from procthor.generation.hallway_room_specs import (
            HALLWAY_HOUSE_2BR_1BA, HALLWAY_HOUSE_3BR_2BA, HALLWAY_HOUSE_4BR_2BA,
            NO_HALLWAY_1BR_1BA, NO_HALLWAY_2BR_1BA,
        )
        spec_map_by_bedrooms = {
            1: [(1, 1, NO_HALLWAY_1BR_1BA)],
            2: [(2, 1, HALLWAY_HOUSE_2BR_1BA), (2, 1, NO_HALLWAY_2BR_1BA)],
            3: [(3, 2, HALLWAY_HOUSE_3BR_2BA)],
            4: [(4, 2, HALLWAY_HOUSE_4BR_2BA)],
        }
        matching_specs = []
        if bedrooms in spec_map_by_bedrooms:
            for br, ba, spec in spec_map_by_bedrooms[bedrooms]:
                if bathrooms is None or ba == bathrooms:
                    matching_specs.append(spec)
        if matching_specs:
            room_spec_sampler = RoomSpecSampler(matching_specs)
            ba_str = f"{bathrooms}BA" if bathrooms else "any BA"
            logger.info(f"Using {bedrooms}BR/{ba_str} specs ({len(matching_specs)} options)")
        else:
            logger.warning(f"No specific spec for {bedrooms}BR/{bathrooms}BA, using ALL_ROOM_SPEC_SAMPLER")
            room_spec_sampler = ALL_ROOM_SPEC_SAMPLER
    else:
        room_spec_sampler = ALL_ROOM_SPEC_SAMPLER

    houses = []
    failed_count = 0

    try:
        with multiprocessing.Pool(processes=num_workers) as pool:
            # Use imap_unordered for real-time progress updates
            results = pool.imap_unordered(
                _worker_args_wrapper,
                [
                    (
                        i,
                        split,
                        seed,
                        max_retries,
                        save_images,
                        image_dir,
                        bedrooms,
                        bathrooms,
                        grid_size,
                    )
                    for i in range(num_houses)
                ],
            )

            # Collect results with progress bar
            for house_data, success in tqdm(results, total=num_houses, desc="Generating houses", unit="house"):
                if success and house_data is not None:
                    houses.append(house_data)
                else:
                    failed_count += 1
    finally:
        # Cleanup all worker generators
        for pid, gen in _house_generators.items():
            try:
                if gen.controller is not None:
                    gen.controller.stop()
            except Exception as e:
                logger.warning(f"Error stopping controller for PID {pid}: {e}")
        _house_generators.clear()

    if failed_count > 0:
        logger.warning(f"Failed to generate {failed_count} houses out of {num_houses} requested.")

    return houses


def save_dataset(houses: List[Dict], output_path: str) -> None:
    """Save houses to a gzipped JSON file.

    Args:
        houses: List of house data dictionaries.
        output_path: Output file path (should end with .json.gz).
    """
    if not output_path.endswith(".json.gz"):
        output_path = f"{output_path}.json.gz"

    logger.info(f"Saving {len(houses)} houses to {output_path}...")
    with gzip.open(output_path, "wt", encoding="UTF-8") as f:
        json.dump(houses, f)
    logger.info(f"Dataset saved successfully.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a dataset of procedurally generated houses.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n", "--num-houses",
        type=int,
        default=1000,
        help="Number of houses to generate.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="dataset.json.gz",
        help="Output file path (gzipped JSON).",
    )
    parser.add_argument(
        "-s", "--split",
        type=str,
        choices=["train", "val", "test"],
        default="train",
        help="Data split to use for generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=10,
        help="Maximum retries per house on generation failure.",
    )
    parser.add_argument(
        "-i", "--save-images",
        action="store_true",
        help="Save a PNG floorplan for each generated house.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="./images/",
        help="Directory to save images (only used with --save-images).",
    )
    parser.add_argument(
        "--bedrooms",
        type=int,
        default=None,
        help="Constrain to houses with exactly this many bedrooms.",
    )
    parser.add_argument(
        "--bathrooms",
        type=int,
        default=None,
        help="Constrain to houses with exactly this many bathrooms.",
    )
    parser.add_argument(
        "-w", "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes for parallel generation (default: 1, single-threaded).",
    )
    parser.add_argument(
        "--grid-size",
        type=float,
        default=0.25,
        help="Navigation grid size for agent pose generation (default: 0.25).",
    )

    args = parser.parse_args()

    logger.info(f"Generating {args.num_houses} houses (split={args.split}, seed={args.seed}, num_workers={args.num_workers})")
    if args.save_images:
        logger.info(f"Saving images to {args.image_dir}")

    # Choose single-threaded or parallel generation
    if args.num_workers == 1:
        houses = generate_houses(
            num_houses=args.num_houses,
            split=args.split,
            seed=args.seed,
            max_retries=args.max_retries,
            save_images=args.save_images,
            image_dir=args.image_dir,
            bedrooms=args.bedrooms,
            bathrooms=args.bathrooms,
            grid_size=args.grid_size,
        )
    else:
        houses = generate_houses_parallel(
            num_houses=args.num_houses,
            num_workers=args.num_workers,
            split=args.split,
            seed=args.seed,
            max_retries=args.max_retries,
            save_images=args.save_images,
            image_dir=args.image_dir,
            bedrooms=args.bedrooms,
            bathrooms=args.bathrooms,
            grid_size=args.grid_size,
        )

    if not houses:
        logger.error("No houses were generated successfully.")
        return 1

    save_dataset(houses, args.output)
    logger.info(f"Successfully generated {len(houses)} houses.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

