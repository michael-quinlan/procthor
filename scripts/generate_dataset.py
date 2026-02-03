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
import os
import sys
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from tqdm import tqdm

from procthor.generation import HouseGenerator
from procthor.generation.hallway_room_specs import HALLWAY_ROOM_SPEC_SAMPLER

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
) -> List[Dict]:
    """Generate a list of house data dictionaries.

    Args:
        num_houses: Number of houses to generate.
        split: Data split to use ('train', 'val', or 'test').
        seed: Optional random seed for reproducibility.
        max_retries: Maximum retries per house on failure.
        save_images: If True, save a PNG floorplan for each house.
        image_dir: Directory to save images (if save_images is True).

    Returns:
        List of house data dictionaries.
    """
    house_generator = HouseGenerator(
        split=split,
        seed=seed,
        room_spec_sampler=HALLWAY_ROOM_SPEC_SAMPLER,
    )

    # Create image directory if needed
    if save_images:
        os.makedirs(image_dir, exist_ok=True)

    houses = []
    failed_count = 0
    house_count = 0  # Track successful houses for image numbering

    try:
        for i in tqdm(range(num_houses), desc="Generating houses", unit="house"):
            retries = 0
            while retries < max_retries:
                try:
                    house, _ = house_generator.sample()
                    house.validate(house_generator.controller)

                    # Skip houses with warnings and retry
                    if house.data.get("metadata", {}).get("warnings"):
                        retries += 1
                        continue

                    # Validate final room proportions (after all processing)
                    passed, reason = validate_final_proportions(house)
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

    args = parser.parse_args()

    logger.info(f"Generating {args.num_houses} houses (split={args.split}, seed={args.seed})")
    if args.save_images:
        logger.info(f"Saving images to {args.image_dir}")

    houses = generate_houses(
        num_houses=args.num_houses,
        split=args.split,
        seed=args.seed,
        max_retries=args.max_retries,
        save_images=args.save_images,
        image_dir=args.image_dir,
    )

    if not houses:
        logger.error("No houses were generated successfully.")
        return 1

    save_dataset(houses, args.output)
    logger.info(f"Successfully generated {len(houses)} houses.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

