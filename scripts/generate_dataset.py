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
import sys
from typing import Dict, List, Optional

from tqdm import tqdm

from procthor.generation import HouseGenerator, PROCTHOR10K_ROOM_SPEC_SAMPLER

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_houses(
    num_houses: int,
    split: str,
    seed: Optional[int] = None,
    max_retries: int = 3,
) -> List[Dict]:
    """Generate a list of house data dictionaries.

    Args:
        num_houses: Number of houses to generate.
        split: Data split to use ('train', 'val', or 'test').
        seed: Optional random seed for reproducibility.
        max_retries: Maximum retries per house on failure.

    Returns:
        List of house data dictionaries.
    """
    house_generator = HouseGenerator(
        split=split,
        seed=seed,
        room_spec_sampler=PROCTHOR10K_ROOM_SPEC_SAMPLER,
    )

    houses = []
    failed_count = 0

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

                    houses.append(house.data)
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
        default=3,
        help="Maximum retries per house on generation failure.",
    )

    args = parser.parse_args()

    logger.info(f"Generating {args.num_houses} houses (split={args.split}, seed={args.seed})")

    houses = generate_houses(
        num_houses=args.num_houses,
        split=args.split,
        seed=args.seed,
        max_retries=args.max_retries,
    )

    if not houses:
        logger.error("No houses were generated successfully.")
        return 1

    save_dataset(houses, args.output)
    logger.info(f"Successfully generated {len(houses)} houses.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

