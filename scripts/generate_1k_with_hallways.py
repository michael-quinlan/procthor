"""Generate a 1000-house dataset with hallways in 30% of houses.

Uses the same room distribution as the 10K dataset, but adds hallways to 30% of houses.
"""
import json
import os
import random
from datetime import datetime
from pathlib import Path

from ai2thor.controller import Controller
from procthor.constants import PROCTHOR_INITIALIZATION
from procthor.generation import HouseGenerator, PROCTHOR10K_ROOM_SPEC_SAMPLER
from procthor.generation.hallway_room_specs import create_hallway_room_spec
from procthor.generation.room_specs import RoomSpec, RoomSpecSampler

# Configuration
NUM_HOUSES = 1000
HALLWAY_PERCENTAGE = 0.30
OUTPUT_DIR = Path("dataset_1k_with_hallways")
SPLIT = "train"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Hallway specs matching common configurations from 10K dataset
HALLWAY_SPECS = [
    # 2 bedroom configs
    create_hallway_room_spec("hallway-2br-1ba", num_bedrooms=2, num_bathrooms=1),
    create_hallway_room_spec("hallway-2br-2ba", num_bedrooms=2, num_bathrooms=2),
    # 3 bedroom configs
    create_hallway_room_spec("hallway-3br-1ba", num_bedrooms=3, num_bathrooms=1),
    create_hallway_room_spec("hallway-3br-2ba", num_bedrooms=3, num_bathrooms=2),
    create_hallway_room_spec("hallway-3br-3ba", num_bedrooms=3, num_bathrooms=3),
    # 4 bedroom configs
    create_hallway_room_spec("hallway-4br-2ba", num_bedrooms=4, num_bathrooms=2),
    create_hallway_room_spec("hallway-4br-3ba", num_bedrooms=4, num_bathrooms=3),
]

HALLWAY_SAMPLER = RoomSpecSampler(HALLWAY_SPECS)


def generate_houses():
    """Generate NUM_HOUSES houses with HALLWAY_PERCENTAGE having hallways."""
    print(f"Starting generation at {datetime.now()}")
    print(f"Target: {NUM_HOUSES} houses, {HALLWAY_PERCENTAGE*100:.0f}% with hallways")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Calculate how many of each type
    num_hallway = int(NUM_HOUSES * HALLWAY_PERCENTAGE)
    num_regular = NUM_HOUSES - num_hallway

    print(f"Generating {num_regular} regular houses + {num_hallway} hallway houses")

    # Create list of (use_hallway, index) to shuffle
    house_configs = [(False, i) for i in range(num_regular)]
    house_configs += [(True, i) for i in range(num_hallway)]
    random.shuffle(house_configs)

    # Stats tracking
    stats = {
        "total": 0,
        "regular": 0,
        "hallway": 0,
        "failed": 0,
        "room_specs": {},
    }

    # Create a single shared controller to avoid spawning many Unity windows
    print("Initializing AI2-THOR controller...")
    controller = Controller(quality="Low", **PROCTHOR_INITIALIZATION)
    print("Controller ready.")
    print()

    for idx, (use_hallway, _) in enumerate(house_configs):
        seed = idx + 1000  # Offset seed to get different results

        if use_hallway:
            room_spec = HALLWAY_SAMPLER.sample()
            spec_type = "hallway"
        else:
            room_spec = PROCTHOR10K_ROOM_SPEC_SAMPLER.sample()
            spec_type = "regular"

        # Reuse the same controller for all houses
        generator = HouseGenerator(
            split=SPLIT,
            seed=seed,
            room_spec=room_spec,
            controller=controller,
        )

        try:
            house, _ = generator.sample()

            # Save house
            filename = OUTPUT_DIR / f"house_{idx:04d}_{spec_type}.json"
            with open(filename, "w") as f:
                f.write(house.to_json())

            stats["total"] += 1
            stats[spec_type] += 1

            spec_id = room_spec.room_spec_id
            stats["room_specs"][spec_id] = stats["room_specs"].get(spec_id, 0) + 1

            print(f"[{idx + 1}/{NUM_HOUSES}] {spec_type} ({spec_id}) ✓")

        except Exception as e:
            stats["failed"] += 1
            print(f"Failed house {idx} ({spec_type}): {e}")

    # Clean up controller
    controller.stop()

    # Save stats
    stats_file = OUTPUT_DIR / "stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print()
    print(f"Done! Generated {stats['total']} houses")
    print(f"  Regular: {stats['regular']}")
    print(f"  Hallway: {stats['hallway']}")
    print(f"  Failed: {stats['failed']}")
    print(f"Stats saved to: {stats_file}")


if __name__ == "__main__":
    generate_houses()

