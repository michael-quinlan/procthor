#!/usr/bin/env python3
"""Generate sample houses using hallway-based room specs.

This script generates 1-4 sample houses using HALLWAY_ROOM_SPEC_SAMPLER
and saves a floorplan visualization to floorplan.png.

Usage:
    python scripts/generate_samples.py
"""

import random
import sys

import matplotlib.pyplot as plt
import numpy as np

from procthor.generation import HouseGenerator
from procthor.generation.hallway_room_specs import HALLWAY_ROOM_SPEC_SAMPLER


def visualize_floorplan(floorplan: np.ndarray, room_type_map: dict, ax: plt.Axes) -> None:
    """Visualize a floorplan with room type labels.

    Args:
        floorplan: 2D numpy array with room IDs
        room_type_map: Mapping from room ID to room type name
        ax: Matplotlib axes to draw on
    """
    # Color palette for different rooms
    colors = plt.cm.Set3(np.linspace(0, 1, 12))

    room_ids = np.unique(floorplan.flatten())
    for i, room_id in enumerate(room_ids):
        if room_id == 0 or room_id == 1:  # Skip outdoor/empty room IDs
            continue
        coords = np.argwhere(floorplan == room_id)
        if len(coords) == 0:
            continue
        room_type = room_type_map.get(room_id, f"Room {room_id}")
        ax.scatter(
            coords[:, 1], coords[:, 0],
            label=f"{room_type} ({room_id})",
            c=[colors[i % len(colors)]],
            s=50,
            alpha=0.8
        )

    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    ax.grid(True, alpha=0.3)


def main():
    """Generate sample houses and save floorplan visualization."""
    print("Initializing house generator with HALLWAY_ROOM_SPEC_SAMPLER...")

    try:
        # Create house generator with hallway-based room specs
        house_generator = HouseGenerator(
            split="train",
            seed=42,
            room_spec_sampler=HALLWAY_ROOM_SPEC_SAMPLER,
        )
    except Exception as e:
        print(f"Error initializing house generator: {e}")
        sys.exit(1)

    # Generate 1-4 houses
    num_houses = random.randint(1, 4)
    print(f"Generating {num_houses} sample house(s)...")

    # Set up figure with subplots
    fig, axes = plt.subplots(1, num_houses, figsize=(6 * num_houses, 6))
    if num_houses == 1:
        axes = [axes]

    houses = []
    for i in range(num_houses):
        print(f"\nGenerating house {i + 1}/{num_houses}...")

        try:
            # Sample a new room spec for each house
            house_generator.room_spec = None
            house, partial_houses = house_generator.sample()
            houses.append(house)

            # Get the floorplan from partial house structure
            partial_house = list(partial_houses.values())[0]
            floorplan = partial_house.house_structure.floorplan
            room_type_map = partial_house.room_spec.room_type_map

            print(f"  Room spec: {house.room_spec.room_spec_id}")
            print(f"  Rooms: {list(room_type_map.values())}")
            print(f"  Floorplan shape: {floorplan.shape}")

            # Visualize floorplan
            visualize_floorplan(floorplan, room_type_map, axes[i])
            axes[i].set_title(f"House {i + 1}: {house.room_spec.room_spec_id}")

        except Exception as e:
            print(f"  Error generating house {i + 1}: {e}")
            axes[i].text(0.5, 0.5, f"Error: {e}", ha="center", va="center")
            axes[i].set_title(f"House {i + 1}: Generation Failed")

    # Save the floorplan visualization
    plt.tight_layout()
    output_file = "floorplan.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nFloorplan visualization saved to: {output_file}")

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"Successfully generated {len(houses)} house(s)")
    print(f"Output: {output_file}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()

