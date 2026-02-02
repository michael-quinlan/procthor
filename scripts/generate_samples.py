#!/usr/bin/env python3
"""Generate sample houses using hallway-based room specs.

This script generates sample houses using HALLWAY_ROOM_SPEC_SAMPLER
and saves a floorplan visualization to floorplan.png.

Usage:
    python scripts/generate_samples.py
"""

import random
import sys

import matplotlib.pyplot as plt

from procthor.generation import HouseGenerator
from procthor.generation.hallway_room_specs import HALLWAY_ROOM_SPEC_SAMPLER

# Room type colors (matching original branch)
ROOM_TYPE_TO_COLOR = {
    "Kitchen": "#ffd6e7",
    "LivingRoom": "#d9f7be",
    "Bedroom": "#fff1b8",
    "Bathroom": "#bae7ff",
    "Hallway": "#d9d9d9",
}

# Conversion constant
SQM_TO_SQFT = 10.7639


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


def main():
    """Generate sample houses and save floorplan visualization."""
    print("Generating sample houses with hallways...")

    # Generate 4 houses with different seeds
    num_houses = 4
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    seeds = [42, 123, 456, 789]
    
    for i, seed in enumerate(seeds):
        print(f"\nGenerating house {i + 1}/{num_houses} (seed={seed})...")

        generator = None
        try:
            room_spec = HALLWAY_ROOM_SPEC_SAMPLER.sample()
            generator = HouseGenerator(split="train", seed=seed, room_spec=room_spec)
            house, _ = generator.sample()

            room_types = [r.room_type for r in house.rooms.values()]
            print(f"  Spec: {room_spec.room_spec_id}")
            print(f"  Rooms: {room_types}")

            title = f"{room_spec.room_spec_id} (seed={seed})\n{len(house.rooms)} rooms"
            plot_house(house, axes[i], title)

        except Exception as e:
            print(f"  Error: {e}")
            axes[i].text(0.5, 0.5, f"Error:\n{str(e)[:50]}",
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f"House {i + 1} (Failed)")
        finally:
            # Ensure the AI2THOR controller is properly cleaned up to avoid zombie windows
            if generator is not None and generator.controller is not None:
                generator.controller.stop()

    plt.tight_layout()
    output_file = "floorplan.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
