"""Visualize multiple generated ProcTHOR house floorplans."""
import matplotlib.pyplot as plt
from procthor.generation import HouseGenerator, PROCTHOR10K_ROOM_SPEC_SAMPLER
from procthor.generation.hallway_room_specs import create_hallway_room_spec
from procthor.generation.room_specs import RoomSpec
from procthor.utils.types import LeafRoom, MetaRoom

room_type_to_color = {
    "Kitchen": "#ffd6e7",
    "LivingRoom": "#d9f7be",
    "Bedroom": "#fff1b8",
    "Bathroom": "#bae7ff",
    "Hallway": "#d9d9d9",
}

def plot_house(house, ax, title):
    """Plot a single house floorplan."""
    for room_id, room in house.rooms.items():
        poly = room.room_polygon.polygon.exterior.coords
        xs = [p[0] for p in poly]
        zs = [p[1] for p in poly]
        ax.fill(xs, zs, room_type_to_color.get(room.room_type, "#cccccc"))
        ax.plot(xs, zs, "#000000", linewidth=2)

        centroid = room.room_polygon.polygon.centroid
        ax.text(centroid.x, centroid.y, f"{room.room_type}\n({room_id})",
                ha='center', va='center', fontsize=7, fontweight='bold')

    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')


import random

def create_custom_spec(num_bedrooms: int, num_bathrooms: int, spec_id: str) -> RoomSpec:
    """Create a house spec with specified number of bedrooms and bathrooms.

    Always includes: 1 Kitchen, 1 LivingRoom, 1 Hallway
    """
    room_id = 2

    # Public rooms
    kitchen = LeafRoom(room_id=room_id, ratio=2, room_type="Kitchen")
    room_id += 1

    living = LeafRoom(room_id=room_id, ratio=3, room_type="LivingRoom")
    room_id += 1

    hallway = LeafRoom(room_id=room_id, ratio=1, room_type="Hallway")
    room_id += 1

    # Private rooms
    private_rooms = []
    for i in range(num_bedrooms):
        bedroom = LeafRoom(room_id=room_id, ratio=2, room_type="Bedroom")
        room_id += 1
        private_rooms.append(bedroom)

    for i in range(num_bathrooms):
        bathroom = LeafRoom(room_id=room_id, ratio=1, room_type="Bathroom")
        room_id += 1
        private_rooms.append(bathroom)

    # Structure: [Public zone] - Hallway - [Private zone]
    public = MetaRoom(ratio=5, children=[kitchen, living])
    private = MetaRoom(ratio=num_bedrooms * 2 + num_bathrooms, children=private_rooms)
    house = MetaRoom(ratio=10 + len(private_rooms), children=[public, hallway, private])

    # Scale dims based on total room count
    total_rooms = 3 + num_bedrooms + num_bathrooms
    dim_size = 8 + total_rooms // 2

    return RoomSpec(
        room_spec_id=spec_id,
        sampling_weight=1,
        spec=[house],
        dims=lambda d=dim_size: (d, d),
    )


# Generate 6 houses: 3 bedrooms, 2 bathrooms each (8 rooms total)
# 1 LivingRoom + 1 Kitchen + 1 Hallway + 3 Bedrooms + 2 Bathrooms = 8 rooms
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

configs = [
    (3, 2, 100, "3 Bed / 2 Bath (1)"),
    (3, 2, 200, "3 Bed / 2 Bath (2)"),
    (3, 2, 300, "3 Bed / 2 Bath (3)"),
    (3, 2, 400, "3 Bed / 2 Bath (4)"),
    (3, 2, 500, "3 Bed / 2 Bath (5)"),
    (3, 2, 600, "3 Bed / 2 Bath (6)"),
]

for idx, (num_bedrooms, num_bathrooms, seed, title) in enumerate(configs):
    print(f"Generating {title}...")

    room_spec = create_custom_spec(num_bedrooms, num_bathrooms, f"custom-{seed}")
    gen = HouseGenerator(split="train", seed=seed, room_spec=room_spec)

    try:
        house, _ = gen.sample()
        plot_house(house, axes[idx], f"{title}\n({len(house.rooms)} rooms, with hallway)")
    except Exception as e:
        axes[idx].text(0.5, 0.5, f"Failed:\n{str(e)[:50]}",
                       ha='center', va='center', transform=axes[idx].transAxes)
        axes[idx].set_title(f"{title} (Failed)")

plt.tight_layout()
plt.savefig("floorplans_gallery.png", dpi=150, bbox_inches="tight")
print("Saved: floorplans_gallery.png")

