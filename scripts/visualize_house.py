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
    """Plot a single house floorplan with doors."""
    # Draw rooms
    for room_id, room in house.rooms.items():
        poly = room.room_polygon.polygon.exterior.coords
        xs = [p[0] for p in poly]
        zs = [p[1] for p in poly]
        ax.fill(xs, zs, room_type_to_color.get(room.room_type, "#cccccc"))
        ax.plot(xs, zs, "#000000", linewidth=2)

        centroid = room.room_polygon.polygon.centroid
        ax.text(centroid.x, centroid.y, f"{room.room_type}\n({room_id})",
                ha='center', va='center', fontsize=7, fontweight='bold')

    # Draw doors on top of walls
    if "doors" in house.data and house.data["doors"]:
        for door in house.data["doors"]:
            # Get wall coordinates from wall0 id: "wall|room_id|x0|z0|x1|z1"
            wall_id = door.get("wall0", "")
            parts = wall_id.split("|")
            if len(parts) >= 6:
                try:
                    x0, z0, x1, z1 = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])

                    # holePolygon x values are positions along the wall
                    if "holePolygon" in door and len(door["holePolygon"]) >= 2:
                        hole_start = float(door["holePolygon"][0]["x"])
                        hole_end = float(door["holePolygon"][1]["x"])

                        # Calculate wall direction
                        wall_len = ((x1 - x0)**2 + (z1 - z0)**2)**0.5
                        if wall_len > 0:
                            dx = (x1 - x0) / wall_len
                            dz = (z1 - z0) / wall_len

                            # Door start and end positions along the wall
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
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')


import random

def create_custom_spec(num_bedrooms: int, num_bathrooms: int, spec_id: str) -> RoomSpec:
    """Create a house spec with specified number of bedrooms and bathrooms.

    Always includes: 1 Kitchen, 1 LivingRoom, 1 Hallway

    Structure ensures:
    - Hallway connects public and private zones
    - At least one bathroom is in the hallway zone (for guest access)
    - Bedrooms connect to hallway
    """
    room_id = 2

    # Public rooms (Kitchen + LivingRoom)
    kitchen = LeafRoom(room_id=room_id, ratio=2, room_type="Kitchen")
    room_id += 1

    living = LeafRoom(room_id=room_id, ratio=3, room_type="LivingRoom")
    room_id += 1

    hallway = LeafRoom(room_id=room_id, ratio=1, room_type="Hallway")
    room_id += 1

    # First bathroom goes with hallway (guest bathroom - public access)
    guest_bathroom = LeafRoom(room_id=room_id, ratio=1, room_type="Bathroom")
    room_id += 1

    # Bedrooms
    bedrooms = []
    for i in range(num_bedrooms):
        bedroom = LeafRoom(room_id=room_id, ratio=2, room_type="Bedroom")
        room_id += 1
        bedrooms.append(bedroom)

    # Additional bathrooms go with bedrooms (en-suite style)
    extra_bathrooms = []
    for i in range(num_bathrooms - 1):
        bathroom = LeafRoom(room_id=room_id, ratio=1, room_type="Bathroom")
        room_id += 1
        extra_bathrooms.append(bathroom)

    # Structure:
    # [Public: Kitchen + LivingRoom] adjacent to [Hallway zone: Hallway + Guest Bath] adjacent to [Private: Bedrooms + Extra Baths]
    public = MetaRoom(ratio=5, children=[kitchen, living])
    hallway_zone = MetaRoom(ratio=2, children=[hallway, guest_bathroom])

    private_rooms = bedrooms + extra_bathrooms
    if private_rooms:
        private = MetaRoom(ratio=num_bedrooms * 2 + len(extra_bathrooms), children=private_rooms)
        house = MetaRoom(ratio=10 + len(private_rooms), children=[public, hallway_zone, private])
    else:
        house = MetaRoom(ratio=7, children=[public, hallway_zone])

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

