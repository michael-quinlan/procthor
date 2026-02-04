"""Visualize the saved hallway houses."""
import gzip
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, FancyBboxPatch
import matplotlib.patches as mpatches

room_type_to_color = {
    "Kitchen": "#ffd6e7",
    "LivingRoom": "#d9f7be",
    "Bedroom": "#fff1b8",
    "Bathroom": "#bae7ff",
    "Hallway": "#d9d9d9",
}

# 1 meter = 3.28084 feet, so 1 sq meter = 10.7639 sq feet
SQ_METERS_TO_SQ_FEET = 10.7639

def polygon_area(poly):
    """Calculate area of polygon using shoelace formula."""
    n = len(poly)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += poly[i]["x"] * poly[j]["z"]
        area -= poly[j]["x"] * poly[i]["z"]
    return abs(area) / 2.0

def plot_house_from_data(house_data, ax, title):
    """Plot a house from its JSON data."""
    total_sqft = 0

    # Draw rooms
    for room in house_data["rooms"]:
        room_id = room["id"]
        room_type = room["roomType"]
        poly = room["floorPolygon"]

        xs = [p["x"] for p in poly] + [poly[0]["x"]]
        zs = [p["z"] for p in poly] + [poly[0]["z"]]

        color = room_type_to_color.get(room_type, "#cccccc")
        ax.fill(xs, zs, color, alpha=0.8)
        ax.plot(xs, zs, "#000000", linewidth=2)

        # Calculate centroid for label
        cx = sum(p["x"] for p in poly) / len(poly)
        cz = sum(p["z"] for p in poly) / len(poly)

        # Calculate area in sq feet
        area_sqm = polygon_area(poly)
        area_sqft = area_sqm * SQ_METERS_TO_SQ_FEET
        total_sqft += area_sqft

        # Shorten room_id for display
        short_id = room_id.replace("room|", "")
        ax.text(cx, cz, f"{room_type}\n({short_id})\n{area_sqft:.0f} sqft",
                ha='center', va='center', fontsize=7, fontweight='bold')
    
    # Draw doors
    for door in house_data.get("doors", []):
        door_id = door["id"]
        wall0 = door.get("wall0", "")
        wall1 = door.get("wall1", "")
        
        # Parse wall info to get door position
        # Wall format: "wall|room_id|x1|z1|x2|z2"
        if wall0:
            parts = wall0.split("|")
            if len(parts) >= 5:
                try:
                    x1, z1, x2, z2 = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                    # Door is somewhere along this wall
                    door_x = (x1 + x2) / 2
                    door_z = (z1 + z2) / 2
                    ax.plot(door_x, door_z, 's', color='brown', markersize=8, zorder=5)
                except (ValueError, IndexError):
                    pass
    
    ax.set_title(f"{title}\nTotal: {total_sqft:.0f} sqft", fontsize=10, fontweight='bold')
    ax.set_aspect('equal')
    ax.axis('off')

    return total_sqft


# Load the saved houses
with gzip.open("hallway_houses.json.gz", "rt") as f:
    dataset = json.load(f)

houses = dataset["train"]
print(f"Loaded {len(houses)} houses")

# Create figure
fig, axes = plt.subplots(1, len(houses), figsize=(6 * len(houses), 6))
if len(houses) == 1:
    axes = [axes]

for idx, house_data in enumerate(houses):
    spec_id = house_data.get("metadata", {}).get("roomSpecId", f"House {idx+1}")
    num_rooms = len(house_data.get("rooms", []))
    num_doors = len(house_data.get("doors", []))
    
    title = f"{spec_id}\n({num_rooms} rooms, {num_doors} doors)"
    plot_house_from_data(house_data, axes[idx], title)

# Add legend
legend_patches = [mpatches.Patch(color=color, label=room_type) 
                  for room_type, color in room_type_to_color.items()]
fig.legend(handles=legend_patches, loc='lower center', ncol=5, fontsize=9)

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
plt.savefig("saved_houses_floorplans.png", dpi=150, bbox_inches="tight")
print("Saved: saved_houses_floorplans.png")

