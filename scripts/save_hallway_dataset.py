"""Generate and save houses with hallways in ProcTHOR-10K compatible format.

The output can be loaded locally like:
    import gzip, json
    with gzip.open("hallway_houses.json.gz", "rt") as f:
        dataset = json.load(f)
    house = dataset["train"][0]  # Same format as prior.load_dataset("procthor-10k")
"""
import gzip
import json

from procthor.generation import HouseGenerator
from procthor.generation.room_specs import RoomSpec
from procthor.utils.types import LeafRoom, MetaRoom


def create_hallway_spec(num_bedrooms: int, num_bathrooms: int, spec_id: str) -> RoomSpec:
    """Create a house spec with hallway structure."""
    room_id = 2

    kitchen = LeafRoom(room_id=room_id, ratio=2, room_type="Kitchen")
    room_id += 1

    living = LeafRoom(room_id=room_id, ratio=3, room_type="LivingRoom")
    room_id += 1

    hallway = LeafRoom(room_id=room_id, ratio=1, room_type="Hallway")
    room_id += 1

    guest_bathroom = LeafRoom(room_id=room_id, ratio=1, room_type="Bathroom")
    room_id += 1

    bedrooms = []
    for i in range(num_bedrooms):
        bedroom = LeafRoom(room_id=room_id, ratio=2, room_type="Bedroom")
        room_id += 1
        bedrooms.append(bedroom)

    extra_bathrooms = []
    for i in range(num_bathrooms - 1):
        bathroom = LeafRoom(room_id=room_id, ratio=1, room_type="Bathroom")
        room_id += 1
        extra_bathrooms.append(bathroom)

    public = MetaRoom(ratio=5, children=[kitchen, living])
    hallway_zone = MetaRoom(ratio=2, children=[hallway, guest_bathroom])
    
    private_rooms = bedrooms + extra_bathrooms
    if private_rooms:
        private = MetaRoom(ratio=num_bedrooms * 2 + len(extra_bathrooms), children=private_rooms)
        house = MetaRoom(ratio=10 + len(private_rooms), children=[public, hallway_zone, private])
    else:
        house = MetaRoom(ratio=7, children=[public, hallway_zone])

    total_rooms = 3 + num_bedrooms + num_bathrooms
    dim_size = 8 + total_rooms // 2

    return RoomSpec(
        room_spec_id=spec_id,
        sampling_weight=1,
        spec=[house],
        dims=lambda d=dim_size: (d, d),
    )


def generate_houses(seeds: list, num_bedrooms: int = 3, num_bathrooms: int = 2):
    """Generate houses for given seeds."""
    houses = []
    for seed in seeds:
        print(f"  Generating seed {seed}...")
        spec_id = f"hallway-{num_bedrooms}bed-{num_bathrooms}bath-{seed}"
        room_spec = create_hallway_spec(num_bedrooms, num_bathrooms, spec_id)
        
        gen = HouseGenerator(split="train", seed=seed, room_spec=room_spec)
        try:
            house, _ = gen.sample()
            house.validate(gen.controller)
            houses.append(house.data)
            print(f"    ✓ Success: {len(house.rooms)} rooms")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
        
        if gen.controller:
            gen.controller.stop()
    
    return houses


if __name__ == "__main__":
    # Seeds that worked in visualization (indices 2, 3, 4 = seeds 303, 404, 505)
    working_seeds = [303, 404, 505]
    
    # Also try a few more seeds to get more variety
    extra_seeds = [606, 707, 808, 909, 1010]
    
    print("Generating houses with hallways...")
    print("=" * 50)
    
    all_houses = []
    
    print("\nUsing known working seeds:")
    all_houses.extend(generate_houses(working_seeds))
    
    print("\nTrying additional seeds:")
    all_houses.extend(generate_houses(extra_seeds))
    
    print("=" * 50)
    print(f"Successfully generated {len(all_houses)} houses")
    
    if all_houses:
        # Save in ProcTHOR-10K format: {"train": [...], "val": [...], "test": [...]}
        dataset = {
            "train": all_houses,
            "val": [],
            "test": []
        }
        
        # Save as gzipped JSON (same as procthor-10k)
        output_file = "hallway_houses.json.gz"
        with gzip.open(output_file, "wt", encoding="UTF-8") as f:
            json.dump(dataset, f)
        print(f"\nSaved to: {output_file}")
        
        # Also save uncompressed for easy viewing
        output_json = "hallway_houses.json"
        with open(output_json, "w") as f:
            json.dump(dataset, f, indent=2)
        print(f"Saved to: {output_json}")
        
        print(f"\nTo load:")
        print("  import gzip, json")
        print(f'  with gzip.open("{output_file}", "rt") as f:')
        print("      dataset = json.load(f)")
        print("  house = dataset['train'][0]")

