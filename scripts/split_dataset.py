"""Convert a gzipped dataset to individual JSON files organized by house type.

Reads a gzipped JSON dataset and extracts individual houses into separate files.
Determines house type (hallway vs regular) based on room_spec_id.

Usage:
    uv run python scripts/split_dataset.py [--input INPUT] [--output OUTPUT]

Example:
    uv run python scripts/split_dataset.py \\
        --input ~/code/michael-quinlan/procthor/new_5k.json.gz \\
        --output ~/code/michael-quinlan/procthor/dataset_5k_with_hallways/
"""
import argparse
import gzip
import json
from pathlib import Path


def determine_house_type(house_dict: dict) -> str:
    """Determine if a house is hallway or regular based on room_spec_id.
    
    Args:
        house_dict: House dictionary with metadata
        
    Returns:
        "hallway" if room_spec_id starts with "hallway-", else "regular"
    """
    metadata = house_dict.get("metadata", {})
    room_spec_id = metadata.get("roomSpecId", "")
    
    if room_spec_id.startswith("hallway-"):
        return "hallway"
    return "regular"


def split_dataset(input_file: Path, output_dir: Path) -> None:
    """Split a gzipped dataset into individual house JSON files.
    
    Args:
        input_file: Path to gzipped JSON dataset
        output_dir: Directory to save individual house files
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize stats
    stats = {
        "total": 0,
        "hallway": 0,
        "regular": 0,
        "room_specs": {},
    }
    
    print(f"Reading dataset from: {input_file}")
    
    # Load gzipped dataset
    with gzip.open(input_file, "rt", encoding="UTF-8") as f:
        data = json.load(f)
    
    # Handle both formats: dict with splits or plain list
    if isinstance(data, dict):
        # Format: {"train": [...], "val": [...], "test": [...]}
        houses = []
        for split in ["train", "val", "test"]:
            if split in data:
                houses.extend(data[split])
    else:
        # Format: plain list
        houses = data
    
    print(f"Found {len(houses)} houses")
    print()
    
    # Process each house
    for idx, house_dict in enumerate(houses):
        house_type = determine_house_type(house_dict)
        
        # Save house file
        filename = output_dir / f"house_{idx:05d}_{house_type}.json"
        with open(filename, "w") as f:
            json.dump(house_dict, f)
        
        # Update stats
        stats["total"] += 1
        stats[house_type] += 1
        
        metadata = house_dict.get("metadata", {})
        room_spec_id = metadata.get("roomSpecId", "unknown")
        stats["room_specs"][room_spec_id] = stats["room_specs"].get(room_spec_id, 0) + 1
        
        if (idx + 1) % 100 == 0 or idx == len(houses) - 1:
            print(f"[{idx + 1}/{len(houses)}] {house_type} ({room_spec_id}) ✓")
    
    # Save stats
    stats_file = output_dir / "stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    print()
    print(f"✓ Saved {stats['total']} houses to: {output_dir}")
    print(f"  - Hallway: {stats['hallway']}")
    print(f"  - Regular: {stats['regular']}")
    print(f"  - Stats: {stats_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert gzipped dataset to individual JSON files"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path.home() / "code/michael-quinlan/procthor/new_5k.json.gz",
        help="Input gzipped JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / "code/michael-quinlan/procthor/dataset_5k_with_hallways",
        help="Output directory for individual house files",
    )
    
    args = parser.parse_args()
    
    # Expand home directory
    input_file = args.input.expanduser()
    output_dir = args.output.expanduser()
    
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return
    
    split_dataset(input_file, output_dir)


if __name__ == "__main__":
    main()

