# Agent Setup and Execution Guide

This document provides instructions for AI agents to set up and run the ProcTHOR project.

## Project Overview

ProcTHOR is a procedural generation system for creating interactive 3D house environments compatible with AI2-THOR.

## Environment Setup

### Prerequisites

- Python 3.8+ (tested with 3.10+)
- UV package manager (https://docs.astral.sh/uv/)

### Installation

1. **Install dependencies using UV:**
   ```bash
   uv sync
   ```
   This will:
   - Create a `.venv` virtual environment
   - Install all dependencies from `pyproject.toml`
   - Generate `uv.lock` for reproducible builds

2. **Verify installation:**
   ```bash
   uv run python -c "from procthor.generation import HouseGenerator; print('Success')"
   ```

## Running Scripts

All scripts should be executed using `uv run` to ensure the correct environment:

```bash
uv run python <script_path> [arguments]
```

### Generate Dataset Example

The primary test command for verifying the setup:

```bash
uv run python scripts/generate_dataset.py --num-houses 100 --output dataset.json.gz --save-images --image-dir ./images/
```

**Arguments:**
- `--num-houses`: Number of houses to generate (default: 1000)
- `--output`: Output file path for gzipped JSON (default: dataset.json.gz)
- `--split`: Data split - 'train', 'val', or 'test' (default: train)
- `--seed`: Random seed for reproducibility (optional)
- `--max-retries`: Max retries per house on failure (default: 10)
- `--save-images`: Save PNG floorplans for each house
- `--image-dir`: Directory for saving images (default: ./images/)

### Other Available Scripts

- `scripts/example.py` - Generate a single house
- `scripts/visualize_house.py` - Visualize multiple house floorplans
- `scripts/show_saved_houses.py` - Display saved houses from dataset

## Key Dependencies

- **ai2thor**: AI2-THOR simulator integration
- **numpy, scipy**: Numerical computing
- **matplotlib**: Visualization
- **trimesh, shapely**: Geometry processing
- **pandas**: Data manipulation
- **networkx**: Graph algorithms
- **python-sat, python-fcl**: Constraint solving and collision detection
- **tqdm**: Progress bars
- **moviepy**: Video/animation support

## Troubleshooting

### Module Not Found Errors

If you encounter `ModuleNotFoundError`, run `uv sync` again to ensure all dependencies are installed.

### Virtual Environment Issues

If the `.venv` directory becomes corrupted:
```bash
rm -rf .venv uv.lock
uv sync
```

### AI2-THOR Build Warnings

Warnings about unavailable builds are normal and can be safely ignored.

## Development Notes

- The project uses `pyproject.toml` for dependency management (UV-compatible)
- All code should be run through `uv run` for consistency
- The `uv.lock` file ensures reproducible environments across machines

