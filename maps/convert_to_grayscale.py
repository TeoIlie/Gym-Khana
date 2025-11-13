#!/usr/bin/env python3
"""
Convert map image PNG to grayscale for use in the simulator.

Usage:
    python3 convert_to_grayscale.py --map Drift
"""
from PIL import Image
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convert RGB to grayscale for use in simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--map", type=str, default="Drift", help="Map name (default: Drift)")

    args = parser.parse_args()

    # Paths
    script_dir = Path(__file__).parent
    map_path = script_dir / args.map
    map_name = args.map
    img_path = map_path / f"{map_name}.png"

    if not map_path.exists():
        raise ValueError(f"ERROR: Map directory not found: {map_path}")

    # Load the RGB image
    img = Image.open(img_path)
    print(f"Original - Mode: {img.mode}, Size: {img.size}")

    # Convert to grayscale
    img_gray = img.convert("L")
    print(f"Converted - Mode: {img_gray.mode}, Size: {img_gray.size}")

    # Save back as grayscale
    img_gray.save(img_path)
    print("Saved grayscale version to Drift.png")


if __name__ == "__main__":
    main()
