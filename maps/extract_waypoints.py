#!/usr/bin/env python3
"""
Extract centerline waypoints from a track map image.

This script implements steps 2.1-2.4 of the Drift map extraction plan:
1. Load and prepare the track image
2. Extract skeleton (medial axis) as centerline
3. Order skeleton pixels into sequential path
4. Measure path length and calculate required waypoints

Usage:
    python3 extract_waypoints.py --map Drift [--visualize] [--spacing SPACING_VALUE]
"""

import argparse
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
from skimage.measure import label
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from pathlib import Path
import csv

# Default flag values
DEFAULT_MAP = "Drift"
DEFAULT_SPACING = 1.0


class SkeletonTracer:
    """Trace a skeleton image into an ordered sequential path."""

    def __init__(self, skeleton):
        """
        Args:
            skeleton: Binary 2D numpy array where True = skeleton pixels
        """
        self.skeleton = skeleton.astype(bool)
        self.visited = np.zeros_like(skeleton, dtype=bool)

    def get_neighbors(self, y, x):
        """Get 8-connected neighbors of pixel (y, x) that are on skeleton."""
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue

                ny, nx = y + dy, x + dx

                # Check bounds
                if 0 <= ny < self.skeleton.shape[0] and 0 <= nx < self.skeleton.shape[1]:
                    if self.skeleton[ny, nx] and not self.visited[ny, nx]:
                        neighbors.append((ny, nx))

        return neighbors

    def trace_from_point(self, start_y, start_x, max_points=None):
        """
        Trace skeleton starting from (start_y, start_x).

        Args:
            start_y, start_x: Starting pixel coordinates
            max_points: Maximum points to trace (None = unlimited)

        Returns:
            List of (y, x) coordinates in order
        """
        path = [(start_y, start_x)]
        self.visited[start_y, start_x] = True

        current_y, current_x = start_y, start_x

        while True:
            if max_points and len(path) >= max_points:
                break

            neighbors = self.get_neighbors(current_y, current_x)

            if len(neighbors) == 0:
                # No unvisited neighbors - path complete
                break

            # Choose the first available neighbor
            # For a clean skeleton, there should be at most 2 neighbors
            current_y, current_x = neighbors[0]
            path.append((current_y, current_x))
            self.visited[current_y, current_x] = True

        return path


def load_and_prepare_image(map_path, map_name):
    """
    Step 2.1: Load and prepare track image.

    Args:
        map_path: Path to map directory

    Returns:
        track_mask: Binary numpy array (1=track, 0=walls)
        img_array: Original grayscale image array
    """
    print("Step 2.1: Loading and preparing image...")

    img_path = map_path / f"{map_name}.png"
    if not img_path.exists():
        raise FileNotFoundError(f"Map image not found: {img_path}")

    # Load image
    img = Image.open(img_path)
    img_gray = img.convert("L")  # Convert to grayscale
    img_array = np.array(img_gray)

    print(f"  Image size: {img_array.shape[1]} × {img_array.shape[0]} pixels")

    # Binarize: track=1 (white), walls=0 (black)
    track_mask = (img_array > 127).astype(np.uint8)

    track_pixels = np.sum(track_mask)
    total_pixels = track_mask.size
    track_percent = 100.0 * track_pixels / total_pixels

    print(f"  Track pixels: {track_pixels:,} ({track_percent:.1f}%)")

    return track_mask, img_array


def extract_skeleton(track_mask):
    """
    Step 2.2: Extract skeleton (centerline) using morphological skeletonization.

    Args:
        track_mask: Binary numpy array (1=track, 0=walls)

    Returns:
        skeleton: Binary numpy array (True=centerline pixels)
    """
    print("\nStep 2.2: Extracting skeleton...")

    # Compute medial axis (skeleton)
    skeleton = skeletonize(track_mask)

    skeleton_pixels = np.sum(skeleton)
    print(f"  Skeleton pixels: {skeleton_pixels:,}")

    # Check connectivity
    labeled = label(skeleton, connectivity=2)
    num_components = labeled.max()
    print(f"  Connected components: {num_components}")

    if num_components > 1:
        print("  WARNING: Skeleton has multiple disconnected components!")
        print("           Using largest component...")

        # Find largest component
        component_sizes = [(i, np.sum(labeled == i)) for i in range(1, num_components + 1)]
        largest_component = max(component_sizes, key=lambda x: x[1])[0]

        # Keep only largest component
        skeleton = labeled == largest_component
        skeleton_pixels = np.sum(skeleton)
        print(f"  Largest component pixels: {skeleton_pixels:,}")

    return skeleton


def order_skeleton_path(skeleton):
    """
    Step 2.3: Order skeleton pixels into sequential path.

    Args:
        skeleton: Binary numpy array (True=centerline pixels)

    Returns:
        ordered_path: Numpy array of shape (N, 2) with (y, x) coordinates
    """
    print("\nStep 2.3: Ordering skeleton into sequential path...")

    # Find skeleton coordinates
    skeleton_coords = np.argwhere(skeleton)  # Returns [[y1,x1], [y2,x2], ...]

    if len(skeleton_coords) == 0:
        raise ValueError("Skeleton is empty!")

    print(f"  Total skeleton pixels: {len(skeleton_coords)}")

    # Find starting point (topmost, then leftmost)
    start_idx = np.argmin(skeleton_coords[:, 0])  # Smallest y (top of image)
    start_y, start_x = skeleton_coords[start_idx]

    print(f"  Starting point: ({start_x}, {start_y}) [x, y in pixels]")

    # Trace path
    tracer = SkeletonTracer(skeleton)
    path = tracer.trace_from_point(start_y, start_x)

    print(f"  Traced path length: {len(path)} pixels")

    # Convert to numpy array
    ordered_path = np.array(path)  # Shape: (N, 2) with (y, x)

    # Check if we traced the full skeleton
    traced_percent = 100.0 * len(path) / len(skeleton_coords)
    print(f"  Coverage: {traced_percent:.1f}% of skeleton")

    if traced_percent < 95:
        print("  WARNING: Did not trace full skeleton!")
        print("           Track may have branches or disconnections.")

    return ordered_path


def measure_path_and_calculate_waypoints(ordered_path, resolution, target_spacing):
    """
    Step 2.4: Measure path length and calculate required number of waypoints.

    Args:
        ordered_path: Numpy array of shape (N, 2) with (y, x) coordinates in pixels
        resolution: Meters per pixel
        target_spacing: Target spacing between waypoints in meters

    Returns:
        Dictionary with:
            - total_length_px: Total path length in pixels
            - total_length_m: Total path length in meters
            - cumulative_distance_px: Cumulative distance array
            - num_waypoints: Recommended number of waypoints
            - waypoint_spacing_m: Actual spacing that will be used
    """
    print("\nStep 2.4: Measuring path length and calculating waypoint count...")

    # Calculate distances between consecutive points
    # ordered_path is (y, x), so we need Euclidean distance
    deltas = np.diff(ordered_path, axis=0)  # Shape: (N-1, 2)
    distances = np.sqrt(np.sum(deltas**2, axis=1))  # Shape: (N-1,)

    # Cumulative distance
    cumulative_distance_px = np.concatenate([[0], np.cumsum(distances)])
    total_length_px = cumulative_distance_px[-1]

    # Convert to meters
    total_length_m = total_length_px * resolution

    print(f"  Total path length: {total_length_px:.1f} pixels = {total_length_m:.2f} meters")

    # Calculate number of waypoints
    num_waypoints = int(np.round(total_length_m / target_spacing))

    # Ensure at least minimum number of waypoints
    if num_waypoints < 10:
        print(f"  WARNING: Only {num_waypoints} waypoints calculated.")
        print(f"           Using minimum of 10 waypoints.")
        num_waypoints = 10

    # Calculate actual spacing that will be achieved
    actual_spacing = total_length_m / num_waypoints

    print(f"  Target waypoint spacing: {target_spacing:.3f} meters")
    print(f"  Number of waypoints: {num_waypoints}")
    print(f"  Actual waypoint spacing: {actual_spacing:.3f} meters")

    # Check loop closure (distance from last point to first)
    loop_closure_px = np.linalg.norm(ordered_path[-1] - ordered_path[0])
    loop_closure_m = loop_closure_px * resolution

    print(f"  Loop closure gap: {loop_closure_px:.1f} pixels = {loop_closure_m:.3f} meters")

    if loop_closure_m > 2 * target_spacing:
        print("  WARNING: Large loop closure gap detected!")
        print("           Track may not be a closed loop.")

    return {
        "total_length_px": total_length_px,
        "total_length_m": total_length_m,
        "cumulative_distance_px": cumulative_distance_px,
        "num_waypoints": num_waypoints,
        "waypoint_spacing_m": actual_spacing,
        "loop_closure_m": loop_closure_m,
    }


def subsample_waypoints(ordered_path, num_waypoints):
    """
    Step 2.5a: Subsample ordered path to target number of waypoints.

    Uses simple subsampling (every Nth point) to preserve skeleton geometry
    and maintain excellent loop closure.

    Args:
        ordered_path: Numpy array of shape (N, 2) with (y, x) coordinates in pixels
        num_waypoints: Target number of waypoints

    Returns:
        subsampled_path: Numpy array of shape (num_waypoints, 2)
    """
    print(f"\nStep 2.5a: Subsampling to {num_waypoints} waypoints...")

    num_skeleton_points = len(ordered_path)
    subsample_factor = num_skeleton_points // num_waypoints

    # Ensure we have at least some subsampling
    if subsample_factor < 1:
        subsample_factor = 1

    # Take every Nth point
    indices = np.arange(0, num_skeleton_points, subsample_factor)

    waypoints_px = ordered_path[indices]

    print(f"  Subsampled from {num_skeleton_points} to {len(waypoints_px)} waypoints")
    print(f"  Subsample factor: every {subsample_factor} pixels")

    return waypoints_px


def calculate_track_widths(waypoints_px, track_mask, resolution):
    """
    Step 2.5b: Calculate track width at each waypoint using distance transform.

    Uses distance transform for efficient width calculation. The distance transform
    gives the distance from each track pixel to the nearest boundary.

    Args:
        waypoints_px: Numpy array of shape (N, 2) with (y, x) coordinates in pixels
        track_mask: Binary track mask (1=track, 0=walls)
        resolution: Meters per pixel

    Returns:
        Tuple of (w_tr_right, w_tr_left) as numpy arrays in meters
    """
    print(f"\nStep 2.5b: Calculating track widths using distance transform...")

    # Compute distance transform: value = distance to nearest boundary (in pixels)
    distance_map = distance_transform_edt(track_mask)

    w_tr_right = []
    w_tr_left = []

    for i, (y_px, x_px) in enumerate(waypoints_px):
        # Sample distance map value at waypoint
        # This gives distance to nearest boundary in pixels
        center_dist_px = distance_map[int(y_px), int(x_px)]

        # Convert to meters
        # Using symmetric width (simplified approach)
        width_m = center_dist_px * resolution

        w_tr_right.append(width_m)
        w_tr_left.append(width_m)

    w_tr_right = np.array(w_tr_right)
    w_tr_left = np.array(w_tr_left)

    avg_width = np.mean(w_tr_right) + np.mean(w_tr_left)
    print(f"  Average track width: {avg_width:.3f} meters")
    print(f"  Width range: {2*np.min(w_tr_right):.3f} - {2*np.max(w_tr_right):.3f} meters")

    return w_tr_right, w_tr_left


def convert_to_world_coordinates(waypoints_px, image_shape, resolution, origin):
    """
    Step 2.6: Convert pixel coordinates to world coordinates.

    Transforms from image pixel coordinates to simulation world coordinates.
    Critical: y-axis flip is required because image y=0 is top,
    but world y increases upward.

    Args:
        waypoints_px: Numpy array of shape (N, 2) with (y, x) coordinates in pixels
        image_shape: Tuple of (height, width) of the image
        resolution: Meters per pixel
        origin: Origin from YAML file [x, y, z]

    Returns:
        waypoints_world: Numpy array of shape (N, 2) with (x, y) in meters
    """
    print(f"\nStep 2.6: Converting to world coordinates...")

    image_height, image_width = image_shape

    waypoints_world = []

    for y_px, x_px in waypoints_px:
        # Image coordinates: origin top-left, y-axis down
        # World coordinates: origin at map center, y-axis up

        # Flip y-axis
        y_flipped = (image_height - 1) - y_px

        # Convert to world coordinates
        x_m = origin[0] + x_px * resolution
        y_m = origin[1] + y_flipped * resolution

        waypoints_world.append([x_m, y_m])

    waypoints_world = np.array(waypoints_world)

    print(f"  World coordinate range:")
    print(f"    X: [{np.min(waypoints_world[:, 0]):.3f}, {np.max(waypoints_world[:, 0]):.3f}] meters")
    print(f"    Y: [{np.min(waypoints_world[:, 1]):.3f}, {np.max(waypoints_world[:, 1]):.3f}] meters")

    return waypoints_world


def write_centerline_csv(waypoints_world, w_tr_right, w_tr_left, output_path):
    """
    Step 2.7: Generate centerline CSV file.

    Writes waypoints in the format expected by the F1TENTH Gym simulator.

    Args:
        waypoints_world: Numpy array of shape (N, 2) with (x, y) in meters
        w_tr_right: Right track widths in meters
        w_tr_left: Left track widths in meters
        output_path: Path to output CSV file
    """
    print(f"\nStep 2.7: Writing centerline CSV...")

    with open(output_path, "w", newline="") as f:
        # Write header comment line directly (not using csv.writer to avoid quotes)
        f.write("# x_m, y_m, w_tr_right_m, w_tr_left_m\n")

        writer = csv.writer(f)

        # Write waypoints
        for i in range(len(waypoints_world)):
            x_m, y_m = waypoints_world[i]
            w_right = w_tr_right[i]
            w_left = w_tr_left[i]

            writer.writerow([f"{x_m}", f"{y_m}", f"{w_right:.1f}", f"{w_left:.1f}"])

        # Write empty line at end (to match format of other tracks)
        writer.writerow([])

    print(f"  Saved centerline to: {output_path}")
    print(f"  Total waypoints: {len(waypoints_world)}")


def validate_centerline(waypoints_world):
    """
    Step 3.2: Verify track properties.

    Args:
        waypoints_world: Numpy array of shape (N, 2) with (x, y) in meters

    Returns:
        Dictionary with validation metrics
    """
    print(f"\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    # Check waypoint spacing
    spacings = np.sqrt(np.sum(np.diff(waypoints_world, axis=0) ** 2, axis=1))
    mean_spacing = np.mean(spacings)
    std_spacing = np.std(spacings)
    variation_pct = 100 * std_spacing / mean_spacing

    print(f"Waypoint spacing:")
    print(f"  Mean: {mean_spacing:.4f} m")
    print(f"  Std:  {std_spacing:.4f} m")
    print(f"  Variation: {variation_pct:.1f}%")
    print(f"  Range: [{np.min(spacings):.4f}, {np.max(spacings):.4f}] m")

    # Check loop closure
    loop_gap = np.linalg.norm(waypoints_world[-1] - waypoints_world[0])
    print(f"\nLoop closure gap: {loop_gap:.4f} m")

    if loop_gap > 0.1:
        print("  WARNING: Loop closure gap is large!")
    else:
        print("  ✓ Excellent loop closure")

    # Check total length
    total_length = np.sum(spacings) + loop_gap
    print(f"\nTotal track length: {total_length:.2f} m")

    print("=" * 70)

    return {
        "mean_spacing": mean_spacing,
        "std_spacing": std_spacing,
        "variation_pct": variation_pct,
        "loop_gap": loop_gap,
        "total_length": total_length,
    }


def visualize_centerline(img_array, waypoints_px, waypoints_world, origin, resolution, output_path):
    """
    Step 3.1: Visualize waypoints on image.

    Creates visualization showing the final centerline overlaid on the track image.

    Args:
        img_array: Original grayscale image
        waypoints_px: Waypoint coordinates in pixels (y, x)
        waypoints_world: Waypoint coordinates in world frame (x, y)
        origin: Origin from YAML
        resolution: Meters per pixel
        output_path: Path to save visualization
    """
    print(f"\nCreating centerline visualization...")

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.imshow(img_array, cmap="gray")

    # Plot the centerline
    ax.plot(waypoints_px[:, 1], waypoints_px[:, 0], "r-", linewidth=2, label="Centerline")

    # Mark start point (green)
    ax.scatter(
        waypoints_px[0, 1],
        waypoints_px[0, 0],
        c="green",
        s=100,
        marker="o",
        zorder=10,
        label="Start",
    )

    # Mark end point (blue)
    ax.scatter(
        waypoints_px[-1, 1],
        waypoints_px[-1, 0],
        c="blue",
        s=100,
        marker="s",
        zorder=10,
        label="End",
    )

    # Mark every 10th waypoint for density visualization
    every_n = max(1, len(waypoints_px) // 30)
    ax.scatter(
        waypoints_px[::every_n, 1],
        waypoints_px[::every_n, 0],
        c="yellow",
        s=20,
        marker="o",
        alpha=0.6,
        zorder=5,
    )

    ax.set_title(f"Drift Track Centerline ({len(waypoints_px)} waypoints)")
    ax.legend()
    ax.axis("equal")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved visualization to: {output_path}")


def visualize_skeleton(img_array, skeleton, ordered_path, output_path):
    """
    Visualize the skeleton and ordered path overlaid on the original image.

    Args:
        img_array: Original grayscale image
        skeleton: Binary skeleton array
        ordered_path: Ordered path array (y, x)
        output_path: Path to save visualization
    """
    print(f"\nCreating skeleton visualization...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    axes[0].imshow(img_array, cmap="gray")
    axes[0].set_title("Original Track Image")
    axes[0].axis("equal")

    # Skeleton overlay
    axes[1].imshow(img_array, cmap="gray")
    skeleton_overlay = np.zeros((*skeleton.shape, 3))
    skeleton_overlay[skeleton, 0] = 1  # Red for skeleton
    axes[1].imshow(skeleton_overlay, alpha=0.6)
    axes[1].set_title(f"Skeleton ({np.sum(skeleton)} pixels)")
    axes[1].axis("equal")

    # Ordered path
    axes[2].imshow(img_array, cmap="gray")
    # Plot path as line
    axes[2].plot(ordered_path[:, 1], ordered_path[:, 0], "r-", linewidth=2, label="Ordered Path")
    # Mark start point
    axes[2].scatter(
        ordered_path[0, 1],
        ordered_path[0, 0],
        c="green",
        s=100,
        marker="o",
        zorder=10,
        label="Start",
    )
    # Mark end point
    axes[2].scatter(
        ordered_path[-1, 1],
        ordered_path[-1, 0],
        c="blue",
        s=100,
        marker="s",
        zorder=10,
        label="End",
    )
    axes[2].set_title(f"Ordered Path ({len(ordered_path)} points)")
    axes[2].legend()
    axes[2].axis("equal")

    for ax in axes:
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved visualization to: {output_path}")

    # Show plot if running interactively
    # plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Extract centerline waypoints from track map",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--map", type=str, default=DEFAULT_MAP, help=f"Map name (default: {DEFAULT_MAP})")
    parser.add_argument("--visualize", action="store_true", help="Create visualization plots")
    parser.add_argument(
        "--spacing",
        type=float,
        default=DEFAULT_SPACING,
        help=f"Target waypoint spacing in meters (default: {DEFAULT_SPACING})",
    )

    args = parser.parse_args()

    # Paths
    map_name = args.map
    script_dir = Path(__file__).parent
    map_path = script_dir / map_name

    if not map_path.exists():
        print(f"ERROR: Map directory not found: {map_path}")
        return 1

    print("=" * 70)
    print(f"EXTRACTING WAYPOINTS FOR {map_name.upper()} TRACK")
    print("=" * 70)

    # Load map config to get resolution
    yaml_path = map_path / f"{map_name}_map.yaml"
    if not yaml_path.exists():
        print(f"ERROR: Map YAML not found: {yaml_path}")
        return 1

    # Parse YAML to get resolution and origin
    resolution = None
    origin = None
    with open(yaml_path, "r") as f:
        for line in f:
            if line.startswith("resolution:"):
                resolution = float(line.split(":")[1].strip())
            elif line.startswith("origin:"):
                # Parse origin list [x, y, z]
                origin_str = line.split(":", 1)[1].strip()
                origin = eval(origin_str)  # Parse Python list literal

    if resolution is None:
        print("ERROR: Resolution not found in the yaml file")
        return 1

    if origin is None:
        print("ERROR: Origin not found in the yaml file")
        return 1

    print(f"\nMap: {map_name}")
    print(f"Resolution: {resolution:.6f} m/px")
    print(f"Origin: {origin}")
    print(f"Target spacing: {args.spacing} m")

    # Step 2.1: Load and prepare image
    track_mask, img_array = load_and_prepare_image(map_path, map_name)

    # Step 2.2: Extract skeleton
    skeleton = extract_skeleton(track_mask)

    # Step 2.3: Order skeleton path
    ordered_path = order_skeleton_path(skeleton)

    # Step 2.4: Measure and calculate waypoints
    path_info = measure_path_and_calculate_waypoints(ordered_path, resolution, args.spacing)

    # Step 2.5: Subsample and calculate track widths
    waypoints_px = subsample_waypoints(ordered_path, path_info["num_waypoints"])
    w_tr_right, w_tr_left = calculate_track_widths(waypoints_px, track_mask, resolution)

    # Step 2.6: Convert to world coordinates
    waypoints_world = convert_to_world_coordinates(waypoints_px, img_array.shape, resolution, origin)

    # Step 2.7: Write centerline CSV
    csv_path = map_path / f"{map_name}_centerline.csv"
    write_centerline_csv(waypoints_world, w_tr_right, w_tr_left, csv_path)

    # Step 3.2: Validation
    validation_metrics = validate_centerline(waypoints_world)

    # Summary
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Track length:        {validation_metrics['total_length']:.2f} meters")
    print(f"Waypoints created:   {len(waypoints_world)}")
    print(
        f"Waypoint spacing:    {validation_metrics['mean_spacing']:.4f} ± {validation_metrics['std_spacing']:.4f} meters"
    )
    print(f"Loop closure gap:    {validation_metrics['loop_gap']:.4f} meters")
    print(f"\nOutput file:         {csv_path}")
    print("=" * 70)

    # Visualizations
    if args.visualize:
        # Create generation subfolder
        generation_path = map_path / "generation"
        generation_path.mkdir(exist_ok=True)

        # Skeleton extraction visualization
        skeleton_vis_path = generation_path / "skeleton_extraction.png"
        visualize_skeleton(img_array, skeleton, ordered_path, skeleton_vis_path)

        # Final centerline visualization
        centerline_vis_path = generation_path / "centerline_final.png"
        visualize_centerline(
            img_array,
            waypoints_px,
            waypoints_world,
            origin,
            resolution,
            centerline_vis_path,
        )

    print("\nAll steps (2.1-2.7) complete!")
    print(f"Centerline saved to: {csv_path}")

    return 0


if __name__ == "__main__":
    exit(main())
