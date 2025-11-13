# Drift Map Implementation Plan

## Overview

Create map configuration files for the Drift track to match the F1TENTH Gym format used by other tracks like Spielberg. The Drift track is a compact drift/practice course (~20m length) with higher pixel density than the racing circuits.

## Key Parameters (Calculated from Measurements)

### Resolution Calculation
- **Spielberg**: 37 px = 2.2 m → resolution = 0.05946 m/px (yaml shows 0.05796)
- **Drift**: 127 px = 1.0 m → **resolution = 0.00787 m/px**

The Drift track has **8× finer resolution** than Spielberg (smaller real-world pixel size).

### Map Dimensions
- **Image size**: 746 × 912 pixels (width × height)
- **Map size**: 5.87 × 7.18 meters
- **Origin** (to center map): `[-2.94, -3.59, 0.0]`

### Track Characteristics
- **Average width**: 1.0 meter
- **Estimated length**: ~18-20 meters
- **Target waypoints**: ~250-350 (dense sampling for accuracy)
- **Waypoint spacing**: ~60-80mm (much denser than Spielberg's 400mm)

## Implementation Steps

### Step 1: Create Drift_map.yaml

**File**: `maps/Drift/Drift_map.yaml`

```yaml
image: Drift.png
resolution: 0.00787
origin: [-2.9358420000000003, -3.5905439999999998, 0.0]
negate: 0
occupied_thresh: 0.45
free_thresh: 0.196
```

**Notes:**
- `resolution`: 1.0 / 127 = 0.007874... (use full precision)
- `origin`: Calculated to center the track around (0, 0) in world coordinates
- Other parameters: Copy from Spielberg (standard occupancy grid values)

### Step 2: Extract Track Centerline (Dense Sampling)

**Goal**: Generate densely-spaced ordered waypoints directly from skeleton, preserving loop closure.

**Key Design Decision**: Use **dense sampling directly from skeleton** instead of sparse sampling + resampling. This:
- Preserves perfect loop closure from skeleton
- Maintains accurate track width measurements
- Avoids interpolation artifacts at sharp curves

#### 2.1 Load and Prepare Image
```python
from PIL import Image
import numpy as np

# Load image
img = Image.open('maps/Drift/Drift.png')
img_gray = img.convert('L')  # Convert to grayscale
img_array = np.array(img_gray)

# Binarize: track=1 (white), walls=0 (black)
track_mask = (img_array > 127).astype(np.uint8)
```

**Important**: Ensure the PNG has black background and only the inside track is white. The skeleton will trace the centerline of the white region.

#### 2.2 Extract Skeleton (Centerline)
```python
from skimage.morphology import skeletonize

# Compute medial axis (skeleton)
skeleton = skeletonize(track_mask)

# skeleton is now a binary image with 1-pixel-wide centerline
# True where centerline exists, False elsewhere
```

**Why skeleton?**
- Gives mathematically optimal centerline (equidistant from both boundaries)
- Single-pixel width simplifies path tracing
- Handles variable track width automatically

#### 2.3 Order Skeleton Pixels into Sequential Path

The skeleton gives unordered pixels. Trace them in order:

```python
# Find skeleton pixel coordinates
skeleton_coords = np.argwhere(skeleton)  # Returns [[y1,x1], [y2,x2], ...]

# Find starting point (e.g., topmost point)
start_idx = np.argmin(skeleton_coords[:, 0])  # Smallest y (top of image)
start_point = skeleton_coords[start_idx]

# Trace path by following neighbors
ordered_path = trace_skeleton_path(skeleton, start_point)
```

**Tracing algorithm**:
1. Start at initial point
2. Mark as visited
3. Find unvisited neighbor (8-connected: up/down/left/right/diagonals)
4. Move to neighbor, add to path
5. Repeat until back near start (closed loop) or no neighbors

**Expected result**: 2000-2500 ordered waypoint pixels for ~18m track

#### 2.4 Subsample to Target Density

Instead of resampling, **subsample** the ordered path to target waypoint count:

```python
# Determine target number of waypoints
target_num_waypoints = 300  # Dense sampling: ~60mm spacing

# Calculate subsample indices
num_skeleton_points = len(ordered_path)
subsample_factor = num_skeleton_points // target_num_waypoints

# Take every Nth point
indices = np.arange(0, num_skeleton_points, subsample_factor)
waypoints_px = ordered_path[indices]

print(f"Subsampled from {num_skeleton_points} to {len(waypoints_px)} waypoints")
```

**Why subsample instead of resample?**
- Preserves actual skeleton geometry
- Maintains excellent loop closure
- No interpolation errors at curves

#### 2.5 Calculate Track Width at Each Waypoint

Use distance transform for efficient width calculation:

```python
from scipy.ndimage import distance_transform_edt

# Compute distance transform: value = distance to nearest boundary
distance_map = distance_transform_edt(track_mask)

w_tr_right = []
w_tr_left = []

for i, (y_px, x_px) in enumerate(waypoints_px):
    # Calculate tangent direction using neighboring waypoints
    if i == 0:
        prev_point = waypoints_px[-1]  # Wrap to end
        next_point = waypoints_px[1]
    elif i == len(waypoints_px) - 1:
        prev_point = waypoints_px[i-1]
        next_point = waypoints_px[0]  # Wrap to start
    else:
        prev_point = waypoints_px[i-1]
        next_point = waypoints_px[i+1]

    # Tangent vector (forward direction)
    tangent = next_point - prev_point
    tangent = tangent / np.linalg.norm(tangent)

    # Normal vectors (perpendicular to tangent)
    # In image coordinates (y down), perpendicular is [dy, dx] → [-dx, dy] or [dx, -dy]
    normal_right = np.array([tangent[1], -tangent[0]])   # 90° clockwise
    normal_left = np.array([-tangent[1], tangent[0]])    # 90° counter-clockwise

    # Sample distance map value at waypoint
    # This gives distance to nearest boundary
    center_dist_px = distance_map[int(y_px), int(x_px)]

    # Use symmetric width (simplified approach)
    # For asymmetric width, cast short rays along normals
    w_tr_right.append(center_dist_px * resolution)
    w_tr_left.append(center_dist_px * resolution)

w_tr_right = np.array(w_tr_right)
w_tr_left = np.array(w_tr_left)
```

**Simplified approach**: Just use the distance_map value at each waypoint for both left and right widths. This assumes symmetric track width, which is reasonable for a simple drift course.

**Why not ray casting?**
- Distance transform is faster and simpler
- At sharp curves with subsampled waypoints, ray casting can give wrong directions
- For this application, symmetric width is sufficient

#### 2.6 Convert to World Coordinates

Transform from image pixel coordinates to simulation world coordinates:

```python
image_height, image_width = img_array.shape
resolution = 0.00787
origin = [-2.9358420000000003, -3.5905439999999998, 0.0]

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
```

**Critical**: The y-axis flip is essential because image y=0 is top, but world y increases upward.

#### 2.7 Generate Drift_centerline.csv

```python
import csv

with open('maps/Drift/Drift_centerline.csv', 'w', newline='') as f:
    writer = csv.writer(f)

    # Write header
    writer.writerow(['# x_m, y_m, w_tr_right_m, w_tr_left_m'])

    # Write waypoints
    for i in range(len(waypoints_world)):
        x_m, y_m = waypoints_world[i]
        w_right = w_tr_right[i]
        w_left = w_tr_left[i]

        writer.writerow([f'{x_m}', f'{y_m}', f'{w_right:.1f}', f'{w_left:.1f}'])

    # Write empty line at end (to match Spielberg format)
    writer.writerow([])
```

**Format notes**:
- Header line with comment
- Coordinates: full precision (Python default)
- Widths: 1 decimal place
- Empty line at end

### Step 3: Validation

#### 3.1 Visualize Waypoints on Image

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 12))
ax.imshow(img_array, cmap='gray')

# Convert world coords back to pixels for plotting
waypoints_plot_x = (waypoints_world[:, 0] - origin[0]) / resolution
waypoints_plot_y = image_height - 1 - (waypoints_world[:, 1] - origin[1]) / resolution

ax.plot(waypoints_plot_x, waypoints_plot_y, 'r-', linewidth=2, label='Centerline')
ax.scatter(waypoints_plot_x[0], waypoints_plot_y[0], c='green', s=100, marker='o', label='Start')
ax.scatter(waypoints_plot_x[-1], waypoints_plot_y[-1], c='blue', s=100, marker='s', label='End')

ax.set_title('Drift Track Centerline')
ax.legend()
ax.axis('equal')
plt.savefig('maps/Drift/centerline_visualization.png', dpi=150)
```

#### 3.2 Verify Track Properties

```python
# Check waypoint spacing
spacings = np.sqrt(np.sum(np.diff(waypoints_world, axis=0)**2, axis=1))
print(f"Waypoint spacing: mean={np.mean(spacings):.4f}m, std={np.std(spacings):.4f}m")
print(f"Spacing variation: {100*np.std(spacings)/np.mean(spacings):.1f}%")

# Check track width
avg_width = np.mean(w_tr_right) + np.mean(w_tr_left)
print(f"Average track width: {avg_width:.3f}m (target: 1.0m)")

# Check loop closure
loop_gap = np.linalg.norm(waypoints_world[-1] - waypoints_world[0])
print(f"Loop closure gap: {loop_gap:.4f}m")

# Check total length
total_length = np.sum(spacings) + loop_gap  # Include closure segment
print(f"Total track length: {total_length:.2f}m")
```

**Expected values**:
- Spacing: 0.060 ± 0.010 m (60mm ± 10mm variation is acceptable)
- Track width: 0.9-1.1 m
- Loop closure: < 0.1 m (excellent with direct subsampling)
- Track length: 17-20 m

#### 3.3 Test in Simulator

```python
import gymnasium as gym

# Try loading the map
env = gym.make('f1tenth_gym:f1tenth-v0',
               config={'map': 'Drift', 'num_agents': 1})

obs, info = env.reset()
print("Map loaded successfully!")
env.close()
```

## Expected Output Files

### 1. `maps/Drift/Drift_map.yaml`
Already created (see Step 1)

### 2. `maps/Drift/Drift_centerline.csv`
- **Rows**: ~300-350 waypoints
- **Format**: `x_m, y_m, w_tr_right_m, w_tr_left_m`
- **Spacing**: ~60-80mm between waypoints
- **Example**:
```
# x_m, y_m, w_tr_right_m, w_tr_left_m
-1.234567, 2.345678, 0.5, 0.5
-1.230123, 2.349876, 0.5, 0.5
...
(empty line at end)
```

### 3. `maps/Drift/centerline_visualization.png` (validation)
- Visual overlay of centerline waypoints on track image
- Green dot: start point
- Blue square: end point
- Red line: centerline path

## Common Issues and Solutions

### Issue 1: Poor loop closure after processing
**Symptom**: Loop closure gap > 0.1m, end waypoint far from start
**Cause**: Resampling or interpolation breaks the skeleton's natural loop
**Solution**: Use direct subsampling instead of interpolation. Never use `interp1d` or spline fitting - just take every Nth skeleton point with `indices = np.arange(0, N, step)`.

### Issue 2: Skeleton has branches or spurious pixels
**Symptom**: Multiple disconnected components, path tracing gets stuck
**Solution**: Clean skeleton using morphological operations:
```python
from skimage.morphology import remove_small_objects
skeleton_clean = remove_small_objects(skeleton, min_size=50)
```
Or ensure the input PNG has clean white track on black background.

### Issue 3: Spikes in track width at sharp curves
**Symptom**: Width jumps to 2-3m at certain waypoints
**Cause**: With subsampled waypoints, tangent calculation at curves uses distant points, giving wrong normal direction
**Solution**: Calculate widths on the **dense skeleton before subsampling**, then subsample waypoints and widths together using the same indices.

### Issue 4: Track width all wrong (too large or too small)
**Symptom**: All widths are 2x or 0.5x expected
**Cause**: Distance transform not properly converted to meters
**Solution**: Verify `width_m = distance_px * resolution`. Check that `resolution = 0.00787` is correct.

### Issue 5: Waypoint spacing is very uneven
**Symptom**: Some waypoints 10mm apart, others 200mm apart
**Cause**: Skeleton has uneven pixel density in different regions
**Solution**: This is acceptable within 10% variation. If worse, consider using cumulative distance subsampling:
```python
# Calculate cumulative distance
deltas = np.diff(ordered_path, axis=0)
distances = np.sqrt(np.sum(deltas**2, axis=1))
cum_dist = np.concatenate([[0], np.cumsum(distances)])

# Sample at even distances
target_distances = np.linspace(0, cum_dist[-1], num_waypoints, endpoint=False)
indices = [np.argmin(np.abs(cum_dist - d)) for d in target_distances]
waypoints_px = ordered_path[indices]
```

### Issue 6: Coordinate transformation looks wrong in simulator
**Symptom**: Track appears flipped or offset
**Solution**: Double-check y-axis flip: `y_flipped = (height - 1) - y_px`. Verify origin sign (should be negative for centering). Test with a known point like image center.

## Implementation Checklist

- [ ] Calculate and verify resolution (1.0m / 127px = 0.00787)
- [ ] Calculate origin to center map
- [ ] Create Drift_map.yaml with correct parameters
- [ ] Load Drift.png and create binary track mask (white track, black background)
- [ ] Extract skeleton using skeletonize
- [ ] Trace skeleton into ordered path
- [ ] Verify skeleton coverage (should be 100% or close)
- [ ] Subsample to ~300 waypoints using simple indexing (no interpolation)
- [ ] Calculate track width using distance transform at each waypoint
- [ ] Convert waypoints to world coordinates (with proper y-flip)
- [ ] Write Drift_centerline.csv in correct format
- [ ] Visualize centerline overlaid on image
- [ ] Verify waypoint spacing (~60mm, variation < 20%)
- [ ] Verify track width (~1.0m)
- [ ] Verify loop closure (< 0.1m)
- [ ] Test map loading in simulator
- [ ] Commit files to repository

## Summary

The Drift track is a compact 18-20 meter drift course with 8× higher pixel density than Spielberg. Key differences from Spielberg:
- **Much smaller**: ~6m × 7m vs Spielberg's 57m × 69m
- **Higher resolution**: 0.00787 m/px vs 0.058 m/px
- **More waypoints**: ~300 vs Spielberg's 865
- **Denser spacing**: ~60mm vs 400mm (for precision at small scale)
- **Same format**: Compatible with existing F1TENTH Gym infrastructure

**Critical insight**: Use **direct subsampling** instead of resampling to preserve skeleton geometry and maintain perfect loop closure. Calculate widths on dense skeleton before subsampling to avoid tangent errors at curves.
