# Maps
## Format
All maps are stored here, each in its own directory `"MAP_NAME"`, with these key files:
1. `MAP_NAME_centerline.csv` stores the centerline waypoints in the row format `x_m, y_m, w_tr_right_m, w_tr_left_m`
2. `MAP_NAME_map.yaml` stores configuration details. Resolution is calculated as real-life track width in metres divided by track length in pixels in the PNG image
3. `MAP_NAME_raceline.csv` stores the raceline waypoints in the row format `s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2`
4. `MAP_NAME.png` is a grayscale PNG image representing the occupancy grid - black is occupied, and white is track

## Generating a new map
New maps are created as a PNG in a tool like Photoshop, and then the provided scripts convert them into formats appropriate for use in this simulator
1. First, the png must be generated. This can be done in Photoshop by tracing the inner, outer borders with the **Curvature Pen** tool, using **Threshold** to convert to black and white, and exporting as PNG to a new folder `/maps/MAP_NAME/MAP_NAME.png`. Though not necessary, rendering looks better if the aspect ratio of the map PNG is 1:1
2. Then, `convert_to_grayscale.py` is used to convert the PNG to grayscale format for interpretation by the simulator
3. Finally, `extract_waypoints.py` takes the PNG, uses `skeletonize` to extract the centerline, and saves the waypoints to the appropriate file. 
    1. Note that for this to work, only the inside (not the outside) of the track must be white, so the `skeletonize` algorithm recognizes this as the only empty space. This can easily be done using the **Paint Bucket** tool in Photoshop to fill the outer space with solid black. 
    2. Note also that the `skeletonize` generates very many waypoints which are downsampled. The precision of the downsampling can be manually configured with the `--spacing SPACING_VALUE` flag. Using a finer precision creates more points, but may be ragged due to `skeletonize` imprecision. Less precision may follow the track more loosely, but be smoother due to the cubic spline fitted to the waypoints. The image generated as `centerline_final.png` does not use a cubic spline, to see the actual track defined by a series of waypoints, load the track into the `gym` environment with rendering enabled. Experiment with the spacing value (default `0.1`) to find the best middle-ground