from __future__ import annotations
import pathlib
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import yaml
from PIL import Image
from PIL.Image import Transpose
from yamldataclassconfig.config import YamlDataClassConfig

from . import Raceline
from .cubic_spline import CubicSpline2D
from .utils import find_track_dir
from ..rendering import EnvRenderer


@dataclass
class TrackSpec(YamlDataClassConfig):
    name: Optional[str]
    image: Optional[str]
    resolution: float
    origin: Tuple[float, float, float]
    negate: int
    occupied_thresh: float
    free_thresh: float


@dataclass
class Track:
    spec: TrackSpec
    filepath: Optional[str]
    ext: Optional[str]
    occupancy_map: np.ndarray
    centerline: Raceline
    raceline: Raceline

    def __init__(
        self,
        spec: TrackSpec,
        occupancy_map: np.ndarray,
        filepath: Optional[str] = None,
        ext: Optional[str] = None,
        centerline: Optional[Raceline] = None,
        raceline: Optional[Raceline] = None,
    ):
        """
        Initialize track object.

        Parameters
        ----------
        spec : TrackSpec
            track specification
        filepath : str
            path to the track image
        ext : str
            file extension of the track image file
        occupancy_map : np.ndarray
            occupancy grid map
        centerline : Raceline, optional
            centerline of the track, by default None
        raceline : Raceline, optional
            raceline of the track, by default None
        """
        self.spec = spec
        self.filepath = filepath
        self.ext = ext
        self.occupancy_map = occupancy_map
        self.centerline = centerline
        self.raceline = raceline

        # Render handle for lookahead curvature visualization
        self.lookahead_render = None

        # Debug: Store last projected point for visualization
        self.debug_projected_point = None
        self.debug_vehicle_point = None
        self.debug_render_projected = None
        self.debug_render_vehicle = None
        self.debug_render_line = None

    @staticmethod
    def load_spec(track: str, filespec: str) -> TrackSpec:
        """
        Load track specification from yaml file.

        Parameters
        ----------
        track : str
            name of the track
        filespec : str
            path to the yaml file

        Returns
        -------
        TrackSpec
            track specification
        """
        with open(filespec, "r") as yaml_stream:
            map_metadata = yaml.safe_load(yaml_stream)
            track_spec = TrackSpec(name=track, **map_metadata)
        return track_spec

    @staticmethod
    def from_track_name(track: str, track_scale: float = 1.0) -> Track:
        """
        Load track from track name.

        Parameters
        ----------
        track : str
            name of the track
        track_scale : float, optional
            scale of the track, by default 1.0

        Returns
        -------
        Track
            track object

        Raises
        ------
        FileNotFoundError
            if the track cannot be loaded
        """
        try:
            track_dir = find_track_dir(track)
            track_spec = Track.load_spec(
                track=track, filespec=str(track_dir / f"{track_dir.stem}_map.yaml")
            )
            track_spec.resolution = track_spec.resolution * track_scale
            track_spec.origin = (
                track_spec.origin[0] * track_scale,
                track_spec.origin[1] * track_scale,
                track_spec.origin[2],
            )

            # load occupancy grid
            map_filename = pathlib.Path(track_spec.image)
            image = Image.open(track_dir / str(map_filename)).transpose(
                Transpose.FLIP_TOP_BOTTOM
            )

            occupancy_map = np.array(image).astype(np.float32)
            occupancy_map[occupancy_map <= 128] = 0.0
            occupancy_map[occupancy_map > 128] = 255.0

            # if exists, load centerline
            if (track_dir / f"{track}_centerline.csv").exists():
                centerline = Raceline.from_centerline_file(
                    track_dir / f"{track}_centerline.csv",
                    track_scale=track_scale,
                )
            else:
                centerline = None

            # if exists, load raceline
            if (track_dir / f"{track}_raceline.csv").exists():
                raceline = Raceline.from_raceline_file(
                    track_dir / f"{track}_raceline.csv",
                    track_scale=track_scale,
                )
            else:
                raceline = centerline

            return Track(
                spec=track_spec,
                filepath=str((track_dir / map_filename.stem).absolute()),
                ext=map_filename.suffix,
                occupancy_map=occupancy_map,
                centerline=centerline,
                raceline=raceline,
            )
        except Exception as ex:
            print(ex)
            raise FileNotFoundError(f"It could not load track {track}") from ex

    @staticmethod
    def from_track_path(path: pathlib.Path, track_scale: float = 1.0) -> Track:
        """
        Load track from track path.

        Parameters
        ----------
        path : pathlib.Path
            path to the track yaml file

        Returns
        -------
        Track
            track object

        Raises
        ------
        FileNotFoundError
            if the track cannot be loaded
        """
        try:
            if type(path) is str:
                path = pathlib.Path(path)
        
            track_spec = Track.load_spec(
                track=path.stem, filespec=path
            )
            track_spec.resolution = track_spec.resolution * track_scale
            track_spec.origin = (
                track_spec.origin[0] * track_scale,
                track_spec.origin[1] * track_scale,
                track_spec.origin[2],
            )

            # load occupancy grid
            # Image path is from path + image name from track_spec
            image_path = path.parent / track_spec.image  
            image = Image.open(image_path).transpose(Transpose.FLIP_TOP_BOTTOM)
            occupancy_map = np.array(image).astype(np.float32)
            occupancy_map[occupancy_map <= 128] = 0.0
            occupancy_map[occupancy_map > 128] = 255.0

            # if exists, load centerline
            if (path / f"{path.stem}_centerline.csv").exists():
                centerline = Raceline.from_centerline_file(path / f"{path.stem}_centerline.csv")
            else:
                centerline = None

            # if exists, load raceline
            if (path / f"{path.stem}_raceline.csv").exists():
                raceline = Raceline.from_raceline_file(path / f"{path.stem}_raceline.csv")
            else:
                raceline = centerline

            return Track(
                spec=track_spec,
                filepath=str(path.absolute()),
                ext=image_path.suffix,
                occupancy_map=occupancy_map,
                centerline=centerline,
                raceline=raceline,
            )
        except Exception as ex:
            print(ex)
            raise FileNotFoundError(f"It could not load track {path}") from ex

    @staticmethod
    def from_refline(x: np.ndarray, y: np.ndarray, velx: np.ndarray):
        """
        Create an empty track reference line.

        Parameters
        ----------
        x : np.ndarray
            x-coordinates of the waypoints
        y : np.ndarray
            y-coordinates of the waypoints
        velx : np.ndarray
            velocities at the waypoints

        Returns
        -------
        track: Track
            track object
        """
        ds = 0.1
        resolution = 0.05
        margin_perc = 0.1

        spline = CubicSpline2D(x=x, y=y)
        ss, xs, ys, yaws, ks, vxs = [], [], [], [], [], []
        for i_s in np.arange(0, spline.s[-1], ds):
            xi, yi = spline.calc_position(i_s)
            yaw = spline.calc_yaw(i_s)
            k = spline.calc_curvature(i_s)

            # find closest waypoint
            closest = np.argmin(np.hypot(x - xi, y - yi))
            v = velx[closest]

            xs.append(xi)
            ys.append(yi)
            yaws.append(yaw)
            ks.append(k)
            ss.append(i_s)
            vxs.append(v)

        refline = Raceline(
            ss=np.array(ss).astype(np.float32),
            xs=np.array(xs).astype(np.float32),
            ys=np.array(ys).astype(np.float32),
            psis=np.array(yaws).astype(np.float32),
            kappas=np.array(ks).astype(np.float32),
            velxs=np.array(vxs).astype(np.float32),
            accxs=np.zeros_like(ss).astype(np.float32),
            spline=spline,
        )

        min_x, max_x = np.min(xs), np.max(xs)
        min_y, max_y = np.min(ys), np.max(ys)
        x_range = max_x - min_x
        y_range = max_y - min_y
        occupancy_map = 255.0 * np.ones(
            (
                int((1 + 2 * margin_perc) * x_range / resolution),
                int((1 + 2 * margin_perc) * y_range / resolution),
            ),
            dtype=np.float32,
        )
        # origin is the bottom left corner
        origin = (min_x - margin_perc * x_range, min_y - margin_perc * y_range, 0.0)

        track_spec = TrackSpec(
            name=None,
            image=None,
            resolution=resolution,
            origin=origin,
            negate=False,
            occupied_thresh=0.65,
            free_thresh=0.196,
        )

        return Track(
            spec=track_spec,
            filepath=None,
            ext=None,
            occupancy_map=occupancy_map,
            raceline=refline,
            centerline=refline,
        )

    def frenet_to_cartesian(self, s, ey, ephi, use_raceline=False):
        """
        Convert Frenet coordinates to Cartesian coordinates.

        s: distance along the raceline
        ey: lateral deviation
        ephi: heading deviation

        returns:
            x: x-coordinate
            y: y-coordinate
            psi: yaw angle
        """
        line = self.raceline if use_raceline else self.centerline
        x, y = line.spline.calc_position(s)
        psi = line.spline.calc_yaw(s)

        # Adjust x,y by shifting along the normal vector
        x -= ey * np.sin(psi)
        y += ey * np.cos(psi)

        # Adjust psi by adding the heading deviation
        psi += ephi

        return x, y, psi

    def cartesian_to_frenet(self, x, y, phi, use_raceline=False, s_guess=0, precise=False, debug=False):
        """
        WARNING: Only use "precise=True" if you pass an excellent s_guess.
        More work required to validate this even works.

        Convert Cartesian coordinates to Frenet coordinates.

        x: x-coordinate (Cartesian coord from std_state["x"])
        y: y-coordinate (Cartesian coord from std_state["y"])
        phi: yaw angle (from std_state["yaw"])
        use_raceline: Calculate with respect to raceline or centerline
        s_guess: Initial guess for arc length calculation
        debug: Enable debug logging and validation

        returns:
            s: arc length distance along the centerline/raceline
            ey: lateral deviation from centerline/raceline
            ephi: heading deviation (angle between vehicle and track heading) in radians
        """
        line = self.raceline if use_raceline else self.centerline

        # Store vehicle position for visualization
        self.debug_vehicle_point = np.array([x, y], dtype=np.float32)

        # Optimization-based arclength calculation

        if precise:
            # This is more precise, but causes errors - the search sometimes does not find the correct spot on the map
            s, ey = line.spline.calc_arclength(x, y, s_guess) 
        else:
            # This is slightly less accurate, but the global search doesn't cause errors
            s, ey = line.spline.calc_arclength_inaccurate(x, y) # slightly more inaccurate, but much faster

        # Wrap around
        s = s % line.spline.s[-1]

        # Use the normal to calculate the signed lateral deviation
        yaw = line.spline.calc_yaw(s)
        normal = np.asarray([-np.sin(yaw), np.cos(yaw)])
        x_eval, y_eval = line.spline.calc_position(s)
        dx = x - x_eval
        dy = y - y_eval
        distance_sign = np.sign(np.dot([dx, dy], normal))
        ey = ey * distance_sign

        # Store projected point for visualization
        self.debug_projected_point = np.array([x_eval, y_eval], dtype=np.float32)

        # Validation: Check if optimization likely failed
        actual_distance = np.hypot(dx, dy)

        if debug:
            track_length = line.spline.s[-1]

            # Always compute robust method for comparison when using precise mode
            if precise:
                # Validate optimization against global search
                s_robust, ey_robust = line.spline.calc_arclength_inaccurate(x, y)
                s_robust = s_robust % track_length

                s_error = abs(s - s_robust)
                ey_error = abs(ey - ey_robust)

                print(f"[DEBUG Frenet] Method: PRECISE (optimization)")
                print(f"[DEBUG Frenet] Vehicle: ({x:.2f}, {y:.2f}) -> Projected: ({x_eval:.2f}, {y_eval:.2f})")
                print(f"[DEBUG Frenet] s_guess={s_guess:.2f}m → s_optimized={s:.2f}m")
                print(f"[DEBUG Frenet] Validation: s_robust={s_robust:.2f}m (global search)")
                print(f"[DEBUG Frenet] ey_precise={ey:.3f}m, ey_robust={ey_robust:.3f}m")
                print(f"[DEBUG Frenet] Discrepancy: Δs={s_error:.2f}m, Δey={ey_error:.3f}m, dist={actual_distance:.3f}m")

                # Optimization failure detection
                if s_error > 20.0:
                    print(f"[WARNING] Optimization converged to WRONG location! ({s_error:.2f}m error)")
                    print(f"[WARNING] → Likely stuck in wrong local minimum. Use precise=False for robustness.")
                elif s_error > 5.0:
                    print(f"[INFO] Moderate discrepancy ({s_error:.2f}m). Optimization may be in different basin.")

                if ey_error > 1.0:
                    print(f"[WARNING] Large lateral deviation error ({ey_error:.3f}m)")

            else:
                # Robust method is ground truth - just report results
                print(f"[DEBUG Frenet] Method: ROBUST (global search)")
                print(f"[DEBUG Frenet] Vehicle: ({x:.2f}, {y:.2f}) -> Projected: ({x_eval:.2f}, {y_eval:.2f})")
                print(f"[DEBUG Frenet] s={s:.2f}m, ey={ey:.3f}m, distance={actual_distance:.3f}m")
                print(f"[DEBUG Frenet] Note: s_guess parameter is ignored by robust method")

            # Common warnings (applies to both methods)
            if actual_distance > 10.0:
                print(f"[WARNING] Vehicle FAR from track! Distance={actual_distance:.2f}m (typical track width ~10m)")
                print(f"[WARNING] → Vehicle may be off-track or in collision")

        phi = phi - yaw
        return s, ey, np.arctan2(np.sin(phi), np.cos(phi))

    def render_centerline(self, e: EnvRenderer) -> None:
        """
        Render the track centerline.

        The centerline represents the geometric center of the track and is rendered in green.

        Parameters
        ----------
        e : EnvRenderer
            Environment renderer object.
        """
        GREEN = (0, 255, 0)
        if self.centerline is not None:
            self.centerline.render_waypoints(e, color=GREEN)

    def render_raceline(self, e: EnvRenderer) -> None:
        """
        Render the track raceline.

        The raceline represents the optimal racing line through the track and is rendered in red.

        Parameters
        ----------
        e : EnvRenderer
            Environment renderer object.
        """
        RED = (255, 0, 0)
        if self.raceline is not None:
            self.raceline.render_waypoints(e, color=RED)

    def render_both_lines(self, e: EnvRenderer) -> None:
        """
        Render both the centerline and raceline.

        This is a convenience method that renders both the centerline (green) and raceline (red).

        Parameters
        ----------
        e : EnvRenderer
            Environment renderer object.
        """
        self.render_centerline(e)
        self.render_raceline(e)

    def render_frenet_projection(
        self,
        e: EnvRenderer,
        vehicle_color: tuple[int, int, int] = (255, 0, 255),  # Magenta for vehicle
        projected_color: tuple[int, int, int] = (0, 255, 255),  # Cyan for projected point
        line_color: tuple[int, int, int] = (255, 128, 0),  # Orange for connection line
        size: int = 8
    ) -> None:
        """
        Render debug visualization of Frenet coordinate projection.

        Shows:
        - Vehicle position (magenta)
        - Projected point on centerline (cyan)
        - Line connecting them (orange)

        This helps diagnose if cartesian_to_frenet is finding the correct closest point.

        Parameters
        ----------
        e : EnvRenderer
            Environment renderer object
        vehicle_color : tuple[int, int, int]
            RGB color for vehicle position marker
        projected_color : tuple[int, int, int]
            RGB color for projected point marker
        line_color : tuple[int, int, int]
            RGB color for connection line
        size : int
            Size of rendered points in pixels
        """
        if self.debug_vehicle_point is None or self.debug_projected_point is None:
            return  # No data to render yet

        # Render vehicle position
        vehicle_pt = self.debug_vehicle_point.reshape(1, 2)
        if self.debug_render_vehicle is None:
            self.debug_render_vehicle = e.render_points(vehicle_pt, color=vehicle_color, size=size)
        else:
            if hasattr(self.debug_render_vehicle, 'setData'):
                self.debug_render_vehicle.setData(vehicle_pt[:, 0], vehicle_pt[:, 1])
            else:
                self.debug_render_vehicle = e.render_points(vehicle_pt, color=vehicle_color, size=size)

        # Render projected point
        projected_pt = self.debug_projected_point.reshape(1, 2)
        if self.debug_render_projected is None:
            self.debug_render_projected = e.render_points(projected_pt, color=projected_color, size=size)
        else:
            if hasattr(self.debug_render_projected, 'setData'):
                self.debug_render_projected.setData(projected_pt[:, 0], projected_pt[:, 1])
            else:
                self.debug_render_projected = e.render_points(projected_pt, color=projected_color, size=size)

        # Render connection line (2 points: vehicle and projected)
        line_pts = np.vstack([self.debug_vehicle_point, self.debug_projected_point])
        if self.debug_render_line is None:
            self.debug_render_line = e.render_lines(
                line_pts,  # Shape: (2, 2) - two points with x,y coords
                color=line_color,
                size=2
            )
        else:
            if hasattr(self.debug_render_line, 'setData'):
                self.debug_render_line.setData(line_pts[:, 0], line_pts[:, 1])
            else:
                self.debug_render_line = e.render_lines(
                    line_pts,
                    color=line_color,
                    size=2
                )

    def render_lookahead_curvatures(
        self,
        e: EnvRenderer,
        vehicle_s: float,
        n_points: int,
        ds: float,
        color: tuple[int, int, int] = (0, 0, 255),
        size: int = 6
    ) -> None:
        """
        Render lookahead curvature sampling points ahead of vehicle.

        Visualizes the points where curvature is sampled for drift control,
        matching the behavior of sample_lookahead_curvatures() in observation.py.

        Parameters
        ----------
        e : EnvRenderer
            Environment renderer object
        vehicle_s : float
            Current arc length position of vehicle on centerline (meters)
        n_points : int, optional
            Number of lookahead points to render (default 10)
        ds : float, optional
            Spacing between points in meters (default 0.3m = 30cm)
        color : tuple[int, int, int], optional
            RGB color tuple for points (default yellow: (255, 255, 0))
        size : int, optional
            Size of rendered points in pixels (default 4)

        Notes
        -----
        - Points are sampled starting at (vehicle_s + ds), not at current position
        - For closed tracks, sampling wraps around using modulo arithmetic
        - Requires centerline with valid spline to be initialized
        """
        # Check if centerline exists and has spline
        if self.centerline is None or not hasattr(self.centerline, 'spline'):
            return  # Early return if no centerline to visualize

        centerline = self.centerline
        track_length = centerline.spline.s[-1]  # Total length for wrap-around

        # Collect points to render
        points = []
        for i in range(n_points):
            # Lookahead arc length (wrap around for closed tracks)
            # Match the formula from observation.py line 58
            s_ahead = (vehicle_s + ((i + 1) * ds)) % track_length

            # Convert s to x,y position
            try:
                x, y = centerline.spline.calc_position(s_ahead)
                points.append([x, y])
            except Exception:
                # Skip invalid points gracefully
                continue

        # Render all valid points
        if len(points) > 0:
            points_array = np.array(points, dtype=np.float32)

            # Handle caching to prevent point accumulation
            if self.lookahead_render is None:
                # First render: create new plot item
                self.lookahead_render = e.render_points(points_array, color=color, size=size)
            else:
                # Update existing plot item using setData()
                # This works for both PyQt (uses setData) and Pygame (returns None, so recreates)
                if hasattr(self.lookahead_render, 'setData'):
                    # PyQt renderer: update existing PlotDataItem
                    self.lookahead_render.setData(points_array[:, 0], points_array[:, 1])
                else:
                    # Pygame renderer: returns None, so always recreate
                    self.lookahead_render = e.render_points(points_array, color=color, size=size)
