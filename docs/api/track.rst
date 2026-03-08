Track System
============

The track system handles loading racing circuits, managing centerline and raceline data, and coordinate conversions.

Source: ``gymkhana/envs/track/``

Track loading
-------------

Tracks are loaded from map files following the ROS map convention:

- A ``.yaml`` metadata file with ``resolution`` (m/pixel) and ``origin`` fields
- A single-channel black-and-white image (black = obstacles, white = free space)

Both files must share the same base name and directory.

Map management
--------------

Maps are managed through a two-tier system:

- **Local** ``maps/`` **directory**: git submodule from https://github.com/TeoIlie/F1TENTH_Racetracks — used for development
- **User cache** ``~/.gymkhana/maps/``: auto-downloaded from GitHub releases for pip-installed users

The download URL is configured in ``gymkhana/envs/track/track_utils.py::MAPS_URL``.

Frenet coordinates
------------------

The track system supports conversion between Cartesian and Frenet (arc-length based) coordinate frames.

``frenet_to_cartesian()``
   Convert Frenet coordinates to Cartesian. Available in ``gymkhana/envs/track/track.py``.

This is useful for initializing vehicles at specific positions along the track using ``env.reset(options={"poses": ...})``.

Centerline and raceline
------------------------

Each track includes:

- **Centerline**: the geometric center of the track, used for Frenet projection and progress measurement
- **Raceline**: an optimized racing line (when available), rendered in red when ``render_track_lines`` is enabled

Both use cubic spline interpolation for smooth trajectories.
