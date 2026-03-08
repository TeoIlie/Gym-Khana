Laser Scan Simulator Models
======================================

This file contains all numba just-in-time compiled function for the 2D laser scan models. The core of the laser scan simulation is the Euclidean distance transform of the map image provided. See more details about the algorithm here: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html#scipy.ndimage.distance_transform_edt. Then, ray tracing is used to create the beams of the 2D lasers scan.

See ``gymkhana/envs/laser_models.py`` for the full source.