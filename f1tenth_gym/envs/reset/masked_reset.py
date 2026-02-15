from abc import abstractmethod

import numpy as np

from ..track import Raceline, Track
from .reset_fn import ResetFn
from .reset_utils import sample_around_waypoint


class MaskedResetFn(ResetFn):
    @abstractmethod
    def get_mask(self) -> np.ndarray:
        pass

    def __init__(
        self,
        track: Track,
        line_type: str,
        num_agents: int,
        move_laterally: bool,
        min_dist: float,
        max_dist: float,
    ):
        if line_type not in ["centerline", "raceline"]:
            raise ValueError(f"Invalid line_type: '{line_type}'. Must be 'centerline' or 'raceline'.")

        self.track = track
        self.line_type = line_type
        self.n_agents = num_agents
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.move_laterally = move_laterally
        self.mask = self.get_mask()

    @property
    def reference_line(self) -> Raceline:
        """Dynamically access the current active reference line from track."""
        return getattr(self.track, self.line_type)

    def sample(self) -> np.ndarray:
        waypoint_id = np.random.choice(np.where(self.mask)[0])
        poses = sample_around_waypoint(
            reference_line=self.reference_line,
            waypoint_id=waypoint_id,
            n_agents=self.n_agents,
            min_dist=self.min_dist,
            max_dist=self.max_dist,
            move_laterally=self.move_laterally,
        )
        return poses


class GridResetFn(MaskedResetFn):
    def __init__(
        self,
        track: Track,
        line_type: str,
        num_agents: int,
        move_laterally: bool = True,
        use_centerline: bool = True,
        shuffle: bool = True,
        start_width: float = 1.0,
        min_dist: float = 1.5,
        max_dist: float = 2.5,
    ):
        self.start_width = start_width
        self.shuffle = shuffle

        super().__init__(
            track=track,
            line_type=line_type,
            num_agents=num_agents,
            move_laterally=move_laterally,
            min_dist=min_dist,
            max_dist=max_dist,
        )

    def get_mask(self) -> np.ndarray:
        # approximate the nr waypoints in the starting line
        step_size = self.reference_line.length / self.reference_line.n
        n_wps = int(self.start_width / step_size)

        mask = np.zeros(self.reference_line.n)
        mask[:n_wps] = 1
        return mask.astype(bool)

    def sample(self) -> np.ndarray:
        poses = super().sample()

        if self.shuffle:
            np.random.shuffle(poses)

        return poses


class AllTrackResetFn(MaskedResetFn):
    def __init__(
        self,
        track: Track,
        line_type: str,
        num_agents: int,
        move_laterally: bool = True,
        shuffle: bool = True,
        min_dist: float = 1.5,
        max_dist: float = 2.5,
    ):
        super().__init__(
            track=track,
            line_type=line_type,
            num_agents=num_agents,
            move_laterally=move_laterally,
            min_dist=min_dist,
            max_dist=max_dist,
        )
        self.shuffle = shuffle

    def get_mask(self) -> np.ndarray:
        return np.ones(self.reference_line.n).astype(bool)

    def sample(self) -> np.ndarray:
        poses = super().sample()

        if self.shuffle:
            np.random.shuffle(poses)

        return poses
