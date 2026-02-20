"""
Curriculum learning callback for recovery training.

Gradually expands recovery state ranges as the agent demonstrates competence,
measured by a rolling success rate from info["recovered"].
"""

from collections import deque
from dataclasses import dataclass, field

from stable_baselines3.common.callbacks import BaseCallback

import wandb


@dataclass
class CurriculumRange:
    """A single curriculum range that expands symmetrically from initial to max bounds."""

    name: str
    initial_lo: float
    initial_hi: float
    max_lo: float
    max_hi: float
    n_stages: int

    # Runtime state (set in __post_init__)
    increment: float = field(init=False)
    current_lo: float = field(init=False)
    current_hi: float = field(init=False)

    def __post_init__(self):
        """
        increment, current_lo, and current_hi must be set conditionally, and after input validation
        """
        expand_hi = self.max_hi - self.initial_hi
        expand_lo = self.initial_lo - self.max_lo

        if abs(expand_hi - expand_lo) > 1e-6:
            raise ValueError(
                f"Range '{self.name}': asymmetric expansion. "
                f"hi expands by {expand_hi:.6f}, lo expands by {expand_lo:.6f}. "
                f"These must be equal."
            )

        if self.n_stages <= 0:
            raise ValueError(f"n_stages must be positive, got {self.n_stages}")

        self.increment = expand_hi / self.n_stages
        self.current_lo = self.initial_lo
        self.current_hi = self.initial_hi

    def expand(self) -> bool:
        """Expand range by one increment. Returns True if range actually changed."""
        if self.is_at_max():
            return False

        self.current_lo = max(self.current_lo - self.increment, self.max_lo)
        self.current_hi = min(self.current_hi + self.increment, self.max_hi)
        return True

    def get_range(self) -> list[float]:
        return [self.current_lo, self.current_hi]

    def is_at_max(self) -> bool:
        return abs(self.current_lo - self.max_lo) < 1e-9 and abs(self.current_hi - self.max_hi) < 1e-9


class CurriculumLearningCallback(BaseCallback):
    """
    SB3 callback that implements curriculum learning for recovery training.

    Tracks a rolling episode success rate and expands recovery state ranges
    when the agent reaches the success threshold.
    """

    def __init__(
        self,
        v_range: CurriculumRange,  # range of velocity values
        beta_range: CurriculumRange,  # range of sideslip values
        r_range: CurriculumRange,  # range of yaw rate value
        yaw_range: CurriculumRange,  # range of yaw (heading error) values
        n_stages: int,  # number of curriculum stages to traverse the ranges
        window_size: int = 500,  # number of episodes across which to evaluate success
        success_threshold: float = 0.8,  # threshold success to progress to next stage
        min_episodes_between_expansions: int = 1000,  # hysteresis: min episodes to wait per stage
        max_curriculum_timestep: int | None = None,  # maximum timesteps for learning
        log_freq: int = 10_000,  # frequency of logging
        verbose: int = 0,  # verbosity
    ):
        super().__init__(verbose)
        self.ranges = {
            "v": v_range,
            "beta": beta_range,
            "r": r_range,
            "yaw": yaw_range,
        }
        self.n_stages = n_stages
        self.window_size = window_size
        self.success_threshold = success_threshold
        if min_episodes_between_expansions < window_size:
            raise ValueError(
                f"min_episodes_between_expansions ({min_episodes_between_expansions}) must be >= "
                f"window_size ({window_size}), otherwise the window may not be full at expansion time."
            )

        self.min_episodes_between_expansions = min_episodes_between_expansions
        self.max_curriculum_timestep = max_curriculum_timestep
        self.log_freq = log_freq

        self.success_window: deque[bool] = deque(
            maxlen=window_size
        )  # creates a sliding window of max length window_size
        self.current_stage = 0
        self.episodes_since_expansion = 0
        self._last_log_timestep = 0

    def _on_training_start(self) -> None:
        """Push initial (narrow) ranges to all envs before the first rollout."""
        self._push_ranges_to_envs()
        self._log_expansion_event("initial")

    def _on_step(self) -> bool:
        dones = self.locals["dones"]  # contains an array of booleans for each env, indicated done status
        infos = self.locals[
            "infos"
        ]  # contains a dictionary for each env, with the "recovered" key indicating successful recovery

        # for done episodes, append recovered state to sliding window, and increment episodes since expansion
        for i, done in enumerate(dones):
            if done:
                recovered = infos[i].get("recovered", False)
                self.success_window.append(recovered)
                self.episodes_since_expansion += 1

        if self._should_expand():
            self._expand_ranges()

        if self.num_timesteps - self._last_log_timestep >= self.log_freq:
            self._log_metrics()
            self._last_log_timestep = self.num_timesteps

        return True

    def _should_expand(self) -> bool:
        # cannot expand past the final stage
        if self.current_stage >= self.n_stages:
            return False

        # hysteresis: min episodes must pass before next expansion
        if self.episodes_since_expansion < self.min_episodes_between_expansions:
            return False

        success_rate = sum(self.success_window) / len(self.success_window)

        # success_rate must exceed threshold
        if success_rate < self.success_threshold:
            return False

        # the maximum timestep must not be reached
        if self.max_curriculum_timestep is not None and self.num_timesteps > self.max_curriculum_timestep:
            return False

        return True

    def _expand_ranges(self) -> None:
        self.current_stage += 1

        for r in self.ranges.values():
            r.expand()

        self.success_window.clear()
        self.episodes_since_expansion = 0

        self._push_ranges_to_envs()
        self._log_expansion_event("expand")

    def _push_ranges_to_envs(self) -> None:
        self.training_env.env_method(
            "set_recovery_ranges",
            self.ranges["v"].get_range(),
            self.ranges["beta"].get_range(),
            self.ranges["r"].get_range(),
            self.ranges["yaw"].get_range(),
        )

    def _log_expansion_event(self, event: str) -> None:
        print(
            f"[Curriculum {event}] stage={self.current_stage}/{self.n_stages} "
            f"v={self.ranges['v'].get_range()} "
            f"beta={self.ranges['beta'].get_range()} "
            f"r={self.ranges['r'].get_range()} "
            f"yaw={self.ranges['yaw'].get_range()}"
        )

    def _log_metrics(self) -> None:
        success_rate = sum(self.success_window) / len(self.success_window) if self.success_window else 0.0

        metrics = {
            "curriculum/success_rate": success_rate,
            "curriculum/stage": self.current_stage,
            "curriculum/expansion_count": self.current_stage,
        }

        for name, r in self.ranges.items():
            metrics[f"curriculum/{name}_lo"] = r.current_lo
            metrics[f"curriculum/{name}_hi"] = r.current_hi

        wandb.log(metrics, step=self.num_timesteps)


def make_curriculum_callback(config: dict) -> CurriculumLearningCallback | None:
    """
    Factory that builds a CurriculumLearningCallback from a config dict.

    Returns None if curriculum is disabled.
    """
    if not config.get("enabled", False):
        return None

    n_stages = config["n_stages"]

    def _make_range(name: str, values: list[float]) -> CurriculumRange:
        initial_lo, initial_hi, max_lo, max_hi = values
        return CurriculumRange(
            name=name,
            initial_lo=initial_lo,
            initial_hi=initial_hi,
            max_lo=max_lo,
            max_hi=max_hi,
            n_stages=n_stages,
        )

    # Only forward keys present in config; missing keys fall back to __init__ defaults
    optional_keys = (
        "window_size",
        "success_threshold",
        "min_episodes_between_expansions",
        "max_curriculum_timestep",
        "log_freq",
    )
    kwargs = {k: config[k] for k in optional_keys if k in config}

    return CurriculumLearningCallback(
        v_range=_make_range("v", config["v_range"]),
        beta_range=_make_range("beta", config["beta_range"]),
        r_range=_make_range("r", config["r_range"]),
        yaw_range=_make_range("yaw", config["yaw_range"]),
        n_stages=n_stages,
        **kwargs,
    )
