from __future__ import annotations

import logging
import pathlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import yaml

from .video_recorder import VideoRecorder

# default output directory for frames saved via EnvRenderer.save_frame(), relative to the cwd
SCREENSHOT_DIR = pathlib.Path("figures") / "frames"
# default output directory for videos recorded via EnvRenderer.start_recording(), relative to the cwd
VIDEO_DIR = pathlib.Path("figures") / "videos"
# render window title, suffixed with a recording indicator while a video is recorded
WINDOW_TITLE = "Gym-Khana"


@dataclass
class RenderSpec:
    window_size: int
    zoom_in_factor: float
    focus_on: str
    car_tickness: int
    show_wheels: bool
    show_info: Optional[bool] = True
    show_ctr_debug: Optional[bool] = False
    show_obs_debug: Optional[bool] = False
    vehicle_palette: Optional[list[str]] = None
    render_type: Optional[str] = "pygame"
    screenshot_scale: Optional[int] = 4
    video_fps: Optional[int] = 30
    video_scale: Optional[int] = 1

    def __init__(
        self,
        window_size: int = 800,
        focus_on: str = None,
        zoom_in_factor: float = 1.0,
        car_tickness: int = 1,
        show_wheels: bool = False,
        show_info: bool = True,
        show_ctr_debug: bool = False,
        show_obs_debug: bool = False,
        vehicle_palette: list[str] = None,
        render_type: str = "pygame",
        screenshot_scale: int = 4,
        video_fps: int = 30,
        video_scale: int = 1,
    ) -> None:
        """
        Initialize rendering specification.

        Parameters
        ----------
        window_size : int, optional
            size of the square window, by default 800
        focus_on : str, optional
            focus on a specific vehicle, by default None
        zoom_in_factor : float, optional
            zoom in factor, by default 1.0 (no zoom)
        car_tickness : int, optional
            thickness of the car in pixels, by default 1
        show_wheels : bool, optional
            toggle rendering of line segments for wheels, by default False
        show_info : bool, optional
            toggle rendering of text instructions, by default True
        vehicle_palette : list, optional
            list of colors for rendering vehicles according to their id, by default None
        screenshot_scale : int, optional
            resolution multiplier over the window size for frames saved via
            :meth:`EnvRenderer.save_frame`, by default 4
        video_fps : int, optional
            playback frames per second of videos recorded via :meth:`EnvRenderer.start_recording`.
            Frames are sampled in sim time, so playback is real time. By default 30
        video_scale : int, optional
            resolution multiplier over the window size for recorded videos, by default 1
        """
        self.window_size = window_size
        self.focus_on = focus_on
        self.zoom_in_factor = zoom_in_factor
        self.car_tickness = car_tickness
        self.show_wheels = show_wheels
        self.show_info = show_info
        self.show_ctr_debug = show_ctr_debug
        self.show_obs_debug = show_obs_debug
        self.vehicle_palette = vehicle_palette or ["#984ea3"]
        self.render_type = render_type
        self.screenshot_scale = screenshot_scale
        self.video_fps = video_fps
        self.video_scale = video_scale

    @staticmethod
    def from_yaml(yaml_file: str | pathlib.Path, overrides: Optional[dict] = None):
        """
        Load rendering specification from a yaml file, optionally overriding fields.

        Parameters
        ----------
        yaml_file : str | pathlib.Path
            path to the yaml file
        overrides : dict, optional
            dict of field values that take precedence over the yaml contents

        Returns
        -------
        RenderSpec
            rendering specification object
        """
        with open(yaml_file, "r") as yaml_stream:
            try:
                config = yaml.safe_load(yaml_stream)
            except yaml.YAMLError as ex:
                raise ValueError(f"Failed to parse render config YAML at {yaml_file}: {ex}") from ex
        if overrides:
            config.update(overrides)
        return RenderSpec(**config)


class EnvRenderer(ABC):
    """
    Abstract class for rendering the environment.
    """

    # whether save_frame() and recorded videos can render above the on-screen resolution
    supports_supersampling: bool = True

    def __init__(self) -> None:
        # video recording state, see start_recording()
        self._recorder: Optional[VideoRecorder] = None
        self._last_video_frame_idx: Optional[int] = None

    @abstractmethod
    def update(self, state: Any) -> None:
        """
        Update the state to be rendered.
        This is called at every rendering call.

        Parameters
        ----------
        state : Any
            state to be rendered, e.g. a list of vehicle states
        """
        raise NotImplementedError()

    @abstractmethod
    def render(self):
        """
        Render the current state in a frame.
        """
        raise NotImplementedError()

    @abstractmethod
    def render_lines(
        self,
        points: list | np.ndarray,
        color: Optional[tuple[int, int, int]] = (0, 0, 255),
        size: Optional[int] = 1,
    ):
        """
        Render a sequence of lines segments.

        Parameters
        ----------
        points : list | np.ndarray
            list of points to render
        color : tuple[int, int, int], optional
            color as rgb tuple, by default blue (0, 0, 255)
        size : int, optional
            size of the line, by default 1
        """
        raise NotImplementedError()

    @abstractmethod
    def render_closed_lines(
        self,
        points: list | np.ndarray,
        color: Optional[tuple[int, int, int]] = (0, 0, 255),
        size: Optional[int] = 1,
    ):
        """
        Render a closed loop of lines (draw a line between the last and the first point).

        Parameters
        ----------
        points : list | np.ndarray
            list of points to render
        color : tuple[int, int, int], optional
            color as rgb tuple, by default blue (0, 0, 255)
        size : int, optional
            size of the line, by default 1
        """
        raise NotImplementedError()

    @abstractmethod
    def render_text(
        self,
        text: str,
        position: tuple[float, float],
        color: Optional[tuple[int, int, int]] = (255, 255, 255),
        font_size: Optional[int] = 12,
        anchor: Optional[str] = "center",
    ):
        """
        Render text at world coordinates.

        Parameters
        ----------
        text : str
            text string to render
        position : tuple[float, float]
            world coordinate position (x, y) for text placement
        color : tuple[int, int, int], optional
            RGB color tuple, by default white (255, 255, 255)
        font_size : int, optional
            font size in points, by default 12
        anchor : str, optional
            text anchor point ('center', 'left', 'right'), by default 'center'
        """
        raise NotImplementedError()

    def save_frame(self, path: Optional[str] = None, scale: Optional[int] = None) -> str:
        """
        Save the currently rendered frame to an image file.

        Parameters
        ----------
        path : str, optional
            output file path. By default a timestamped png under ``figures/frames/``.
        scale : int, optional
            resolution multiplier over the on-screen window size. By default the
            ``screenshot_scale`` field of the rendering spec.

        Returns
        -------
        str
            path of the written file
        """
        raise NotImplementedError()

    def resolve_frame_path(self, path: Optional[str] = None) -> pathlib.Path:
        """
        Build the output path for :meth:`save_frame`, creating parent directories.

        See :meth:`resolve_output_path` for the naming and collision rules.

        Parameters
        ----------
        path : str, optional
            requested output path, or None to use the default location and name

        Returns
        -------
        pathlib.Path
            a path that does not yet exist, with its parent directory created
        """
        return self.resolve_output_path(path, default_dir=SCREENSHOT_DIR, prefix="frame", suffix=".png")

    def resolve_output_path(
        self, path: Optional[str], default_dir: pathlib.Path, prefix: str, suffix: str
    ) -> pathlib.Path:
        """
        Build an output file path, creating parent directories.

        A default name is derived from the current sim time. Existing files are never
        overwritten: a ``-1``, ``-2``, ... suffix is appended on collision, so files
        captured at the same sim time (e.g. across episode resets) do not clobber each other.

        Parameters
        ----------
        path : str, optional
            requested output path, or None to use the default location and name
        default_dir : pathlib.Path
            directory used when `path` is None
        prefix : str
            file name prefix used when `path` is None
        suffix : str
            file extension used when `path` is None

        Returns
        -------
        pathlib.Path
            a path that does not yet exist, with its parent directory created
        """
        if path is not None:
            out_path = pathlib.Path(path)
        else:
            # sim_time is None until the first update() call
            sim_time = getattr(self, "sim_time", None) or 0.0
            out_path = default_dir / f"{prefix}_t{sim_time:07.2f}s{suffix}"

        out_path.parent.mkdir(parents=True, exist_ok=True)

        stem, suffix, index = out_path.stem, out_path.suffix, 1
        while out_path.exists():
            out_path = out_path.with_name(f"{stem}-{index}{suffix}")
            index += 1

        return out_path

    @property
    def is_recording(self) -> bool:
        """Whether a video is currently being recorded."""
        return self._recorder is not None

    def start_recording(self, path: Optional[str] = None) -> str:
        """
        Start recording the rendered frames to a video file.

        Frames are sampled in sim time at the ``video_fps`` field of the rendering spec, so the
        video plays back in real time regardless of the timestep or render mode. The recording
        continues across episode resets until :meth:`stop_recording` is called.

        Parameters
        ----------
        path : str, optional
            output file path. By default a timestamped mp4 under ``figures/videos/``.

        Returns
        -------
        str
            path of the video being recorded
        """
        if self.is_recording:
            logging.warning(f"Already recording to {self._recorder.path}")
            return str(self._recorder.path)

        if not self.supports_supersampling and self.render_spec.video_scale not in (None, 1):
            logging.warning("This renderer cannot supersample; recording the video at window resolution instead.")

        out_path = self.resolve_output_path(path, default_dir=VIDEO_DIR, prefix="video", suffix=".mp4")
        self._recorder = VideoRecorder(out_path, fps=self.render_spec.video_fps)
        self._last_video_frame_idx = None  # capture the next rendered frame
        self._set_window_title(f"{WINDOW_TITLE} ● REC")

        print(f"Recording video to {out_path.resolve()}")
        return str(out_path)

    def stop_recording(self) -> Optional[str]:
        """
        Stop recording and finalize the video file.

        Returns
        -------
        Optional[str]
            path of the written video, or None if not recording or no frame was recorded
        """
        if not self.is_recording:
            return None

        recorder, self._recorder = self._recorder, None
        self._set_window_title(WINDOW_TITLE)
        out_path = recorder.close()

        if out_path is not None:
            duration = recorder.frame_count / recorder.fps
            print(f"Saved {recorder.frame_count}-frame ({duration:.2f}s) video to {pathlib.Path(out_path).resolve()}")
        return out_path

    def toggle_recording(self) -> Optional[str]:
        """
        Start recording if not recording, stop otherwise.

        Returns
        -------
        Optional[str]
            path of the video being recorded or just written
        """
        if self.is_recording:
            return self.stop_recording()
        return self.start_recording()

    def _record_frame_if_due(self) -> None:
        """
        Write the current frame to the video if recording and a new video frame slot has started.

        Sim time is divided into ``1 / video_fps`` slots and at most one frame is written per slot,
        so an episode reset (sim time jumping back) or rendering slower than the video fps needs no
        special handling. Called by the backends at the end of every render() call. Errors stop the
        recording (keeping the frames written so far) instead of interrupting the simulation.
        """
        if not self.is_recording or self.sim_time is None or not self.draw_flag:
            return

        try:
            frame_idx = int(self.sim_time * self._recorder.fps + 1e-9)
            if frame_idx != self._last_video_frame_idx:
                self._recorder.write(self._capture_frame())
                self._last_video_frame_idx = frame_idx
        except Exception as ex:
            logging.error(f"Failed to record video frame, stopping the recording: {ex}")
            self.stop_recording()

    def _capture_frame(self) -> np.ndarray:
        """
        Capture the current window for video recording.

        Returns
        -------
        np.ndarray
            BGR image of shape (H, W, 3) and dtype uint8
        """
        raise NotImplementedError()

    def _set_window_title(self, title: str) -> None:
        """
        Set the render window title, used for the recording indicator since it is not part of the frames.

        Parameters
        ----------
        title : str
            window title
        """
        pass

    @staticmethod
    def _run_key_action(action: Callable[[], Any], description: str) -> None:
        """
        Run a key-bound action, logging instead of raising on failure.

        A failed save or recording must never interrupt the simulation loop, and an unhandled
        exception in a Qt event handler aborts the process.

        Parameters
        ----------
        action : Callable[[], Any]
            action to run
        description : str
            what the action does, for the error message (e.g. "save frame")
        """
        try:
            action()
        except Exception as ex:
            logging.error(f"Failed to {description}: {ex}")

    @abstractmethod
    def close(self):
        """
        Close the rendering window.
        """
        raise NotImplementedError()
