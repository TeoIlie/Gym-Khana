from __future__ import annotations

import logging
import pathlib
from typing import Optional

import cv2
import numpy as np


class VideoRecorder:
    """
    Stream BGR uint8 frames to a video file with cv2.VideoWriter (mp4v codec).

    The writer is opened lazily on the first frame, since the frame size is not known before.
    The file is only playable once :meth:`close` has been called.
    """

    def __init__(self, path: pathlib.Path, fps: int) -> None:
        """
        Initialize the video recorder.

        Parameters
        ----------
        path : pathlib.Path
            output video file path
        fps : int
            playback frames per second of the video
        """
        self.path = pathlib.Path(path)
        self.fps = int(fps)
        self.frame_count = 0
        self._writer: Optional[cv2.VideoWriter] = None
        self._size: Optional[tuple[int, int]] = None  # (width, height)

    def write(self, frame: np.ndarray) -> None:
        """
        Append a frame to the video.

        Parameters
        ----------
        frame : np.ndarray
            BGR image of shape (H, W, 3) and dtype uint8

        Raises
        ------
        RuntimeError
            if the video writer cannot be opened
        """
        # some codecs and players reject odd frame sizes, so round down to even
        height, width = frame.shape[:2]
        even_size = (width - width % 2, height - height % 2)

        if self._writer is None:
            self._size = even_size
            self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self._size)
            if not self._writer.isOpened():
                self._writer = None
                raise RuntimeError(f"Could not open a video writer for {self.path}")

        if even_size == self._size:
            frame = frame[: self._size[1], : self._size[0]]
        else:
            # the window was resized mid-recording, the writer needs a fixed frame size
            frame = cv2.resize(frame, dsize=self._size, interpolation=cv2.INTER_AREA)

        self._writer.write(np.ascontiguousarray(frame))
        self.frame_count += 1

    def close(self) -> Optional[str]:
        """
        Finalize the video file. Safe to call more than once.

        Returns
        -------
        Optional[str]
            path of the written video, or None if no frame was ever written
        """
        if self._writer is not None:
            self._writer.release()
            self._writer = None

        if self.frame_count == 0:
            self.path.unlink(missing_ok=True)
            logging.warning(f"No frames were recorded, {self.path} was not written.")
            return None

        return str(self.path)
