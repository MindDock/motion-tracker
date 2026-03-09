"""Unified video processing pipeline for files and camera input."""

from typing import Optional, Callable, List, Iterator, Tuple
from dataclasses import dataclass, field
import cv2
import numpy as np
import time

from .pose_estimator import PoseEstimator, PoseResult


@dataclass
class FrameContext:
    """Context for a processed frame."""
    frame_index: int
    timestamp_ms: float
    frame: np.ndarray
    pose_result: Optional[PoseResult]
    fps: float = 0.0

    @property
    def timestamp_sec(self) -> float:
        return self.timestamp_ms / 1000.0


class VideoSource:
    """Abstraction over video file or camera input."""

    def __init__(self, source=0, width: int = 1280, height: int = 720):
        self.source = source
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None

    @property
    def is_file(self) -> bool:
        return isinstance(self.source, str)

    @property
    def total_frames(self) -> int:
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def source_fps(self) -> float:
        if self._cap is None:
            return 30.0
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self.source)
        if not self.is_file:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return self._cap.isOpened()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None
        return self._cap.read()

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.release()


FrameCallback = Callable[[FrameContext], Optional[bool]]


class VideoProcessor:
    """Process video frames through a configurable pipeline.

    Supports both real-time camera and offline video file processing.
    Frame callbacks (hooks) are called in order for each frame.
    """

    def __init__(
        self,
        estimator: PoseEstimator,
        source: VideoSource,
        flip_camera: bool = True,
        skip_frames: int = 0,
    ):
        self.estimator = estimator
        self.source = source
        self.flip_camera = flip_camera
        self.skip_frames = skip_frames
        self._hooks: List[FrameCallback] = []

    def add_hook(self, callback: FrameCallback):
        """Add a frame processing hook. Hooks run in order per frame."""
        self._hooks.append(callback)

    def process_frames(self) -> Iterator[FrameContext]:
        """Iterate over processed frames. Yields FrameContext for each frame."""
        frame_index = 0
        prev_time = time.time()

        while True:
            ret, frame = self.source.read()
            if not ret or frame is None:
                break

            # Skip frames if configured (for faster offline processing)
            if self.skip_frames > 0 and frame_index % (self.skip_frames + 1) != 0:
                frame_index += 1
                continue

            # Flip for camera mirror effect
            if self.flip_camera and not self.source.is_file:
                frame = cv2.flip(frame, 1)

            # Pose estimation
            pose_result = self.estimator.process_frame(frame)

            # Calculate FPS
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            # Timestamp from video file or wall clock
            if self.source.is_file:
                timestamp_ms = frame_index * (1000.0 / self.source.source_fps)
            else:
                timestamp_ms = time.time() * 1000

            ctx = FrameContext(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                frame=frame,
                pose_result=pose_result,
                fps=fps,
            )

            # Run hooks
            stop = False
            for hook in self._hooks:
                result = hook(ctx)
                if result is False:
                    stop = True
                    break

            yield ctx

            if stop:
                break

            frame_index += 1

    def run_interactive(self, window_name: str = "Motion Tracker"):
        """Run with OpenCV window for interactive use."""
        for ctx in self.process_frames():
            cv2.imshow(window_name, ctx.frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()
