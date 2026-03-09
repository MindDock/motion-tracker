"""Angular velocity and acceleration analysis for motion dynamics."""

from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np

from .pose_estimator import PoseResult
from .angle_calculator import AngleCalculator


class VelocityAnalyzer:
    """Compute angular velocity and acceleration from pose sequences.

    Tracks per-joint angle changes over time to detect fast movements,
    explosive actions, and motion intensity.
    """

    def __init__(
        self,
        buffer_size: int = 60,
        fps: float = 30.0,
        smoothing_window: int = 3,
    ):
        self.buffer_size = buffer_size
        self.fps = fps
        self.smoothing_window = smoothing_window
        self.calculator = AngleCalculator(use_3d=True)

        self._angle_buffers: Dict[str, deque] = {}
        self._time_buffer: deque = deque(maxlen=buffer_size)
        self._frame_count = 0

    def set_fps(self, fps: float):
        self.fps = fps

    def update(self, pose_result: PoseResult, timestamp_ms: Optional[float] = None):
        """Add a new frame's pose data."""
        if timestamp_ms is None:
            timestamp_ms = self._frame_count * (1000.0 / self.fps)

        self._time_buffer.append(timestamp_ms)
        self._frame_count += 1

        angles = self.calculator.calculate_all_angles(pose_result)
        for joint, angle in angles.items():
            if angle is not None:
                if joint not in self._angle_buffers:
                    self._angle_buffers[joint] = deque(maxlen=self.buffer_size)
                self._angle_buffers[joint].append(angle)

    def get_angular_velocity(self, joint: str) -> Optional[float]:
        """Get current angular velocity in degrees/second."""
        buf = self._angle_buffers.get(joint)
        if buf is None or len(buf) < 2:
            return None

        dt = (1.0 / self.fps) if self.fps > 0 else 1.0 / 30.0
        # Use smoothed finite difference
        window = min(self.smoothing_window, len(buf) - 1)
        angles = list(buf)
        diffs = [angles[-(i)] - angles[-(i + 1)] for i in range(1, window + 1)]
        avg_diff = np.mean(diffs)
        return float(avg_diff / dt)

    def get_angular_acceleration(self, joint: str) -> Optional[float]:
        """Get current angular acceleration in degrees/second^2."""
        buf = self._angle_buffers.get(joint)
        if buf is None or len(buf) < 3:
            return None

        dt = (1.0 / self.fps) if self.fps > 0 else 1.0 / 30.0
        angles = list(buf)
        # Second derivative via central difference
        v1 = angles[-1] - angles[-2]
        v0 = angles[-2] - angles[-3]
        return float((v1 - v0) / (dt * dt))

    def get_all_velocities(self) -> Dict[str, float]:
        """Get angular velocities for all tracked joints."""
        result = {}
        for joint in self._angle_buffers:
            v = self.get_angular_velocity(joint)
            if v is not None:
                result[joint] = v
        return result

    def get_all_accelerations(self) -> Dict[str, float]:
        """Get angular accelerations for all tracked joints."""
        result = {}
        for joint in self._angle_buffers:
            a = self.get_angular_acceleration(joint)
            if a is not None:
                result[joint] = a
        return result

    def get_motion_intensity(self) -> float:
        """Get overall motion intensity (RMS of all angular velocities)."""
        velocities = self.get_all_velocities()
        if not velocities:
            return 0.0
        return float(np.sqrt(np.mean([v ** 2 for v in velocities.values()])))

    def get_peak_joint(self) -> Optional[Tuple[str, float]]:
        """Get the joint with the highest absolute angular velocity."""
        velocities = self.get_all_velocities()
        if not velocities:
            return None
        joint = max(velocities, key=lambda k: abs(velocities[k]))
        return joint, velocities[joint]

    def get_keypoint_velocity(
        self,
        pose_history: List[PoseResult],
        keypoint_name: str,
    ) -> Optional[float]:
        """Calculate linear velocity of a keypoint using world coordinates."""
        if len(pose_history) < 2:
            return None

        curr = pose_history[-1].get_keypoint(keypoint_name)
        prev = pose_history[-2].get_keypoint(keypoint_name)

        if curr is None or prev is None:
            return None

        curr_w = curr.world_coords()
        prev_w = prev.world_coords()
        if curr_w is None or prev_w is None:
            return None

        dt = 1.0 / self.fps
        displacement = np.linalg.norm(curr_w - prev_w)
        return float(displacement / dt)

    def clear(self):
        self._angle_buffers.clear()
        self._time_buffer.clear()
        self._frame_count = 0
