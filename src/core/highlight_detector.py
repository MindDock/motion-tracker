"""Pluggable highlight detection framework.

Detects exciting moments in motion sequences by combining multiple
signal sources. Each signal contributes a score; frames with combined
scores above a threshold become highlight candidates.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
import numpy as np

from .pose_estimator import PoseResult
from .velocity_analyzer import VelocityAnalyzer


@dataclass
class HighlightCandidate:
    """A candidate highlight moment."""
    frame_index: int
    timestamp_ms: float
    score: float
    signals: Dict[str, float] = field(default_factory=dict)
    label: str = ""

    @property
    def timestamp_sec(self) -> float:
        return self.timestamp_ms / 1000.0


class SignalSource(ABC):
    """Abstract signal source for highlight detection.

    Subclass this to create custom signals (e.g., explosive motion,
    jump detection, specific action patterns).
    """

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    @abstractmethod
    def compute(
        self,
        pose_result: PoseResult,
        velocity_analyzer: VelocityAnalyzer,
        frame_index: int,
        timestamp_ms: float,
    ) -> float:
        """Compute signal strength for current frame.

        Returns:
            Signal strength (0.0 = no signal, 1.0 = maximum signal)
        """
        pass


class MotionIntensitySignal(SignalSource):
    """Detects high overall motion intensity."""

    def __init__(self, threshold: float = 300.0, weight: float = 1.0):
        super().__init__("motion_intensity", weight)
        self.threshold = threshold

    def compute(self, pose_result, velocity_analyzer, frame_index, timestamp_ms) -> float:
        intensity = velocity_analyzer.get_motion_intensity()
        return min(1.0, intensity / self.threshold)


class ExplosiveMotionSignal(SignalSource):
    """Detects explosive/sudden motion via peak angular acceleration."""

    def __init__(self, threshold: float = 5000.0, weight: float = 1.5):
        super().__init__("explosive_motion", weight)
        self.threshold = threshold

    def compute(self, pose_result, velocity_analyzer, frame_index, timestamp_ms) -> float:
        accels = velocity_analyzer.get_all_accelerations()
        if not accels:
            return 0.0
        peak = max(abs(a) for a in accels.values())
        return min(1.0, peak / self.threshold)


class PostureChangeSignal(SignalSource):
    """Detects large posture changes (e.g., jumps, lunges)."""

    def __init__(self, threshold: float = 500.0, weight: float = 1.0):
        super().__init__("posture_change", weight)
        self.threshold = threshold
        self._prev_hip_y: Optional[float] = None

    def compute(self, pose_result, velocity_analyzer, frame_index, timestamp_ms) -> float:
        if pose_result is None:
            return 0.0

        left_hip = pose_result.get_keypoint('left_hip')
        right_hip = pose_result.get_keypoint('right_hip')
        if left_hip is None or right_hip is None:
            return 0.0

        hip_y = (left_hip.y + right_hip.y) / 2
        if self._prev_hip_y is None:
            self._prev_hip_y = hip_y
            return 0.0

        delta = abs(hip_y - self._prev_hip_y) * 10000  # scale up normalized coords
        self._prev_hip_y = hip_y
        return min(1.0, delta / self.threshold)


class ArmSwingSignal(SignalSource):
    """Detects fast arm swing (key for racket sports)."""

    def __init__(self, threshold: float = 600.0, weight: float = 2.0):
        super().__init__("arm_swing", weight)
        self.threshold = threshold

    def compute(self, pose_result, velocity_analyzer, frame_index, timestamp_ms) -> float:
        arm_joints = ['left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder']
        velocities = []
        for joint in arm_joints:
            v = velocity_analyzer.get_angular_velocity(joint)
            if v is not None:
                velocities.append(abs(v))
        if not velocities:
            return 0.0
        peak = max(velocities)
        return min(1.0, peak / self.threshold)


class HighlightDetector:
    """Detect highlights by combining pluggable signal sources.

    Usage:
        detector = HighlightDetector()
        detector.add_signal(MotionIntensitySignal())
        detector.add_signal(ArmSwingSignal())

        # Per frame:
        detector.update(pose_result, velocity_analyzer, frame_idx, ts)

        # After processing:
        highlights = detector.get_highlights(top_n=10)
    """

    def __init__(
        self,
        score_threshold: float = 0.5,
        min_gap_frames: int = 30,
        context_frames: int = 15,
    ):
        self.score_threshold = score_threshold
        self.min_gap_frames = min_gap_frames
        self.context_frames = context_frames
        self._signals: List[SignalSource] = []
        self._candidates: List[HighlightCandidate] = []

    def add_signal(self, signal: SignalSource):
        """Register a signal source."""
        self._signals.append(signal)

    def update(
        self,
        pose_result: PoseResult,
        velocity_analyzer: VelocityAnalyzer,
        frame_index: int,
        timestamp_ms: float,
    ):
        """Process one frame and record candidate if score is high enough."""
        if pose_result is None:
            return

        signal_values = {}
        total_weight = 0.0
        weighted_sum = 0.0

        for signal in self._signals:
            value = signal.compute(
                pose_result, velocity_analyzer, frame_index, timestamp_ms
            )
            signal_values[signal.name] = value
            weighted_sum += value * signal.weight
            total_weight += signal.weight

        score = weighted_sum / total_weight if total_weight > 0 else 0.0

        if score >= self.score_threshold:
            self._candidates.append(HighlightCandidate(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                score=score,
                signals=signal_values,
            ))

    def get_highlights(
        self,
        top_n: int = 10,
        label_fn: Optional[callable] = None,
    ) -> List[HighlightCandidate]:
        """Get top-N highlights with non-maximum suppression.

        Args:
            top_n: Maximum number of highlights to return
            label_fn: Optional function to label highlights based on signals

        Returns:
            List of highlight candidates sorted by score descending
        """
        if not self._candidates:
            return []

        # Sort by score descending
        sorted_candidates = sorted(self._candidates, key=lambda c: c.score, reverse=True)

        # Non-maximum suppression: remove candidates too close together
        selected = []
        for candidate in sorted_candidates:
            if len(selected) >= top_n:
                break
            too_close = any(
                abs(candidate.frame_index - s.frame_index) < self.min_gap_frames
                for s in selected
            )
            if not too_close:
                if label_fn:
                    candidate.label = label_fn(candidate)
                selected.append(candidate)

        return sorted(selected, key=lambda c: c.frame_index)

    def clear(self):
        self._candidates.clear()

    @property
    def all_candidates(self) -> List[HighlightCandidate]:
        return list(self._candidates)
