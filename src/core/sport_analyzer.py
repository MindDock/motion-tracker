"""Base class for sport-specific motion analysis.

Provides a standard interface for analyzing sport movements, detecting
actions, generating corrections, and finding highlights. Subclass this
for each sport (badminton, tennis, basketball, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np

from .pose_estimator import PoseResult
from .angle_calculator import AngleCalculator
from .velocity_analyzer import VelocityAnalyzer
from .highlight_detector import HighlightDetector, SignalSource


@dataclass
class ActionTemplate:
    """Template defining a standard sport action for comparison."""
    name: str
    description: str
    key_joints: List[str]
    angle_sequences: Dict[str, List[float]] = field(default_factory=dict)
    ideal_angles: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # (joint, min_angle, max_angle) for correctness checks

    def set_from_recording(self, angle_history: List[Dict[str, Optional[float]]]):
        """Build template from recorded angle history."""
        if not angle_history:
            return
        all_joints = set()
        for angles in angle_history:
            all_joints.update(angles.keys())

        for joint in all_joints:
            seq = [a.get(joint) or 0.0 for a in angle_history]
            self.angle_sequences[joint] = seq


@dataclass
class CorrectionItem:
    """A single correction suggestion."""
    joint: str
    message: str
    severity: str  # 'info', 'warning', 'error'
    current_angle: Optional[float] = None
    ideal_range: Optional[Tuple[float, float]] = None
    deviation: float = 0.0


@dataclass
class ActionDetection:
    """Result of detecting a specific action."""
    action_name: str
    confidence: float
    frame_index: int
    timestamp_ms: float
    matched_template: Optional[str] = None


class SportAnalyzer(ABC):
    """Abstract base class for sport-specific analyzers."""

    def __init__(self, name: str):
        self.name = name
        self.calculator = AngleCalculator(use_3d=True)
        self.velocity_analyzer = VelocityAnalyzer()
        self.highlight_detector = HighlightDetector()
        self._templates: Dict[str, ActionTemplate] = {}
        self._action_history: List[ActionDetection] = []

        # Let subclasses configure
        self._setup_signals()
        self._setup_templates()

    @abstractmethod
    def _setup_signals(self):
        """Register highlight signal sources. Called during __init__."""
        pass

    @abstractmethod
    def _setup_templates(self):
        """Register action templates. Called during __init__."""
        pass

    @abstractmethod
    def detect_action(
        self,
        pose_result: PoseResult,
        frame_index: int,
        timestamp_ms: float,
    ) -> Optional[ActionDetection]:
        """Detect which action is being performed in the current frame."""
        pass

    @abstractmethod
    def generate_corrections(
        self,
        pose_result: PoseResult,
        action_name: Optional[str] = None,
    ) -> List[CorrectionItem]:
        """Generate correction suggestions for the current pose."""
        pass

    def register_template(self, template: ActionTemplate):
        """Register an action template."""
        self._templates[template.name] = template

    def get_template(self, name: str) -> Optional[ActionTemplate]:
        return self._templates.get(name)

    def list_templates(self) -> List[str]:
        return list(self._templates.keys())

    def update(
        self,
        pose_result: PoseResult,
        frame_index: int,
        timestamp_ms: float,
    ):
        """Process one frame through all analyzers."""
        if pose_result is None or not pose_result.is_valid():
            return

        self.velocity_analyzer.update(pose_result, timestamp_ms)
        self.highlight_detector.update(
            pose_result, self.velocity_analyzer, frame_index, timestamp_ms
        )

        action = self.detect_action(pose_result, frame_index, timestamp_ms)
        if action is not None:
            self._action_history.append(action)

    def get_highlights(self, top_n: int = 10):
        return self.highlight_detector.get_highlights(top_n=top_n)

    def get_action_history(self) -> List[ActionDetection]:
        return list(self._action_history)

    def get_summary(self) -> Dict[str, Any]:
        """Get analysis summary."""
        action_counts: Dict[str, int] = {}
        for action in self._action_history:
            action_counts[action.action_name] = action_counts.get(action.action_name, 0) + 1

        return {
            'sport': self.name,
            'total_actions_detected': len(self._action_history),
            'action_counts': action_counts,
            'highlights_count': len(self.highlight_detector.all_candidates),
        }

    def clear(self):
        self.velocity_analyzer.clear()
        self.highlight_detector.clear()
        self._action_history.clear()
