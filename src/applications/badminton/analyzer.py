"""Badminton-specific motion analyzer.

Implements the SportAnalyzer interface for badminton, combining
action detection, highlight detection, and motion correction.
"""

from typing import Dict, List, Optional
from collections import deque

from ...core.pose_estimator import PoseResult
from ...core.sport_analyzer import (
    SportAnalyzer, ActionDetection, CorrectionItem, ActionTemplate,
)
from ...core.highlight_detector import (
    MotionIntensitySignal,
    ExplosiveMotionSignal,
    ArmSwingSignal,
    PostureChangeSignal,
)
from ...core.analysis_logger import AnalysisLogger
from .actions import BadmintonActions
from .correction import CorrectionEngine


class SmashDetectionSignal:
    """Custom signal tuned for badminton smash detection."""

    def __init__(self, weight: float = 2.5):
        from ...core.highlight_detector import SignalSource
        self._base = type(
            'SmashSignal',
            (SignalSource,),
            {'compute': self._compute, '__init__': lambda s, **kw: SignalSource.__init__(s, 'smash_detection', weight)},
        )()

    def _compute(self, pose_result, velocity_analyzer, frame_index, timestamp_ms) -> float:
        # Smash = high arm velocity + shoulder angle > 130 + explosive accel
        arm_v = velocity_analyzer.get_angular_velocity('right_elbow')
        shoulder_v = velocity_analyzer.get_angular_velocity('right_shoulder')

        if arm_v is None or shoulder_v is None:
            return 0.0

        # Combined arm speed
        speed = abs(arm_v) + abs(shoulder_v)

        # Check shoulder is raised
        from ...core.angle_calculator import AngleCalculator
        calc = AngleCalculator(use_3d=True)
        shoulder_angle = calc.calculate_joint_angle(pose_result, 'right_shoulder')

        if shoulder_angle is not None and shoulder_angle > 120 and speed > 400:
            return min(1.0, speed / 800.0)

        return 0.0

    @property
    def signal(self):
        return self._base


class BadmintonAnalyzer(SportAnalyzer):
    """Badminton motion analyzer.

    Detects badminton strokes, identifies highlights (smashes, rallies),
    and provides per-frame motion correction feedback.

    Usage:
        analyzer = BadmintonAnalyzer()

        for frame_ctx in video_processor.process_frames():
            analyzer.update(frame_ctx.pose_result, frame_ctx.frame_index, frame_ctx.timestamp_ms)

            # Get real-time corrections
            corrections = analyzer.generate_corrections(frame_ctx.pose_result)

            # Get current detected action
            action = analyzer.current_action

        # After processing
        highlights = analyzer.get_highlights(top_n=10)
        summary = analyzer.get_summary()
    """

    def __init__(
        self,
        racket_hand: str = 'right',
        detection_cooldown: int = 15,
        min_action_velocity: float = 100.0,
        logger: Optional[AnalysisLogger] = None,
    ):
        self.racket_hand = racket_hand
        self.detection_cooldown = detection_cooldown
        self.min_action_velocity = min_action_velocity
        self.correction_engine = CorrectionEngine()
        self.logger = logger
        self._current_action: Optional[ActionDetection] = None
        self._frames_since_detection = 0
        self._recent_angles: deque = deque(maxlen=90)  # ~3 sec at 30fps
        self._last_logged_action: Optional[str] = None

        super().__init__(name='badminton')

    def _setup_signals(self):
        """Configure highlight signals tuned for badminton."""
        self.highlight_detector.add_signal(
            ArmSwingSignal(threshold=400.0, weight=2.0)
        )
        self.highlight_detector.add_signal(
            ExplosiveMotionSignal(threshold=3000.0, weight=1.5)
        )
        self.highlight_detector.add_signal(
            MotionIntensitySignal(threshold=250.0, weight=1.0)
        )
        self.highlight_detector.add_signal(
            PostureChangeSignal(threshold=300.0, weight=0.8)
        )

        # Smash-specific signal
        smash_sig = SmashDetectionSignal(weight=2.5)
        self.highlight_detector.add_signal(smash_sig.signal)

    def _setup_templates(self):
        """Load all badminton action templates."""
        for name, template in BadmintonActions.all_templates().items():
            self.register_template(template)

    def detect_action(
        self,
        pose_result: PoseResult,
        frame_index: int,
        timestamp_ms: float,
    ) -> Optional[ActionDetection]:
        """Detect badminton action in current frame."""
        self._frames_since_detection += 1

        # Cooldown to avoid rapid-fire detections
        if self._frames_since_detection < self.detection_cooldown:
            return None

        # Get current angles
        angles = self.calculator.calculate_all_angles(pose_result)
        self._recent_angles.append(angles)

        # Get arm velocity for action classification
        arm_velocity = 0.0
        for joint in BadmintonActions.RACKET_ARM_JOINTS:
            v = self.velocity_analyzer.get_angular_velocity(joint)
            if v is not None:
                arm_velocity = max(arm_velocity, abs(v))

        # Only try detection when there's meaningful movement
        if arm_velocity < self.min_action_velocity:
            # Check if it's a ready stance (low velocity expected)
            result = BadmintonActions.detect_action_type(
                angles, arm_velocity, {'ready_stance': self._templates.get('ready_stance')}
            )
            if result:
                name, confidence = result
                self._current_action = ActionDetection(
                    action_name=name,
                    confidence=confidence,
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    matched_template=name,
                )
                return self._current_action
            return None

        # Detect active strokes
        result = BadmintonActions.detect_action_type(
            angles, arm_velocity, self._templates
        )

        if result is None:
            return None

        name, confidence = result
        self._frames_since_detection = 0

        detection = ActionDetection(
            action_name=name,
            confidence=confidence,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            matched_template=name,
        )
        self._current_action = detection
        return detection

    def generate_corrections(
        self,
        pose_result: PoseResult,
        action_name: Optional[str] = None,
    ) -> List[CorrectionItem]:
        """Generate corrections for current pose.

        If action_name is provided, compares against that template.
        Otherwise uses the most recently detected action.
        """
        if action_name is None and self._current_action is not None:
            action_name = self._current_action.action_name

        if action_name is None:
            return []

        template = self.get_template(action_name)
        if template is None:
            return []

        return self.correction_engine.analyze(pose_result, template)

    def get_sequence_report(
        self,
        action_name: str,
    ) -> Optional[Dict]:
        """Generate a report comparing recent frames to a template."""
        template = self.get_template(action_name)
        if template is None or not self._recent_angles:
            return None

        return self.correction_engine.compare_sequences(
            list(self._recent_angles), template
        )

    def update(self, pose_result, frame_index: int, timestamp_ms: float):
        """Process one frame and emit log events."""
        super().update(pose_result, frame_index, timestamp_ms)

        if self.logger is None or pose_result is None or not pose_result.is_valid():
            return

        # Log per-frame metrics at configured interval
        angles = self.calculator.calculate_all_angles(pose_result)
        intensity = self.velocity_analyzer.get_motion_intensity()
        self.logger.log_frame(frame_index, timestamp_ms, angles=angles, motion_intensity=intensity)

        # Log action detection (only when action changes)
        action = self._current_action
        if action is not None and action.action_name != self._last_logged_action:
            self.logger.log_action(
                frame_index, timestamp_ms,
                action=action.action_name,
                confidence=action.confidence,
            )
            self._last_logged_action = action.action_name

            # Log corrections for this action
            corrections = self.generate_corrections(pose_result, action.action_name)
            if corrections:
                self.logger.log_correction(frame_index, timestamp_ms, corrections)

        # Log new highlight candidates as they are detected
        candidates = self.highlight_detector.all_candidates
        if candidates:
            latest = candidates[-1]
            if latest.frame_index == frame_index:
                self.logger.log_highlight(
                    frame_index, timestamp_ms,
                    label=latest.label or "精彩瞬间",
                    score=latest.score,
                    signals=latest.signals,
                )

    @property
    def current_action(self) -> Optional[ActionDetection]:
        return self._current_action

    def get_highlights(self, top_n: int = 10):
        """Get highlights with badminton-specific labels."""
        def label_fn(candidate):
            signals = candidate.signals
            if signals.get('smash_detection', 0) > 0.5:
                return '扣杀'
            if signals.get('arm_swing', 0) > 0.7:
                return '大力击球'
            if signals.get('posture_change', 0) > 0.6:
                return '大幅移动'
            if signals.get('explosive_motion', 0) > 0.6:
                return '爆发动作'
            return '精彩瞬间'

        return self.highlight_detector.get_highlights(
            top_n=top_n, label_fn=label_fn
        )
