"""Badminton motion correction engine.

Compares player's current pose against ideal templates and generates
actionable correction suggestions.
"""

from typing import Dict, List, Optional, Tuple
from ...core.pose_estimator import PoseResult
from ...core.angle_calculator import AngleCalculator
from ...core.sport_analyzer import ActionTemplate, CorrectionItem


# Human-readable joint names for feedback
JOINT_DISPLAY_NAMES = {
    'right_shoulder': '持拍肩',
    'right_elbow': '持拍肘',
    'right_wrist': '持拍腕',
    'left_shoulder': '非持拍肩',
    'left_elbow': '非持拍肘',
    'left_wrist': '非持拍腕',
    'right_hip': '右髋',
    'left_hip': '左髋',
    'right_knee': '右膝',
    'left_knee': '左膝',
    'right_ankle': '右踝',
    'left_ankle': '左踝',
}

# Correction message templates
CORRECTION_MESSAGES = {
    'too_low': '{joint_name}角度过小({current:.0f}°)，应在{min:.0f}°-{max:.0f}°之间，请适当抬高',
    'too_high': '{joint_name}角度过大({current:.0f}°)，应在{min:.0f}°-{max:.0f}°之间，请适当降低',
}


class CorrectionEngine:
    """Generate correction suggestions by comparing pose to templates."""

    def __init__(
        self,
        warning_threshold: float = 10.0,
        error_threshold: float = 25.0,
    ):
        self.warning_threshold = warning_threshold
        self.error_threshold = error_threshold
        self.calculator = AngleCalculator(use_3d=True)

    def analyze(
        self,
        pose_result: PoseResult,
        template: ActionTemplate,
    ) -> List[CorrectionItem]:
        """Compare pose against template and generate corrections.

        Args:
            pose_result: Current pose
            template: Action template to compare against

        Returns:
            List of correction items sorted by severity
        """
        corrections = []
        angles = self.calculator.calculate_all_angles(pose_result)

        for joint, (min_angle, max_angle) in template.ideal_angles.items():
            current = angles.get(joint)
            if current is None:
                continue

            joint_name = JOINT_DISPLAY_NAMES.get(joint, joint)

            if current < min_angle:
                deviation = min_angle - current
                severity = self._classify_severity(deviation)
                message = CORRECTION_MESSAGES['too_low'].format(
                    joint_name=joint_name,
                    current=current,
                    min=min_angle,
                    max=max_angle,
                )
                corrections.append(CorrectionItem(
                    joint=joint,
                    message=message,
                    severity=severity,
                    current_angle=current,
                    ideal_range=(min_angle, max_angle),
                    deviation=deviation,
                ))
            elif current > max_angle:
                deviation = current - max_angle
                severity = self._classify_severity(deviation)
                message = CORRECTION_MESSAGES['too_high'].format(
                    joint_name=joint_name,
                    current=current,
                    min=min_angle,
                    max=max_angle,
                )
                corrections.append(CorrectionItem(
                    joint=joint,
                    message=message,
                    severity=severity,
                    current_angle=current,
                    ideal_range=(min_angle, max_angle),
                    deviation=deviation,
                ))

        # Sort by severity: error > warning > info
        severity_order = {'error': 0, 'warning': 1, 'info': 2}
        corrections.sort(key=lambda c: (severity_order.get(c.severity, 3), -c.deviation))

        return corrections

    def compare_sequences(
        self,
        recorded_angles: List[Dict[str, Optional[float]]],
        template: ActionTemplate,
    ) -> Dict[str, Dict]:
        """Compare a recorded angle sequence against template.

        Returns per-joint statistics showing average deviation and worst frame.
        """
        results = {}

        for joint, (min_a, max_a) in template.ideal_angles.items():
            deviations = []
            worst_frame = 0
            worst_dev = 0.0

            for i, frame_angles in enumerate(recorded_angles):
                angle = frame_angles.get(joint)
                if angle is None:
                    continue

                if angle < min_a:
                    dev = min_a - angle
                elif angle > max_a:
                    dev = angle - max_a
                else:
                    dev = 0.0

                deviations.append(dev)
                if dev > worst_dev:
                    worst_dev = dev
                    worst_frame = i

            if deviations:
                import numpy as np
                results[joint] = {
                    'avg_deviation': float(np.mean(deviations)),
                    'max_deviation': worst_dev,
                    'worst_frame': worst_frame,
                    'in_range_pct': sum(1 for d in deviations if d == 0) / len(deviations) * 100,
                    'joint_name': JOINT_DISPLAY_NAMES.get(joint, joint),
                }

        return results

    def _classify_severity(self, deviation: float) -> str:
        if deviation >= self.error_threshold:
            return 'error'
        elif deviation >= self.warning_threshold:
            return 'warning'
        return 'info'
