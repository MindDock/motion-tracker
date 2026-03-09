"""Badminton standard action templates and detection logic."""

from typing import Dict, List, Optional, Tuple
from ...core.sport_analyzer import ActionTemplate


class BadmintonActions:
    """Library of standard badminton action templates.

    Each template defines key joints and ideal angle ranges for a specific
    stroke or movement. Templates can be extended by recording real players.
    """

    # Key joints for badminton analysis
    RACKET_ARM_JOINTS = [
        'right_shoulder', 'right_elbow', 'right_wrist',
    ]
    NON_RACKET_ARM_JOINTS = [
        'left_shoulder', 'left_elbow', 'left_wrist',
    ]
    LOWER_BODY_JOINTS = [
        'left_hip', 'right_hip', 'left_knee', 'right_knee',
        'left_ankle', 'right_ankle',
    ]
    ALL_KEY_JOINTS = RACKET_ARM_JOINTS + NON_RACKET_ARM_JOINTS + LOWER_BODY_JOINTS

    @classmethod
    def overhead_clear(cls) -> ActionTemplate:
        """High clear / overhead stroke template."""
        return ActionTemplate(
            name='overhead_clear',
            description='高远球 - 将球高高击向对方后场的过顶击球',
            key_joints=cls.RACKET_ARM_JOINTS + ['right_hip', 'right_knee'],
            ideal_angles={
                # (min, max) angle range at the key moment
                'right_shoulder': (140.0, 180.0),  # arm raised high
                'right_elbow': (140.0, 180.0),     # arm extended
                'right_hip': (150.0, 180.0),       # body upright or slight lean back
                'right_knee': (150.0, 180.0),      # legs relatively straight
            },
        )

    @classmethod
    def smash(cls) -> ActionTemplate:
        """Smash stroke template."""
        return ActionTemplate(
            name='smash',
            description='扣杀 - 大力过顶下压击球',
            key_joints=cls.RACKET_ARM_JOINTS + ['right_hip', 'left_knee'],
            ideal_angles={
                'right_shoulder': (130.0, 180.0),  # arm high and forward
                'right_elbow': (90.0, 160.0),      # more bent than clear at contact
                'right_hip': (130.0, 170.0),       # body tilted forward
                'left_knee': (120.0, 170.0),       # front leg bracing
            },
        )

    @classmethod
    def forehand_drive(cls) -> ActionTemplate:
        """Forehand drive template."""
        return ActionTemplate(
            name='forehand_drive',
            description='正手平抽 - 身体中部高度的平击球',
            key_joints=cls.RACKET_ARM_JOINTS + ['right_hip'],
            ideal_angles={
                'right_shoulder': (60.0, 120.0),   # arm at side
                'right_elbow': (100.0, 160.0),     # slightly bent
                'right_hip': (150.0, 180.0),       # body upright
            },
        )

    @classmethod
    def backhand_drive(cls) -> ActionTemplate:
        """Backhand drive template."""
        return ActionTemplate(
            name='backhand_drive',
            description='反手平抽 - 非持拍侧的平击球',
            key_joints=cls.RACKET_ARM_JOINTS + ['left_hip'],
            ideal_angles={
                'right_shoulder': (40.0, 100.0),   # arm across body
                'right_elbow': (80.0, 140.0),      # bent
                'left_hip': (140.0, 180.0),        # body turned
            },
        )

    @classmethod
    def drop_shot(cls) -> ActionTemplate:
        """Drop shot template."""
        return ActionTemplate(
            name='drop_shot',
            description='吊球 - 轻柔的过顶击球，落点靠近球网',
            key_joints=cls.RACKET_ARM_JOINTS,
            ideal_angles={
                'right_shoulder': (130.0, 170.0),  # similar to clear
                'right_elbow': (120.0, 170.0),     # extended but controlled
            },
        )

    @classmethod
    def lunge_step(cls) -> ActionTemplate:
        """Forward lunge step (common footwork)."""
        return ActionTemplate(
            name='lunge_step',
            description='弓步上网 - 向前跨步接近网前球',
            key_joints=cls.LOWER_BODY_JOINTS,
            ideal_angles={
                'right_knee': (70.0, 110.0),      # front knee deeply bent
                'left_knee': (140.0, 180.0),      # back leg extended
                'right_hip': (70.0, 120.0),       # hip flexed
            },
        )

    @classmethod
    def ready_stance(cls) -> ActionTemplate:
        """Basic ready position."""
        return ActionTemplate(
            name='ready_stance',
            description='准备姿势 - 等待接球的中立站位',
            key_joints=cls.ALL_KEY_JOINTS,
            ideal_angles={
                'right_knee': (130.0, 160.0),     # slightly bent
                'left_knee': (130.0, 160.0),      # slightly bent
                'right_hip': (140.0, 170.0),      # slight forward lean
                'left_hip': (140.0, 170.0),
                'right_shoulder': (20.0, 60.0),   # arms in front
                'left_shoulder': (20.0, 60.0),
                'right_elbow': (90.0, 140.0),     # arms ready
                'left_elbow': (90.0, 140.0),
            },
        )

    @classmethod
    def serve_forehand(cls) -> ActionTemplate:
        """Forehand serve template."""
        return ActionTemplate(
            name='serve_forehand',
            description='正手发球 - 下手发球',
            key_joints=cls.RACKET_ARM_JOINTS + ['right_hip', 'right_knee'],
            ideal_angles={
                'right_shoulder': (30.0, 80.0),   # arm swings low to high
                'right_elbow': (130.0, 170.0),    # relatively straight
                'right_hip': (150.0, 180.0),      # upright
                'right_knee': (150.0, 180.0),     # standing
            },
        )

    @classmethod
    def all_templates(cls) -> Dict[str, ActionTemplate]:
        """Get all built-in templates."""
        return {
            'overhead_clear': cls.overhead_clear(),
            'smash': cls.smash(),
            'forehand_drive': cls.forehand_drive(),
            'backhand_drive': cls.backhand_drive(),
            'drop_shot': cls.drop_shot(),
            'lunge_step': cls.lunge_step(),
            'ready_stance': cls.ready_stance(),
            'serve_forehand': cls.serve_forehand(),
        }

    @classmethod
    def detect_action_type(
        cls,
        angles: Dict[str, Optional[float]],
        arm_velocity: float,
        templates: Optional[Dict[str, ActionTemplate]] = None,
    ) -> Optional[Tuple[str, float]]:
        """Detect which action best matches current angles.

        Args:
            angles: Current joint angles
            arm_velocity: Current peak arm angular velocity
            templates: Templates to match against (defaults to all built-in)

        Returns:
            (action_name, confidence) or None
        """
        if templates is None:
            templates = cls.all_templates()

        best_match = None
        best_score = 0.0

        for name, template in templates.items():
            if not template.ideal_angles:
                continue

            match_count = 0
            total = 0
            for joint, (min_a, max_a) in template.ideal_angles.items():
                angle = angles.get(joint)
                if angle is None:
                    continue
                total += 1
                if min_a <= angle <= max_a:
                    match_count += 1
                else:
                    # Partial credit for being close
                    dist = min(abs(angle - min_a), abs(angle - max_a))
                    if dist < 15:
                        match_count += 0.5

            if total == 0:
                continue

            score = match_count / total

            # Boost overhead actions when arm velocity is high
            if name in ('smash', 'overhead_clear') and arm_velocity > 200:
                score *= 1.2
            elif name == 'ready_stance' and arm_velocity < 50:
                score *= 1.1

            score = min(1.0, score)

            if score > best_score:
                best_score = score
                best_match = name

        if best_match and best_score >= 0.5:
            return best_match, best_score

        return None
