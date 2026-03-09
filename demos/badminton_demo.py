#!/usr/bin/env python3
"""Badminton motion analysis demo.

Analyze badminton video (file or camera) for:
- Action detection (smash, clear, drive, lunge, etc.)
- Highlight / exciting moment detection
- Real-time motion correction feedback

Usage:
    # Analyze a video file
    python demos/badminton_demo.py --video path/to/badminton.mp4

    # Live camera analysis
    python demos/badminton_demo.py --camera 0

    # Analyze video and export highlights
    python demos/badminton_demo.py --video match.mp4 --export-highlights

Controls (interactive mode):
    q - Quit
    h - Show/hide highlight markers
    c - Toggle correction overlay
    t - Cycle through action templates for comparison
    s - Save screenshot
    r - Start/stop recording for template creation
    space - Pause/resume (video file only)
"""

import sys
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backends.mediapipe_backend import MediaPipeBackend
from src.core.video_processor import VideoProcessor, VideoSource, FrameContext
from src.core.angle_calculator import AngleCalculator
from src.core.analysis_logger import AnalysisLogger
from src.visualization.skeleton_renderer import SkeletonRenderer
from src.visualization.text_renderer import put_chinese_text
from src.applications.badminton.analyzer import BadmintonAnalyzer
from src.applications.badminton.actions import BadmintonActions


class BadmintonDemo:
    """Interactive badminton analysis demo."""

    def __init__(self, source, racket_hand='right', skip_frames=0, log_file: str = None):
        self.source = source
        self.racket_hand = racket_hand
        self.video_source = VideoSource(source)
        self.estimator = MediaPipeBackend(
            model_complexity=1,
            static_image_mode=isinstance(source, str),
        )
        self.renderer = SkeletonRenderer()
        self.calculator = AngleCalculator(use_3d=True)

        # Build log file name if not given
        if log_file is None:
            ts = int(time.time())
            src_name = Path(source).stem if isinstance(source, str) else "camera"
            log_file = f"logs/badminton_{src_name}_{ts}.jsonl"

        self.logger = AnalysisLogger(log_file=log_file, console=True, frame_interval=30)
        self.analyzer = BadmintonAnalyzer(racket_hand=racket_hand, logger=self.logger)
        self.skip_frames = skip_frames

        # UI state
        self.show_corrections = True
        self.show_highlights_bar = True
        self.paused = False
        self.selected_template_idx = 0
        self.template_names = list(BadmintonActions.all_templates().keys())

        # Recording state for custom templates
        self.recording = False
        self.recorded_angles = []
        self.record_start_time = 0.0

    # Action name mapping for display
    ACTION_NAMES_CN = {
        'overhead_clear': '高远球',
        'smash': '扣杀',
        'forehand_drive': '正手平抽',
        'backhand_drive': '反手平抽',
        'drop_shot': '吊球',
        'lunge_step': '弓步上网',
        'ready_stance': '准备姿势',
        'serve_forehand': '正手发球',
    }

    def run(self, export_highlights: bool = False):
        """Run the demo."""
        print("=" * 60)
        print("动作追踪 - 羽毛球动作分析")
        print("=" * 60)

        if not self.video_source.open():
            print("[错误] 无法打开视频源")
            return 1

        if not self.estimator.initialize():
            print("[错误] 无法初始化姿态估计器")
            return 1

        self.analyzer.velocity_analyzer.set_fps(self.video_source.source_fps)

        source_str = str(self.source)
        self.logger.session_start(
            source=source_str,
            fps=self.video_source.source_fps,
            total_frames=self.video_source.total_frames,
            racket_hand=self.racket_hand,
        )

        print(f"视频帧率: {self.video_source.source_fps:.1f}")
        if self.video_source.is_file:
            print(f"总帧数: {self.video_source.total_frames}")
        print("\n操作: q=退出, c=动作指正, h=精彩瞬间, t=切换模板, s=截图, r=录制, 空格=暂停")
        print("=" * 60)

        try:
            self._process_loop(export_highlights)
        finally:
            self.video_source.release()
            cv2.destroyAllWindows()
            self.estimator.release()
            summary = self.analyzer.get_summary()
            self.logger.session_end(summary=summary)
            self.logger.close()

        return 0

    def _process_loop(self, export_highlights: bool):
        frame_index = 0
        total = self.video_source.total_frames if self.video_source.is_file else 0

        while True:
            if not self.paused:
                ret, frame = self.video_source.read()
                if not ret or frame is None:
                    break

                if not self.video_source.is_file:
                    frame = cv2.flip(frame, 1)

                # Skip frames for faster processing
                if self.skip_frames > 0 and frame_index % (self.skip_frames + 1) != 0:
                    frame_index += 1
                    continue

                timestamp_ms = frame_index * (1000.0 / self.video_source.source_fps)

                # Pose estimation
                pose_result = self.estimator.process_frame(frame)

                if pose_result and pose_result.is_valid():
                    # Update analyzer
                    self.analyzer.update(pose_result, frame_index, timestamp_ms)

                    # Calculate angles for display
                    angles = self.calculator.calculate_all_angles(pose_result)

                    # Render skeleton
                    frame = self.renderer.render(frame, pose_result, angles)

                    # Recording
                    if self.recording:
                        self.recorded_angles.append(angles)
                        self._draw_recording_indicator(frame)

                    # Draw action detection
                    self._draw_action_info(frame)

                    # Draw corrections
                    if self.show_corrections and pose_result:
                        corrections = self.analyzer.generate_corrections(pose_result)
                        self._draw_corrections(frame, corrections)

                # Draw progress bar for video files
                if self.video_source.is_file and total > 0:
                    self._draw_progress_bar(frame, frame_index, total)

                # Draw FPS
                self._draw_fps_counter(frame, frame_index)

                frame_index += 1
                self._last_frame = frame
            else:
                frame = self._last_frame.copy() if hasattr(self, '_last_frame') else np.zeros((720, 1280, 3), dtype=np.uint8)
                put_chinese_text(frame, "已暂停", (frame.shape[1] // 2 - 60, 15),
                                 color=(0, 255, 255), font_size=36, bg_color=(0, 0, 0))

            cv2.imshow('Badminton Analysis', frame)  # macOS window title doesn't support CJK well, keep English
            key = cv2.waitKey(1) & 0xFF
            if self._handle_key(key):
                break

        # Post-processing
        self._print_summary()

        if export_highlights:
            self._export_highlights()

    def _draw_action_info(self, frame):
        action = self.analyzer.current_action
        if action is None:
            return

        display_name = self.ACTION_NAMES_CN.get(action.action_name, action.action_name)
        confidence_pct = action.confidence * 100

        text = f"动作: {display_name}  {confidence_pct:.0f}%"
        color = (0, 255, 0) if action.confidence > 0.7 else (0, 200, 255)

        put_chinese_text(frame, text, (10, 10), color=color, font_size=24, bg_color=(0, 0, 0))

    # Joint name mapping for overlay display
    JOINT_NAMES_CN = {
        'right_shoulder': '持拍肩', 'right_elbow': '持拍肘', 'right_wrist': '持拍腕',
        'left_shoulder': '非持拍肩', 'left_elbow': '非持拍肘', 'left_wrist': '非持拍腕',
        'right_hip': '右髋', 'left_hip': '左髋',
        'right_knee': '右膝', 'left_knee': '左膝',
        'right_ankle': '右踝', 'left_ankle': '左踝',
    }

    def _draw_corrections(self, frame, corrections):
        if not corrections:
            if self.analyzer.current_action:
                put_chinese_text(frame, "姿势良好", (10, 42),
                                 color=(0, 255, 0), font_size=18, bg_color=(0, 0, 0))
            return

        y = 42
        for i, c in enumerate(corrections[:5]):
            if c.severity == 'error':
                color = (0, 0, 255)
                marker = "!!"
            elif c.severity == 'warning':
                color = (0, 165, 255)
                marker = "!"
            else:
                color = (200, 200, 200)
                marker = "·"

            joint_cn = self.JOINT_NAMES_CN.get(c.joint, c.joint)
            if c.current_angle is not None and c.ideal_range is not None:
                text = f"{marker} {joint_cn}: {c.current_angle:.0f}° (标准 {c.ideal_range[0]:.0f}°-{c.ideal_range[1]:.0f}°)"
            else:
                text = f"{marker} {c.message[:50]}"

            put_chinese_text(frame, text, (10, y), color=color, font_size=16, bg_color=(0, 0, 0))
            y += 24

    def _draw_progress_bar(self, frame, current, total):
        h, w = frame.shape[:2]
        bar_h = 6
        bar_y = h - bar_h - 2
        progress = current / max(total, 1)

        cv2.rectangle(frame, (0, bar_y), (w, h), (40, 40, 40), -1)
        cv2.rectangle(frame, (0, bar_y), (int(w * progress), h), (0, 180, 0), -1)

        # Time display
        current_sec = current / self.video_source.source_fps
        total_sec = total / self.video_source.source_fps
        time_text = f"{self._format_time(current_sec)} / {self._format_time(total_sec)}"
        cv2.putText(frame, time_text, (w - 160, bar_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def _draw_fps_counter(self, frame, frame_index):
        h, w = frame.shape[:2]
        cv2.putText(frame, f"F: {frame_index}", (w - 120, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    def _draw_recording_indicator(self, frame):
        elapsed = time.time() - self.record_start_time
        frames = len(self.recorded_angles)
        w = frame.shape[1]
        text = f"录制中  {frames}帧  {elapsed:.1f}秒"
        put_chinese_text(frame, text, (w - 260, 10), color=(0, 0, 255), font_size=20, bg_color=(0, 0, 0))
        if int(elapsed * 3) % 2 == 0:
            cv2.circle(frame, (w - 275, 22), 8, (0, 0, 255), -1)

    def _handle_key(self, key) -> bool:
        """Handle keyboard input. Returns True to quit."""
        if key == ord('q'):
            return True
        elif key == ord('c'):
            self.show_corrections = not self.show_corrections
            print(f"动作指正: {'开启' if self.show_corrections else '关闭'}")
        elif key == ord('h'):
            self.show_highlights_bar = not self.show_highlights_bar
            print(f"精彩瞬间: {'开启' if self.show_highlights_bar else '关闭'}")
        elif key == ord('t'):
            self.selected_template_idx = (self.selected_template_idx + 1) % len(self.template_names)
            name = self.template_names[self.selected_template_idx]
            cn_name = self.ACTION_NAMES_CN.get(name, name)
            print(f"当前对比模板: {cn_name}")
        elif key == ord('s'):
            if hasattr(self, '_last_frame'):
                path = f"badminton_{int(time.time())}.png"
                cv2.imwrite(path, self._last_frame)
                print(f"截图已保存: {path}")
        elif key == ord('r'):
            if self.recording:
                self.recording = False
                if self.recorded_angles:
                    self._save_recorded_template()
                self.recorded_angles = []
            else:
                self.recording = True
                self.recorded_angles = []
                self.record_start_time = time.time()
                print("开始录制...再次按 'r' 停止")
        elif key == ord(' ') and self.video_source.is_file:
            self.paused = not self.paused
        return False

    def _save_recorded_template(self):
        """Save recorded angles as a custom template."""
        from src.core.sport_analyzer import ActionTemplate
        name = f"custom_{int(time.time())}"
        template = ActionTemplate(
            name=name,
            description=f"自定义录制模板 ({len(self.recorded_angles)} 帧)",
            key_joints=BadmintonActions.ALL_KEY_JOINTS,
        )
        template.set_from_recording(self.recorded_angles)
        self.analyzer.register_template(template)
        self.template_names.append(name)
        print(f"模板已保存: {name} ({len(self.recorded_angles)} 帧)")

    def _print_summary(self):
        summary = self.analyzer.get_summary()
        highlights = self.analyzer.get_highlights(top_n=10)

        print("\n" + "=" * 60)
        print("分析报告")
        print("=" * 60)
        print(f"检测到动作总数: {summary['total_actions_detected']}")

        if summary['action_counts']:
            print("\n动作统计:")
            for action, count in sorted(summary['action_counts'].items(),
                                         key=lambda x: x[1], reverse=True):
                cn_name = self.ACTION_NAMES_CN.get(action, action)
                print(f"  {cn_name:15s}: {count} 次")

        if highlights:
            print(f"\n精彩瞬间 (共{len(highlights)}个):")
            for i, hl in enumerate(highlights, 1):
                time_str = self._format_time(hl.timestamp_sec)
                label = hl.label or "精彩瞬间"
                print(f"  {i:2d}. [{time_str}] {label} (得分: {hl.score:.2f})")

        print("=" * 60)

    SIGNAL_NAMES_CN = {
        'motion_intensity': '运动强度',
        'explosive_motion': '爆发力',
        'arm_swing': '挥臂速度',
        'posture_change': '体位变化',
        'smash_detection': '扣杀检测',
    }

    def _export_highlights(self):
        """Export highlight timestamps to a file."""
        highlights = self.analyzer.get_highlights(top_n=20)
        if not highlights:
            print("未检测到精彩瞬间")
            return

        path = "badminton_highlights.txt"
        with open(path, 'w') as f:
            f.write("羽毛球动作分析 - 精彩瞬间\n")
            f.write("=" * 40 + "\n\n")
            for i, hl in enumerate(highlights, 1):
                time_str = self._format_time(hl.timestamp_sec)
                label = hl.label or "精彩瞬间"
                f.write(f"{i}. [{time_str}] {label} (得分: {hl.score:.2f})\n")
                for sig_name, sig_val in hl.signals.items():
                    if sig_val > 0.1:
                        cn_sig = self.SIGNAL_NAMES_CN.get(sig_name, sig_name)
                        f.write(f"   - {cn_sig}: {sig_val:.2f}\n")
                f.write("\n")

        print(f"精彩瞬间已导出: {path}")

    @staticmethod
    def _format_time(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"


def parse_args():
    parser = argparse.ArgumentParser(description='羽毛球动作分析')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--video', type=str, help='视频文件路径')
    source.add_argument('--camera', type=int, default=None, help='摄像头设备ID')
    parser.add_argument('--racket-hand', choices=['left', 'right'], default='right',
                        help='持拍手 (默认: right)')
    parser.add_argument('--skip-frames', type=int, default=0,
                        help='每N帧处理一帧 (加速离线分析)')
    parser.add_argument('--export-highlights', action='store_true',
                        help='导出精彩瞬间到文件')
    parser.add_argument('--log-file', type=str, default=None,
                        help='分析日志输出路径 (默认: logs/badminton_<源>_<时间>.jsonl)')
    parser.add_argument('--no-console-log', action='store_true',
                        help='关闭控制台日志输出（仅写文件）')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.video:
        source = args.video
    elif args.camera is not None:
        source = args.camera
    else:
        source = 0  # default camera

    demo = BadmintonDemo(
        source=source,
        racket_hand=args.racket_hand,
        skip_frames=args.skip_frames,
        log_file=args.log_file,
    )
    if args.no_console_log:
        demo.logger.console = False

    return demo.run(export_highlights=args.export_highlights)


if __name__ == '__main__':
    sys.exit(main())
