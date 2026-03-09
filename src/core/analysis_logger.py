"""Real-time structured analysis logging.

Outputs analysis events to console and/or file as they happen.
Supports both human-readable text and machine-readable JSON Lines formats.
"""

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO


class LogLevel(Enum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3


class EventType(Enum):
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    ACTION_DETECTED = "action_detected"
    CORRECTION = "correction"
    HIGHLIGHT = "highlight"
    FRAME = "frame"
    SUMMARY = "summary"


@dataclass
class LogEvent:
    event_type: str
    timestamp_ms: float
    frame_index: int
    data: Dict[str, Any] = field(default_factory=dict)
    wall_time: str = field(default_factory=lambda: datetime.now().isoformat(timespec='milliseconds'))

    def to_json(self) -> str:
        return json.dumps({
            "event": self.event_type,
            "ts_ms": round(self.timestamp_ms, 1),
            "frame": self.frame_index,
            "wall_time": self.wall_time,
            **self.data,
        }, ensure_ascii=False)

    def to_text(self) -> str:
        ts = self._format_ts(self.timestamp_ms)
        event = self.event_type.upper().ljust(16)
        data_str = "  ".join(f"{k}={v}" for k, v in self.data.items())
        return f"[{ts}] {event} {data_str}"

    @staticmethod
    def _format_ts(ms: float) -> str:
        total_sec = int(ms / 1000)
        m, s = divmod(total_sec, 60)
        ms_part = int(ms % 1000)
        return f"{m:02d}:{s:02d}.{ms_part:03d}"


class AnalysisLogger:
    """Real-time analysis event logger.

    Emits structured events as analysis runs. Supports:
    - Console output (human-readable)
    - File output (JSON Lines, one event per line)
    - Both simultaneously

    Usage:
        logger = AnalysisLogger(log_file="analysis.jsonl", console=True)
        logger.session_start(source="match.mp4", fps=30.0)

        # Per frame:
        logger.log_frame(frame_index=100, timestamp_ms=3333.0, angles={...})
        logger.log_action(frame_index=100, timestamp_ms=3333.0, action="扣杀", confidence=0.92)
        logger.log_correction(frame_index=100, timestamp_ms=3333.0, corrections=[...])
        logger.log_highlight(frame_index=100, timestamp_ms=3333.0, label="扣杀", score=0.87)

        logger.session_end(summary={...})
    """

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

    def __init__(
        self,
        log_file: Optional[str] = None,
        console: bool = True,
        fmt: str = "text",          # "text" | "json" | "both"
        min_level: LogLevel = LogLevel.INFO,
        frame_interval: int = 30,   # Log frame events every N frames (0 = disabled)
    ):
        self.console = console
        self.fmt = fmt
        self.min_level = min_level
        self.frame_interval = frame_interval
        self._file: Optional[TextIO] = None
        self._event_count = 0
        self._session_start_wall = time.time()

        if log_file:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(path, 'w', encoding='utf-8', buffering=1)  # line-buffered

    # ── Public logging API ──────────────────────────────────────────

    def session_start(self, source: str, fps: float, **kwargs):
        event = LogEvent(
            event_type=EventType.SESSION_START.value,
            timestamp_ms=0.0,
            frame_index=0,
            data={"source": source, "fps": fps, **kwargs},
        )
        self._emit(event, prefix="")
        self._print_separator("分析开始")

    def session_end(self, summary: Dict[str, Any]):
        elapsed = time.time() - self._session_start_wall
        event = LogEvent(
            event_type=EventType.SESSION_END.value,
            timestamp_ms=0.0,
            frame_index=0,
            data={"elapsed_sec": round(elapsed, 1), **summary},
        )
        self._print_separator("分析结束")
        self._emit(event, prefix="")

    def log_frame(
        self,
        frame_index: int,
        timestamp_ms: float,
        angles: Optional[Dict[str, float]] = None,
        motion_intensity: Optional[float] = None,
    ):
        """Log per-frame metrics (throttled by frame_interval)."""
        if self.frame_interval <= 0:
            return
        if frame_index % self.frame_interval != 0:
            return

        data: Dict[str, Any] = {}
        if motion_intensity is not None:
            data["intensity"] = round(motion_intensity, 1)
        if angles:
            # Only log key angles to keep logs concise
            key_joints = ['right_shoulder', 'right_elbow', 'right_knee', 'left_knee']
            data["angles"] = {
                j: round(v, 1) for j, v in angles.items()
                if j in key_joints and v is not None
            }

        event = LogEvent(
            event_type=EventType.FRAME.value,
            timestamp_ms=timestamp_ms,
            frame_index=frame_index,
            data=data,
        )
        self._emit(event, level=LogLevel.DEBUG)

    def log_action(
        self,
        frame_index: int,
        timestamp_ms: float,
        action: str,
        confidence: float,
    ):
        cn_name = self.ACTION_NAMES_CN.get(action, action)
        event = LogEvent(
            event_type=EventType.ACTION_DETECTED.value,
            timestamp_ms=timestamp_ms,
            frame_index=frame_index,
            data={
                "action": action,
                "action_cn": cn_name,
                "confidence": round(confidence, 3),
            },
        )
        self._emit(event, prefix=f"🏸 动作识别  {cn_name} ({confidence*100:.0f}%)")

    def log_correction(
        self,
        frame_index: int,
        timestamp_ms: float,
        corrections: List[Any],   # List[CorrectionItem]
    ):
        if not corrections:
            return

        items = []
        for c in corrections:
            item: Dict[str, Any] = {
                "joint": c.joint,
                "severity": c.severity,
                "message": c.message,
            }
            if c.current_angle is not None:
                item["current_angle"] = round(c.current_angle, 1)
            if c.ideal_range:
                item["ideal_range"] = [c.ideal_range[0], c.ideal_range[1]]
            items.append(item)

        event = LogEvent(
            event_type=EventType.CORRECTION.value,
            timestamp_ms=timestamp_ms,
            frame_index=frame_index,
            data={"count": len(items), "items": items},
        )

        # Console: print each correction on its own line
        lines = []
        for c in corrections:
            severity_marker = {"error": "  !!  ", "warning": "   !  ", "info": "   ·  "}.get(c.severity, "      ")
            lines.append(f"{severity_marker}{c.message}")

        self._emit(
            event,
            prefix=f"动作指正  {len(corrections)}条",
            extra_lines=lines,
        )

    def log_highlight(
        self,
        frame_index: int,
        timestamp_ms: float,
        label: str,
        score: float,
        signals: Optional[Dict[str, float]] = None,
    ):
        ts = LogEvent._format_ts(timestamp_ms)
        event = LogEvent(
            event_type=EventType.HIGHLIGHT.value,
            timestamp_ms=timestamp_ms,
            frame_index=frame_index,
            data={
                "label": label,
                "score": round(score, 3),
                "signals": {k: round(v, 3) for k, v in (signals or {}).items() if v > 0.1},
            },
        )
        self._emit(event, prefix=f"★  精彩瞬间  {label}  得分={score:.2f}")

    def log_summary(self, summary: Dict[str, Any]):
        event = LogEvent(
            event_type=EventType.SUMMARY.value,
            timestamp_ms=0.0,
            frame_index=0,
            data=summary,
        )
        self._emit(event, prefix="")

    def close(self):
        if self._file:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── Internal ────────────────────────────────────────────────────

    def _emit(
        self,
        event: LogEvent,
        level: LogLevel = LogLevel.INFO,
        prefix: Optional[str] = None,
        extra_lines: Optional[List[str]] = None,
    ):
        if level.value < self.min_level.value:
            return

        self._event_count += 1

        # File: always JSON Lines
        if self._file:
            self._file.write(event.to_json() + "\n")

        # Console: human-readable
        if self.console:
            ts = LogEvent._format_ts(event.timestamp_ms)
            frame_tag = f"F{event.frame_index:05d}"

            if prefix is not None:
                text = prefix
            else:
                text = event.to_text()

            if text:
                print(f"  [{ts}] [{frame_tag}] {text}")
            if extra_lines:
                for line in extra_lines:
                    print(f"             {line}")

    def _print_separator(self, label: str):
        if self.console:
            wall = datetime.now().strftime("%H:%M:%S")
            print(f"\n{'─' * 55}")
            print(f"  {label}  {wall}")
            print(f"{'─' * 55}")
