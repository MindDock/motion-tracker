"""Chinese/Unicode text rendering on OpenCV frames via PIL."""

from typing import Tuple, Optional
import numpy as np
import cv2
import os

from PIL import Image, ImageDraw, ImageFont

# Font search order for Chinese support on macOS / Linux / Windows
_FONT_SEARCH_PATHS = [
    # macOS
    '/System/Library/Fonts/PingFang.ttc',
    '/System/Library/Fonts/STHeiti Medium.ttc',
    '/System/Library/Fonts/STHeiti Light.ttc',
    '/Library/Fonts/Arial Unicode.ttf',
    # Linux
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    # Windows
    'C:/Windows/Fonts/msyh.ttc',
    'C:/Windows/Fonts/simsun.ttc',
]

_font_cache: dict = {}


def _find_cjk_font() -> Optional[str]:
    """Find an available CJK font on the system."""
    for path in _FONT_SEARCH_PATHS:
        if os.path.exists(path):
            return path
    return None


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    """Get a cached PIL font at the given size."""
    if size in _font_cache:
        return _font_cache[size]

    font_path = _find_cjk_font()
    if font_path:
        font = ImageFont.truetype(font_path, size)
    else:
        font = ImageFont.load_default()

    _font_cache[size] = font
    return font


def put_chinese_text(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
    font_size: int = 20,
    bg_color: Optional[Tuple[int, int, int]] = None,
    bg_padding: int = 4,
) -> np.ndarray:
    """Draw Unicode/Chinese text on an OpenCV frame.

    Args:
        frame: BGR OpenCV image (modified in-place and returned)
        text: Text string (supports Chinese, English, mixed)
        position: (x, y) top-left corner of the text
        color: BGR text color
        font_size: Font size in pixels
        bg_color: Optional background rectangle color (BGR)
        bg_padding: Padding around text for background rectangle

    Returns:
        The frame with text drawn on it
    """
    # Convert BGR -> RGB for PIL
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(font_size)

    # Get text bounding box
    bbox = draw.textbbox(position, text, font=font)

    # Draw background if requested
    if bg_color is not None:
        bg_rgb = (bg_color[2], bg_color[1], bg_color[0])  # BGR -> RGB
        draw.rectangle(
            [
                bbox[0] - bg_padding,
                bbox[1] - bg_padding,
                bbox[2] + bg_padding,
                bbox[3] + bg_padding,
            ],
            fill=bg_rgb,
        )

    # Draw text (convert BGR color to RGB)
    rgb_color = (color[2], color[1], color[0])
    draw.text(position, text, font=font, fill=rgb_color)

    # Convert RGB -> BGR back to OpenCV
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    np.copyto(frame, result)
    return frame


def get_text_size(text: str, font_size: int = 20) -> Tuple[int, int]:
    """Get the pixel dimensions of rendered text.

    Returns:
        (width, height) in pixels
    """
    font = _get_font(font_size)
    dummy = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]
