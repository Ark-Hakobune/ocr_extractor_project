from __future__ import annotations

import time
from pathlib import Path
from typing import Generator, Tuple

import cv2
import numpy as np

from .config import Rect

try:
    import mss
except ImportError:
    mss = None


def union_rect(a: Rect, b: Rect | None) -> Rect:
    if b is None:
        return a
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left = min(ax, bx)
    top = min(ay, by)
    right = max(ax + aw, bx + bw)
    bottom = max(ay + ah, by + bh)
    return (left, top, right - left, bottom - top)


def to_local_rect(roi: Rect | None, base: Rect) -> Rect | None:
    if roi is None:
        return None
    x, y, w, h = roi
    bx, by, _, _ = base
    return (x - bx, y - by, w, h)


class ScreenCapture:
    def __init__(self, roi: Rect, interval_sec: float = 0.03) -> None:
        if mss is None:
            raise RuntimeError("screen 模式需要 mss：pip install mss")
        self.roi = roi
        self.interval_sec = max(0.005, float(interval_sec))

    def frames(self) -> Generator[Tuple[int, float, np.ndarray, float], None, None]:
        x, y, w, h = self.roi
        monitor = {"left": x, "top": y, "width": w, "height": h}
        index = 0
        start = time.time()
        with mss.mss() as sct:
            while True:
                shot = sct.grab(monitor)
                roi_frame = np.array(shot)[:, :, :3].copy()
                yield index, time.time() - start, roi_frame, 0.0
                index += 1
                time.sleep(self.interval_sec)


class VideoCaptureSource:
    def __init__(self, video_path: Path, roi: Rect, sample_every_n_frames: int = 1) -> None:
        self.video_path = video_path
        self.roi = roi
        self.sample_every_n_frames = max(1, sample_every_n_frames)

    def frames(self) -> Generator[Tuple[int, float, np.ndarray, float], None, None]:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {self.video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 30.0
        frame_index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % self.sample_every_n_frames != 0:
                    frame_index += 1
                    continue
                x, y, w, h = self.roi
                roi_frame = frame[y:y+h, x:x+w].copy()
                yield frame_index, frame_index / fps, roi_frame, 0.0
                frame_index += 1
        finally:
            cap.release()
