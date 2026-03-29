from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np

from .config import Rect

try:
    import mss
except ImportError:
    mss = None


@dataclass
class CaptureFrame:
    frame_index: int
    timestamp_sec: float
    text_image: np.ndarray
    speaker_image: Optional[np.ndarray]
    source: str
    diff_score: float = 0.0


class ScreenCapture:
    def __init__(
        self,
        text_roi: Rect,
        speaker_roi: Optional[Rect] = None,
        interval_sec: float = 0.03,
    ) -> None:
        if mss is None:
            raise RuntimeError("screen 模式需要 mss：pip install mss")
        self.text_roi = text_roi
        self.speaker_roi = speaker_roi
        self.interval_sec = interval_sec

    @staticmethod
    def _grab_roi(sct, roi: Rect) -> np.ndarray:
        x, y, w, h = roi
        monitor = {"left": int(x), "top": int(y), "width": int(w), "height": int(h)}
        shot = sct.grab(monitor)
        return np.array(shot)[:, :, :3].copy()

    def frames(self) -> Generator[CaptureFrame, None, None]:
        frame_index = 0
        start = time.time()
        with mss.mss() as sct:
            while True:
                text_image = self._grab_roi(sct, self.text_roi)
                speaker_image = self._grab_roi(sct, self.speaker_roi) if self.speaker_roi is not None else None
                yield CaptureFrame(
                    frame_index=frame_index,
                    timestamp_sec=time.time() - start,
                    text_image=text_image,
                    speaker_image=speaker_image,
                    source="screen",
                    diff_score=0.0,
                )
                frame_index += 1
                time.sleep(self.interval_sec)


class VideoCaptureSource:
    def __init__(
        self,
        video_path: Path,
        text_roi: Rect,
        speaker_roi: Optional[Rect] = None,
        sample_every_n_frames: int = 1,
    ) -> None:
        self.video_path = video_path
        self.text_roi = text_roi
        self.speaker_roi = speaker_roi
        self.sample_every_n_frames = max(1, sample_every_n_frames)

    @staticmethod
    def _crop(frame: np.ndarray, roi: Rect) -> np.ndarray:
        x, y, w, h = roi
        return frame[y : y + h, x : x + w].copy()

    def frames(self) -> Generator[CaptureFrame, None, None]:
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
                text_image = self._crop(frame, self.text_roi)
                speaker_image = self._crop(frame, self.speaker_roi) if self.speaker_roi is not None else None
                yield CaptureFrame(
                    frame_index=frame_index,
                    timestamp_sec=frame_index / fps,
                    text_image=text_image,
                    speaker_image=speaker_image,
                    source="video",
                    diff_score=0.0,
                )
                frame_index += 1
        finally:
            cap.release()
