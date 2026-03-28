from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Tuple

Rect = Tuple[int, int, int, int]


@dataclass
class ScreenConfig:
    interval_sec: float = 0.25
    diff_threshold: float = 2.0
    save_changed_frames_only: bool = True


@dataclass
class VideoConfig:
    sample_every_n_frames: int = 1
    diff_threshold: float = 2.0
    save_changed_frames_only: bool = True


@dataclass
class OCRConfig:
    lang: str = "ch"
    use_gpu: bool = False
    scale: float = 2.0
    binarize: bool = False
    invert: bool = False
    min_text_len: int = 2
    min_score: float = 0.55
    similarity_threshold: int = 94


@dataclass
class AppConfig:
    output_dir: str = "outputs"
    debug_dir: Optional[str] = None
    speaker_roi: Optional[Rect] = None
    text_roi: Optional[Rect] = None
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            output_dir=data.get("output_dir", "outputs"),
            debug_dir=data.get("debug_dir"),
            speaker_roi=_tuple_or_none(data.get("speaker_roi")),
            text_roi=_tuple_or_none(data.get("text_roi")),
            screen=ScreenConfig(**data.get("screen", {})),
            video=VideoConfig(**data.get("video", {})),
            ocr=OCRConfig(**data.get("ocr", {})),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")


def _tuple_or_none(value):
    if value is None:
        return None
    if len(value) != 4:
        raise ValueError("ROI 必须包含 4 个整数: x, y, w, h")
    return tuple(int(v) for v in value)
