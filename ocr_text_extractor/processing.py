from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np
from rapidfuzz import fuzz

from .config import Rect
from .utils import format_record, is_ui_noise, sanitize_dialogue, sanitize_speaker


@dataclass
class OCRRecord:
    index: int
    timestamp_sec: float
    source: str
    speaker: str
    text: str
    score: float
    frame_index: Optional[int] = None
    diff_score: Optional[float] = None
    image_name: Optional[str] = None


@dataclass
class CapturedFrame:
    frame_index: int
    timestamp_sec: float
    image_path: Path
    source: str
    diff_score: Optional[float] = None


class ImagePreprocessor:
    def __init__(self, scale: float = 2.0, binarize: bool = False, invert: bool = False) -> None:
        self.scale = scale
        self.binarize = binarize
        self.invert = invert

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return image

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if self.scale and self.scale != 1.0:
            gray = cv2.resize(
                gray,
                None,
                fx=self.scale,
                fy=self.scale,
                interpolation=cv2.INTER_CUBIC,
            )

        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        if self.binarize:
            gray = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                11,
            )

        if self.invert:
            gray = 255 - gray

        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class Deduplicator:
    def __init__(self, similarity_threshold: int = 94, window_size: int = 5) -> None:
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size
        self.recent: List[str] = []

    def is_duplicate(self, text: str) -> bool:
        if not text:
            return True
        for prev in self.recent[-self.window_size :]:
            if fuzz.ratio(text, prev) >= self.similarity_threshold:
                return True
        self.recent.append(text)
        if len(self.recent) > self.window_size * 3:
            self.recent = self.recent[-self.window_size :]
        return False


class OutputWriter:
    def __init__(self, out_dir: Path, run_name: str) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.raw_path = self.out_dir / f"{run_name}.jsonl"
        self.clean_path = self.out_dir / f"{run_name}.txt"
        self.raw_fh = self.raw_path.open("w", encoding="utf-8")
        self.clean_fh = self.clean_path.open("w", encoding="utf-8")

    def write(self, record: OCRRecord) -> None:
        self.raw_fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        line = f"[{record.speaker}] {record.text}" if record.speaker else record.text
        self.clean_fh.write(line + "\n")
        self.raw_fh.flush()
        self.clean_fh.flush()

    def close(self) -> None:
        self.raw_fh.close()
        self.clean_fh.close()


class FrameFilter:
    def __init__(
        self,
        min_text_mask_ratio: float = 0.003,
        min_edge_ratio: float = 0.005,
        hash_size: int = 8,
    ) -> None:
        self.min_text_mask_ratio = min_text_mask_ratio
        self.min_edge_ratio = min_edge_ratio
        self.hash_size = hash_size
        self._seen_hashes: set[str] = set()

    def looks_like_text(self, image: np.ndarray) -> bool:
        if image is None or image.size == 0:
            return False
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        h, w = gray.shape[:2]
        if h == 0 or w == 0:
            return False

        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        white_ratio = min(float((binary > 0).mean()), float((binary == 0).mean()))
        edges = cv2.Canny(blur, 50, 150)
        edge_ratio = float((edges > 0).mean())
        return white_ratio >= self.min_text_mask_ratio and edge_ratio >= self.min_edge_ratio

    def compute_hash(self, image: np.ndarray) -> str:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        resized = cv2.resize(gray, (self.hash_size, self.hash_size), interpolation=cv2.INTER_AREA)
        avg = float(resized.mean())
        bits = (resized >= avg).astype(np.uint8).flatten()
        return hashlib.sha1(bits.tobytes()).hexdigest()

    def keep(self, image: np.ndarray) -> tuple[bool, str]:
        if not self.looks_like_text(image):
            return False, "no_text"
        img_hash = self.compute_hash(image)
        if img_hash in self._seen_hashes:
            return False, "duplicate"
        self._seen_hashes.add(img_hash)
        return True, "ok"


class BatchCaptureSession:
    def __init__(self, root_dir: Path, source_name: str) -> None:
        self.root_dir = root_dir
        self.source_name = source_name
        self.raw_dir = self.root_dir / "captured_raw"
        self.filtered_dir = self.root_dir / "captured_filtered"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.filtered_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.root_dir / "capture_index.jsonl"
        self.meta_fh = self.meta_path.open("w", encoding="utf-8")
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    def save_frame(self, frame_index: int, timestamp_sec: float, image: np.ndarray, diff_score: Optional[float] = None) -> CapturedFrame:
        filename = f"{frame_index:06d}_{int(timestamp_sec * 1000):010d}.png"
        path = self.raw_dir / filename
        cv2.imwrite(str(path), image)
        rec = {
            "frame_index": frame_index,
            "timestamp_sec": round(timestamp_sec, 6),
            "source": self.source_name,
            "diff_score": diff_score,
            "image_name": filename,
        }
        self.meta_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.meta_fh.flush()
        self._count += 1
        return CapturedFrame(frame_index, timestamp_sec, path, self.source_name, diff_score)

    def close(self) -> None:
        self.meta_fh.close()


class BatchPostProcessor:
    def __init__(
        self,
        ocr_backend,
        preprocessor: ImagePreprocessor,
        deduplicator: Deduplicator,
        output_dir: Path,
        run_name: str,
        speaker_roi: Optional[Rect],
        min_text_len: int = 2,
        min_score: float = 0.55,
        filterer: Optional[FrameFilter] = None,
    ) -> None:
        self.ocr_backend = ocr_backend
        self.preprocessor = preprocessor
        self.deduplicator = deduplicator
        self.output_dir = output_dir
        self.run_name = run_name
        self.speaker_roi = speaker_roi
        self.min_text_len = min_text_len
        self.min_score = min_score
        self.filterer = filterer or FrameFilter()

    def _copy_filtered(self, src: Path, dst_dir: Path) -> Path:
        dst = dst_dir / src.name
        if src.resolve() != dst.resolve():
            image = cv2.imread(str(src))
            if image is not None:
                cv2.imwrite(str(dst), image)
        return dst

    def filter_frames(self, frames: Iterable[CapturedFrame], filtered_dir: Path, log=None) -> List[CapturedFrame]:
        kept: List[CapturedFrame] = []
        filtered_dir.mkdir(parents=True, exist_ok=True)
        for frame in frames:
            image = cv2.imread(str(frame.image_path))
            if image is None:
                if log:
                    log(f"筛除 {frame.image_path.name}: 读取失败")
                continue
            keep, reason = self.filterer.keep(image)
            if not keep:
                if log:
                    log(f"筛除 {frame.image_path.name}: {reason}")
                continue
            dst = self._copy_filtered(frame.image_path, filtered_dir)
            kept.append(CapturedFrame(frame.frame_index, frame.timestamp_sec, dst, frame.source, frame.diff_score))
        return kept

    def run_ocr(self, frames: Iterable[CapturedFrame], log=None) -> tuple[Path, Path, int]:
        writer = OutputWriter(self.output_dir, self.run_name)
        record_index = 0
        try:
            for frame in frames:
                image = cv2.imread(str(frame.image_path))
                if image is None:
                    continue

                speaker = ""
                if self.speaker_roi is not None:
                    speaker_crop = crop(image, self.speaker_roi)
                    speaker_img = self.preprocessor(speaker_crop)
                    speaker, _ = self.ocr_backend.recognize(speaker_img)
                    speaker = sanitize_speaker(speaker)

                text_img = self.preprocessor(image)
                text, score = self.ocr_backend.recognize(text_img)
                text = sanitize_dialogue(text)

                if len(text) < self.min_text_len:
                    continue
                if score < self.min_score:
                    continue
                if is_ui_noise(text):
                    continue
                key = f"{speaker}|{text}" if speaker else text
                if self.deduplicator.is_duplicate(key):
                    continue

                record = OCRRecord(
                    index=record_index,
                    timestamp_sec=round(frame.timestamp_sec, 3),
                    source=frame.source,
                    speaker=speaker,
                    text=text,
                    score=round(score, 4),
                    frame_index=frame.frame_index,
                    diff_score=round(frame.diff_score, 4) if frame.diff_score is not None else None,
                    image_name=frame.image_path.name,
                )
                writer.write(record)
                if log:
                    log(format_record(record))
                record_index += 1
        finally:
            writer.close()
        return writer.clean_path, writer.raw_path, record_index


def crop(frame: np.ndarray, roi: Rect) -> np.ndarray:
    if frame is None or frame.size == 0:
        return np.empty((0, 0, 3), dtype=np.uint8)

    x, y, w, h = roi
    fh, fw = frame.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(fw, x + w)
    y2 = min(fh, y + h)

    if x2 <= x1 or y2 <= y1:
        if frame.ndim == 2:
            return np.empty((0, 0), dtype=frame.dtype)
        return np.empty((0, 0, frame.shape[2]), dtype=frame.dtype)

    return frame[y1:y2, x1:x2].copy()

