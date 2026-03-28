from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .utils import normalize_text

try:
    from paddleocr import PaddleOCR
except ImportError as exc:
    PaddleOCR = None  # type: ignore
    OCR_IMPORT_ERROR = exc
else:
    OCR_IMPORT_ERROR = None


class OCRBackend:
    def __init__(self, lang: str = "ch", use_gpu: bool = False) -> None:
        if PaddleOCR is None:
            raise RuntimeError(
                "PaddleOCR 未安装。请先安装依赖，例如：\n"
                "pip install paddleocr paddlepaddle opencv-python mss rapidfuzz numpy"
            ) from OCR_IMPORT_ERROR
        self.ocr = PaddleOCR(use_angle_cls=False, lang=lang, use_gpu=use_gpu, show_log=False)

    def recognize(self, image: np.ndarray) -> Tuple[str, float]:
        result = self.ocr.ocr(image, cls=False)
        if not result:
            return "", 0.0

        lines: List[str] = []
        scores: List[float] = []
        for block in result:
            if not block:
                continue
            for item in block:
                if not item or len(item) < 2:
                    continue
                text, score = item[1][0], float(item[1][1])
                text = normalize_text(text)
                if text:
                    lines.append(text)
                    scores.append(score)

        if not lines:
            return "", 0.0
        merged = " ".join(lines).strip()
        avg_score = float(sum(scores) / len(scores)) if scores else 0.0
        return merged, avg_score
