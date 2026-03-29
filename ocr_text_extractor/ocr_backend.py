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
    def __init__(self, lang: str = "ch") -> None:
        if PaddleOCR is None:
            raise RuntimeError(
                "PaddleOCR 未安装。请先安装依赖，例如：\n"
                "pip install paddleocr paddlepaddle opencv-python mss rapidfuzz numpy"
            ) from OCR_IMPORT_ERROR
        self.ocr = PaddleOCR(use_angle_cls=False, lang=lang)

    def recognize(self, image: np.ndarray) -> Tuple[str, float]:
        result = self.ocr.predict(image)
        if not result:
            return "", 0.0

        lines: List[str] = []
        scores: List[float] = []
        for item in result:
            # 新版 PaddleOCR（dict）
            if isinstance(item, dict):
                texts = item.get("rec_texts", [])
                scs = item.get("rec_scores", [])
                for t, s in zip(texts, scs):
                    t = normalize_text(str(t))
                    if t:
                        lines.append(t)
                        scores.append(float(s))

            # 旧版 PaddleOCR（兼容）
            elif isinstance(item, (list, tuple)):
                for sub in item:
                    if (
                        isinstance(sub, (list, tuple))
                        and len(sub) >= 2
                        and isinstance(sub[1], (list, tuple))
                        and len(sub[1]) >= 2
                    ):
                        t = normalize_text(str(sub[1][0]))
                        s = float(sub[1][1])
                        if t:
                            lines.append(t)
                            scores.append(s)

        if not lines:
            return "", 0.0
        merged = " ".join(lines).strip()
        avg_score = float(sum(scores) / len(scores)) if scores else 0.0
        return merged, avg_score
