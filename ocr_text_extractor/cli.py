from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .config import AppConfig
from .gui import run_gui
from .ocr_backend import OCRBackend
from .processing import BatchCaptureSession, BatchPostProcessor, Deduplicator, FrameFilter, ImagePreprocessor
from .sources import ScreenCapture, VideoCaptureSource, to_local_rect, union_rect


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OCR 文本提取工具")
    sub = parser.add_subparsers(dest="mode", required=True)

    gui = sub.add_parser("gui", help="启动图形界面")
    gui.add_argument("--config", default="config.json")

    run = sub.add_parser("run", help="按配置文件运行")
    run.add_argument("--config", default="config.json")
    run.add_argument("--source", choices=["screen", "video"], required=True)
    run.add_argument("--video", help="video 模式时的视频路径")
    run.add_argument("--max-frames", type=int, help="screen 模式下最大抓取帧数，用于命令行测试")
    return parser


def execute_from_config(config_path: str | Path, source_type: str, video_path: str | None = None, max_frames: int | None = None) -> int:
    cfg = AppConfig.load(config_path)
    if cfg.text_roi is None:
        raise RuntimeError("必须在配置文件中设置 text_roi")

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg.output_dir)
    session_root = (Path(cfg.debug_dir) if cfg.debug_dir else output_dir / "sessions") / run_name
    session = BatchCaptureSession(session_root, source_type)
    try:
        if source_type == "screen":
            capture_roi = union_rect(cfg.text_roi, cfg.speaker_roi)
            local_speaker_roi = to_local_rect(cfg.speaker_roi, capture_roi)
            source = ScreenCapture(capture_roi, cfg.screen.interval_sec)
        else:
            if not video_path:
                raise RuntimeError("video 模式必须提供 --video")
            local_speaker_roi = cfg.speaker_roi
            source = VideoCaptureSource(Path(video_path), cfg.text_roi, cfg.video.sample_every_n_frames)

        count = 0
        for frame_index, timestamp_sec, frame, diff_score in source.frames():
            session.save_frame(frame_index, timestamp_sec, frame, diff_score)
            count += 1
            if max_frames is not None and count >= max_frames:
                break
    finally:
        session.close()

    backend = OCRBackend(lang=cfg.ocr.lang, use_gpu=cfg.ocr.use_gpu)
    processor = BatchPostProcessor(
        ocr_backend=backend,
        preprocessor=ImagePreprocessor(scale=cfg.ocr.scale, binarize=cfg.ocr.binarize, invert=cfg.ocr.invert),
        deduplicator=Deduplicator(similarity_threshold=cfg.ocr.similarity_threshold),
        output_dir=output_dir,
        run_name=run_name,
        speaker_roi=local_speaker_roi,
        min_text_len=cfg.ocr.min_text_len,
        min_score=cfg.ocr.min_score,
        filterer=FrameFilter(),
    )

    raw_frames = []
    for image_path in sorted(session.raw_dir.glob("*.png")):
        parts = image_path.stem.split("_")
        frame_idx = int(parts[0]) if parts else 0
        ts_ms = int(parts[1]) if len(parts) > 1 else 0
        raw_frames.append((frame_idx, ts_ms / 1000.0, image_path))

    frame_objs = [
        type("FrameObj", (), {
            "frame_index": idx,
            "timestamp_sec": ts,
            "image_path": path,
            "source": source_type,
            "diff_score": 0.0,
        })()
        for idx, ts, path in raw_frames
    ]
    kept = processor.filter_frames(frame_objs, session.filtered_dir)
    processor.run_ocr(kept)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "gui":
        run_gui(args.config)
        return 0
    return execute_from_config(args.config, args.source, args.video, args.max_frames)


if __name__ == "__main__":
    raise SystemExit(main())
