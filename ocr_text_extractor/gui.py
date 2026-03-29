from __future__ import annotations

import queue
import threading
from datetime import datetime
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from .config import AppConfig
from .ocr_backend import OCRBackend
from .processing import (
    BatchCaptureSession,
    BatchPostProcessor,
    CapturedFrame,
    Deduplicator,
    FrameFilter,
    ImagePreprocessor,
)
from .sources import ScreenCapture, VideoCaptureSource

try:
    import mss
except ImportError:
    mss = None


class AppGUI:
    def __init__(self, root: tk.Tk, config_path: Path) -> None:
        self.root = root
        self.root.title("OCR 文本提取工具")
        self.config_path = config_path
        self.config = AppConfig.load(config_path) if config_path.exists() else AppConfig()
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.stop_flag = threading.Event()

        self.mode_var = tk.StringVar(value="screen")
        self.video_var = tk.StringVar(value="")
        self.out_dir_var = tk.StringVar(value=self.config.output_dir)
        self.debug_dir_var = tk.StringVar(value=self.config.debug_dir or "")
        self.text_roi_var = tk.StringVar(value=self._fmt_roi(self.config.text_roi))
        self.speaker_roi_var = tk.StringVar(value=self._fmt_roi(self.config.speaker_roi))
        self.interval_var = tk.StringVar(value=str(self.config.screen.interval_sec))
        self.sample_every_var = tk.StringVar(value=str(self.config.video.sample_every_n_frames))
        self.similarity_var = tk.StringVar(value=str(self.config.ocr.similarity_threshold))
        self.min_score_var = tk.StringVar(value=str(self.config.ocr.min_score))

        self._build()
        self._poll_logs()

    def _build(self) -> None:
        frm = ttk.Frame(self.root, padding=10)
        frm.grid(sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        row = 0
        ttk.Label(frm, text="模式").grid(row=row, column=0, sticky="w")
        ttk.Combobox(frm, textvariable=self.mode_var, values=["screen", "video"], width=12, state="readonly").grid(row=row, column=1, sticky="ew")
        row += 1

        ttk.Label(frm, text="视频文件").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.video_var, width=60).grid(row=row, column=1, sticky="ew")
        ttk.Button(frm, text="选择", command=self._pick_video).grid(row=row, column=2, sticky="ew")
        row += 1

        ttk.Label(frm, text="输出目录").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.out_dir_var, width=60).grid(row=row, column=1, sticky="ew")
        ttk.Button(frm, text="选择", command=self._pick_output_dir).grid(row=row, column=2, sticky="ew")
        row += 1

        ttk.Label(frm, text="会话目录").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.debug_dir_var, width=60).grid(row=row, column=1, sticky="ew")
        ttk.Button(frm, text="选择", command=self._pick_debug_dir).grid(row=row, column=2, sticky="ew")
        row += 1

        ttk.Label(frm, text="正文 ROI").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.text_roi_var, width=60).grid(row=row, column=1, sticky="ew")
        ttk.Button(frm, text="框选", command=lambda: self._select_roi("text")).grid(row=row, column=2, sticky="ew")
        row += 1

        ttk.Label(frm, text="姓名 ROI").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.speaker_roi_var, width=60).grid(row=row, column=1, sticky="ew")
        ttk.Button(frm, text="框选", command=lambda: self._select_roi("speaker")).grid(row=row, column=2, sticky="ew")
        row += 1

        ttk.Label(frm, text="截屏频率(秒)").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.interval_var).grid(row=row, column=1, sticky="ew")
        row += 1

        ttk.Label(frm, text="视频抽帧步长").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.sample_every_var).grid(row=row, column=1, sticky="ew")
        row += 1

        ttk.Label(frm, text="文本去重阈值").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.similarity_var).grid(row=row, column=1, sticky="ew")
        row += 1

        ttk.Label(frm, text="最低 OCR 分数").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.min_score_var).grid(row=row, column=1, sticky="ew")
        row += 1

        btns = ttk.Frame(frm)
        btns.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(8, 8))
        ttk.Button(btns, text="保存配置", command=self._save_config).pack(side="left", padx=4)
        ttk.Button(btns, text="开始抓取", command=self._start).pack(side="left", padx=4)
        ttk.Button(btns, text="停止", command=self._stop).pack(side="left", padx=4)
        ttk.Button(btns, text="处理已有图片", command=self._process_existing_session).pack(side="left", padx=4)
        row += 1

        self.log_text = tk.Text(frm, width=100, height=24)
        self.log_text.grid(row=row, column=0, columnspan=3, sticky="nsew")
        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(row, weight=1)

    def _fmt_roi(self, roi) -> str:
        return "" if not roi else ",".join(str(v) for v in roi)

    def _parse_roi(self, text: str):
        text = text.strip()
        if not text:
            return None
        parts = [int(x.strip()) for x in text.split(",")]
        if len(parts) != 4:
            raise ValueError("ROI 需要 4 个整数，格式 x,y,w,h")
        return tuple(parts)

    def _pick_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4;*.mkv;*.avi;*.mov"), ("All", "*.*")])
        if path:
            self.video_var.set(path)

    def _pick_output_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.out_dir_var.set(path)

    def _pick_debug_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.debug_dir_var.set(path)

    def _grab_screen(self) -> np.ndarray:
        if mss is None:
            raise RuntimeError("screen 模式需要 mss")
        with mss.mss() as sct:
            mon = sct.monitors[1]
            shot = sct.grab(mon)
            return np.array(shot)[:, :, :3].copy()

    def _select_roi(self, target: str):
        try:
            if self.mode_var.get() == "video":
                video = self.video_var.get().strip()
                if not video:
                    raise RuntimeError("请先选择视频文件")
                cap = cv2.VideoCapture(video)
                ok, frame = cap.read()
                cap.release()
                if not ok:
                    raise RuntimeError("无法读取视频首帧")
                image = frame
            else:
                image = self._grab_screen()
            rect = cv2.selectROI("选择区域", image, showCrosshair=True, fromCenter=False)
            cv2.destroyAllWindows()
            x, y, w, h = map(int, rect)
            if w <= 0 or h <= 0:
                return
            value = f"{x},{y},{w},{h}"
            if target == "text":
                self.text_roi_var.set(value)
            else:
                self.speaker_roi_var.set(value)
        except Exception as exc:
            messagebox.showerror("错误", str(exc))

    def _save_config(self):
        self.config.output_dir = self.out_dir_var.get().strip() or "outputs"
        self.config.debug_dir = self.debug_dir_var.get().strip() or None
        self.config.text_roi = self._parse_roi(self.text_roi_var.get())
        self.config.speaker_roi = self._parse_roi(self.speaker_roi_var.get())
        self.config.screen.interval_sec = float(self.interval_var.get())
        self.config.video.sample_every_n_frames = int(self.sample_every_var.get())
        self.config.ocr.similarity_threshold = int(self.similarity_var.get())
        self.config.ocr.min_score = float(self.min_score_var.get())
        self.config.save(self.config_path)
        messagebox.showinfo("成功", f"配置已保存到 {self.config_path}")

    def _start(self):
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("提示", "任务正在运行")
            return
        try:
            self._save_config()
        except Exception as exc:
            messagebox.showerror("错误", str(exc))
            return
        self.stop_flag.clear()
        self.worker = threading.Thread(target=self._run_worker, daemon=True)
        self.worker.start()

    def _stop(self):
        self.stop_flag.set()
        self.log_queue.put("收到停止信号，开始收尾并处理已抓取图片。")

    def _build_processor(self, cfg: AppConfig, output_dir: Path, run_name: str) -> BatchPostProcessor:
        backend = OCRBackend(lang=cfg.ocr.lang)
        return BatchPostProcessor(
            ocr_backend=backend,
            preprocessor=ImagePreprocessor(
                scale=cfg.ocr.scale,
                binarize=cfg.ocr.binarize,
                invert=cfg.ocr.invert,
            ),
            deduplicator=Deduplicator(similarity_threshold=cfg.ocr.similarity_threshold),
            output_dir=output_dir,
            run_name=run_name,
            min_text_len=cfg.ocr.min_text_len,
            min_score=cfg.ocr.min_score,
            filterer=FrameFilter(),
        )

    def _load_raw_frames(self, session_root: Path, source_name: str) -> list[CapturedFrame]:
        raw_text_dir = session_root / "captured_raw" / "text"
        raw_speaker_dir = session_root / "captured_raw" / "speaker"
        frames: list[CapturedFrame] = []
        for text_path in sorted(raw_text_dir.glob("*.png")):
            parts = text_path.stem.split("_")
            frame_idx = int(parts[0]) if parts else 0
            ts_ms = int(parts[1]) if len(parts) > 1 else 0
            speaker_path = raw_speaker_dir / text_path.name
            if not speaker_path.exists():
                speaker_path = None
            frames.append(
                CapturedFrame(
                    frame_index=frame_idx,
                    timestamp_sec=ts_ms / 1000.0,
                    text_image_path=text_path,
                    speaker_image_path=speaker_path,
                    source=source_name,
                    diff_score=0.0,
                )
            )
        return frames

    def _run_worker(self):
        session = None
        try:
            cfg = AppConfig.load(self.config_path)
            if cfg.text_roi is None:
                raise RuntimeError("必须先设置正文 ROI")

            mode = self.mode_var.get()
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(cfg.output_dir)
            session_root = (Path(cfg.debug_dir) if cfg.debug_dir else output_dir / "sessions") / run_name
            session = BatchCaptureSession(session_root, mode)

            if mode == "screen":
                source = ScreenCapture(cfg.text_roi, cfg.speaker_roi, cfg.screen.interval_sec)
                self.log_queue.put(f"开始高频抓取 screen，会话目录={session_root}")
            else:
                video = self.video_var.get().strip()
                if not video:
                    raise RuntimeError("video 模式需要选择视频文件")
                source = VideoCaptureSource(Path(video), cfg.text_roi, cfg.speaker_roi, cfg.video.sample_every_n_frames)
                self.log_queue.put(f"开始抓取 video: {video}，会话目录={session_root}")

            for frame in source.frames():
                if self.stop_flag.is_set():
                    break
                saved = session.save_frame(
                    frame_index=frame.frame_index,
                    timestamp_sec=frame.timestamp_sec,
                    text_image=frame.text_image,
                    speaker_image=frame.speaker_image,
                    diff_score=frame.diff_score,
                )
                if saved.frame_index % 20 == 0:
                    self.log_queue.put(f"已抓取 {saved.frame_index + 1} 帧")

            session.close()
            self.log_queue.put(f"抓取结束，共保存 {session.count} 张原始正文图片，开始筛选。")

            processor = self._build_processor(cfg, output_dir, run_name)
            raw_frames = self._load_raw_frames(session_root, mode)
            kept = processor.filter_frames(
                raw_frames,
                session.filtered_text_dir,
                session.filtered_speaker_dir,
                log=self.log_queue.put,
            )
            self.log_queue.put(f"筛选完成，保留 {len(kept)} 张图片，开始 OCR。")
            txt_path, jsonl_path, record_count = processor.run_ocr(kept, log=self.log_queue.put)
            self.log_queue.put(f"OCR 完成，共输出 {record_count} 条文本。")
            self.log_queue.put(f"文本文件: {txt_path}")
            self.log_queue.put(f"结构化文件: {jsonl_path}")
            self.log_queue.put("任务结束。")

        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.log_queue.put(f"错误: {exc}")
        finally:
            if session is not None:
                try:
                    session.close()
                except Exception:
                    pass

    def _poll_logs(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_text.insert("end", str(msg) + "\n")
                self.log_text.see("end")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_logs)

    def _process_existing_session(self):
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("提示", "当前已有任务在运行")
            return

        session_dir = filedialog.askdirectory(title="选择已有抓图会话目录")
        if not session_dir:
            return

        try:
            self._save_config()
        except Exception:
            return

        self.stop_flag.clear()
        self.worker = threading.Thread(
            target=self._run_existing_session_worker,
            args=(Path(session_dir),),
            daemon=True,
        )
        self.worker.start()

    def _run_existing_session_worker(self, session_dir: Path):
        try:
            cfg = AppConfig.load(self.config_path)
            if cfg.text_roi is None:
                raise RuntimeError("必须先设置正文 ROI")

            output_dir = Path(cfg.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            run_name = session_dir.name

            raw_text_dir = session_dir / "captured_raw" / "text"
            if not raw_text_dir.exists():
                raise RuntimeError(f"未找到抓图目录: {raw_text_dir}")

            self.log_queue.put(f"开始处理已有图片，会话目录: {session_dir}")

            processor = self._build_processor(cfg, output_dir, run_name)
            raw_frames = self._load_raw_frames(session_dir, self.mode_var.get())
            self.log_queue.put(f"读取到原始图片 {len(raw_frames)} 张")

            filtered_text_dir = session_dir / "captured_filtered" / "text"
            filtered_speaker_dir = session_dir / "captured_filtered" / "speaker"
            kept = processor.filter_frames(
                raw_frames,
                filtered_text_dir,
                filtered_speaker_dir,
                log=self.log_queue.put,
            )
            self.log_queue.put(f"筛选完成，保留 {len(kept)} 张图片，开始 OCR。")
            txt_path, jsonl_path, record_count = processor.run_ocr(kept, log=self.log_queue.put)
            self.log_queue.put(f"OCR 完成，共输出 {record_count} 条文本。")
            self.log_queue.put(f"文本文件: {txt_path}")
            self.log_queue.put(f"结构化文件: {jsonl_path}")
            self.log_queue.put("任务结束。")

        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.log_queue.put(f"错误: {exc}")


def run_gui(config_path: str | Path = "config.json") -> None:
    root = tk.Tk()
    root.geometry("900x700")
    AppGUI(root, Path(config_path))
    root.mainloop()
