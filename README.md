# OCR 文本提取工具

## 当前工作流

现在的流程改成了三阶段：

1. 高频抓取
   - screen 模式按 `screen.interval_sec` 高频截图
   - video 模式按 `video.sample_every_n_frames` 抽帧
   - 抓到的每一张正文区图片都会保存到当前会话目录下的 `captured_raw`

2. 停止后筛选
   - 你主动点击“停止并处理”后，程序停止抓取
   - 对已保存图片做两类筛选：
     - 无文字内容
     - 完全重复内容
   - 通过筛选的图片会复制到 `captured_filtered`

3. OCR 与导出
   - 对筛选后的全部图片统一做 OCR
   - 结果写到 `output_dir`
   - 每次输出文件名都使用处理时间，例如：
     - `20260401_153012.txt`
     - `20260401_153012.jsonl`
   - 不会覆盖旧结果

## 目录结构

一次运行后，典型目录如下：

```text
output_dir/
  20260401_153012.txt
  20260401_153012.jsonl
  sessions/
    20260401_153012/
      capture_index.jsonl
      captured_raw/
      captured_filtered/
```

如果配置了 `debug_dir`，会话目录会建在 `debug_dir` 下。

## 启动 GUI

```bash
python main.py gui --config config.json
```

## 直接命令行运行

### screen 模式

```bash
python main.py run --config config.json --source screen --max-frames 300
```

### video 模式

```bash
python main.py run --config config.json --source video --video your_video.mp4
```

## 配置说明

### 高频抓取

- `screen.interval_sec`
  - 控制 screen 模式截图频率
  - 默认 `0.03` 秒一次

- `video.sample_every_n_frames`
  - 控制视频抽帧步长
  - 默认 `1`，即每帧都抓

### OCR

- `ocr.scale`
  - 预处理放大倍数
- `ocr.min_score`
  - OCR 最低置信度
- `ocr.similarity_threshold`
  - 文本去重阈值

## 使用建议

如果目标是尽量保留快进时出现的每一句：

1. screen 模式下把 `interval_sec` 维持在 `0.02 ~ 0.05`
2. 快进抓取时不要让窗口被遮挡
3. 先抓，再停，再 OCR，不要边抓边 OCR
4. 最终以会话目录中的 `captured_raw` 和 `captured_filtered` 进行核查
